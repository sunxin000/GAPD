"""GAPD teacher logprob computation during rollout.

Builds gold-hint-augmented teacher prompts and computes prompt_logprobs
via the rollout vLLM servers (same model = self-distillation). Only
response-position teacher logprobs are returned, since the JSD loss
operates exclusively on response tokens.

Key design: teacher prompt is constructed at the token-id level by
inserting pre-tokenized hint block ids before the assistant_prefix ids.
This avoids calling the HuggingFace fast tokenizer at inference time,
which would cause "RuntimeError: Already borrowed" under concurrent
access from multiple AgentLoopWorker coroutines.
"""

import asyncio
import logging
from uuid import uuid4

import torch

from kbqa_r1.utils.tokenizer_lock import global_tokenizer_lock

logger = logging.getLogger(__name__)

# Cache for the tokenized hint template per tokenizer.
# Keyed by tokenizer hash to handle different tokenizers.
_hint_cache = {}


def _get_hint_token_ids(tokenizer, action_hint_text: str) -> list[int]:
    """Tokenize the hint block once and cache the result.

    Returns the token ids for:
        HINT_PREFIX + action_hint + HINT_SUFFIX
    """
    cache_key = (hash(tokenizer), action_hint_text)
    if cache_key in _hint_cache:
        return _hint_cache[cache_key]

    HINT_PREFIX = (
        "\n[Reference action sequence. "
        "Do not mention or repeat this in your response, "
        "just silently use it as guidance.]\n"
    )
    HINT_SUFFIX = "\n[End reference]\n"

    hint_text = HINT_PREFIX + action_hint_text.strip() + HINT_SUFFIX
    with global_tokenizer_lock:
        hint_ids = tokenizer.encode(hint_text, add_special_tokens=False)
    _hint_cache[cache_key] = hint_ids
    return hint_ids


def _find_sublist_index(lst: list, sub: list) -> int:
    """Find the first occurrence of sub in lst. Returns -1 if not found."""
    n, m = len(lst), len(sub)
    for i in range(n - m + 1):
        if lst[i:i + m] == sub:
            return i
    return -1


def _find_last_sublist_index(lst: list, sub: list) -> int:
    """Find the last occurrence of sub in lst. Returns -1 if not found."""
    n, m = len(lst), len(sub)
    for i in range(n - m, -1, -1):
        if lst[i:i + m] == sub:
            return i
    return -1


def _build_teacher_prompt_ids(
    prompt_ids: list[int],
    hint_ids: list[int],
    assistant_prefix_ids_list: list[list[int]],
) -> list[int]:
    """Build teacher prompt token ids by inserting hint before assistant_prefix.

    Finds the last occurrence of any assistant_prefix in prompt_ids and
    inserts hint_ids before it. This is the token-id equivalent of
    build_teacher_prompt's text-level string replacement.

    Args:
        prompt_ids: Student prompt token ids.
        hint_ids: Pre-tokenized hint block token ids.
        assistant_prefix_ids_list: List of possible assistant_prefix token id
            sequences, ordered by preference (longest first).

    Returns:
        Teacher prompt token ids with hint inserted.
    """
    # Find the last assistant_prefix (for multi-turn, inject before the
    # last assistant turn which is where the model generates)
    best_pos = -1
    for prefix_ids in assistant_prefix_ids_list:
        pos = _find_last_sublist_index(prompt_ids, prefix_ids)
        if pos != -1:
            best_pos = pos
            break  # Use first (most specific) match found

    if best_pos == -1:
        logger.warning("GAPD: Could not find assistant_prefix in prompt ids, "
                       "using unmodified prompt")
        return prompt_ids

    # Insert hint_ids before the assistant_prefix
    return prompt_ids[:best_pos] + hint_ids + prompt_ids[best_pos:]


def _truncate_action_hint(action_hint: str, max_turns: int) -> str:
    """Truncate action hint to the first max_turns actions.

    Each line in the action hint is one action (e.g., Find_relation, Merge, etc.).
    When gapd_teacher_max_turns is set, only the first N actions should be given
    as hints, so the teacher doesn't see future-turn privileged information for
    turns it won't supervise.
    """
    if max_turns <= 0:
        return action_hint
    lines = [ln for ln in action_hint.strip().split('\n') if ln.strip()]
    if len(lines) <= max_turns:
        return action_hint
    return '\n'.join(lines[:max_turns])


async def compute_gapd_teacher_logprobs(
    server_manager,
    prompt_ids: list[int],
    response_ids: list[int],
    gold_function_list,
    tokenizer,
    top_k: int = 128,
    max_length: int = 16384,
    assistant_prefix_ids_list: list[list[int]] | None = None,
    action_hint: str | None = None,
    gapd_teacher_max_turns: int = -1,
) -> tuple[torch.Tensor, torch.Tensor, list[int]] | None:
    """Compute teacher logprobs with gold-hint-augmented prompts via vLLM.

    Returns teacher data at response positions only (since JSD loss only
    uses response tokens). Shape: (resp_len, top_k) each.

    This function does NOT call the HuggingFace fast tokenizer at inference
    time, avoiding "Already borrowed" errors under concurrent access.

    Args:
        server_manager: AsyncLLMServerManager for the rollout vLLM servers.
        prompt_ids: Student prompt token ids.
        response_ids: Student response token ids.
        gold_function_list: Gold function_list (list[str] or None). Used as
            fallback when action_hint is not provided.
        tokenizer: HuggingFace tokenizer (only used for initial hint tokenization).
        top_k: Number of top-k logprobs to request from vLLM.
        max_length: Max teacher sequence length.
        assistant_prefix_ids_list: Pre-tokenized assistant prefix ids.
        action_hint: Pre-computed action hint string (e.g., from dataset's
            extra_info.action_hint). When available, this is used directly
            instead of regenerating from gold_function_list, ensuring
            consistency with the rejection-sampling pipeline.
        gapd_teacher_max_turns: Max number of turns to provide hints for.
            If > 0, only the first N actions are included in the hint.
            -1 means all actions (no truncation).

    Returns:
        (teacher_ids, teacher_logprobs, teacher_prompt_ids) at response positions,
        teacher tensors each shape (resp_len, top_k), or None if no gold data.
    """
    # Determine action hint: prefer pre-computed, fall back to function_list
    if action_hint is not None and action_hint.strip():
        hint_text = action_hint.strip()
    elif gold_function_list is not None and len(gold_function_list) > 0:
        from .teacher_prompt_builder import function_list_to_action_hint
        hint_text = function_list_to_action_hint(gold_function_list)
        if not hint_text.strip():
            return None
    else:
        return None

    # Truncate hint to only include actions for the supervised turns
    if gapd_teacher_max_turns > 0:
        hint_text = _truncate_action_hint(hint_text, gapd_teacher_max_turns)
        if not hint_text.strip():
            return None

    try:
        # 1. Tokenize hint block (cached, avoids repeated tokenizer calls)
        hint_ids = _get_hint_token_ids(tokenizer, hint_text)

        # 3. Build teacher prompt at token-id level
        if assistant_prefix_ids_list is None:
            assistant_prefix_ids_list = _get_assistant_prefix_ids(tokenizer)

        teacher_prompt_ids = _build_teacher_prompt_ids(
            prompt_ids, hint_ids, assistant_prefix_ids_list,
        )

        # 4. Concatenate teacher prompt + student response
        full_sequence_ids = teacher_prompt_ids + response_ids

        if len(full_sequence_ids) > max_length:
            available = max_length - len(teacher_prompt_ids)
            if available <= 0:
                logger.warning("GAPD: teacher prompt exceeds max_length; skipping teacher logprobs")
                return None
            full_sequence_ids = teacher_prompt_ids + response_ids[:available]

        # 5. Send to vLLM with prompt_logprobs
        sampling_params = {
            "max_tokens": 1,
            "temperature": 1.0,
            "prompt_logprobs": top_k,
        }

        teacher_output = await server_manager.generate(
            request_id=uuid4().hex,
            prompt_ids=full_sequence_ids,
            sampling_params=sampling_params,
        )

        # 6. Extract prompt_logprobs from vLLM response
        all_prompt_ids = teacher_output.extra_fields.get("prompt_ids", None)
        all_prompt_logprobs = teacher_output.extra_fields.get("prompt_logprobs", None)

        if all_prompt_ids is None or all_prompt_logprobs is None:
            logger.warning("GAPD: vLLM did not return prompt_logprobs")
            return None

        all_prompt_ids = torch.tensor(all_prompt_ids, dtype=torch.int32)
        all_prompt_logprobs = torch.tensor(all_prompt_logprobs, dtype=torch.float32)

        # 7. Extract only response-position logprobs
        resp_start = len(teacher_prompt_ids) - 1
        resp_len = len(response_ids)

        if resp_start + resp_len > all_prompt_logprobs.shape[0]:
            resp_len = all_prompt_logprobs.shape[0] - resp_start

        if resp_len <= 0:
            logger.warning("GAPD: No response positions in teacher logprobs")
            return None

        teacher_ids_resp = all_prompt_ids[resp_start:resp_start + resp_len]
        teacher_logprobs_resp = all_prompt_logprobs[resp_start:resp_start + resp_len]

        return teacher_ids_resp, teacher_logprobs_resp, teacher_prompt_ids

    except Exception as e:
        logger.warning(f"GAPD teacher logprob computation failed: {e}")
        return None


# Cache for assistant prefix token ids per tokenizer
_prefix_cache = {}


def _get_assistant_prefix_ids(tokenizer) -> list[list[int]]:
    """Get pre-tokenized assistant prefix ids (cached per tokenizer)."""
    cache_key = hash(tokenizer)
    if cache_key in _prefix_cache:
        return _prefix_cache[cache_key]

    # Common assistant prefixes for chat templates
    prefix_texts = [
        "<|im_start|>assistant\n",
        "<|im_start|>assistant",
        "<|start_header_id|>assistant<|end_header_id|>",
    ]

    result = []
    for text in prefix_texts:
        with global_tokenizer_lock:
            ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            result.append(ids)

    _prefix_cache[cache_key] = result
    return result


async def compute_gapd_teacher_logprobs_per_turn(
    server_manager,
    prompt_ids: list[int],
    response_ids: list[int],
    matched_turns: list[dict],
    tokenizer,
    top_k: int = 20,
    max_length: int = 16384,
    assistant_prefix_ids_list: list[list[int]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[dict]] | None:
    """Compute per-turn teacher logprobs for state-anchored GAPD.

    For each matched turn, builds a separate teacher prompt containing only
    that turn's hint, calls vLLM prompt_logprobs, and stitches the response-
    position teacher data into a single (resp_len, top_k) tensor.

    Args:
        server_manager: AsyncLLMServerManager for rollout vLLM servers.
        prompt_ids: Final prompt token ids (after truncation).
        response_ids: Final response token ids (after truncation).
        matched_turns: List of dicts with keys:
            gen_start: absolute position in all_ids of turn's first gen token.
            gen_end: absolute position in all_ids after turn's last gen token.
            teacher_hint: Action hint string for this turn.
        tokenizer: HuggingFace tokenizer.
        top_k: Number of top-k teacher logprobs to request.
        max_length: Max teacher sequence length.
        assistant_prefix_ids_list: Pre-tokenized assistant prefix ids.

    Returns:
        (teacher_ids, teacher_logprobs, prompt_debug). Teacher tensors each
        have shape (resp_len, top_k); prompt_debug contains per-turn teacher
        prompt ids and hints for offline inspection.
        or None if no matched turns or all calls fail.
    """
    if not matched_turns:
        return None

    if assistant_prefix_ids_list is None:
        assistant_prefix_ids_list = _get_assistant_prefix_ids(tokenizer)

    prompt_len = len(prompt_ids)
    resp_len = len(response_ids)
    all_ids = prompt_ids + response_ids

    async def _call_for_turn(mt: dict) -> tuple[int, int, torch.Tensor, torch.Tensor, dict] | None:
        """Call vLLM for a single matched turn.

        Returns: (rel_start, rel_end, teacher_ids_turn, teacher_logprobs_turn, debug)
        """
        gen_start = mt["gen_start"]
        gen_end = mt["gen_end"]
        teacher_hint = mt["teacher_hint"]

        # Convert to relative positions within response_ids
        rel_start = gen_start - prompt_len
        rel_end = gen_end - prompt_len
        if rel_start < 0 or rel_start >= resp_len:
            return None
        rel_end = min(rel_end, resp_len)
        gen_len = rel_end - rel_start
        if gen_len <= 0:
            return None

        # Build teacher prompt: everything before this turn's generation
        prompt_before_turn = all_ids[:gen_start]

        # Tokenize hint block
        hint_ids = _get_hint_token_ids(tokenizer, teacher_hint)

        # Insert hint before assistant_prefix
        teacher_prompt_ids = _build_teacher_prompt_ids(
            prompt_before_turn, hint_ids, assistant_prefix_ids_list,
        )
        hint_inserted = len(teacher_prompt_ids) == len(prompt_before_turn) + len(hint_ids)
        prompt_debug = {
            "rel_start": rel_start,
            "rel_end": rel_end,
            "gen_len": gen_len,
            "teacher_hint": teacher_hint,
            "hint_token_count": len(hint_ids),
            "prompt_before_turn_len": len(prompt_before_turn),
            "teacher_prompt_len": len(teacher_prompt_ids),
            "hint_inserted": hint_inserted,
            "teacher_prompt_ids": teacher_prompt_ids,
        }

        # Append this turn's generated tokens
        turn_gen_ids = response_ids[rel_start:rel_end]
        full_sequence_ids = teacher_prompt_ids + turn_gen_ids

        if len(full_sequence_ids) > max_length:
            available = max_length - len(teacher_prompt_ids)
            if available <= 0:
                return None
            full_sequence_ids = teacher_prompt_ids + turn_gen_ids[:available]
            gen_len = available

        sampling_params = {
            "max_tokens": 1,
            "temperature": 1.0,
            "prompt_logprobs": top_k,
        }

        try:
            teacher_output = await server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=full_sequence_ids,
                sampling_params=sampling_params,
            )
        except Exception as e:
            logger.warning(f"GAPD per-turn teacher call failed: {e}")
            return None

        all_prompt_ids = teacher_output.extra_fields.get("prompt_ids", None)
        all_prompt_logprobs = teacher_output.extra_fields.get("prompt_logprobs", None)
        if all_prompt_ids is None or all_prompt_logprobs is None:
            logger.warning("GAPD per-turn: vLLM did not return prompt_logprobs")
            return None

        all_prompt_ids = torch.tensor(all_prompt_ids, dtype=torch.int32)
        all_prompt_logprobs = torch.tensor(all_prompt_logprobs, dtype=torch.float32)

        # Extract response-position logprobs (same indexing as single-turn path)
        resp_start = len(teacher_prompt_ids) - 1
        if resp_start + gen_len > all_prompt_logprobs.shape[0]:
            gen_len = all_prompt_logprobs.shape[0] - resp_start

        if gen_len <= 0:
            return None

        teacher_ids_turn = all_prompt_ids[resp_start:resp_start + gen_len]
        teacher_logprobs_turn = all_prompt_logprobs[resp_start:resp_start + gen_len]

        prompt_debug["rel_end"] = rel_start + gen_len
        prompt_debug["gen_len"] = gen_len
        return rel_start, rel_start + gen_len, teacher_ids_turn, teacher_logprobs_turn, prompt_debug

    # Call all turns concurrently (vLLM batches internally)
    results = await asyncio.gather(*[_call_for_turn(mt) for mt in matched_turns])

    # Stitch results into full-response tensors
    teacher_ids_full = torch.zeros(resp_len, top_k, dtype=torch.int32)
    teacher_logprobs_full = torch.full((resp_len, top_k), -100.0, dtype=torch.float32)
    prompt_debug = []

    valid_count = 0
    for res in results:
        if res is None:
            continue
        t_start, t_end, t_ids, t_logprobs, debug = res
        actual_len = min(t_end - t_start, t_ids.shape[0])
        if actual_len <= 0:
            continue
        teacher_ids_full[t_start:t_start + actual_len] = t_ids[:actual_len]
        teacher_logprobs_full[t_start:t_start + actual_len] = t_logprobs[:actual_len]
        prompt_debug.append(debug)
        valid_count += 1

    if valid_count == 0:
        return None

    return teacher_ids_full, teacher_logprobs_full, prompt_debug
