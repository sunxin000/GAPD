"""
KBQA S-Expression Agent Loop for verl 0.8.0.

Replaces the old batch-synchronized SExprLLMGenerationManager.run_llm_loop()
with a per-sample async coroutine that integrates with verl's agent loop framework.

Each sample runs independently — fast SPARQL queries don't wait for slow ones.
The LLM server handles continuous batching internally.
"""

import asyncio
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, List, Optional, Set
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    register,
)
from verl.utils.ray_utils import get_event_loop

# Use the process-wide tokenizer lock shared across ALL modules to prevent
# "RuntimeError: Already borrowed" from the Rust-backed fast tokenizer.
from kbqa_r1.utils.tokenizer_lock import safe_decode as _safe_decode
from kbqa_r1.utils.tokenizer_lock import safe_encode as _safe_encode
from verl.workers.rollout.replica import TokenOutput

try:
    from kbqa_r1.gapd.gold_trace_builder import GoldTraceStep
    from kbqa_r1.gapd.gold_trace_builder import _extract_mid_set_from_result
    from kbqa_r1.gapd.state_matcher import jaccard_similarity
    from kbqa_r1.gapd.state_matcher import match_state_to_gold_step

    _STATE_MATCHING_AVAILABLE = True
except ImportError:
    _STATE_MATCHING_AVAILABLE = False

    @dataclass
    class GoldTraceStep:
        step_index: int
        prev_mid_set: Set[str]
        action_function: str
        next_mid_set: Set[str]
        per_turn_hint: str

    def match_state_to_gold_step(current_state, gold_trace):
        return None

    def jaccard_similarity(a, b):
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def _extract_mid_set_from_result(execution_result) -> Set[str]:
        mid_set = set()
        for row in getattr(execution_result, "results", []) or []:
            values = row.values() if isinstance(row, dict) else row
            for value in values:
                if isinstance(value, str) and (value.startswith("m.") or value.startswith("g.")):
                    mid_set.add(value)
        return mid_set

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@dataclass
class TurnExecutionResult:
    """Execution result for a single agent loop turn."""

    observation_text: str
    is_successful: bool
    mid_set: Set[str]
    error_message: Optional[str] = None


# Lazy-initialized shared components (per-worker process, not per-sample)
_shared_components = {}


def _get_shared_components(config, tokenizer):
    """Lazy-init heavy KBQA components once per worker process.

    RelationRetrieval loads SimCSE indices and SPARQL connections which are
    expensive.  We cache them at module level so all agent loop instances
    created by the same AgentLoopWorker reuse them.
    """
    global _shared_components
    if _shared_components:
        return _shared_components

    from kbqa_r1.sexpr.action_parser import ActionParser
    from kbqa_r1.sexpr.relation_retrieval import RelationRetrieval
    from kbqa_r1.sexpr.sexpr_executor import SExprExecutor
    from kbqa_r1.sexpr import SExprGenerator, SExprValidator
    from kbqa_r1.llm_agent.sexpr_action_processor import SExprActionProcessor
    from kbqa_r1.llm_agent.sexpr_result_processor import SExprResultProcessor
    from kbqa_r1.llm_agent.sexpr_state_manager import SExprStateManager
    from kbqa_r1.llm_agent.sexpr_multi_result_formatter import SExprMultiResultFormatter
    from kbqa_r1.sparql.sparql_manager import SPARQLConfig

    sexpr_cfg = config.get("sexpr_config", {})
    dataset_type = os.getenv("DATASET_TYPE", "webqsp")

    sparql_kwargs = {}
    if config.get("odbc_config"):
        sparql_kwargs["odbc_config"] = dict(config.odbc_config)
    sparql_kwargs["sparql_url"] = config.get("sparql", {}).get("url", "http://localhost:8000/execute")
    sparql_kwargs["sparql_batch_size"] = config.get("sparql_batch_size", 128)
    sparql_kwargs["sparql_max_concurrent"] = config.get("sparql_max_concurrent", 16)
    sparql_kwargs["use_odbc"] = config.get("use_odbc", True)
    sparql_kwargs["use_aioodbc"] = config.get("use_aioodbc", False)
    sparql_config = SPARQLConfig(**sparql_kwargs)

    action_parser = ActionParser()
    sexpr_generator = SExprGenerator()
    sexpr_validator = SExprValidator() if sexpr_cfg.get("enable_sexpr_validation", True) else None
    relation_retrieval = RelationRetrieval(sparql_config=sparql_config, dataset=dataset_type)
    sexpr_executor = SExprExecutor(sparql_config, dataset_type=dataset_type)

    state_manager = SExprStateManager()
    action_processor = SExprActionProcessor(relation_retrieval, state_manager)

    model_name = getattr(tokenizer, "name_or_path", None)
    # SExprResultProcessor needs a config-like object with max_obs_length
    class _MiniCfg:
        def __init__(self, c):
            self.max_obs_length = c.data.get("max_obs_length", 3072)
    result_processor = SExprResultProcessor(_MiniCfg(config), relation_retrieval, model_name=model_name, tokenizer=tokenizer)
    result_processor.action_processor = action_processor

    multi_result_formatter = SExprMultiResultFormatter(result_processor)

    _shared_components = {
        "action_parser": action_parser,
        "sexpr_generator": sexpr_generator,
        "sexpr_validator": sexpr_validator,
        "relation_retrieval": relation_retrieval,
        "sexpr_executor": sexpr_executor,
        "state_manager": state_manager,
        "action_processor": action_processor,
        "result_processor": result_processor,
        "multi_result_formatter": multi_result_formatter,
        "dataset_type": dataset_type,
        "max_turns": config.get("max_turns", 6),
    }
    logger.info(f"[KBQA-AgentLoop] Initialized shared components for dataset={dataset_type}")
    return _shared_components


def _parse_candidate_entities(prompt_text: str):
    """Extract candidate entities from prompt text.

    Expected fragment: Candidate Entities: ['Barack Obama' (m.02mjmr), ...]
    """
    m = re.search(r"(Candidate Entities|Entities):\s*\[(.*?)\]", prompt_text, re.DOTALL)
    if not m:
        return []
    inner = m.group(2)
    pairs = re.findall(r"'([^']+)'\s*\(([^)]+)\)", inner)
    return [(name, eid) for name, eid in pairs]


def _compact_mid_set(mid_set: Set[str], limit: int = 25) -> list[str]:
    return sorted(mid_set)[:limit]


def _state_match_candidates(current_state: Set[str], gold_trace: List[GoldTraceStep], limit: int = 5):
    candidates = []
    current_size = len(current_state)
    for idx, step in enumerate(gold_trace):
        prev_size = len(step.prev_mid_set)
        intersection_size = len(current_state & step.prev_mid_set)
        union_size = len(current_state | step.prev_mid_set)
        student_coverage = (
            intersection_size / current_size
            if current_size > 0
            else (1.0 if prev_size == 0 else 0.0)
        )
        gold_coverage = (
            intersection_size / prev_size
            if prev_size > 0
            else (1.0 if current_size == 0 else 0.0)
        )
        score = jaccard_similarity(current_state, step.prev_mid_set)
        candidates.append({
            "gold_idx": idx,
            "score": float(score),
            "prev_size": prev_size,
            "next_size": len(step.next_mid_set),
            "intersection_size": intersection_size,
            "union_size": union_size,
            "student_coverage": float(student_coverage),
            "gold_coverage": float(gold_coverage),
            "prev_mid_set": _compact_mid_set(step.prev_mid_set),
            "next_mid_set": _compact_mid_set(step.next_mid_set),
            "action_function": step.action_function,
            "teacher_hint": step.per_turn_hint,
        })
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates[:limit]


def _extract_first_action_text(text: str) -> str:
    match = re.search(r"<action>\s*(.*?)\s*</action>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return re.sub(r"\s+", " ", match.group(1)).strip()
    answer = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)
    if answer:
        return "<answer> " + re.sub(r"\s+", " ", answer.group(1)).strip()
    return re.sub(r"\s+", " ", text).strip()[:500]


def _find_action_token_span(text: str, gen_ids: list[int], tokenizer) -> Optional[tuple[int, int]]:
    """Return token offsets for the first <action>...</action> span in one turn."""
    match = re.search(r"<action\b[^>]*>.*?</action>", text, re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    prefix_ids = _safe_encode(tokenizer, text[: match.start()], add_special_tokens=False)
    action_ids = _safe_encode(tokenizer, text[match.start() : match.end()], add_special_tokens=False)
    start = min(len(prefix_ids), len(gen_ids))
    end = min(start + len(action_ids), len(gen_ids))
    if start >= end:
        return None
    return start, end


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _has_expression_ref(text: Optional[str]) -> bool:
    return bool(text and re.search(r"\bexpression\d+\b", text, re.IGNORECASE))


@register("kbqa_sexpr")
class KBQASExprAgentLoop(AgentLoopBase):
    """Per-sample async agent loop for KBQA S-Expression generation.

    Replaces SExprLLMGenerationManager with the verl 0.8.0 agent loop pattern.
    Each sample runs as an independent coroutine — no batch synchronization.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_turns = self.config.get("max_turns", 6)
        self.response_length = self.rollout_config.response_length
        self.prompt_length = self.rollout_config.prompt_length
        # Total context budget for multi-turn dialogue (prompt + response).
        # response_length is the total response padding target (e.g., 10240 for
        # multi-turn). Per-turn vLLM generation is limited by
        # multi_turn.max_tokens_per_turn (default 1024), set in the vLLM server.
        self.max_total_length = self.prompt_length + self.response_length
        # Token-level truncation limit for observations (matches old run_llm_loop).
        # Prevents long SPARQL results from consuming too much context.
        self.max_obs_length = self.config.data.get("max_obs_length", 3072)
        # GAPD turn limit: only apply teacher distillation to the first N turns.
        # -1 means all turns (default, no turn filtering).
        self.gapd_teacher_max_turns = getattr(self.rollout_config, "gapd_teacher_max_turns", -1)
        # Enable state-anchored per-turn GAPD when gold_trace is available.
        self.gapd_use_state_matching = getattr(self.rollout_config, "gapd_use_state_matching", True)
        # Conservative gates for all-turn GAPD ablations.
        self.gapd_skip_same_next_state = _as_bool(
            getattr(self.rollout_config, "gapd_skip_same_next_state", False)
        )
        self.gapd_skip_expression_id_risk = _as_bool(
            getattr(self.rollout_config, "gapd_skip_expression_id_risk", False)
        )
        self.gapd_supervise_action_only = _as_bool(
            getattr(self.rollout_config, "gapd_supervise_action_only", False)
        )

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        validate = _as_bool(kwargs.get("validate", False))
        gapd_val_teacher_debug_candidate = validate and _as_bool(
            kwargs.get("_gapd_val_teacher_debug_candidate", False)
        )
        messages = list(kwargs["raw_prompt"])
        request_id = uuid4().hex
        loop = get_event_loop()

        # Initialize heavy components (cached per worker process)
        comp = _get_shared_components(self.config, self.tokenizer)
        action_parser = comp["action_parser"]
        sexpr_generator = comp["sexpr_generator"]
        sexpr_validator = comp["sexpr_validator"]
        sexpr_executor = comp["sexpr_executor"]
        state_manager = comp["state_manager"]
        action_processor = comp["action_processor"]
        result_processor = comp["result_processor"]

        # Per-sample state — use sample_index from trajectory_info as sample_id
        sample_id = kwargs.get("sample_index", id(asyncio.current_task()) % (2**31))
        state_manager.clear_sample_state(sample_id)

        # Decode the prompt to extract candidate entities
        prompt_ids = await self.apply_chat_template(messages)
        prompt_text = await loop.run_in_executor(
            None, lambda: _safe_decode(self.tokenizer, prompt_ids)
        )
        candidate_entities = _parse_candidate_entities(prompt_text)
        if candidate_entities:
            state_manager.set_sample_entities(sample_id, candidate_entities)

        # Parse gold trace for state-anchored per-turn GAPD
        gold_trace: Optional[List[GoldTraceStep]] = None
        if self.gapd_use_state_matching and _STATE_MATCHING_AVAILABLE:
            extra_info = kwargs.get("extra_info", None)
            gold_trace_raw = None
            if isinstance(extra_info, dict):
                gold_trace_raw = extra_info.get("gold_trace", None)
            if gold_trace_raw is None:
                rm = kwargs.get("reward_model", None)
                if isinstance(rm, dict) and "ground_truth" in rm:
                    gt = rm["ground_truth"]
                    if hasattr(gt, "get"):
                        gold_trace_raw = gt.get("gold_trace", None)
            if gold_trace_raw:
                try:
                    gold_trace = [
                        GoldTraceStep(
                            step_index=step.get("step_index", i),
                            prev_mid_set=set(step.get("prev_mid_set", [])),
                            action_function=step.get("action_function", ""),
                            next_mid_set=set(step.get("next_mid_set", [])),
                            per_turn_hint=step.get("per_turn_hint", ""),
                        )
                        for i, step in enumerate(gold_trace_raw)
                    ]
                except Exception as e:
                    logger.warning(f"[KBQA-AgentLoop] Failed to parse gold_trace: {e}")

        use_state_matching = _STATE_MATCHING_AVAILABLE and gold_trace is not None and len(gold_trace) > 0
        gapd_missing_gold_trace = (
            (not validate) or gapd_val_teacher_debug_candidate
        ) and self.gapd_use_state_matching and not use_state_matching

        # Track the initial prompt length for final output split
        initial_prompt_len = len(prompt_ids)

        # Accumulate full token sequence and response mask
        all_ids = list(prompt_ids)
        response_mask: list[int] = []
        response_logprobs: list[float] = []
        metrics: dict[str, Any] = {}
        num_turns = 0
        last_response_text = ""

        # Track turn boundaries for GAPD distillation.
        # When gold_trace is available, state matching determines which turns
        # receive teacher supervision. Otherwise fall back to turn-count limit.
        track_gapd_turns = ((not validate) or gapd_val_teacher_debug_candidate) and (
            use_state_matching or (self.gapd_teacher_max_turns > 0)
        )
        gapd_turn_mask: list[int] = [] if track_gapd_turns else None  # type: ignore[assignment]
        gapd_turn_ids: list[int] = []
        gapd_matched_turns: list[dict] = []  # metadata for per-turn teacher logprob computation
        gapd_state_match_debug: list[dict] = []
        rollout_debug: list[dict] = []
        rollout_states: list[dict] = []  # per-turn execution state anchors
        used_gold_indices: set[int] = set()
        gapd_drop_counts = {
            "answer_turn": 0,
            "same_next_state": 0,
            "expression_id_risk": 0,
            "no_action_span": 0,
        }
        gapd_drop_counts_turn1 = {
            "answer_turn": 0,
            "same_next_state": 0,
            "expression_id_risk": 0,
            "no_action_span": 0,
        }
        gapd_initial_supervised_turns = 0

        # Add stop tokens so LLM stops at </action> or </answer>
        gen_params = dict(sampling_params)
        gen_params["stop"] = ["</action>", "</answer>"]
        gen_params["include_stop_str_in_output"] = True

        for turn in range(self.max_turns):
            # --- State matching for THIS turn (before generation) ---
            current_state = rollout_states[-1]["mid_set"] if rollout_states else set()
            matched_gold_idx = None
            teacher_hint = None
            match_debug = None
            if use_state_matching:
                matched_gold_idx = match_state_to_gold_step(current_state, gold_trace)
                if matched_gold_idx is not None:
                    teacher_hint = gold_trace[matched_gold_idx].per_turn_hint
                candidates = _state_match_candidates(current_state, gold_trace)
                match_threshold = 0.95
                best_candidate = candidates[0] if candidates else None
                second_candidate = candidates[1] if len(candidates) > 1 else None
                best_score = best_candidate.get("score") if best_candidate else None
                second_best_score = second_candidate.get("score") if second_candidate else None
                score_margin = (
                    float(best_score - second_best_score)
                    if best_score is not None and second_best_score is not None
                    else None
                )
                matched_rank = (
                    next(
                        (rank + 1 for rank, c in enumerate(candidates) if c["gold_idx"] == matched_gold_idx),
                        None,
                    )
                    if matched_gold_idx is not None
                    else None
                )
                num_candidates_above_threshold = sum(
                    1 for c in candidates if c.get("score", 0.0) >= match_threshold
                )
                match_debug = {
                    "turn": turn + 1,
                    "current_state_size": len(current_state),
                    "current_mid_set": _compact_mid_set(current_state),
                    "matched_gold_idx": matched_gold_idx,
                    "matched_score": (
                        next((c["score"] for c in candidates if c["gold_idx"] == matched_gold_idx), None)
                        if matched_gold_idx is not None
                        else None
                    ),
                    "threshold": match_threshold,
                    "best_gold_idx": best_candidate.get("gold_idx") if best_candidate else None,
                    "best_score": best_score,
                    "second_best_score": second_best_score,
                    "score_margin": score_margin,
                    "num_candidates_above_threshold": num_candidates_above_threshold,
                    "matched_rank": matched_rank,
                    "best_student_coverage": best_candidate.get("student_coverage") if best_candidate else None,
                    "best_gold_coverage": best_candidate.get("gold_coverage") if best_candidate else None,
                    "best_action_function": best_candidate.get("action_function") if best_candidate else None,
                    "already_used": matched_gold_idx in used_gold_indices if matched_gold_idx is not None else False,
                    "teacher_hint": teacher_hint,
                    "candidates": candidates,
                }

            # --- Generate ---
            output: TokenOutput = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=all_ids,
                sampling_params=gen_params,
            )
            gen_ids = output.token_ids
            num_turns += 1

            gen_start = len(all_ids)
            all_ids += gen_ids
            gen_end = len(all_ids)
            response_mask += [1] * len(gen_ids)
            gapd_turn_ids += [turn + 1] * len(gen_ids)
            is_supervised = False
            if track_gapd_turns:
                if use_state_matching:
                    is_supervised = (
                        matched_gold_idx is not None
                        and matched_gold_idx not in used_gold_indices
                        and (
                            self.gapd_teacher_max_turns <= 0
                            or matched_gold_idx < self.gapd_teacher_max_turns
                        )
                    )
                    if is_supervised:
                        gapd_initial_supervised_turns += 1
                    gapd_turn_mask += [1 if is_supervised else 0] * len(gen_ids)
                    if is_supervised:
                        used_gold_indices.add(matched_gold_idx)
                        gapd_matched_turns.append({
                            "gen_start": gen_start,
                            "gen_end": gen_end,
                            "teacher_hint": teacher_hint,
                        })
                else:
                    # Fall back to turn-count limit
                    gapd_turn_mask += [1 if turn < self.gapd_teacher_max_turns else 0] * len(gen_ids)
            if output.log_probs:
                response_logprobs += output.log_probs

            def _drop_current_gapd_supervision(reason: str):
                nonlocal is_supervised
                if not (track_gapd_turns and use_state_matching and is_supervised and gapd_turn_mask is not None):
                    return
                rel_start = gen_start - initial_prompt_len
                rel_end = gen_end - initial_prompt_len
                for idx in range(max(0, rel_start), min(len(gapd_turn_mask), rel_end)):
                    gapd_turn_mask[idx] = 0
                if (
                    gapd_matched_turns
                    and gapd_matched_turns[-1].get("gen_start") == gen_start
                    and gapd_matched_turns[-1].get("gen_end") == gen_end
                ):
                    gapd_matched_turns.pop()
                if matched_gold_idx is not None:
                    used_gold_indices.discard(matched_gold_idx)
                is_supervised = False
                if reason in gapd_drop_counts:
                    gapd_drop_counts[reason] += 1
                    if turn == 0:
                        gapd_drop_counts_turn1[reason] += 1
                if match_debug is not None:
                    match_debug["supervision_dropped_reason"] = reason
                    match_debug["supervised"] = False

            # --- Decode and check termination ---
            last_response_text = await loop.run_in_executor(
                None, lambda ids=gen_ids: _safe_decode(self.tokenizer, ids, skip_special_tokens=False)
            )
            has_answer = "<answer>" in last_response_text.lower()
            if has_answer and use_state_matching and is_supervised and gapd_turn_mask is not None:
                # GAPD teacher hints supervise reasoning actions, not final answers.
                # If the student decides to answer from the matched state, do not
                # force the next gold action onto the answer span.
                _drop_current_gapd_supervision("answer_turn")
            action_token_span = None
            if self.gapd_supervise_action_only and is_supervised and gapd_turn_mask is not None:
                action_token_span = _find_action_token_span(last_response_text, gen_ids, self.tokenizer)
                if action_token_span is None:
                    _drop_current_gapd_supervision("no_action_span")
                else:
                    rel_start = gen_start - initial_prompt_len
                    rel_end = gen_end - initial_prompt_len
                    for idx in range(max(0, rel_start), min(len(gapd_turn_mask), rel_end)):
                        gapd_turn_mask[idx] = 0
                    action_rel_start = rel_start + action_token_span[0]
                    action_rel_end = rel_start + action_token_span[1]
                    for idx in range(max(0, action_rel_start), min(len(gapd_turn_mask), action_rel_end)):
                        gapd_turn_mask[idx] = 1
            if match_debug is not None:
                match_debug["student_action"] = _extract_first_action_text(last_response_text)
                match_debug["gen_start"] = gen_start
                match_debug["gen_end"] = gen_end
                if action_token_span is not None:
                    rel_start = gen_start - initial_prompt_len
                    match_debug["action_only_rel_start"] = rel_start + action_token_span[0]
                    match_debug["action_only_rel_end"] = rel_start + action_token_span[1]
                if track_gapd_turns and use_state_matching:
                    match_debug["supervised"] = bool(is_supervised)
                gapd_state_match_debug.append(match_debug)
            student_action_text = _extract_first_action_text(last_response_text)
            if (
                self.gapd_skip_expression_id_risk
                and is_supervised
                and (_has_expression_ref(student_action_text) or _has_expression_ref(teacher_hint))
            ):
                _drop_current_gapd_supervision("expression_id_risk")
            rollout_entry = {
                "turn": turn + 1,
                "assistant_rel_start": gen_start - initial_prompt_len,
                "assistant_rel_end": gen_end - initial_prompt_len,
                "assistant_text": last_response_text,
                "student_action": student_action_text,
                "has_answer": bool(has_answer),
                "supervised": bool(is_supervised),
                "matched_gold_idx": matched_gold_idx,
                "teacher_hint": teacher_hint,
                "current_state_size": len(current_state),
                "current_mid_set": _compact_mid_set(current_state),
                "gapd_action_only": bool(self.gapd_supervise_action_only),
            }
            if action_token_span is not None:
                rel_start = gen_start - initial_prompt_len
                rollout_entry["action_only_rel_start"] = rel_start + action_token_span[0]
                rollout_entry["action_only_rel_end"] = rel_start + action_token_span[1]
            rollout_debug.append(rollout_entry)

            if has_answer:
                break
            if len(all_ids) >= self.max_total_length:
                break

            # --- Parse actions ---
            actions = action_parser.parse_actions_from_text(last_response_text)
            if not actions:
                # Old run_llm_loop returns a guidance observation and continues the loop,
                # giving the model another chance to produce valid actions.
                is_final_turn = turn == self.max_turns - 1
                info_block = "<information>\nNo valid actions found. Please provide reasoning actions.\n</information>"
                observation = result_processor._add_guidance_prompt(info_block, is_final_turn)
                obs_ids = await loop.run_in_executor(
                    None, lambda obs=observation: _safe_encode(self.tokenizer, obs, add_special_tokens=False)
                )
                original_obs_token_count = len(obs_ids)
                if len(obs_ids) > self.max_obs_length:
                    obs_ids = obs_ids[:self.max_obs_length]
                dumped_observation_text = observation
                if len(obs_ids) != original_obs_token_count:
                    dumped_observation_text = await loop.run_in_executor(
                        None, lambda ids=obs_ids: _safe_decode(self.tokenizer, ids, skip_special_tokens=False)
                    )
                if len(all_ids) + len(obs_ids) >= self.max_total_length:
                    rollout_debug[-1].update({
                        "observation_text": dumped_observation_text,
                        "observation_token_count": len(obs_ids),
                        "observation_original_token_count": original_obs_token_count,
                        "observation_appended": False,
                        "observation_drop_reason": "max_total_length",
                        "execution_successful": False,
                        "execution_error": "no_valid_actions",
                    })
                    break
                obs_start = len(all_ids)
                all_ids += obs_ids
                obs_end = len(all_ids)
                response_mask += [0] * len(obs_ids)
                gapd_turn_ids += [0] * len(obs_ids)
                if track_gapd_turns:
                    gapd_turn_mask += [0] * len(obs_ids)
                if response_logprobs:
                    response_logprobs += [0.0] * len(obs_ids)
                # Record empty state for this turn (no valid action -> no state change)
                rollout_states.append({
                    "turn": turn,
                    "mid_set": current_state,
                    "is_successful": False,
                })
                if gapd_state_match_debug and gapd_state_match_debug[-1].get("turn") == turn + 1:
                    gapd_state_match_debug[-1].update({
                        "execution_successful": False,
                        "next_state_size": len(current_state),
                        "next_mid_set": _compact_mid_set(current_state),
                        "execution_error": "no_valid_actions",
                    })
                rollout_debug[-1].update({
                    "observation_rel_start": obs_start - initial_prompt_len,
                    "observation_rel_end": obs_end - initial_prompt_len,
                    "observation_text": dumped_observation_text,
                    "observation_token_count": len(obs_ids),
                    "observation_original_token_count": original_obs_token_count,
                    "observation_appended": True,
                    "execution_successful": False,
                    "next_state_size": len(current_state),
                    "next_mid_set": _compact_mid_set(current_state),
                    "execution_error": "no_valid_actions",
                })
                continue

            # --- Process actions (CPU + network bound, run in thread) ---
            turn_result: TurnExecutionResult = await loop.run_in_executor(
                None,
                lambda: self._process_and_execute(
                    actions=actions,
                    sample_id=sample_id,
                    turn=turn,
                    prediction=last_response_text,
                    comp=comp,
                ),
            )

            # Record execution state for next turn's matching
            rollout_states.append({
                "turn": turn,
                "mid_set": turn_result.mid_set,
                "is_successful": turn_result.is_successful,
            })
            if gapd_state_match_debug and gapd_state_match_debug[-1].get("turn") == turn + 1:
                same_next_state = False
                if matched_gold_idx is not None and 0 <= matched_gold_idx < len(gold_trace or []):
                    teacher_next_state = set(gold_trace[matched_gold_idx].next_mid_set)
                    # Empty next states are not informative enough for a safe
                    # equivalence gate, so only skip on non-empty exact matches.
                    same_next_state = bool(teacher_next_state) and set(turn_result.mid_set) == teacher_next_state
                if self.gapd_skip_same_next_state and is_supervised and same_next_state:
                    _drop_current_gapd_supervision("same_next_state")
                gapd_state_match_debug[-1].update({
                    "execution_successful": bool(turn_result.is_successful),
                    "next_state_size": len(turn_result.mid_set),
                    "next_mid_set": _compact_mid_set(turn_result.mid_set),
                    "execution_error": turn_result.error_message,
                    "same_next_state": same_next_state,
                    "supervised": bool(is_supervised),
                })

            # --- Tokenize observation and append with mask=0 ---
            obs_ids = await loop.run_in_executor(
                None, lambda obs=turn_result.observation_text: _safe_encode(self.tokenizer, obs, add_special_tokens=False)
            )
            original_obs_token_count = len(obs_ids)

            # Token-level truncation (matches old run_llm_loop's _process_next_obs).
            # Prevents long SPARQL results from consuming excessive context.
            if len(obs_ids) > self.max_obs_length:
                logger.info(f"[KBQA-AgentLoop] Truncating obs from {len(obs_ids)} to {self.max_obs_length} tokens")
                obs_ids = obs_ids[:self.max_obs_length]
            dumped_observation_text = turn_result.observation_text
            if len(obs_ids) != original_obs_token_count:
                dumped_observation_text = await loop.run_in_executor(
                    None, lambda ids=obs_ids: _safe_decode(self.tokenizer, ids, skip_special_tokens=False)
                )

            if len(all_ids) + len(obs_ids) >= self.max_total_length:
                rollout_debug[-1].update({
                    "supervised": bool(is_supervised),
                    "observation_text": dumped_observation_text,
                    "observation_token_count": len(obs_ids),
                    "observation_original_token_count": original_obs_token_count,
                    "observation_appended": False,
                    "observation_drop_reason": "max_total_length",
                    "execution_successful": bool(turn_result.is_successful),
                    "next_state_size": len(turn_result.mid_set),
                    "next_mid_set": _compact_mid_set(turn_result.mid_set),
                    "execution_error": turn_result.error_message,
                })
                break

            obs_start = len(all_ids)
            all_ids += obs_ids
            obs_end = len(all_ids)
            response_mask += [0] * len(obs_ids)
            gapd_turn_ids += [0] * len(obs_ids)
            if track_gapd_turns:
                gapd_turn_mask += [0] * len(obs_ids)
            if response_logprobs:
                response_logprobs += [0.0] * len(obs_ids)
            rollout_debug[-1].update({
                "supervised": bool(is_supervised),
                "observation_rel_start": obs_start - initial_prompt_len,
                "observation_rel_end": obs_end - initial_prompt_len,
                "observation_text": dumped_observation_text,
                "observation_token_count": len(obs_ids),
                "observation_original_token_count": original_obs_token_count,
                "observation_appended": True,
                "execution_successful": bool(turn_result.is_successful),
                "next_state_size": len(turn_result.mid_set),
                "next_mid_set": _compact_mid_set(turn_result.mid_set),
                "execution_error": turn_result.error_message,
            })

        # If the loop ended without an answer and we have budget, do a final generation
        if "<answer>" not in last_response_text.lower() and len(all_ids) < self.max_total_length:
            output = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=all_ids,
                sampling_params=sampling_params,  # no stop tokens for final turn
            )
            gen_start = len(all_ids)
            all_ids += output.token_ids
            gen_end = len(all_ids)
            response_mask += [1] * len(output.token_ids)
            gapd_turn_ids += [num_turns + 1] * len(output.token_ids)
            if track_gapd_turns:
                # Final generation is beyond the structured turns — no teacher supervision
                gapd_turn_mask += [0] * len(output.token_ids)
            if output.log_probs:
                response_logprobs += output.log_probs
            num_turns += 1
            final_response_text = await loop.run_in_executor(
                None, lambda ids=output.token_ids: _safe_decode(self.tokenizer, ids, skip_special_tokens=False)
            )
            rollout_debug.append({
                "turn": num_turns,
                "assistant_rel_start": gen_start - initial_prompt_len,
                "assistant_rel_end": gen_end - initial_prompt_len,
                "assistant_text": final_response_text,
                "student_action": _extract_first_action_text(final_response_text),
                "has_answer": "<answer>" in final_response_text.lower(),
                "supervised": False,
                "final_generation": True,
            })

        # Clean up state_manager for this sample
        state_manager.clear_sample_state(sample_id)

        # Split into prompt_ids and response_ids
        final_prompt_ids = all_ids[:initial_prompt_len]
        final_response_ids = all_ids[initial_prompt_len:]

        # Truncate to max_total_length - prompt_length (= max response budget),
        # but never exceed the fixed response_length used by verl batching.
        max_response_len = min(self.max_total_length - initial_prompt_len, self.response_length)
        final_response_ids = final_response_ids[:max_response_len]
        response_mask = response_mask[:max_response_len]
        gapd_turn_ids = gapd_turn_ids[:max_response_len]
        if response_logprobs:
            response_logprobs = response_logprobs[:max_response_len]
        if track_gapd_turns:
            gapd_turn_mask = gapd_turn_mask[:max_response_len]

        # --- Compute per-turn teacher logprobs for matched turns ---
        extra = {}
        if gapd_turn_ids:
            extra["gapd_turn_ids"] = gapd_turn_ids
        if track_gapd_turns and gapd_turn_mask is not None:
            extra["gapd_turn_mask"] = gapd_turn_mask
        if gapd_state_match_debug:
            extra["gapd_state_match_debug"] = gapd_state_match_debug
        if rollout_debug:
            extra["rollout_debug"] = rollout_debug
        if gapd_missing_gold_trace:
            # State-matched GAPD is the only safe all-turn mode for KBQA. If a
            # dataset row lacks gold_trace, do not fall back to full-response
            # hint distillation: that would supervise thoughts, observations,
            # and potentially the whole oracle action sequence.
            extra["gapd_skip_teacher"] = True
            extra["gapd_skip_teacher_reason"] = "missing_gold_trace"
        extra["gapd_drop_answer_turns"] = gapd_drop_counts["answer_turn"]
        extra["gapd_drop_same_next_state_turns"] = gapd_drop_counts["same_next_state"]
        extra["gapd_drop_expression_id_risk_turns"] = gapd_drop_counts["expression_id_risk"]
        extra["gapd_drop_no_action_span_turns"] = gapd_drop_counts["no_action_span"]
        extra["gapd_drop_answer_turn1"] = gapd_drop_counts_turn1["answer_turn"]
        extra["gapd_drop_same_next_state_turn1"] = gapd_drop_counts_turn1["same_next_state"]
        extra["gapd_drop_expression_id_risk_turn1"] = gapd_drop_counts_turn1["expression_id_risk"]
        extra["gapd_drop_no_action_span_turn1"] = gapd_drop_counts_turn1["no_action_span"]
        extra["gapd_initial_supervised_turns"] = gapd_initial_supervised_turns
        extra["gapd_final_supervised_turns"] = len(gapd_matched_turns)

        if ((not validate) or gapd_val_teacher_debug_candidate) and use_state_matching and gapd_matched_turns:
            from kbqa_r1.gapd.gapd_rollout_teacher import compute_gapd_teacher_logprobs_per_turn
            teacher_result = await compute_gapd_teacher_logprobs_per_turn(
                server_manager=self.server_manager,
                prompt_ids=final_prompt_ids,
                response_ids=final_response_ids,
                matched_turns=gapd_matched_turns,
                tokenizer=self.tokenizer,
                top_k=getattr(self.rollout_config, "gapd_jsd_top_k", 20),
                max_length=getattr(self.rollout_config, "gapd_teacher_max_length", 16384),
            )
            if teacher_result is not None:
                teacher_ids, teacher_logprobs, teacher_prompt_debug = teacher_result
                teacher_prompt_ids = []
                compact_prompt_debug = []
                for debug in teacher_prompt_debug:
                    compact_debug = dict(debug)
                    teacher_prompt_ids.append(compact_debug.pop("teacher_prompt_ids", None))
                    compact_debug["teacher_prompt_ids_index"] = len(teacher_prompt_ids) - 1
                    compact_prompt_debug.append(compact_debug)
                extra["teacher_ids"] = teacher_ids
                extra["teacher_logprobs"] = teacher_logprobs
                extra["teacher_prompt_ids"] = teacher_prompt_ids
                extra["gapd_teacher_prompt_debug"] = compact_prompt_debug
                extra["gapd_teacher"] = True

        return AgentLoopOutput(
            prompt_ids=final_prompt_ids,
            response_ids=final_response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs if response_logprobs else None,
            num_turns=num_turns,
            metrics=metrics,
            extra_fields=extra,
        )

    # ------------------------------------------------------------------
    # Synchronous helpers (run in thread pool to avoid blocking event loop)
    # ------------------------------------------------------------------

    @staticmethod
    def _process_and_execute(
        actions,
        sample_id: int,
        turn: int,
        prediction: str,
        comp: dict,
    ) -> TurnExecutionResult:
        """Process parsed actions and execute SPARQL.

        This runs in a thread pool because relation retrieval and SPARQL
        execution are synchronous CPU/IO-bound operations.
        """
        from kbqa_r1.sexpr.action_parser import ActionType
        from kbqa_r1.llm_agent.sexpr_utils import SExprUtils

        action_parser = comp["action_parser"]
        action_processor = comp["action_processor"]
        state_manager = comp["state_manager"]
        sexpr_generator = comp["sexpr_generator"]
        sexpr_validator = comp["sexpr_validator"]
        sexpr_executor = comp["sexpr_executor"]
        result_processor = comp["result_processor"]
        max_turns = comp.get("max_turns", 6)

        is_final_turn = turn == max_turns - 1

        # Clear per-turn action states (function_state persists across turns)
        action_processor.clear_sample_action_states(sample_id)

        function_state = state_manager.get_sample_function_state(sample_id)
        candidate_entities = state_manager.get_sample_entities(sample_id) or []

        initial_fs_len = len(function_state)

        # Parse and log actions (matching old [SEXPR] Sample format)
        if actions:
            logger.info(f"[SEXPR] Sample {sample_id}: Parsed {len(actions)} actions from prediction")
            for i, action in enumerate(actions[:2]):
                logger.info(f"[SEXPR]   Action {i+1}: {action.action_type.name} - {action.arguments if action.arguments else 'No args'}")
        else:
            logger.info(f"[SEXPR] Sample {sample_id}: No valid actions found in prediction")

        logger.info(f"[SEXPR] Sample {sample_id}: Using function_state with {len(function_state)} functions")

        # Process actions with relation retrieval
        processed_actions = action_processor.process_actions_with_retrieval(
            actions, function_state, candidate_entities, sample_id=sample_id
        )

        if not processed_actions:
            logger.info(f"[SEXPR] Sample {sample_id}: Failed to process actions with retrieval")
            info_block = "<information>\nFailed to process actions with retrieval.\n</information>"
            return TurnExecutionResult(
                observation_text=result_processor._add_guidance_prompt(info_block, is_final_turn),
                is_successful=False,
                mid_set=set(),
            )

        # Handle multiple FIND_RELATION actions independently
        all_find_rel = all(a.action_type == ActionType.FIND_RELATION for a in processed_actions)
        if len(processed_actions) > 1 and all_find_rel:
            logger.info(f"[SEXPR] Sample {sample_id}: All actions are FIND_RELATION, processing each independently")
            return KBQASExprAgentLoop._handle_multi_find_relation(
                processed_actions, sample_id, turn, comp
            )

        # Check for invalid actions
        invalid = [a for a in processed_actions if not a.is_valid]
        if invalid:
            msgs = []
            for a in invalid:
                err = getattr(a, "error_message", None) or "Invalid action"
                msgs.append(f"Action {a.action_type.name}: {err}")
            info_block = f"<information>\nAction processing failed:\n" + "\n".join(msgs) + "\n</information>"
            return TurnExecutionResult(
                observation_text=result_processor._add_guidance_prompt(info_block, is_final_turn),
                is_successful=False,
                mid_set=set(),
            )

        # Build S-Expression and execute
        return KBQASExprAgentLoop._execute_normal(
            processed_actions, sample_id, initial_fs_len, is_final_turn, prediction, comp
        )

    @staticmethod
    def _execute_normal(
        processed_actions,
        sample_id: int,
        initial_fs_len: int,
        is_final_turn: bool,
        prediction: str,
        comp: dict,
    ) -> TurnExecutionResult:
        """Execute the action list via normal S-Expression path."""
        from kbqa_r1.llm_agent.sexpr_utils import SExprUtils
        state_manager = comp["state_manager"]
        sexpr_generator = comp["sexpr_generator"]
        sexpr_validator = comp["sexpr_validator"]
        sexpr_executor = comp["sexpr_executor"]
        result_processor = comp["result_processor"]
        action_processor = comp["action_processor"]

        persistent_fs = state_manager.get_sample_function_state(sample_id)
        if not persistent_fs:
            info_block = "<information>\nNo function state available.\n</information>"
            return TurnExecutionResult(
                observation_text=result_processor._add_guidance_prompt(info_block, is_final_turn),
                is_successful=False,
                mid_set=set(),
            )

        # Infer target expression
        target_expr = None
        for stmt in reversed(persistent_fs):
            if not isinstance(stmt, str):
                continue
            s = stmt.strip()
            m_stop = re.match(r"^[A-Za-z_]\w*\s*=\s*STOP\(\s*(expression\d+)\s*\)\s*$", s)
            if m_stop:
                target_expr = m_stop.group(1)
                break
            m_lhs = re.match(r"^(expression\d+)\s*=", s)
            if m_lhs and target_expr is None:
                target_expr = m_lhs.group(1)
        if not target_expr:
            target_expr = f"expression{state_manager.get_current_expression_id(sample_id)}"

        sexpr_result = sexpr_generator.generate_sexpr_from_strings(persistent_fs, target_expr)
        if not sexpr_result.is_valid:
            logger.info(f"[SEXPR] Sample {sample_id}: S-Expression generation failed: {sexpr_result.error_message}")
            info_block = f"<information>\nS-Expression generation failed: {sexpr_result.error_message}\n</information>"
            return TurnExecutionResult(
                observation_text=result_processor._add_guidance_prompt(info_block, is_final_turn),
                is_successful=False,
                mid_set=set(),
            )

        if sexpr_validator:
            vr = sexpr_validator.validate(sexpr_result.sexpr)
            if not vr.is_valid:
                info_block = f"<information>\nS-Expression validation failed: {'; '.join(vr.errors)}\n</information>"
                return TurnExecutionResult(
                    observation_text=result_processor._add_guidance_prompt(info_block, is_final_turn),
                    is_successful=False,
                    mid_set=set(),
                )

        # Execute SPARQL
        logger.info(f"[SEXPR-EXEC] Sample {sample_id}: About to execute S-Expression: {sexpr_result.sexpr}")
        try:
            execution_result = sexpr_executor.execute_sexpr(sexpr_result.sexpr, function_state=persistent_fs)
        except RuntimeError as e:
            if "timeout" in str(e).lower() or "Dangerous" in str(e):
                return TurnExecutionResult(
                    observation_text=f"<information>\nSPARQL execution timeout: {e}\n</information>",
                    is_successful=False,
                    mid_set=set(),
                )
            raise

        logger.info(f"[SEXPR] Sample {sample_id}: S-Expression execution - Success: {execution_result.is_successful}")
        if execution_result.is_successful:
            logger.info(f"[SEXPR] Sample {sample_id}: Found {len(execution_result.results)} results")
            mid_list = result_processor.extract_mid_list(execution_result.results)
            mid_set = _extract_mid_set_from_result(execution_result)

            # Transfer per-action relation states
            if hasattr(action_processor, "_per_action_relation_states"):
                result_processor.set_per_action_relation_states(action_processor._per_action_relation_states)

            action_index = None
            if len(processed_actions) == 1:
                action_index = getattr(processed_actions[0], "action_index", 0)

            obs_text = result_processor.build_enriched_info_block(
                execution_result, sexpr_result, mid_list, action_index, sample_id,
                is_final_turn=is_final_turn,
            )
            return TurnExecutionResult(
                observation_text=obs_text,
                is_successful=True,
                mid_set=mid_set,
            )
        else:
            logger.info(f"[SEXPR] Sample {sample_id}: Execution error - {(execution_result.error_message or '')[:100]}")
            # Check timeout
            err_msg = execution_result.error_message or ""
            if any(ind in err_msg.lower() for ind in ("timeout", "hyt00")):
                return TurnExecutionResult(
                    observation_text=f"<information>\nSPARQL execution timeout: {err_msg}\n</information>",
                    is_successful=False,
                    mid_set=set(),
                )

            # Rollback function state on failure
            current_fs_len = len(state_manager.get_sample_function_state(sample_id))
            added = current_fs_len - initial_fs_len
            if added > 0:
                state_manager.rollback_sample_function_state(sample_id, added)
                logger.info(f"[SEXPR] Sample {sample_id}: Execution failed, rolling back {added} function calls")

            return TurnExecutionResult(
                observation_text=f"<information>\nExecution failed: {err_msg}\n</information>",
                is_successful=False,
                mid_set=set(),
            )

    @staticmethod
    def _handle_multi_find_relation(processed_actions, sample_id, turn, comp) -> TurnExecutionResult:
        """Handle multiple FIND_RELATION actions independently."""
        result_processor = comp["result_processor"]
        max_turns = comp.get("max_turns", 6)
        is_final_turn = turn == max_turns - 1

        obs_texts = []
        combined_mid_set: set[str] = set()
        any_success = False
        for action in processed_actions:
            if action.is_valid:
                result = KBQASExprAgentLoop._execute_normal(
                    [action], sample_id, 0, is_final_turn, "", comp
                )
                obs_texts.append(result.observation_text)
                combined_mid_set.update(result.mid_set)
                if result.is_successful:
                    any_success = True
            else:
                err = getattr(action, "error_message", "Invalid action")
                obs_texts.append(f"<information>\nAction FIND_RELATION: {err}\n</information>")

        combined = "\n\n".join(obs_texts)
        if len(obs_texts) > 1:
            info_block = f"<information>\nIndependent FIND_RELATION results:\n\n{combined}\n</information>"
            final_obs = result_processor._add_guidance_prompt(info_block, is_final_turn)
        else:
            final_obs = combined

        return TurnExecutionResult(
            observation_text=final_obs,
            is_successful=any_success,
            mid_set=combined_mid_set,
        )
