"""Teacher prompt builder for GAPD (Gold-Action Policy Distillation).

Injects gold action hints into the teacher prompt between guidance_body
and assistant_prefix, so the teacher sees privileged info but is instructed
not to mention it.  Response tokens are unchanged, so teacher logits
align to student positions via simple [-response_len:] slicing.

Action conversion reuses extract_action_hint_from_response from the
rejection-sampling data pipeline to guarantee consistent expression-ID
management between teacher hints and model-generated actions.
"""

import importlib.util
import logging
import os
import re
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

_extract_action_hint_fn = None

HINT_PREFIX = (
    "\n[Reference action sequence. "
    "Do not mention or repeat this in your response, "
    "just silently use it as guidance.]\n"
)
HINT_SUFFIX = "\n[End reference]\n"


def _get_extract_action_hint_fn():
    """Lazily import extract_action_hint_from_response."""
    global _extract_action_hint_fn
    if _extract_action_hint_fn is not None:
        return _extract_action_hint_fn

    here = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(
        here, '..', '..', 'scripts', 'data_process',
        'prepare_rejection_sampling_dataset.py',
    )
    script_path = os.path.normpath(script_path)
    spec = importlib.util.spec_from_file_location('_rs_dataset', script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _extract_action_hint_fn = mod.extract_action_hint_from_response
    return _extract_action_hint_fn


def _function_list_to_synthetic_response(function_list: List[str]) -> str:
    """Convert s-expr function_list to a synthetic response with <action> tags.

    Handles all WebQSP/GrailQA/GraphQ s-expression patterns including
    JOIN with (R ...) reverse notation, ARG(ARGMAX/ARGMIN, ...),
    CMP with expression references, TC with expression references, etc.

    The output is fed into extract_action_hint_from_response which handles
    expression-ID normalisation, entity inlining, and Extract_entity / Finish
    removal — identical to what the rejection-sampling pipeline does.
    """
    lines = []
    step = 0
    for func in function_list:
        step += 1
        func = func.strip()

        if "START(" in func:
            # START('m.06w2sn5') or START('"Country"')
            m = re.search(r"START\((.*?)\)", func)
            if m:
                entity = m.group(1).strip().strip("'\"")
                lines.append(f"Action{step}: Extract_entity [ {entity} ]")

        elif "JOIN(" in func:
            # JOIN('relation', expression) or JOIN('(R relation)', expression)
            m = re.search(r"JOIN\((.*?),\s*(expression\d*)\)", func)
            if m:
                relation = m.group(1).strip().strip("'\"")
                expr = m.group(2)
                lines.append(f"Action{step}: Find_relation [ {expr} | {relation} ]")

        elif "AND(" in func:
            # AND(expression1, expression2)
            m = re.search(r"AND\((expression\d*),\s*(expression\d*)\)", func)
            if m:
                lines.append(f"Action{step}: Merge [ {m.group(1)} | {m.group(2)} ]")

        elif "ARG(" in func:
            # ARG(ARGMAX, expression, 'relation') or ARG(ARGMIN, expression, 'relation')
            m = re.search(r"ARG\((ARGMAX|ARGMIN),\s*(expression\d*),\s*'(.*?)'\)", func, re.IGNORECASE)
            if m:
                mode = m.group(1).upper()
                if mode == "ARGMAX":
                    mode = "MAX"
                elif mode == "ARGMIN":
                    mode = "MIN"
                expr = m.group(2)
                relation = m.group(3)
                lines.append(f"Action{step}: Order [ {mode} | {expr} | {relation} ]")

        elif "CMP(" in func:
            # CMP('gt', expression, expression1) or CMP(gt, expression, expression1)
            m = re.search(r"CMP\((.*?),\s*(expression\d*),\s*(expression\d*)\)", func)
            if m:
                op = m.group(1).strip().strip("'\"")
                left = m.group(2)
                right = m.group(3)
                lines.append(f"Action{step}: Compare [ {op} | {left} | {right} ]")

        elif "TC(" in func:
            # TC(expression, 'from', '2000') or TC(expression1, 'to', '2010')
            m = re.search(r"TC\((expression\d*),\s*'(.*?)',\s*'(.*?)'\)", func)
            if m:
                expr = m.group(1)
                prop = m.group(2)
                value = m.group(3)
                lines.append(f"Action{step}: Time_constraint [ {expr} | {prop} | {value} ]")

        elif "COUNT(" in func:
            m = re.search(r"COUNT\((expression\d*)\)", func)
            if m:
                lines.append(f"Action{step}: Count [ {m.group(1)} ]")

        elif "STOP(" in func:
            m = re.search(r"STOP\((expression\d*)\)", func)
            if m:
                lines.append(f"Action{step}: Finish [ {m.group(1)} ]")

    return "<action>\n" + "\n".join(lines) + "\n</action>"


def function_list_to_action_hint(
    function_list: List[str],
    extra_info: Optional[dict] = None,
) -> str:
    """Convert function_list to an action-hint string.

    Uses the same extract_action_hint_from_response pipeline as the
    rejection-sampling data preparation, so expression IDs, entity
    inlining and action filtering are all consistent.

    Args:
        function_list: List of s-expression function strings.
        extra_info: Optional dict with 'extracted_entities' or
            'candidate_entities' for entity MID resolution.
            If provided, Extract_entity MIDs are inlined into
            Find_relation actions.
    """
    synthetic_response = _function_list_to_synthetic_response(function_list)
    extract_fn = _get_extract_action_hint_fn()
    return extract_fn(synthetic_response, extra_info=extra_info)


def _make_hint_block(action_hint: str) -> str:
    return HINT_PREFIX + action_hint.strip() + HINT_SUFFIX


def _detect_assistant_prefix(text: str) -> Optional[str]:
    """Detect which assistant prefix is used in the text."""
    for prefix in [
        "<|im_start|>assistant\n",
        "<|im_start|>assistant",
        "<|start_header_id|>assistant<|end_header_id|>",
    ]:
        if prefix in text:
            return prefix
    return None


def build_teacher_prompt(
    student_prompt_text: str,
    function_list: List[str],
) -> str:
    """Build teacher prompt by injecting gold action hint before each assistant_prefix.

    The hint block is inserted between guidance_body and assistant_prefix,
    so the teacher sees privileged info right before generating each turn.
    Response tokens are not modified.
    """
    action_hint = function_list_to_action_hint(function_list)
    if not action_hint.strip():
        return student_prompt_text

    hint_block = _make_hint_block(action_hint)
    assistant_prefix = _detect_assistant_prefix(student_prompt_text)
    if assistant_prefix is None:
        logger.warning("Could not detect assistant prefix in student prompt, "
                       "falling back to unmodified prompt.")
        return student_prompt_text

    return student_prompt_text.replace(assistant_prefix, hint_block + assistant_prefix)


def batch_build_teacher_inputs(
    student_input_ids: torch.Tensor,
    student_attention_mask: torch.Tensor,
    gold_function_lists: List[List[str]],
    tokenizer,
    max_length: int = 16384,
) -> Dict[str, torch.Tensor]:
    """Build tokenized teacher inputs with gold action hints injected.

    For each sample: decode student prompt → inject hint before each
    assistant_prefix → re-tokenize.  Response tokens are unchanged,
    so teacher logits at [-response_len:] align to student positions.

    Returns:
        Dict with 'teacher_input_ids', 'teacher_attention_mask'.
    """
    batch_size = len(gold_function_lists)

    student_texts = tokenizer.batch_decode(student_input_ids, skip_special_tokens=False)

    teacher_texts = []
    for i in range(batch_size):
        func_list = gold_function_lists[i]
        if func_list is not None and len(func_list) > 0:
            teacher_text = build_teacher_prompt(student_texts[i], func_list)
        else:
            teacher_text = student_texts[i]
            logger.warning(f"Sample {i} has no gold function_list, using student prompt as teacher.")
        teacher_texts.append(teacher_text)

    teacher_encoded = tokenizer(
        teacher_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    return {
        "teacher_input_ids": teacher_encoded["input_ids"],
        "teacher_attention_mask": teacher_encoded["attention_mask"],
    }
