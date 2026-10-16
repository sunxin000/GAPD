#!/usr/bin/env python3
"""Build gold execution traces (per-step MID sets) from a with-hint dataset.

This script:
1. Loads a with-hint dataset (jsonl or parquet) containing extra_info.action_hint
2. For each sample, parses the action_hint line by line
3. Uses SExprActionProcessor (with relation retrieval) to execute each action
4. After each action, executes the accumulated function-state as SPARQL
5. Records the intermediate MID set as per-turn state anchors
6. Outputs a JSON mapping sample_id -> List[GoldTraceStep]

Result can be merged into training data via --gold_trace_path in prepare_rl_dataset.py.
"""

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from kbqa_r1.llm_agent.sexpr_action_processor import SExprActionProcessor
from kbqa_r1.llm_agent.sexpr_state_manager import SExprStateManager
from kbqa_r1.sexpr.action_parser import ActionParser, ActionResult
from kbqa_r1.sexpr.relation_retrieval import RelationRetrieval
from kbqa_r1.sexpr.sexpr_executor import ExecutionResult, SExprExecutor
from kbqa_r1.sexpr.sexpr_generator import SExprGenerator
from kbqa_r1.sparql.sparql_manager import SPARQLConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_RE_MID = re.compile(r"[mg]\.[A-Za-z0-9_]+")
MAX_MID_SET_SIZE = 50


@dataclass
class GoldTraceStep:
    step_index: int
    prev_mid_set: List[str]
    action_function: str
    next_mid_set: List[str]
    per_turn_hint: str


def _extract_mid_set_from_result(execution_result: ExecutionResult) -> Set[str]:
    mid_set: Set[str] = set()
    if not execution_result.is_successful or not execution_result.results:
        return mid_set
    for item in execution_result.results:
        if isinstance(item, dict):
            for v in item.values():
                if isinstance(v, str):
                    mid_set.update(_RE_MID.findall(v))
                else:
                    mid_set.update(_RE_MID.findall(str(v)))
        elif isinstance(item, str):
            mid_set.update(_RE_MID.findall(item))
        else:
            mid_set.update(_RE_MID.findall(str(item)))
    return mid_set


def _truncate_mid_set(mid_set: Set[str], max_size: int = MAX_MID_SET_SIZE) -> Set[str]:
    if len(mid_set) <= max_size:
        return mid_set
    return set(sorted(mid_set)[:max_size])


def _to_str_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = value.strip()
        return [value] if value else []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if str(v).strip()]
    if hasattr(value, "tolist"):
        return _to_str_list(value.tolist())
    return [str(value)]


def _normalize_entities(raw_entities) -> List[Tuple[str, str]]:
    if raw_entities is None:
        return []
    if hasattr(raw_entities, "tolist"):
        raw_entities = raw_entities.tolist()
    if isinstance(raw_entities, set):
        raw_entities = list(raw_entities)
    elif not isinstance(raw_entities, (list, tuple)):
        raw_entities = [raw_entities]
    out = []
    for ent in raw_entities:
        if hasattr(ent, "tolist"):
            ent = ent.tolist()
        if isinstance(ent, (list, tuple)) and len(ent) >= 2:
            out.append((str(ent[0]), str(ent[1])))
    return out


def _build_target_expression(function_state: List[str]) -> str:
    if not function_state:
        return "expression"
    for stmt in reversed(function_state):
        stmt = stmt.strip()
        m_stop = re.search(r"STOP\(expression(\d+)\)", stmt)
        if m_stop:
            return f"expression{int(m_stop.group(1))}"
        m_lhs = re.match(r"expression(\d+)\s*=", stmt)
        if m_lhs:
            return f"expression{int(m_lhs.group(1))}"
    return "expression"


def _execute_function_state(
    function_state: List[str],
    sexpr_generator: SExprGenerator,
    sexpr_executor: SExprExecutor,
) -> Set[str]:
    """Execute accumulated function_state and return MID set."""
    if not function_state:
        return set()
    target_expr = _build_target_expression(function_state)
    exec_fs = list(function_state)
    if not any("STOP(" in f for f in exec_fs):
        exec_fs.append(f"{target_expr} = STOP({target_expr})")

    sexpr_result = sexpr_generator.generate_sexpr_from_strings(exec_fs, target_expr)
    if not sexpr_result.is_valid:
        return set()

    try:
        exec_result = sexpr_executor.execute_sexpr(
            sexpr_result.sexpr,
            validate=False,
            function_state=exec_fs,
        )
        return _extract_mid_set_from_result(exec_result)
    except Exception as e:
        logger.debug(f"Execution failed: {e}")
        return set()


def build_sample_gold_trace(
    sample_id: str,
    action_hint: str,
    candidate_entities: List[Tuple[str, str]],
    action_parser: ActionParser,
    action_processor: SExprActionProcessor,
    state_manager: SExprStateManager,
    sexpr_generator: SExprGenerator,
    sexpr_executor: SExprExecutor,
    sample_idx: int = 0,
) -> Optional[List[GoldTraceStep]]:
    """Build gold trace for a single sample by executing actions step-by-step."""
    if not action_hint or not action_hint.strip():
        return None

    # Initialize state
    state_manager.clear_sample_state(sample_idx)
    if candidate_entities:
        state_manager.set_sample_entities(sample_idx, candidate_entities)

    # Parse all actions
    actions = []
    for line in action_hint.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        parsed = action_parser.parse_action(line)
        if parsed:
            actions.append(parsed)

    if not actions:
        return None

    trace: List[GoldTraceStep] = []
    prev_mid_set: Set[str] = set()
    function_state: List[str] = state_manager.get_sample_function_state(sample_idx) or []

    for step_idx, action in enumerate(actions):
        action_type_name = action.action_type.name if hasattr(action.action_type, "name") else str(action.action_type)

        # Skip Finish/STOP actions (no state change)
        if action_type_name.lower() in ("finish", "stop"):
            continue

        hint_line = action.raw_text or str(action)

        # Execute the action through the processor (this builds function_state)
        try:
            _ = action_processor.process_actions_with_retrieval(
                [action],
                function_state.copy(),
                candidate_entities,
                sample_id=sample_idx,
            )
        except Exception as e:
            logger.warning(f"[{sample_id}] Step {step_idx} action processing failed: {e}")
            # Continue: function_state may have been partially updated

        # Re-read updated function_state
        function_state = state_manager.get_sample_function_state(sample_idx) or []

        # Execute accumulated function_state to get MID set
        next_mid_set_raw = _execute_function_state(
            function_state, sexpr_generator, sexpr_executor
        )
        next_mid_set = _truncate_mid_set(next_mid_set_raw)

        trace.append(
            GoldTraceStep(
                step_index=step_idx,
                prev_mid_set=sorted(prev_mid_set),
                action_function=hint_line,
                next_mid_set=sorted(next_mid_set),
                per_turn_hint=hint_line,
            )
        )

        prev_mid_set = next_mid_set

    return trace


def load_with_hint_data(path: str) -> List[Dict]:
    """Load with-hint dataset from jsonl or parquet."""
    if path.endswith(".jsonl"):
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records
    elif path.endswith(".parquet"):
        import pandas as pd
        df = pd.read_parquet(path)
        return df.to_dict("records")
    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]
    else:
        raise ValueError(f"Unsupported format: {path}")


def main():
    parser = argparse.ArgumentParser(description="Build gold execution traces")
    parser.add_argument("--input", type=str, required=True, help="Path to with-hint dataset (jsonl/parquet)")
    parser.add_argument("--output", type=str, required=True, help="Output path (json)")
    parser.add_argument("--dataset", type=str, default="webqsp", choices=["webqsp", "grailqa", "graphq"])
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--sparql_url", type=str, default="http://localhost:8890/sparql")
    parser.add_argument("--log-interval", type=int, default=50, help="Log progress every N samples")
    args = parser.parse_args()

    logger.info(f"Loading dataset from {args.input}...")
    records = load_with_hint_data(args.input)
    if args.limit:
        records = records[:args.limit]
    logger.info(f"Processing {len(records)} samples")

    # Initialize components (same pipeline as run_actions.py)
    sparql_config = SPARQLConfig(sparql_url=args.sparql_url, use_odbc=True)
    sexpr_generator = SExprGenerator()
    sexpr_executor = SExprExecutor(sparql_config, dataset_type=args.dataset)
    state_manager = SExprStateManager()
    relation_retrieval = RelationRetrieval(sparql_config=sparql_config, dataset=args.dataset)
    action_processor = SExprActionProcessor(relation_retrieval, state_manager)
    action_parser = ActionParser()

    gold_traces: Dict[str, List[dict]] = {}
    stats = {"total": 0, "success": 0, "empty": 0, "steps_total": 0, "steps_nonempty": 0}

    for idx, item in enumerate(records):
        extra = item.get("extra_info", {}) or {}
        if isinstance(extra, str):
            try:
                extra = json.loads(extra)
            except Exception:
                extra = {}
        elif hasattr(extra, "tolist"):
            extra = extra.tolist()

        sample_id = extra.get("id", str(idx))
        action_hint = extra.get("action_hint", "")
        candidate_entities = _normalize_entities(extra.get("candidate_entities") or extra.get("extracted_entities"))

        trace = build_sample_gold_trace(
            sample_id=sample_id,
            action_hint=action_hint,
            candidate_entities=candidate_entities,
            action_parser=action_parser,
            action_processor=action_processor,
            state_manager=state_manager,
            sexpr_generator=sexpr_generator,
            sexpr_executor=sexpr_executor,
            sample_idx=idx,
        )

        stats["total"] += 1
        if trace:
            stats["success"] += 1
            gold_traces[sample_id] = [asdict(t) for t in trace]
            for t in trace:
                stats["steps_total"] += 1
                if t.next_mid_set:
                    stats["steps_nonempty"] += 1
                else:
                    stats["empty"] += 1
        else:
            gold_traces[sample_id] = []

        if (idx + 1) % args.log_interval == 0:
            logger.info(
                f"Processed {idx + 1}/{len(records)} samples | "
                f"traces={stats['success']} | "
                f"nonempty_steps={stats['steps_nonempty']}/{stats['steps_total']} ({stats['steps_nonempty']/max(1,stats['steps_total'])*100:.1f}%)"
            )

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(gold_traces, f, ensure_ascii=False, indent=2)

    logger.info(
        f"Done. Saved {len(gold_traces)} traces to {args.output}\n"
        f"  Samples: {stats['total']}\n"
        f"  With trace: {stats['success']}\n"
        f"  Steps total: {stats['steps_total']}\n"
        f"  Steps with non-empty MID set: {stats['steps_nonempty']} ({stats['steps_nonempty']/max(1,stats['steps_total'])*100:.1f}%)\n"
        f"  Steps with empty MID set: {stats['empty']} ({stats['empty']/max(1,stats['steps_total'])*100:.1f}%)"
    )


if __name__ == "__main__":
    main()
