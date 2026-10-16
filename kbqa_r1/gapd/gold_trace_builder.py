"""Gold execution trace builder for per-turn state-anchored GAPD distillation.

Builds gold trace by executing action_hint actions one-by-one using the same
function-state building logic as SExprActionProcessor (run_actions.py path).
This ensures the gold trace aligns with the actual rollout execution path,
rather than directly grouping the ground-truth function_list.

Key difference from old approach:
  OLD: function_list -> group by START/JOIN -> execute sexpr directly
  NEW: action_hint -> parse actions -> simulate action processor function-state
       building -> execute sexpr after each action

The new approach correctly handles expression ID management (implicit START,
chained references, MERGE rollback) and uses the action_hint's inlined entity
MIDs and selected relations, making the trace consistent with rollout behavior.
"""

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Set

from kbqa_r1.sexpr.sexpr_executor import ExecutionResult, SExprExecutor
from kbqa_r1.sexpr.sexpr_generator import SExprGenerator

logger = logging.getLogger(__name__)

_RE_MID = re.compile(r"[mg]\.[A-Za-z0-9_]+")
MAX_MID_SET_SIZE = 50


@dataclass
class GoldTraceStep:
    step_index: int
    prev_mid_set: Set[str]
    action_function: str
    next_mid_set: Set[str]
    per_turn_hint: str


def _extract_mid_set_from_result(execution_result: ExecutionResult) -> Set[str]:
    """Extract pure MID values (e.g. m.02mjmr) from an ExecutionResult."""
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
    """Truncate a MID set to at most max_size elements (sorted for determinism)."""
    if len(mid_set) <= max_size:
        return mid_set
    return set(sorted(mid_set)[:max_size])


def _parse_action_hint_line(line: str) -> tuple[str, List[str]]:
    """Parse one action hint line into (action_type, arguments).

    Supported formats:
      Find_relation [ entity | relation ]
      Merge [ expr1 | expr2 ]
      Order [ mode | expr | relation ]
      Compare [ operator | relation | number ]
      Time_constraint [ relation | time ]
      Count [ expression ]
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return "", []

    bracket_idx = line.find("[")
    if bracket_idx == -1:
        return "", []

    action_type = line[:bracket_idx].strip()
    inner = line[bracket_idx + 1:]
    if inner.endswith("]"):
        inner = inner[:-1]

    parts = [p.strip() for p in inner.split("|")]
    return action_type, parts


def _get_expr_id(token: str) -> Optional[int]:
    """Extract numeric ID from expression token like 'expression1'."""
    if token.startswith("expression"):
        try:
            return int(token.replace("expression", ""))
        except ValueError:
            pass
    return None


def _build_target_expression(function_state: List[str]) -> str:
    """Derive target expression name from the last assignment in function_state."""
    if not function_state:
        return "expression"
    for stmt in reversed(function_state):
        m = re.match(r"(expression\d+)\s*=", stmt.strip())
        if m:
            return m.group(1)
    return "expression"


def _apply_action_to_function_state(
    action_type: str,
    args: List[str],
    function_state: List[str],
    current_expr_id: int,
) -> tuple[List[str], int]:
    """Simulate SExprActionProcessor's function-state building for one action.

    Returns:
        (updated_function_state, updated_current_expr_id)
    """
    if action_type == "Find_relation":
        if len(args) != 2:
            return function_state, current_expr_id
        entity, relation = args[0], args[1]
        expr_id = _get_expr_id(entity)
        if expr_id is not None:
            # Chain: reuse the referenced expression ID
            current_expr_id = expr_id
            fn = f"expression{current_expr_id} = JOIN('{relation}', expression{current_expr_id})"
            function_state.append(fn)
        else:
            # New entity: start a new expression
            current_expr_id += 1
            function_state.append(f"expression{current_expr_id} = START('{entity}')")
            function_state.append(
                f"expression{current_expr_id} = JOIN('{relation}', expression{current_expr_id})"
            )

    elif action_type == "Merge":
        if len(args) != 2:
            return function_state, current_expr_id
        expr1, expr2 = args[0], args[1]
        prev = current_expr_id - 1
        if prev < 1:
            prev = 1
        function_state.append(f"expression{prev} = AND({expr1}, {expr2})")
        current_expr_id = prev

    elif action_type == "Order":
        if len(args) != 3:
            return function_state, current_expr_id
        mode, expr_token, relation = args[0], args[1], args[2]
        expr_id = _get_expr_id(expr_token)
        if expr_id is None:
            # Implicit START for ontology type / literal
            current_expr_id += 1
            function_state.append(f"expression{current_expr_id} = START('{expr_token}')")
            expr_id = current_expr_id
        function_state.append(
            f"expression{expr_id} = ARG('{mode}', expression{expr_id}, '{relation}')"
        )

    elif action_type == "Compare":
        if len(args) != 3:
            return function_state, current_expr_id
        operator, relation, number = args[0], args[1], args[2]
        current_expr_id += 1
        function_state.append(f"expression{current_expr_id} = START('{number}')")
        function_state.append(
            f"expression{current_expr_id} = CMP('{operator}', '{relation}', expression{current_expr_id})"
        )

    elif action_type == "Time_constraint":
        if len(args) != 2:
            return function_state, current_expr_id
        relation, time_val = args[0], args[1]
        function_state.append(
            f"expression{current_expr_id} = TC(expression{current_expr_id}, '{relation}', '{time_val}')"
        )

    elif action_type == "Count":
        if len(args) != 1:
            return function_state, current_expr_id
        expression = args[0]
        function_state.append(f"expression{current_expr_id} = COUNT({expression})")

    else:
        logger.warning(f"[gold_trace] Unknown action type: {action_type}")

    return function_state, current_expr_id


def build_gold_trace(
    sexpr_executor: SExprExecutor,
    action_hint: Optional[str] = None,
    function_list: Optional[List[str]] = None,
    extra_info: Optional[dict] = None,
) -> Optional[List[GoldTraceStep]]:
    """Execute gold actions step-by-step and record per-step MID sets.

    Parses action_hint lines, simulates SExprActionProcessor's function-state
    building logic, and executes the accumulated function-state after each action
    to obtain the intermediate entity set (MID set). This mirrors the actual
    rollout execution path.

    Args:
        sexpr_executor: Initialized SExprExecutor with SPARQL manager.
        action_hint: Pre-computed action hint string (one action per line). This
            is the authoritative source because it has correct entity MID inlining.
            If not provided, returns None. There is intentionally no function_list
            fallback because that path can generate incorrect traces.
        function_list: DEPRECATED and ignored.
        extra_info: DEPRECATED and ignored.

    Returns:
        List of GoldTraceStep (one per non-STOP action), or None if empty.
    """
    if not action_hint or not action_hint.strip():
        logger.warning(
            "[gold_trace] action_hint is required. "
            "function_list fallback has been removed to avoid incorrect traces."
        )
        return None

    sexpr_generator = SExprGenerator()
    hint_lines = [ln.strip() for ln in action_hint.strip().split("\n") if ln.strip()]

    trace: List[GoldTraceStep] = []
    prev_mid_set: Set[str] = set()
    function_state: List[str] = []
    current_expr_id = 0
    action_index = 0

    for line in hint_lines:
        action_type, args = _parse_action_hint_line(line)
        if not action_type:
            continue

        # Skip Finish/STOP actions — they don't change state
        if action_type.lower() in ("finish", "stop"):
            continue

        # Apply action to function_state (mirrors action processor)
        function_state, current_expr_id = _apply_action_to_function_state(
            action_type, args, function_state, current_expr_id
        )

        # Execute to get next_mid_set
        exec_functions = list(function_state)
        target_expr = _build_target_expression(exec_functions)
        if not any("STOP(" in f for f in exec_functions):
            exec_functions.append(f"{target_expr} = STOP({target_expr})")

        sexpr_result = sexpr_generator.generate_sexpr_from_strings(
            exec_functions, target_expr
        )

        if sexpr_result.is_valid:
            try:
                exec_result = sexpr_executor.execute_sexpr(
                    sexpr_result.sexpr,
                    validate=False,
                    function_state=exec_functions,
                )
                next_mid_set = _extract_mid_set_from_result(exec_result)
            except Exception as e:
                logger.warning(
                    f"[gold_trace] SPARQL execution failed at action {action_index}: {e}"
                )
                next_mid_set = set()
        else:
            logger.warning(
                f"[gold_trace] Failed to generate sexpr for action {action_index}: {line}"
            )
            next_mid_set = set()

        next_mid_set = _truncate_mid_set(next_mid_set)

        trace.append(
            GoldTraceStep(
                step_index=action_index,
                prev_mid_set=prev_mid_set,
                action_function=line,
                next_mid_set=next_mid_set,
                per_turn_hint=line,
            )
        )

        prev_mid_set = next_mid_set
        action_index += 1

    return trace
