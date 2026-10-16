# Modified 2026-10-16 for the GAPD release candidate; see CHANGELOG.md.
"""State matcher for aligning rollout execution states with gold trace steps.

Uses Jaccard similarity on entity MID sets to find the best-matching gold step
for a given rollout state. The matched step index is then used to retrieve the
corresponding per-turn teacher hint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Set

if TYPE_CHECKING:
    from kbqa_r1.gapd.gold_trace_builder import GoldTraceStep


def jaccard_similarity(a: Set[str], b: Set[str]) -> float:
    """Compute Jaccard similarity between two sets.

    Returns 1.0 if both sets are empty (identical empty state).
    Returns 0.0 if exactly one is empty.
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0.0


def match_state_to_gold_step(
    current_state: Set[str],
    gold_trace: List[GoldTraceStep],
    threshold: float = 0.95,
) -> Optional[int]:
    """Find the gold step whose prev_mid_set best matches the current state.

    Args:
        current_state: Set of MIDs representing the rollout state before
            the current turn's generation.
        gold_trace: Pre-computed gold execution trace.
        threshold: Minimum Jaccard score to consider a match valid.

    Returns:
        Gold step index if a match exceeds the threshold, else None.
    """
    best_idx: Optional[int] = None
    best_score = -1.0
    for i, step in enumerate(gold_trace):
        score = jaccard_similarity(current_state, step.prev_mid_set)
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx if best_score >= threshold else None
