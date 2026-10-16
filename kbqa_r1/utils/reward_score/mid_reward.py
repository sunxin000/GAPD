import json
import logging
import os
import re

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOG_LEVEL", "INFO"))

# Timeout detection patterns (imported from timeout logic)
TIMEOUT_INDICATORS = (
    'HYT00',
    'timeout',
)

def _is_timeout_error(text: str) -> bool:
    """Check if the text contains timeout error indicators."""
    if not text:
        return False
    text_lower = text.lower()
    return any(indicator.lower() in text_lower for indicator in TIMEOUT_INDICATORS)

def extract_mid_list(solution_str: str) -> list[str] | None:
    """
    Extracts a list of answer values from the <answer> tag in the solution string.
    Handles both entity MIDs (e.g., m.02mjmr) and literal values (e.g., dates, numbers).
    Assumes answers are space-separated within the tag, or potentially a JSON list.
    Handles potential errors gracefully.
    """
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL | re.IGNORECASE)) # Added IGNORECASE

    if not matches:
        return None # No answer tag found

    # Get content of the last answer tag
    content = matches[-1].group(1).strip()

    # Try parsing as JSON list first (more robust)
    try:
        # Simple check if it looks like a list
        if content.startswith('[') and content.endswith(']'):
            # Replace single quotes with double quotes for valid JSON
            content = content.replace("'", '"')
            mid_list = json.loads(content)
            if isinstance(mid_list, list):
                # Ensure all items are strings
                return [str(mid).strip() for mid in mid_list if str(mid).strip()] # Ensure non-empty strings
            else:
                # Parsed but wasn't a list
                 return None
    except json.JSONDecodeError:
        # Not a valid JSON list, try splitting by space
        pass # Fall through to space splitting

    # Fallback: Assume space-separated MIDs if not a JSON list
    # Filter out empty strings that might result from multiple spaces
    mid_list = [mid.strip() for mid in content.split() if mid.strip()] # Ensure non-empty strings
    if not mid_list: # Handle case where answer tag is empty or only spaces
        return None
    return mid_list


def find_in_list(entry, elist):
    """Checks if an entry exists in a list (exact match)."""
    return entry in elist

def calculate_mid_prf1(gold_mid_list: list[str], pred_mid_list: list[str]) -> tuple[float, float, float]:
    """
    Calculates Precision, Recall, and F1 score for MID lists based on exact match.
    Adapted from webqsp_evaluate.py's CalculatePRF1 logic.
    Returns (precision, recall, f1)
    """
    if not gold_mid_list:
        # Handle empty gold list cases based on webqsp_evaluate.py
        precision = 1.0 if not pred_mid_list else 0.0
        recall = 1.0
        f1 = 1.0 if not pred_mid_list else 0.0
        return precision, recall, f1

    if not pred_mid_list:
        # Handle empty prediction list case
        return 1.0, 0.0, 0.0

    # Use sets for efficient comparison - ensure items are hashable (strings are)
    gold_set = set(gold_mid_list)
    pred_set = set(pred_mid_list)

    tp = len(gold_set.intersection(pred_set))
    fp = len(pred_set.difference(gold_set))
    fn = len(gold_set.difference(pred_set))

    if (tp + fp) == 0:
         precision = 1.0 # Avoid division by zero if pred_set is empty (already handled), safety check
    else:
         precision = tp / (tp + fp)

    if (tp + fn) == 0:
        recall = 1.0 # Avoid division by zero if gold_set is empty (already handled)
    else:
        recall = tp / (tp + fn)

    if (precision + recall) == 0:
        f1 = 0.0 # Avoid division by zero
    else:
        f1 = (2 * precision * recall) / (precision + recall)

    return precision, recall, f1


def calculate_best_mid_f1(gold_mid_lists: list[list[str]], pred_mid_list: list[str]) -> tuple[float, float, float]:
    """
    Calculates the best F1 score by comparing predicted MIDs against multiple gold answer candidates.
    Similar to WebQSP evaluation logic where each question can have multiple valid answer sets.
    
    Args:
        gold_mid_lists: List of gold MID lists (multiple possible correct answers)
        pred_mid_list: Single predicted MID list
        
    Returns:
        (best_precision, best_recall, best_f1) - the best scores across all gold candidates
    """
    if not gold_mid_lists:
        # No gold answers provided
        precision = 1.0 if not pred_mid_list else 0.0
        recall = 1.0
        f1 = 1.0 if not pred_mid_list else 0.0
        return precision, recall, f1
    
    if not pred_mid_list:
        # No predictions provided
        return 1.0, 0.0, 0.0
    
    best_f1 = -1.0
    best_precision = 0.0
    best_recall = 0.0
    
    # Try each gold answer candidate and find the best F1 score
    for gold_mid_list in gold_mid_lists:
        if not gold_mid_list:  # Skip empty gold lists
            continue
            
        precision, recall, f1 = calculate_mid_prf1(gold_mid_list, pred_mid_list)
        
        if f1 > best_f1:
            best_f1 = f1
            best_precision = precision
            best_recall = recall
    
    # If no valid gold lists found, return default values
    if best_f1 < 0:
        precision = 1.0 if not pred_mid_list else 0.0
        recall = 1.0
        f1 = 1.0 if not pred_mid_list else 0.0
        return precision, recall, f1
    
    return best_precision, best_recall, best_f1

def compute_mid_reward(solution_str: str, ground_truth, format_score: float = 0.0, correct_reward: float = 1.0, structure_format_score: float = 0.1, training_step: int = 0, timeout_penalty: float = 0.0) -> dict:
    """
    Computes a reward score based on F1 score between predicted and ground truth MID lists.
    Enhanced with progressive format reward for S-Expression training and timeout penalty.
    Supports both single answer list and multiple answer candidates (for WebQSP-style evaluation).

    Args:
        solution_str: The raw output string from the model.
        ground_truth: Either a list of correct MID strings, or a list of lists (multiple answer candidates).
        format_score: Reward value if the format is incorrect (e.g., missing <answer> tag).
        correct_reward: Multiplier for the F1 score when format is correct.
        structure_format_score: Base reward for correct S-Expression format structure.
        training_step: Current training step for progressive reward scaling.
        timeout_penalty: Penalty reward for timeout errors (default: -1.0).

    Returns:
        A dictionary with components:
        - total: final reward used for training
        - mid_f1: F1 score between predicted and gold MIDs
        - structure_reward: additional structure/format reward (0 if not applied)
        - timeout_penalty: penalty applied if timeout detected (0 if no timeout)
    """

    if isinstance(ground_truth, dict):
        ground_truth = ground_truth.get("target", ground_truth.get("answer", []))

    # Check for timeout error first (highest priority)
    if _is_timeout_error(solution_str):
        return {
            'total': timeout_penalty,
            'mid_f1': 0.0,
            'structure_reward': 0.0,
            'timeout_penalty': timeout_penalty,
        }

    #  
    predicted_mids = extract_mid_list(solution_str)

    if predicted_mids is None:
        # No MIDs extracted, return zeros
        return {
            'total': 0.0,
            'mid_f1': 0.0,
            'structure_reward': 0.0,
            'timeout_penalty': 0.0,
        }

    # Determine if ground_truth is a single list or multiple lists
    # logger.warning(f"Ground Truth Type: {type(ground_truth)}, Value: {ground_truth}")
    
    if ground_truth and isinstance(ground_truth[0], list):
        # Multiple answer candidates (WebQSP-style)
        gold_mid_lists = []
        for gt_list in ground_truth:
            cleaned_gt = [str(mid).strip() for mid in gt_list if str(mid).strip()]
            if cleaned_gt:  # Only add non-empty lists
                gold_mid_lists.append(cleaned_gt)
        
        # Calculate best F1 score across all candidates
        precision, recall, f1 = calculate_best_mid_f1(gold_mid_lists, predicted_mids)
    else:
        # Single answer list (traditional approach)
        cleaned_ground_truth = [str(mid).strip() for mid in ground_truth if str(mid).strip()]
        # Calculate F1 score using the logic from webqsp_evaluate.py
        precision, recall, f1 = calculate_mid_prf1(cleaned_ground_truth, predicted_mids)

    # print(f"DEBUG: Predicted MIDs: {predicted_mids}, Gold MIDs: {cleaned_ground_truth}, F1: {f1}")

    # Base reward is F1 score (outcome quality)
    base_reward = f1 * correct_reward
    
    # Only add format reward if the answer is correct (F1 > 0)
    format_reward = 0.0
    if f1 > 0 and structure_format_score != 0:
        # Import here to avoid circular imports
        try:
            from .sexpr_format import is_valid_sexpr_sequence
        except ImportError:
            # Fallback for direct execution
            import sexpr_format
            is_valid_sexpr_sequence = sexpr_format.is_valid_sexpr_sequence
        
        is_valid_format, _ = is_valid_sexpr_sequence(solution_str)
        
        if is_valid_format:
            # Apply full format reward for correct format
            format_reward = structure_format_score
        else:
            # Calculate partial format reward for partial correctness
            format_reward = _calculate_partial_format_reward(solution_str, structure_format_score)
        
        # Apply progressive scaling based on training step
        # format_reward = _apply_progressive_scaling(format_reward, training_step)
    
    # Total reward = base reward + format reward (only if F1 > 0)
    return {
        'total': base_reward + format_reward,
        'mid_f1': float(f1),
        'structure_reward': float(format_reward),
        'timeout_penalty': 0.0,
    }
    
    # return base_reward + format_reward


def _calculate_partial_format_reward(solution_str: str, base_reward: float) -> float:
    """
    Calculate partial format reward based on how many correct tags are present.
    This provides gradual feedback instead of all-or-nothing format validation.
    
    Args:
        solution_str: The solution string to analyze
        base_reward: The base reward for perfect format
        
    Returns:
        Partial reward based on format completeness
    """
    import re

    # Define required tags for S-Expression format
    required_tags = ["think", "action", "information", "answer"]
    tag_scores = {}
    
    # Check for balanced tag pairs
    for tag in required_tags:
        opening_count = len(re.findall(f"<{tag}>", solution_str))
        closing_count = len(re.findall(f"</{tag}>", solution_str))
        
        if opening_count > 0 and closing_count > 0:
            # Both opening and closing tags present
            if opening_count == closing_count:
                tag_scores[tag] = 1.0  # Perfect balance
            else:
                tag_scores[tag] = 0.5  # Partial balance
        elif opening_count > 0 or closing_count > 0:
            # Only one type of tag present
            tag_scores[tag] = 0.2  # Minimal credit
        else:
            tag_scores[tag] = 0.0  # No tags
    
    # Calculate weighted average
    total_score = sum(tag_scores.values()) / len(required_tags)
    
    return base_reward * total_score * 0.3  # Scale down partial rewards


def _apply_progressive_scaling(reward: float, training_step: int) -> float:
    """
    Apply progressive scaling to format rewards based on training progress.
    Early training gets smaller rewards to avoid overwhelming the model.
    
    Args:
        reward: The base reward to scale
        training_step: Current training step
        
    Returns:
        Scaled reward based on training progress
    """
    if training_step <= 0:
        return reward
    
    # Progressive scaling parameters
    warmup_steps = 10  # Steps for warmup period (adjusted for 165 total steps)
    max_scale = 1.0      # Maximum scale factor
    
    if training_step <= warmup_steps:
        # Linear scaling from 0.1 to 1.0 during warmup
        scale_factor = 0.1 + (training_step / warmup_steps) * (max_scale - 0.1)
    else:
        # Full scale after warmup
        scale_factor = max_scale
    
    return reward * scale_factor

