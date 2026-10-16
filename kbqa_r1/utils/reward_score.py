import re
import numpy as np
from typing import List, Dict, Any, Union, Set

def normalize_answer(s):
    """Normalize answer for comparison."""
    if not s:
        return ""
    
    # Convert to lowercase
    s = s.lower()
    
    # Remove articles and punctuation
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'[^\w\s]', ' ', s)
    
    # Replace multiple spaces with a single space
    s = re.sub(r'\s+', ' ', s).strip()
    
    return s

def extract_answer(text):
    """Extract answer from text."""
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    return ""

def compute_f1(prediction_set, ground_truth_set):
    """Compute F1 score."""
    if not prediction_set or not ground_truth_set:
        return 0.0
    
    precision = len(prediction_set.intersection(ground_truth_set)) / len(prediction_set)
    recall = len(prediction_set.intersection(ground_truth_set)) / len(ground_truth_set)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)

def compute_score_em(solution_str: str, ground_truth: Dict[str, Any]) -> float:
    """
    Compute exact match score for KBQA.
    
    Args:
        solution_str: The solution string from the model
        ground_truth: The ground truth dictionary containing target answers
        
    Returns:
        float: The score (0.0 or 1.0)
    """
    # Extract answer from solution string
    predicted_answer = extract_answer(solution_str)
    
    if not predicted_answer:
        return 0.0
    
    # Normalize predicted answer
    normalized_pred = normalize_answer(predicted_answer)
    
    # Get ground truth answers
    target_answers = ground_truth.get("target", [])
    
    # Normalize ground truth answers
    normalized_targets = [normalize_answer(ans) for ans in target_answers]
    
    # Check for exact match
    if normalized_pred in normalized_targets:
        return 1.0
    
    # Check for partial match using F1 score
    pred_set = set(normalized_pred.split())
    
    # Compute F1 scores for each target
    f1_scores = []
    for target in normalized_targets:
        target_set = set(target.split())
        f1_scores.append(compute_f1(pred_set, target_set))
    
    # Return the maximum F1 score
    return max(f1_scores) if f1_scores else 0.0

def compute_score_sparql(solution_str: str, ground_truth: Dict[str, Any]) -> float:
    """
    Compute score for SPARQL query correctness.
    
    Args:
        solution_str: The solution string from the model
        ground_truth: The ground truth dictionary containing target SPARQL query
        
    Returns:
        float: The score (0.0 to 1.0)
    """
    # Extract SPARQL query from solution string
    sparql_match = re.search(r'<sparql>(.*?)</sparql>', solution_str, re.DOTALL)
    if not sparql_match:
        return 0.0
    
    predicted_sparql = sparql_match.group(1).strip()
    
    # Get ground truth SPARQL query
    target_sparql = ground_truth.get("sparql", "")
    
    # Simple string matching for now (can be improved with more sophisticated SPARQL comparison)
    if predicted_sparql == target_sparql:
        return 1.0
    
    # Normalize SPARQL queries for comparison
    normalized_pred = re.sub(r'\s+', ' ', predicted_sparql).strip()
    normalized_target = re.sub(r'\s+', ' ', target_sparql).strip()
    
    if normalized_pred == normalized_target:
        return 1.0
    
    # TODO: Implement more sophisticated SPARQL comparison
    
    return 0.0

def compute_score_combined(solution_str: str, ground_truth: Dict[str, Any]) -> float:
    """
    Compute combined score for KBQA.
    
    Args:
        solution_str: The solution string from the model
        ground_truth: The ground truth dictionary containing target answers and SPARQL query
        
    Returns:
        float: The combined score (0.0 to 1.0)
    """
    # Compute EM score
    em_score = compute_score_em(solution_str, ground_truth)
    
    # Compute SPARQL score
    sparql_score = compute_score_sparql(solution_str, ground_truth)
    
    # Combine scores (weighted average)
    # We prioritize EM score (70%) over SPARQL score (30%)
    return 0.7 * em_score + 0.3 * sparql_score 