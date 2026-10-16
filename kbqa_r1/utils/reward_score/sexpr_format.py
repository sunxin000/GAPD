"""
S-Expression Format Validation Function
Simple format validation for S-Expression generation responses
"""

import re


def is_valid_sexpr_sequence(text):
    """
    Check if the text follows valid S-Expression reasoning sequence format.
    Expected format: <|im_start|>assistant ... <think>...</think> <action>...</action> <information>...</information> <answer>...</answer>
    
    Updated to support multi-round reasoning: think -> action -> information cycles can repeat before final answer.
    """
    # Find the position of "<|im_start|>assistant" with potential whitespace
    # Note: Some chat templates (e.g., LLaMA) may not include this marker.
    # If absent, we validate the entire text instead of failing early.
    assistant_pattern = r"<\|im_start\|>assistant\s*"
    assistant_match = re.search(assistant_pattern, text)
    
    # Extract content starting after the assistant marker if present; otherwise use full text
    start_pos = assistant_match.end() if assistant_match else 0
    content = text[start_pos:]
    
    # Check for balanced tags - S-Expression uses think, action, information, answer
    tags_to_check = ["think", "action", "information", "answer"]
    for tag in tags_to_check:
        opening_count = len(re.findall(f"<{tag}>", content))
        closing_count = len(re.findall(f"</{tag}>", content))
        if opening_count != closing_count:
            return False, f"Mismatch in {tag} tags: {opening_count} opening vs {closing_count} closing tags"
    
    # Check for proper sequence pattern: think -> action -> information -> answer (with multi-round support)
    split_pattern = r"(</?(?:think|action|information|answer)>)"
    parts = re.split(split_pattern, content)
    
    # Track the current position in the expected sequence
    state = "start"  # start -> think -> action -> information -> think -> ... -> answer -> end
    
    # Check each part
    for i, part in enumerate(parts):
        # Skip empty parts
        if not part.strip():
            continue
            
        # Check if this is a tag
        if re.match(r"</?(?:think|action|information|answer)>", part):
            # This is a tag, check if it's valid in the current state
            if part == "<think>" and state in ["start", "information"]:
                state = "in_think"
            elif part == "</think>" and state == "in_think":
                state = "after_think"
            elif part == "<action>" and state == "after_think":
                state = "in_action"
            elif part == "</action>" and state == "in_action":
                state = "after_action"
            elif part == "<information>" and state == "after_action":
                state = "in_information"
            elif part == "</information>" and state == "in_information":
                state = "information"
            elif part == "<answer>" and state in ["after_think", "information"]:
                state = "in_answer"
            elif part == "</answer>" and state == "in_answer":
                state = "end"
            else:
                return False, f"Unexpected tag {part} in state {state}"
        else:
            # This is content, check if it's valid in the current state
            if state in ["in_think", "in_action", "in_information", "in_answer"]:
                # Content is allowed inside tags
                pass
            elif state == "information":
                # Allow guidance_prompt or other helper text after </information>
                # Accept any non-tag text until the next valid tag appears
                # (kept permissive to avoid false negatives when templates inject guidance)
                pass
            elif state in ["start", "after_think", "after_action"]:
                # Only whitespace is allowed between tags in these states
                if part.strip():
                    return False, f"Unexpected content between tags (state: {state})"
            else:
                return False, f"Unexpected content in state {state}"
    
    # Check final state
    if state != "end":
        return False, f"Incomplete sequence, ended in state {state}"
        
    return True, "Valid S-Expression sequence format" 
