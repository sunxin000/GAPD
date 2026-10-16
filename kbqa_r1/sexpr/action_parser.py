"""
Action Parser for converting LLM-generated reasoning actions to function calls
Based on KBQA-o1's action parsing mechanism
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

class ActionType(Enum):
    """Enumeration of supported action types"""
    # EXTRACT_ENTITY = "Extract_entity"
    FIND_RELATION = "Find_relation"
    MERGE = "Merge"
    ORDER = "Order"
    COMPARE = "Compare"
    TIME_CONSTRAINT = "Time_constraint"
    COUNT = "Count"
    # FINISH = "Finish"  # Removed: Not needed for kbqa-r1 training

@dataclass
class ActionResult:
    """Result of action parsing"""
    action_type: ActionType
    arguments: List[str]
    raw_text: str
    step_number: int
    is_valid: bool = True
    error_message: str = None

class ActionParser:
    """
    Parses LLM-generated reasoning actions into structured format
    Supports action types: Find_relation, Merge, Order, Compare, Time_constraint, Count
    Note: Finish action is intentionally excluded as it's not needed for training
    """
    
    def __init__(self):
        # Action pattern mappings - supports both with and without Action prefix
        self.action_patterns = {
            # ActionType.EXTRACT_ENTITY: {
            #     "pattern": r"(?:Action(\d+):\s*|^)Extract_entity\s*\[\s*([^\]]+)\s*\]",
            #     "description": "Extract topic entity from question"
            # },
            ActionType.FIND_RELATION: {
                "pattern": r"(?:Action(\d+):\s*|^)Find_relation\s*\[\s*([^|]+)\s*\|\s*([^\]]+)\s*\]",
                "description": "Find entities connected to the given entity through the specified relation"
            },
            ActionType.MERGE: {
                "pattern": r"(?:Action(\d+):\s*|^)Merge\s*\[\s*([^|]+)\s*\|\s*([^\]]+)\s*\]",
                "description": "Merge two expressions"
            },
            ActionType.ORDER: {
                # Three arguments now: mode | expression_id | relation
                "pattern": r"(?:Action(\d+):\s*|^)Order\s*\[\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^\]]+)\s*\]",
                "description": "Sorting operation for max/min value over a specific expression"
            },
            ActionType.COMPARE: {
                "pattern": r"(?:Action(\d+):\s*|^)Compare\s*\[\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^\]]+)\s*\]",
                "description": "Numerical comparison for range determination"
            },
            ActionType.TIME_CONSTRAINT: {
                "pattern": r"(?:Action(\d+):\s*|^)Time_constraint\s*\[\s*([^|]+)\s*\|\s*([^\]]+)\s*\]",
                "description": "Add time constraint"
            },
            ActionType.COUNT: {
                "pattern": r"(?:Action(\d+):\s*|^)Count\s*\[\s*([^\]]+)\s*\]",
                "description": "Count operation for number determination"
            },
            # ActionType.FINISH: {
            #     "pattern": r"(?:Action(\d+):\s*|^)Finish\s*\[\s*([^\]]+)\s*\]",
            #     "description": "Conclude and output expression"
            # }
            # Removed FINISH action - not needed for training
        }
        
        # Compile regex patterns for efficiency
        self.compiled_patterns = {
            action_type: re.compile(pattern_info["pattern"], re.IGNORECASE)
            for action_type, pattern_info in self.action_patterns.items()
        }
    
    def parse_action(self, text: str) -> Optional[ActionResult]:
        """
        Parse a single action from LLM output text
        
        Args:
            text: Raw text containing action
            
        Returns:
            ActionResult if parsing successful, None otherwise
        """
        text = text.strip()
        
        for action_type, pattern in self.compiled_patterns.items():
            match = pattern.search(text)
            if match:
                try:
                    # Step number might be None if no Action prefix
                    step_number_str = match.group(1)
                    step_number = int(step_number_str) if step_number_str else 0
                    
                    # Extract arguments based on action type
                    if action_type == ActionType.COMPARE:
                        # Three-argument action: operator, relation, number
                        arg1 = match.group(2).strip()  # operator
                        arg2 = match.group(3).strip()  # relation
                        arg3 = match.group(4).strip()  # number
                        arguments = [arg1, arg2, arg3]
                    elif action_type in [ActionType.MERGE, ActionType.TIME_CONSTRAINT, ActionType.FIND_RELATION]:
                        # Two-argument actions
                        arg1 = match.group(2).strip()
                        arg2 = match.group(3).strip()
                        arguments = [arg1, arg2]
                    elif action_type == ActionType.ORDER:
                        # Three-argument Order: mode, expression_id, relation
                        arg1 = match.group(2).strip()
                        arg2 = match.group(3).strip()
                        arg3 = match.group(4).strip()
                        arguments = [arg1, arg2, arg3]
                    else:
                        # Single-argument actions
                        arg = match.group(2).strip()
                        arguments = [arg]
                    
                    return ActionResult(
                        action_type=action_type,
                        arguments=arguments,
                        raw_text=text,
                        step_number=step_number,
                        is_valid=True
                    )
                    
                except Exception as e:
                    logger.warning(f"Error parsing action '{text}': {e}")
                    return ActionResult(
                        action_type=action_type,
                        arguments=[],
                        raw_text=text,
                        step_number=0,
                        is_valid=False,
                        error_message=str(e)
                    )
        
        # No pattern matched
        logger.warning(f"No matching action pattern found for: '{text}'")
        return None
    
    def parse_actions_from_text(self, text: str) -> List[ActionResult]:
        """
        Parse multiple actions from text containing <action></action> tags
        
        Args:
            text: Full text containing actions within <action></action> tags:
                  <think>...</think>
                  <action>
                  Action1: Extract_entity [ m.02mjmr ]
                  Action2: Find_relation [ person.place_of_birth ]
                  Action3: Finish [ expression1 ]
                  </action>
                  <answer>...</answer>
            
        Returns:
            List of ActionResult objects
        """
        # Extract content from <action></action> tags
        action_match = re.search(r'<action>(.*?)</action>', text, re.DOTALL)
        if not action_match:
            logger.debug("No <action></action> tags found in prediction")
            return []
        
        action_content = action_match.group(1).strip()
        
        # Parse each line that starts with "Action"
        actions = []
        lines = action_content.split('\n')
        
        for line in lines:
            line = line.strip()
            # Parse lines that contain action commands in brackets (with or without Action prefix)
            if line and '[' in line and ']' in line:
                action_result = self.parse_action(line)
                if action_result:
                    actions.append(action_result)
        
        return actions
    
    def validate_action_sequence(self, actions: List[ActionResult]) -> Tuple[bool, str]:
         #FIXME, the validate_action used for an action sequence but not a solo action, we should implement a validate function that for a solo action.
        """
        Validate a sequence of actions for logical consistency
        
        Args:
            actions: List of parsed actions
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not actions:
            return False, "Empty action sequence"
        
        # Check if sequence starts with Extract_entity
        # if actions[0].action_type != ActionType.EXTRACT_ENTITY:
        #     return False, "Sequence must start with Extract_entity action"
        
        # Removed: Finish action validation (not needed for training)
        # if actions[-1].action_type != ActionType.FINISH:
        #     return False, "Sequence must end with Finish action"
        
        # Check for duplicate Extract_entity actions (usually not allowed)
        extract_count = sum(1 for action in actions if action.action_type == ActionType.EXTRACT_ENTITY)
        if extract_count > 2:  # Allow some flexibility
            return False, f"Too many Extract_entity actions: {extract_count}"
        
        # Check step numbering
        for i, action in enumerate(actions):
            expected_step = i + 1
            if action.step_number != expected_step:
                return False, f"Invalid step numbering: expected {expected_step}, got {action.step_number}"
        
        return True, "Valid action sequence"
    
    def get_action_description(self, action_type: ActionType) -> str:
        """Get human-readable description of an action type"""
        return self.action_patterns.get(action_type, {}).get("description", "Unknown action")
    
    def format_action_for_display(self, action: ActionResult) -> str:
        """Format action for human-readable display"""
        if len(action.arguments) > 1:
            args_str = " | ".join(action.arguments)
        else:
            args_str = action.arguments[0] if action.arguments else ""
        return f"Action{action.step_number}: {action.action_type.value} [ {args_str} ]"


