"""
Function Builder for converting parsed actions to function call sequences
Based on KBQA-o1's function generation mechanism
"""
import logging
import os
from dataclasses import dataclass
from typing import List

from .action_parser import ActionResult, ActionType
from .constants import COMPARISON_MODE_MAPPING

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

@dataclass
class FunctionCall:
    """Represents a single function call in the sequence"""
    expression_id: str
    function_name: str
    arguments: List[str]
    raw_function: str

class FunctionBuilder:
    """
    Converts parsed actions into function call sequences
    Mirrors the logic from KBQA-o1's prompt_function.py
    """
    
    def __init__(self):
        self.current_expression_id = 0
        self.function_sequence = []
        
    def reset(self):
        """Reset builder state for new sequence"""
        self.current_expression_id = 0
        self.function_sequence = []
    
    def _get_next_expression_id(self) -> str:
        """Get next available expression ID"""
        self.current_expression_id += 1
        return str(self.current_expression_id)
    
    def _get_current_expression_id(self) -> str:
        """Get current expression ID"""
        return str(self.current_expression_id)
    
    def build_start_function(self, action: ActionResult) -> FunctionCall:
        """
        Build START function from Extract_entity action
        
        Args:
            action: Parsed Extract_entity action
            
        Returns:
            FunctionCall object
        """
        if action.action_type != ActionType.EXTRACT_ENTITY:
            raise ValueError(f"Expected Extract_entity action, got {action.action_type}")
        
        entity = action.arguments[0]
        expr_id = self._get_next_expression_id()
        
        # In KBQA-o1, entities are either MIDs (m.xxx, g.xxx) or literals
        # For now, assume entity is already in correct format
        function_call = f"expression{expr_id} = START('{entity}')"
        
        return FunctionCall(
            expression_id=expr_id,
            function_name="START",
            arguments=[entity],
            raw_function=function_call
        )
    
    def build_join_function(self, action: ActionResult, target_expr_id: str = None) -> FunctionCall:
        """
        Build JOIN function from Find_relation action
        
        Args:
            action: Parsed Find_relation action
            target_expr_id: Target expression ID to join with (optional, will use entity from action)
            
        Returns:
            FunctionCall object
        """
        if action.action_type != ActionType.FIND_RELATION:
            raise ValueError(f"Expected Find_relation action, got {action.action_type}")
        
        if len(action.arguments) == 2:
            # New format: [entity, relation] - both are already processed
            entity, relation = action.arguments
            
            # Use the entity from action as the starting point
            expr_id = target_expr_id or self._get_next_expression_id()
            
            # Handle reverse relations (R relation_name)
            if relation.startswith('(R ') and relation.endswith(')'):
                relation_formatted = relation
            elif relation.startswith('R '):
                relation_formatted = f"(R {relation[2:]})" #TODO why 2: ?
            else:
                relation_formatted = relation
            
            # Create JOIN function using the provided entity as expression0
            function_call = f"expression{expr_id} = JOIN('{relation_formatted}', '{entity}')"
            
            return FunctionCall(
                expression_id=expr_id,
                function_name="JOIN",
                arguments=[relation_formatted, entity],
                raw_function=function_call
            )
        elif len(action.arguments) == 1:
            # Old format: [relation] - need to use current expression
            relation = action.arguments[0]
            expr_id = target_expr_id or self._get_current_expression_id()
            
            # Safety check: ensure we don't reference undefined expressions
            if expr_id == "0" or int(expr_id) == 0:
                raise ValueError(f"Cannot JOIN with undefined expression0. Current expression ID: {expr_id}")
            
            # Handle reverse relations (R relation_name)
            if relation.startswith('(R ') and relation.endswith(')'):
                relation_formatted = relation
            elif relation.startswith('R '):
                relation_formatted = f"(R {relation[2:]})"
            else:
                relation_formatted = relation
            
            function_call = f"expression{expr_id} = JOIN('{relation_formatted}', expression{expr_id})"
            
            return FunctionCall(
                expression_id=expr_id,
                function_name="JOIN",
                arguments=[relation_formatted, f"expression{expr_id}"],
                raw_function=function_call
            )
        else:
            raise ValueError(f"Find_relation requires 1 or 2 arguments, got {len(action.arguments)}")
    
    def build_and_function(self, action: ActionResult) -> FunctionCall:
        """
        Build AND function from Merge action
        
        Args:
            action: Parsed Merge action
            
        Returns:
            FunctionCall object
        """
        if action.action_type != ActionType.MERGE:
            raise ValueError(f"Expected Merge action, got {action.action_type}")
        
        expr1, expr2 = action.arguments
        new_expr_id = self._get_next_expression_id()
        
        function_call = f"expression{new_expr_id} = AND({expr1}, {expr2})"
        
        return FunctionCall(
            expression_id=new_expr_id,
            function_name="AND",
            arguments=[expr1, expr2],
            raw_function=function_call
        )
    
    def build_arg_function(self, action: ActionResult, target_expr_id: str = None) -> FunctionCall:
        """
        Build ARG function from Order action
        
        Args:
            action: Parsed Order action
            target_expr_id: Target expression ID
            
        Returns:
            FunctionCall object
        """
        if action.action_type != ActionType.ORDER:
            raise ValueError(f"Expected Order action, got {action.action_type}")
        
        # Three-argument Order: [mode | expression_id | relation]
        if len(action.arguments) == 3:
            mode, expr_token, relation = action.arguments
            # expr_token expected like 'expression3'
            if expr_token.startswith('expression'):
                try:
                    expr_id = expr_token.replace('expression', '')
                except Exception:
                    expr_id = target_expr_id or self._get_current_expression_id()
            else:
                expr_id = target_expr_id or self._get_current_expression_id()
        else:
            mode, relation = action.arguments
            expr_id = target_expr_id or self._get_current_expression_id()
        
        # Map mode to standard format
        mode_mapping = {
            'maximum': 'ARGMAX',
            'max': 'ARGMAX',
            'minimum': 'ARGMIN',
            'min': 'ARGMIN',
            'ARGMAX': 'ARGMAX',
            'ARGMIN': 'ARGMIN'
        }
        
        standardized_mode = mode_mapping.get(mode.upper(), mode.upper())
        
        function_call = f"expression{expr_id} = ARG('{standardized_mode}', expression{expr_id}, '{relation}')"
        
        return FunctionCall(
            expression_id=expr_id,
            function_name="ARG",
            arguments=[standardized_mode, f"expression{expr_id}", relation],
            raw_function=function_call
        )
    
    def build_cmp_function(self, action: ActionResult, target_expr_id: str = None) -> FunctionCall:
        """
        Build CMP function from Compare action
        
        Args:
            action: Parsed Compare action
            target_expr_id: Target expression ID
            
        Returns:
            FunctionCall object
        """
        if action.action_type != ActionType.COMPARE:
            raise ValueError(f"Expected Compare action, got {action.action_type}")
        
        if len(action.arguments) == 3:
            # New format: [operator | relation | number]
            mode, relation, number = action.arguments
            expr_id = self._get_next_expression_id()
            
            # Use common comparison mode mapping
            standardized_mode = COMPARISON_MODE_MAPPING.get(mode, mode.lower())
            
            # Normalize and format number with proper type annotation
            # 1) strip outer quotes if present; 2) ensure value part has no quotes;
            # 3) ensure typed-literal format is value^^datatype_iri (without angle brackets here)
            raw_number = (number or "").strip()
            # Remove surrounding single/double quotes
            if (raw_number.startswith("'") and raw_number.endswith("'")) or (
                raw_number.startswith('"') and raw_number.endswith('"')
            ):
                raw_number = raw_number[1:-1]
            # If already a typed literal, sanitize its components
            if '^^' in raw_number:
                value_part, dtype_part = raw_number.split('^^', 1)
                value_part = value_part.strip().strip('"')
                dtype_part = dtype_part.strip().strip('<>').strip('"')
                number = f'{value_part}^^{dtype_part}'
            else:
                # Infer datatype based on presence of decimal point
                if '.' in raw_number:
                    number = f'{raw_number}^^http://www.w3.org/2001/XMLSchema#float'
                else:
                    number = f'{raw_number}^^http://www.w3.org/2001/XMLSchema#integer'
            
            function_call = f"expression{expr_id} = START('{number}')\nexpression{expr_id} = CMP('{standardized_mode}', '{relation}', expression{expr_id})"
        else:
            # Legacy format: [operator | relation] - assumes expression already exists
            mode, relation = action.arguments
            expr_id = target_expr_id or self._get_current_expression_id()
            
            # Use common comparison mode mapping
            standardized_mode = COMPARISON_MODE_MAPPING.get(mode, mode.lower())
            
            function_call = f"expression{expr_id} = CMP('{standardized_mode}', '{relation}', expression{expr_id})"
        
        return FunctionCall(
            expression_id=expr_id,
            function_name="CMP",
            arguments=[standardized_mode, relation, f"expression{expr_id}"],
            raw_function=function_call
        )
    
    def build_tc_function(self, action: ActionResult, target_expr_id: str = None) -> FunctionCall:
        """
        Build TC (Time Constraint) function from Time_constraint action
        
        Args:
            action: Parsed Time_constraint action
            target_expr_id: Target expression ID
            
        Returns:
            FunctionCall object
        """
        if action.action_type != ActionType.TIME_CONSTRAINT:
            raise ValueError(f"Expected Time_constraint action, got {action.action_type}")
        
        relation, time = action.arguments
        expr_id = target_expr_id or self._get_current_expression_id()
        
        function_call = f"expression{expr_id} = TC(expression{expr_id}, '{relation}', '{time}')"
        
        return FunctionCall(
            expression_id=expr_id,
            function_name="TC",
            arguments=[f"expression{expr_id}", relation, time],
            raw_function=function_call
        )
    
    def build_count_function(self, action: ActionResult, target_expr_id: str = None) -> FunctionCall:
        """
        Build COUNT function from Count action
        
        Args:
            action: Parsed Count action
            target_expr_id: Target expression ID
            
        Returns:
            FunctionCall object
        """
        if action.action_type != ActionType.COUNT:
            raise ValueError(f"Expected Count action, got {action.action_type}")
        
        expr_id = target_expr_id or self._get_current_expression_id()
        
        function_call = f"expression{expr_id} = COUNT(expression{expr_id})"
        
        return FunctionCall(
            expression_id=expr_id,
            function_name="COUNT",
            arguments=[f"expression{expr_id}"],
            raw_function=function_call
        )
    
    # Removed: build_stop_function - Finish action not needed for training
    # def build_stop_function(self, action: ActionResult, target_expr_id: str = None) -> FunctionCall:
    #     """
    #     Build STOP function from Finish action
    #     
    #     Args:
    #         action: Parsed Finish action
    #         target_expr_id: Target expression ID
    #         
    #     Returns:
    #         FunctionCall object
    #     """
    #     if action.action_type != ActionType.FINISH:
    #         raise ValueError(f"Expected Finish action, got {action.action_type}")
    #     
    #     expr_id = target_expr_id or self._get_current_expression_id()
    #     
    #     function_call = f"expression{expr_id} = STOP(expression{expr_id})"
    #     
    #     return FunctionCall(
    #         expression_id=expr_id,
    #         function_name="STOP",
    #         arguments=[f"expression{expr_id}"],
    #         raw_function=function_call
    #     )
    
    def build_function_sequence(self, actions: List[ActionResult]) -> List[FunctionCall]:
        """
        Build complete function sequence from action list
        
        Args:
            actions: List of parsed actions
            
        Returns:
            List of FunctionCall objects
        """
        self.reset()
        functions = []
        
        for action in actions:
            # Skip invalid actions - they should not generate function calls
            if not getattr(action, 'is_valid', True):
                logger.info(f"Skipping invalid action: {action.action_type.name}")
                continue
                
            try:
                # if action.action_type == ActionType.EXTRACT_ENTITY:
                #     func = self.build_start_function(action)
                if action.action_type == ActionType.FIND_RELATION:
                    func = self.build_join_function(action)
                elif action.action_type == ActionType.MERGE:
                    func = self.build_and_function(action)
                elif action.action_type == ActionType.ORDER:
                    func = self.build_arg_function(action)
                elif action.action_type == ActionType.COMPARE:
                    func = self.build_cmp_function(action)
                elif action.action_type == ActionType.TIME_CONSTRAINT:
                    func = self.build_tc_function(action)
                elif action.action_type == ActionType.COUNT:
                    func = self.build_count_function(action)
                # Removed: FINISH action handling - not needed for training
                # elif action.action_type == ActionType.FINISH:
                #     func = self.build_stop_function(action)
                else:
                    logger.warning(f"Unknown action type: {action.action_type}")
                    continue
                
                functions.append(func)
                self.function_sequence.append(func)
                
            except Exception as e:
                logger.error(f"Error building function for action {action}: {e}")
                continue
        
        return functions
    
    def get_function_list_strings(self, functions: List[FunctionCall] = None) -> List[str]:
        """
        Get list of function strings compatible with KBQA-o1 format
        
        Args:
            functions: Optional list of functions, uses internal sequence if None
            
        Returns:
            List of function strings
        """
        if functions is None:
            functions = self.function_sequence
        
        return [func.raw_function for func in functions]


