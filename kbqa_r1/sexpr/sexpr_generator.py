"""
S-Expression Generator for converting function sequences to s-expressions
Based on KBQA-o1's database.py and prompt.py mechanisms
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from .function_builder import FunctionCall

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

class SExpressionExecutionError(Exception):
    """Exception raised when s-expression execution fails"""
    pass

@dataclass 
class SExprResult:
    """Result of s-expression generation"""
    sexpr: str
    is_valid: bool
    error_message: Optional[str] = None
    function_sequence: Optional[List[str]] = None

class SExprGenerator:
    """
    Generates s-expressions from function call sequences
    Mirrors the execution logic from KBQA-o1's utils/prompt.py and utils/database.py
    """
    
    def __init__(self):
        # Define the function execution environment (from KBQA-o1's prompt.py)
        self.exec_globals = {
            'START': self._start_function,
            'JOIN': self._join_function,
            'AND': self._and_function,
            'ARG': self._arg_function,
            'CMP': self._cmp_function,
            'TC': self._tc_function,
            'COUNT': self._count_function,
            'STOP': self._stop_function,
        }
        
    def _start_function(self, entity: str) -> str:
        """START function implementation"""
        return entity
    
    def _join_function(self, relation: str, expression: str) -> str:
        """JOIN function implementation"""
        # Handle both entity strings (quoted) and expression references
        if expression.startswith("'") and expression.endswith("'"):
            # Direct entity reference, remove quotes
            entity = expression[1:-1]
            return f'(JOIN {relation} {entity})'
        else:
            # Expression reference, keep as is
            return f'(JOIN {relation} {expression})'
    
    def _and_function(self, expression: str, sub_expression: str) -> str: #TODO, the most important part
        """AND function implementation"""
        return f'(AND {expression} {sub_expression})'
    
    def _arg_function(self, operator: str, expression: str, relation: str) -> str:
        """ARG function implementation""" #TODO 
        if operator not in ['ARGMAX', 'ARGMIN']:
            raise ValueError(f"Invalid ARG operator: {operator}")
        return f'({operator} {expression} {relation})'
    
    def _cmp_function(self, operator: str, relation: str, expression: str) -> str: #FIXME 
        """CMP function implementation""" 
        return f'({operator} {relation} {expression})'
    
    def _tc_function(self, expression: str, relation: str, entity: str) -> str:
        """TC (Time Constraint) function implementation""" # TIME relation *.*.from  2000
        return f'(TC {expression} {relation} {entity})'
    
    def _count_function(self, expression: str) -> str:
        """COUNT function implementation""" 
        return f'(COUNT {expression})'
    
    def _stop_function(self, expression: str) -> str:
        """STOP function implementation"""
        return expression
    
    def generate_sexpr_from_functions(self, function_calls: List[FunctionCall]) -> SExprResult:
        """
        Generate s-expression from function call sequence
        
        Args:
            function_calls: List of function calls
            
        Returns:
            SExprResult containing the generated s-expression
        """
        if not function_calls:
            return SExprResult(
                sexpr="",
                is_valid=False,
                error_message="Empty function sequence"
            )
        
        # Convert function calls to executable strings
        function_strings = [func.raw_function for func in function_calls]
        
        # Find the target expression (usually the last one)
        target_expr = f"expression{function_calls[-1].expression_id}"
        
        return self.generate_sexpr_from_strings(function_strings, target_expr)
    
    def generate_sexpr_from_strings(self, function_strings: List[str], target_expr: str) -> SExprResult: 
        #FIXME, the function strings are a list of function, we need that each function can be executed, but not execute a function list.
        """
        Generate s-expression from function strings
        Based on KBQA-o1's functions_to_expression in database.py
        
        Args:
            function_strings: List of function definition strings
            target_expr: Target expression to evaluate
            
        Returns:
            SExprResult containing the generated s-expression
        """
        try:
            local_vars = {}
            
            # Execute all function definitions
            #   #TODO
            for func_str in function_strings:
                exec(func_str, self.exec_globals, local_vars) #TODO need a breakpoint here.
            
            # Get the final result
            if target_expr not in local_vars:
                return SExprResult(
                    sexpr="@BAD_EXPRESSION",
                    is_valid=False,
                    error_message=f"Target expression '{target_expr}' not found in local variables",
                    function_sequence=function_strings
                )
            
            result = local_vars[target_expr]
            
            return SExprResult(
                sexpr=result,
                is_valid=True,
                function_sequence=function_strings
            )
            
        except Exception as e:
            logger.error(f"Error generating s-expression: {e}")
            
            # Special handling for expression0 undefined errors
            if "expression0" in str(e) or "name 'expression0' is not defined" in str(e):
                logger.error("Expression0 undefined error detected. This usually means:")
                logger.error("1. The first action is not Extract_entity")
                logger.error("2. Action sequence validation is disabled")
                logger.error("3. Function builder is trying to use undefined expression ID")
                logger.error(f"Function strings: {function_strings}")
                return SExprResult(
                    sexpr="@BAD_EXPRESSION",
                    is_valid=False,
                    error_message=f"Expression0 undefined: {str(e)}. Ensure first action is Extract_entity.",
                    function_sequence=function_strings
                )
            
            return SExprResult(
                sexpr="@BAD_EXPRESSION",
                is_valid=False,
                error_message=str(e),
                function_sequence=function_strings
            )
    
    def generate_sexpr_with_entity_replacement(self, function_strings: List[str], 
                                             target_expr: str, 
                                             entities: List[Tuple[str, str]]) -> SExprResult:
        """
        Generate s-expression with entity label replacement
        Based on KBQA-o1's get_sexpr function in prompt_function.py
        
        Args:
            function_strings: List of function definition strings
            target_expr: Target expression to evaluate
            entities: List of (entity_label, entity_id) tuples
            
        Returns:
            SExprResult containing the generated s-expression with labels
        """
        # First generate basic s-expression
        result = self.generate_sexpr_from_strings(function_strings, target_expr)
        
        if not result.is_valid:
            return result
        
        sexpr = result.sexpr
        
        # Replace entity IDs with labels (from KBQA-o1's get_sexpr)
        for entity_label, entity_id in entities:
            if entity_id in sexpr and entity_label:
                # Replace entity ID with formatted label
                formatted_label = entity_label.replace(" ", "_")
                sexpr = sexpr.replace(f' {entity_id} ', f' <{formatted_label}> ')
                sexpr = sexpr.replace(f' {entity_id})', f' <{formatted_label}>)')
        
        return SExprResult(
            sexpr=sexpr,
            is_valid=True,
            function_sequence=function_strings
        )
    
    def validate_sexpr_syntax(self, sexpr: str) -> Tuple[bool, str]:
        """
        Validate s-expression syntax
        
        Args:
            sexpr: S-expression string to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not sexpr or sexpr == "@BAD_EXPRESSION":
            return False, "Invalid or empty s-expression"
        
        # Check balanced parentheses
        open_count = sexpr.count('(')
        close_count = sexpr.count(')')
        
        if open_count != close_count:
            return False, f"Unbalanced parentheses: {open_count} open, {close_count} close"
        
        # Check for valid operators
        valid_operators = ['JOIN', 'AND', 'ARGMAX', 'ARGMIN', 'COUNT', 'TC', 'le', 'ge', 'lt', 'gt']
        
        # Basic syntax validation - should start with ( for complex expressions
        if sexpr.startswith('('):
            # Extract operator
            parts = sexpr[1:].split()
            if parts and parts[0] not in valid_operators:
                return False, f"Invalid operator: {parts[0]}"
        
        return True, "Valid s-expression syntax"
    
    def get_execution_trace(self, function_strings: List[str]) -> Dict[str, Any]:
        """
        Get detailed execution trace for debugging
        
        Args:
            function_strings: List of function definition strings
            
        Returns:
            Dictionary containing execution trace information
        """
        trace = {
            "functions": function_strings,
            "variables": {},
            "execution_order": [],
            "errors": []
        }
        
        try:
            local_vars = {}
            
            for i, func_str in enumerate(function_strings):
                try:
                    exec(func_str, self.exec_globals, local_vars)
                    trace["execution_order"].append(f"Step {i+1}: {func_str}")
                    trace["variables"][f"step_{i+1}"] = dict(local_vars)
                except Exception as e:
                    trace["errors"].append(f"Step {i+1}: {str(e)}")
                    
        except Exception as e:
            trace["errors"].append(f"Global execution error: {str(e)}")
        
        return trace
