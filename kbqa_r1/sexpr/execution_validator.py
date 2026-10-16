"""
Execution Validator for S-Expression based KBQA-R1
Implements KBQA-o1's critical test_execute_function_list mechanism
"""

import logging
from typing import List, Dict, Optional, Any
import re
from dataclasses import dataclass

# Import existing components
from .sexpr_generator import SExprGenerator
from .sexpr_executor import SExprExecutor
from ..sparql.sparql_manager import SPARQLConfig

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of execution validation"""
    is_valid: bool
    sexpr: Optional[str] = None
    sparql: Optional[str] = None
    results: Optional[List[Any]] = None
    error_message: Optional[str] = None

class ExecutionValidator:
    """
    Validates function sequences by attempting execution
    Implements KBQA-o1's test_execute_function_list mechanism
    """
    
    def __init__(self, sparql_config: SPARQLConfig = None, dataset: str = "WebQSP"):
        self.dataset = dataset
        self.sexpr_generator = SExprGenerator()
        self.sexpr_executor = SExprExecutor(sparql_config) if sparql_config else None
        
        # Performance optimization: cache validation results
        self.validation_cache = {}
        self.cache_enabled = True
        
    def validate_function_list(self, function_list: List[str]) -> ValidationResult:
        """
        Validate if a function list can be executed successfully
        Implements KBQA-o1's test_execute_function_list logic
        
        Args:
            function_list: List of function strings to validate
            
        Returns:
            ValidationResult with validation outcome
        """
        if not function_list:
            return ValidationResult(is_valid=False, error_message="Empty function list")
        
        # Create cache key for performance
        cache_key = None
        if self.cache_enabled:
            cache_key = tuple(function_list)
            if cache_key in self.validation_cache:
                return self.validation_cache[cache_key]
        
        try:
            # Step 1: Add STOP function to complete the sequence (KBQA-o1 pattern)
            function_list_with_stop = self._prepare_function_list_for_validation(function_list)
            
            # Step 2: Generate S-Expression
            target_expr = self._extract_target_expression(function_list_with_stop)
            sexpr_result = self.sexpr_generator.generate_sexpr_from_strings(
                function_list_with_stop, target_expr
            )
            
            if not sexpr_result.is_valid:
                result = ValidationResult(
                    is_valid=False,
                    error_message=f"S-Expression generation failed: {sexpr_result.error_message}"
                )
                if cache_key:
                    self.validation_cache[cache_key] = result
                return result
            
            # Step 3: Execute S-Expression to check validity
            if self.sexpr_executor:
                execution_result = self.sexpr_executor.execute_sexpr(sexpr_result.sexpr, function_state=None)
                
                # Consider validation successful if:
                # 1. Execution is successful, OR
                # 2. Execution fails but generates valid SPARQL (syntax validation)
                is_valid = (execution_result.is_successful or 
                           (execution_result.sparql is not None and 
                            "SELECT" in execution_result.sparql))
                
                result = ValidationResult(
                    is_valid=is_valid,
                    sexpr=sexpr_result.sexpr,
                    sparql=execution_result.sparql,
                    results=execution_result.results if execution_result.is_successful else [],
                    error_message=execution_result.error_message if not is_valid else None
                )
            else:
                # Without executor, just validate S-Expression generation
                result = ValidationResult(
                    is_valid=True,
                    sexpr=sexpr_result.sexpr,
                    error_message=None
                )
            
            # Cache the result
            if cache_key:
                self.validation_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Validation error for function list {function_list}: {e}")
            result = ValidationResult(
                is_valid=False,
                error_message=f"Validation exception: {str(e)}"
            )
            if cache_key:
                self.validation_cache[cache_key] = result
            return result
    
    def _prepare_function_list_for_validation(self, function_list: List[str]) -> List[str]:
        """
        Prepare function list for validation by adding STOP if needed
        Mirrors KBQA-o1's pattern in test_execute_function_list
        """
        function_list_copy = function_list.copy()
        
        # Extract current expression ID from last function
        if function_list_copy:
            last_func = function_list_copy[-1]
            expr_id = self._extract_expression_id(last_func)
            
            # Add STOP function if not already present
            if not any("STOP" in func for func in function_list_copy):
                stop_func = f"expression{expr_id} = STOP(expression{expr_id})"
                function_list_copy.append(stop_func)
        
        return function_list_copy
    
    def _extract_expression_id(self, function_string: str) -> str:
        """Extract expression ID from function string"""
        match = re.search(r"expression(\d*)", function_string)
        return match.group(1) if match else "1"
    
    def _extract_target_expression(self, function_list: List[str]) -> str:
        """Extract target expression for S-Expression generation"""
        if not function_list:
            return "expression1"
        
        # Find the last expression ID
        last_func = function_list[-1]
        expr_id = self._extract_expression_id(last_func)
        return f"expression{expr_id}"
    
    def validate_function_with_addition(self, base_functions: List[str], 
                                      additional_function: str) -> ValidationResult:
        """
        Validate adding a new function to existing sequence
        This is the most common pattern in KBQA-o1's agent.py
        
        Args:
            base_functions: Existing function sequence
            additional_function: New function to add
            
        Returns:
            ValidationResult for the combined sequence
        """
        combined_functions = base_functions + [additional_function]
        return self.validate_function_list(combined_functions)
    
    def is_function_addition_valid(self, base_functions: List[str], 
                                  additional_function: str) -> bool:
        """
        Quick check if adding a function results in valid execution
        Implements the pattern: if len(test_execute_function_list(...)) > 0
        
        Args:
            base_functions: Existing function sequence
            additional_function: New function to add
            
        Returns:
            True if addition results in valid execution
        """
        validation_result = self.validate_function_with_addition(base_functions, additional_function)
        return validation_result.is_valid and len(validation_result.results or []) > 0
    
    def filter_valid_functions(self, base_functions: List[str], 
                              candidate_functions: List[str]) -> List[str]:
        """
        Filter candidate functions to only those that result in valid execution
        
        Args:
            base_functions: Existing function sequence
            candidate_functions: List of candidate functions to test
            
        Returns:
            List of functions that result in valid execution
        """
        valid_functions = []
        
        for candidate in candidate_functions:
            if self.is_function_addition_valid(base_functions, candidate):
                valid_functions.append(candidate)
        
        return valid_functions
    
    def validate_with_constraints(self, function_list: List[str], 
                                 min_results: int = 1, 
                                 max_results: int = None) -> ValidationResult:
        """
        Validate function list with result count constraints
        
        Args:
            function_list: Function sequence to validate
            min_results: Minimum number of results required
            max_results: Maximum number of results allowed
            
        Returns:
            ValidationResult considering constraints
        """
        base_validation = self.validate_function_list(function_list)
        
        if not base_validation.is_valid:
            return base_validation
        
        result_count = len(base_validation.results or [])
        
        # Check constraints
        if result_count < min_results:
            return ValidationResult(
                is_valid=False,
                sexpr=base_validation.sexpr,
                sparql=base_validation.sparql,
                results=base_validation.results,
                error_message=f"Too few results: {result_count} < {min_results}"
            )
        
        if max_results and result_count > max_results:
            return ValidationResult(
                is_valid=False,
                sexpr=base_validation.sexpr,
                sparql=base_validation.sparql,
                results=base_validation.results,
                error_message=f"Too many results: {result_count} > {max_results}"
            )
        
        return base_validation
    
    def clear_cache(self):
        """Clear validation cache"""
        self.validation_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            "cache_size": len(self.validation_cache),
            "cache_enabled": self.cache_enabled
        }
    
    def enable_cache(self, enabled: bool = True):
        """Enable or disable caching"""
        self.cache_enabled = enabled
        if not enabled:
            self.clear_cache()

class ValidationMode:
    """Different validation modes for different use cases"""
    
    STRICT = "strict"      # Must return results
    LENIENT = "lenient"    # Valid SPARQL is enough
    SYNTAX_ONLY = "syntax" # Only check S-Expression syntax
    
class AdvancedExecutionValidator(ExecutionValidator):
    """
    Advanced validator with additional features
    """
    
    def __init__(self, sparql_config: SPARQLConfig = None, dataset: str = "WebQSP", 
                 validation_mode: str = ValidationMode.STRICT):
        super().__init__(sparql_config, dataset)
        self.validation_mode = validation_mode
        
    def validate_with_mode(self, function_list: List[str]) -> ValidationResult:
        """Validate with specific mode"""
        base_result = self.validate_function_list(function_list)
        
        if self.validation_mode == ValidationMode.SYNTAX_ONLY:
            # Only check if S-Expression can be generated
            return ValidationResult(
                is_valid=base_result.sexpr is not None,
                sexpr=base_result.sexpr,
                error_message="Syntax validation only"
            )
        
        elif self.validation_mode == ValidationMode.LENIENT:
            # Valid if SPARQL can be generated
            return ValidationResult(
                is_valid=base_result.sparql is not None,
                sexpr=base_result.sexpr,
                sparql=base_result.sparql,
                error_message=base_result.error_message if base_result.sparql is None else None
            )
        
        else:  # STRICT mode
            return base_result
    
    def batch_validate(self, function_lists: List[List[str]]) -> List[ValidationResult]:
        """Validate multiple function lists efficiently"""
        results = []
        
        for func_list in function_lists:
            result = self.validate_with_mode(func_list)
            results.append(result)
        
        return results




