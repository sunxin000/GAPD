"""
S-Expression Validator for validating generated s-expressions
Provides validation rules and error checking mechanisms
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """Validation strictness levels"""
    BASIC = "basic"      # Basic syntax checking
    STANDARD = "standard"  # Standard semantic validation
    STRICT = "strict"    # Strict validation with entity/relation checking

@dataclass
class ValidationResult:
    """Result of validation check"""
    is_valid: bool
    level: ValidationLevel
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]

class SExprValidator:
    """
    Validates s-expressions for correctness and safety
    Prevents generation of resource-intensive or invalid queries
    """
    
    def __init__(self):
        # Valid operators in s-expressions
        self.valid_operators = {
            'JOIN', 'AND', 'ARGMAX', 'ARGMIN', 'COUNT', 'TC',
            'le', 'ge', 'lt', 'gt'  # Comparison operators
        }
        
        # Dangerous patterns that could cause resource issues
        self.dangerous_patterns = [
            r'\(\s*JOIN\s+\?\w+\s+\?\w+\s*\)',  # JOIN with two variables
            r'\(\s*AND\s+\?\w+\s+\?\w+\s*\)',   # AND with two variables
        ]
        
        # Maximum nesting depth to prevent stack overflow
        self.max_nesting_depth = 10
        
        # Maximum expression length
        self.max_expression_length = 1000
    
    def validate(self, sexpr: str, level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationResult:
        """
        Validate s-expression at specified level
        
        Args:
            sexpr: S-expression to validate
            level: Validation level
            
        Returns:
            ValidationResult with validation outcome
        """
        errors = []
        warnings = []
        suggestions = []
        
        # Basic validation always performed
        basic_result = self._validate_basic_syntax(sexpr)
        errors.extend(basic_result.errors)
        warnings.extend(basic_result.warnings)
        suggestions.extend(basic_result.suggestions)
        
        if level in [ValidationLevel.STANDARD, ValidationLevel.STRICT]:
            # Standard semantic validation
            semantic_result = self._validate_semantics(sexpr)
            errors.extend(semantic_result.errors)
            warnings.extend(semantic_result.warnings)
            suggestions.extend(semantic_result.suggestions)
            
            # Safety validation
            safety_result = self._validate_safety(sexpr)
            errors.extend(safety_result.errors)
            warnings.extend(safety_result.warnings)
            suggestions.extend(safety_result.suggestions)
        
        if level == ValidationLevel.STRICT:
            # Strict entity/relation validation
            strict_result = self._validate_strict(sexpr)
            errors.extend(strict_result.errors)
            warnings.extend(strict_result.warnings)
            suggestions.extend(strict_result.suggestions)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            level=level,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def _validate_basic_syntax(self, sexpr: str) -> ValidationResult:
        """Validate basic syntax rules"""
        errors = []
        warnings = []
        suggestions = []
        
        # Check for empty or invalid expressions
        if not sexpr or sexpr.strip() == "":
            errors.append("Empty s-expression")
            return ValidationResult(False, ValidationLevel.BASIC, errors, warnings, suggestions)
        
        if sexpr == "@BAD_EXPRESSION":
            errors.append("Invalid expression marker found")
            return ValidationResult(False, ValidationLevel.BASIC, errors, warnings, suggestions)
        
        # Check expression length
        if len(sexpr) > self.max_expression_length:
            errors.append(f"Expression too long: {len(sexpr)} > {self.max_expression_length}")
        
        # Check balanced parentheses
        open_count = sexpr.count('(')
        close_count = sexpr.count(')')
        if open_count != close_count:
            errors.append(f"Unbalanced parentheses: {open_count} open, {close_count} close")
        
        # Check nesting depth
        depth = self._calculate_nesting_depth(sexpr)
        if depth > self.max_nesting_depth:
            errors.append(f"Nesting too deep: {depth} > {self.max_nesting_depth}")
        
        # Check for valid operators
        operators = self._extract_operators(sexpr)
        invalid_operators = operators - self.valid_operators
        if invalid_operators:
            #  
            errors.append(f"Invalid operators: {invalid_operators}")
        
        return ValidationResult(len(errors) == 0, ValidationLevel.BASIC, errors, warnings, suggestions)
    
    def _validate_semantics(self, sexpr: str) -> ValidationResult:
        """Validate semantic correctness"""
        errors = []
        warnings = []
        suggestions = []
        
        # Check for proper argument counts
        semantic_errors = self._check_operator_arguments(sexpr)
        errors.extend(semantic_errors)
        
        # Check for entity format
        entity_warnings = self._check_entity_format(sexpr)
        warnings.extend(entity_warnings)
        
        # Check for relation format
        relation_warnings = self._check_relation_format(sexpr)
        warnings.extend(relation_warnings)
        
        return ValidationResult(len(errors) == 0, ValidationLevel.STANDARD, errors, warnings, suggestions)
    
    def _validate_safety(self, sexpr: str) -> ValidationResult:
        """Validate safety to prevent resource-intensive queries"""
        errors = []
        warnings = []
        suggestions = []
        
        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if re.search(pattern, sexpr):
                warnings.append(f"Potentially resource-intensive pattern detected: {pattern}")
        
        # Check for excessive JOINs
        join_count = sexpr.count('JOIN')
        if join_count > 5:
            warnings.append(f"High number of JOINs: {join_count} (may be slow)")
        
        # Check for excessive ANDs
        and_count = sexpr.count('AND')
        if and_count > 3:
            warnings.append(f"High number of ANDs: {and_count} (may be complex)")
        
        return ValidationResult(len(errors) == 0, ValidationLevel.STANDARD, errors, warnings, suggestions)
    
    def _validate_strict(self, sexpr: str) -> ValidationResult:
        """Strict validation with entity/relation checking"""
        errors = []
        warnings = []
        suggestions = []
        
        # Extract entities and check format
        entities = self._extract_entities(sexpr)
        for entity in entities:
            if not self._is_valid_entity_format(entity):
                warnings.append(f"Suspicious entity format: {entity}")
        
        # Extract relations and check format  
        relations = self._extract_relations(sexpr)
        for relation in relations:
            if not self._is_valid_relation_format(relation):
                warnings.append(f"Suspicious relation format: {relation}")
        
        return ValidationResult(len(errors) == 0, ValidationLevel.STRICT, errors, warnings, suggestions)
    
    def _calculate_nesting_depth(self, sexpr: str) -> int:
        """Calculate maximum nesting depth of parentheses"""
        max_depth = 0
        current_depth = 0
        
        for char in sexpr:
            if char == '(':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ')':
                current_depth -= 1
        
        return max_depth
    
    def _extract_operators(self, sexpr: str) -> Set[str]:
        """Extract all operators from s-expression"""
        operators = set()
        
        # Find operators after opening parentheses, but exclude reverse relations
        # Pattern to match operators: (OPERATOR ...) but not (R relation_name)
        pattern = r'\(\s*(\w+)(?:\s+[^)]*)?\)'
        matches = re.findall(pattern, sexpr)
        
        for match in matches:
            # Skip 'R' when it's part of a reverse relation pattern (R relation_name)
            if match == 'R':
                # Check if this R is followed by a relation pattern
                r_pattern = r'\(\s*R\s+(\w+\.\w+\.\w+)'
                if re.search(r_pattern, sexpr):
                    continue  # Skip R as it's part of reverse relation
            operators.add(match)
        
        return operators
    
    def _check_operator_arguments(self, sexpr: str) -> List[str]:
        """Check if operators have correct number of arguments"""
        errors = []
        
        # Define expected argument counts
        arg_counts = {
            'JOIN': 2,    # JOIN relation expression
            'AND': 2,     # AND expr1 expr2
            'ARGMAX': 3,  # ARGMAX expression relation
            'ARGMIN': 3,  # ARGMIN expression relation
            'COUNT': 1,   # COUNT expression
            'TC': 3,      # TC expression relation entity
            'le': 3,      # le relation value expression
            'ge': 3,      # ge relation value expression
            'lt': 3,      # lt relation value expression
            'gt': 3,      # gt relation value expression
        }
        
        # Basic argument count checking (simplified)
        for op, expected_count in arg_counts.items():
            pattern = f'\\(\\s*{op}\\s+'
            matches = re.findall(pattern, sexpr)
            if matches:
                # More sophisticated argument counting would be needed for production
                pass
        
        return errors
    
    def _check_entity_format(self, sexpr: str) -> List[str]:
        """Check entity format in s-expression"""
        warnings = []
        
        # Find potential entities (m.xxx, g.xxx patterns)
        entity_pattern = r'\b[mg]\.\w+'
        entities = re.findall(entity_pattern, sexpr)
        
        for entity in entities:
            if not self._is_valid_entity_format(entity):
                warnings.append(f"Invalid entity format: {entity}")
        
        return warnings
    
    def _check_relation_format(self, sexpr: str) -> List[str]:
        """Check relation format in s-expression"""
        warnings = []
        
        # Find potential relations (domain.type.property patterns)
        relation_pattern = r'\b\w+\.\w+\.\w+'
        relations = re.findall(relation_pattern, sexpr)
        
        for relation in relations:
            if not self._is_valid_relation_format(relation):
                warnings.append(f"Invalid relation format: {relation}")
        
        return warnings
    
    def _extract_entities(self, sexpr: str) -> List[str]:
        """Extract all entities from s-expression"""
        # Pattern for Freebase entities
        entity_pattern = r'\b[mg]\.\w+'
        return re.findall(entity_pattern, sexpr)
    
    def _extract_relations(self, sexpr: str) -> List[str]:
        """Extract all relations from s-expression"""
        # Pattern for Freebase relations
        relation_pattern = r'\b\w+\.\w+\.\w+'
        return re.findall(relation_pattern, sexpr)
    
    def _is_valid_entity_format(self, entity: str) -> bool:
        """Check if entity has valid Freebase format"""
        # Freebase entities: m.xxxxx or g.xxxxx
        return re.match(r'^[mg]\.\w+$', entity) is not None
    
    def _is_valid_relation_format(self, relation: str) -> bool:
        """Check if relation has valid Freebase format"""
        # Freebase relations: domain.type.property
        parts = relation.split('.')
        return len(parts) >= 3 and all(part.isalnum() or '_' in part for part in parts)
    
    def suggest_fixes(self, sexpr: str) -> List[str]:
        """Suggest potential fixes for invalid s-expressions"""
        suggestions = []
        
        validation_result = self.validate(sexpr, ValidationLevel.STANDARD)
        
        if not validation_result.is_valid:
            for error in validation_result.errors:
                if "Unbalanced parentheses" in error:
                    suggestions.append("Check for missing or extra parentheses")
                elif "Invalid operators" in error:
                    suggestions.append("Use only valid operators: " + ", ".join(self.valid_operators))
                elif "Expression too long" in error:
                    suggestions.append("Simplify the expression or break it into smaller parts")
                elif "Nesting too deep" in error:
                    suggestions.append("Reduce nesting depth by restructuring the query")
        
        return suggestions




