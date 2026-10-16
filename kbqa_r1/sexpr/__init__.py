"""
S-Expression processing module for KBQA-R1
This module provides s-expression generation, validation, and execution capabilities
to replace direct SPARQL generation approach.
"""

# Import constants from separate file to avoid circular imports
from .constants import COMPARISON_MODE_MAPPING, COMPARISON_MODE_READABLE_MAPPING

# Import core modules that don't have heavy dependencies
from .action_parser import ActionParser
from .function_builder import FunctionBuilder  
from .sexpr_generator import SExprGenerator
from .sexpr_validator import SExprValidator
from .sexpr_executor import SExprExecutor

# Import modules with heavy dependencies only when needed
from .relation_retrieval import RelationRetrieval
from .execution_validator import ExecutionValidator
from .dynamic_relation_retrieval import DynamicRelationRetrieval

__all__ = [
    'ActionParser',
    'FunctionBuilder', 
    'SExprGenerator',
    'SExprValidator',
    'SExprExecutor',
    'RelationRetrieval',  # Commented out due to heavy dependencies
    'ExecutionValidator',  # Commented out due to heavy dependencies
    'DynamicRelationRetrieval',  # Commented out due to heavy dependencies
    'COMPARISON_MODE_MAPPING',
    'COMPARISON_MODE_READABLE_MAPPING'
]
