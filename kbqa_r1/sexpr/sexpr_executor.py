"""
S-Expression Executor for converting s-expressions to SPARQL and executing them
Integrates with existing SPARQL infrastructure in kbqa-r1 and enhanced KBQA-o1 functionality
"""
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from ..sparql.enhanced_executor import EnhancedSPARQLExecutor
# Import existing SPARQL infrastructure
from ..sparql.sparql_manager import SPARQLConfig, SPARQLExecutionManager
from .sexpr_validator import SExprValidator, ValidationLevel

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@dataclass
class ExecutionResult:
    """Result of s-expression execution"""
    sexpr: str
    sparql: Optional[str]
    results: List[Any]
    is_successful: bool
    error_message: Optional[str] = None
    execution_time: Optional[float] = None

class SExprExecutor:
    """
    Executes s-expressions by converting them to SPARQL and running queries
    Integrates with kbqa-r1's existing SPARQL execution infrastructure
    """
    
    def __init__(self, sparql_config: SPARQLConfig = None, use_enhanced_executor: bool = False, dataset_type: str = "webqsp"):
        self.sparql_manager = SPARQLExecutionManager(sparql_config) if sparql_config else None
        self.validator = SExprValidator()
        self.dataset_type = dataset_type
        # Store contextual fields for error logging
        self._current_input_text: Optional[str] = None
        self._current_prediction_text: Optional[str] = None
        # Disable all disk logging (debug_logs, error contexts, execution_results) when env var set or by default user request.
        # 用户要求: 取消所有文件日志保存（只保留checkpoint）。默认关闭，设置 DISABLE_SEXPR_FILE_LOGS=0 可恢复。
        self.disable_disk_logs = os.getenv("DISABLE_SEXPR_FILE_LOGS", "1").lower() in ("1", "true", "yes")
        
        # Choose appropriate SPARQL converter based on dataset type
        if dataset_type.lower() == "grailqa":
            from ..sparql.sparql_converter_grailqa import \
                SPARQLConverterGrailQA
            self.sparql_converter = SPARQLConverterGrailQA()
        else:  # webqsp, graphq, or default
            from ..sparql.sparql_converter import SPARQLConverter  
            self.sparql_converter = SPARQLConverter()
        
        # Optional enhanced executor for advanced Freebase operations
        self.enhanced_executor = EnhancedSPARQLExecutor() if use_enhanced_executor else None
        
        # Keep original simple conversion for backward compatibility
        self._setup_sparql_conversion()

    def set_error_context(self, input_text: Optional[str] = None, prediction_text: Optional[str] = None):
        """Set current input/prediction context for enhanced error logging."""
        if input_text is not None:
            self._current_input_text = input_text
        if prediction_text is not None:
            self._current_prediction_text = prediction_text
    
    def _setup_sparql_conversion(self):
        """Setup SPARQL conversion functionality"""
        # These would be adapted from KBQA-o1's logic_form_util.py
        # For now, we'll implement basic conversion patterns
        
        self.operator_mappings = {
            'le': '<=',
            'ge': '>=', 
            'lt': '<',
            'gt': '>'
        }
        
        # Namespace prefix for Freebase
        self.freebase_prefix = "PREFIX ns: <http://rdf.freebase.com/ns/>\nPREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
    
    def execute_sexpr(self, sexpr: str, validate: bool = True, function_state: List[str] = None) -> ExecutionResult:
        """
        Execute s-expression by converting to SPARQL and running query
        
        Args:
            sexpr: S-expression to execute
            validate: Whether to validate s-expression first
            function_state: List of function strings for debugging context
            
        Returns:
            ExecutionResult with query results
        """
        if validate:
            validation_result = self.validator.validate(sexpr, ValidationLevel.STANDARD)
            if not validation_result.is_valid:
                return ExecutionResult(
                    sexpr=sexpr,
                    sparql=None,
                    results=[],
                    is_successful=False,
                    error_message=f"Validation failed: {validation_result.errors}"
                )
        
        try:
            # Convert s-expression to SPARQL
            sparql = self.sexpr_to_sparql(sexpr)
            
            if not sparql:
                return ExecutionResult(
                    sexpr=sexpr,
                    sparql=None,
                    results=[],
                    is_successful=False,
                    error_message="Failed to convert s-expression to SPARQL"
                )
            
            # Enhanced safety check: Detect dangerous SPARQL queries that could cause timeouts
            if self._is_dangerous_query(sparql):
                
                error_msg = "DANGEROUS TIMEOUT QUERY DETECTED!\n"
                error_msg += f"S-Expression: {sexpr}\n"
                error_msg += f"Generated SPARQL: {sparql}\n"
                error_msg += "This query contains only FILTER/OPTIONAL clauses without substantial constraints, "
                error_msg += "which would cause a full table scan and timeout."
                
                # Add function_state to error message if available
                if function_state:
                    error_msg += "\nFunction State:\n"
                    for idx, func in enumerate(function_state, 1):
                        error_msg += f"  {idx}. {func}\n"
                
                logger.error(f"[TIMEOUT-DEBUG] {error_msg}")
                
                # Raise an exception with detailed information for debugging
                raise RuntimeError(f"Dangerous timeout query detected: {error_msg}")
            
            # ===== MERGE OPERATION SPARQL DEBUG BREAKPOINT =====
            # Check if this is a merge operation (contains AND in s-expression)
            if 'AND(' in sexpr or 'AND ' in sexpr:
                logger.info("=" * 80)
                logger.info("🔍 MERGE OPERATION SPARQL CONVERSION DEBUG BREAKPOINT")
                logger.info("=" * 80)
                logger.info(f"📝 Original S-Expression: {sexpr}")
                logger.info("🔧 Generated SPARQL Query:")
                logger.info(sparql)
                logger.info("=" * 80)
                
                # Add function state context if available
                if function_state:
                    logger.info("📋 Function State Context:")
                    for idx, func in enumerate(function_state, 1):
                        logger.info(f"  {idx}. {func}")
                    logger.info("=" * 80)
                # Optional: Enable interactive debugging via environment variable
                #  
            # ===== END MERGE OPERATION DEBUG BREAKPOINT =====

            # ===== ARG OPERATION SPARQL DEBUG BREAKPOINT =====
            # Check if this is an ARGMIN/ARGMAX operation
            if ('ARGMIN(' in sexpr or 'ARGMIN ' in sexpr or
                'ARGMAX(' in sexpr or 'ARGMAX ' in sexpr):
                logger.info("=" * 80)
                logger.info("🏁 ARG OPERATION SPARQL CONVERSION DEBUG BREAKPOINT (ARGMIN/ARGMAX)")
                logger.info("=" * 80)
                logger.info(f"📝 Original S-Expression: {sexpr}")
                logger.info("🔧 Generated SPARQL Query:")
                logger.info(sparql)
                logger.info("=" * 80)
                if function_state:
                    logger.info("📋 Function State Context:")
                    for idx, func in enumerate(function_state, 1):
                        logger.info(f"  {idx}. {func}")
                    logger.info("=" * 80)
            # ===== END ARG OPERATION DEBUG BREAKPOINT =====
            
            # ===== COUNT OPERATION SPARQL DEBUG BREAKPOINT =====
            # Check if this is a COUNT operation
            if ('COUNT(' in sexpr or 'COUNT ' in sexpr):
                logger.info("=" * 80)
                logger.info("🧮 COUNT OPERATION SPARQL CONVERSION DEBUG BREAKPOINT")
                logger.info("=" * 80)
                logger.info(f"📝 Original S-Expression: {sexpr}")
                logger.info("🔧 Generated SPARQL Query:")
                logger.info(sparql)
                logger.info("=" * 80)
                if function_state:
                    logger.info("📋 Function State Context:")
                    for idx, func in enumerate(function_state, 1):
                        logger.info(f"  {idx}. {func}")
                    logger.info("=" * 80)
            # ===== END COUNT OPERATION DEBUG BREAKPOINT =====

            # ===== COMPARISON (CMP/le/ge/lt/gt) DEBUG BREAKPOINT =====
            # Check if this is a comparison-related operation
            if ('CMP(' in sexpr or 'CMP ' in sexpr or
                ' le(' in sexpr or sexpr.startswith('le(') or ' le ' in sexpr or
                ' ge(' in sexpr or sexpr.startswith('ge(') or ' ge ' in sexpr or
                ' lt(' in sexpr or sexpr.startswith('lt(') or ' lt ' in sexpr or
                ' gt(' in sexpr or sexpr.startswith('gt(') or ' gt ' in sexpr):
                logger.info("=" * 80)
                logger.info("⚖️  COMPARISON OPERATION SPARQL CONVERSION DEBUG BREAKPOINT (CMP/le/ge/lt/gt)")
                logger.info("=" * 80)
                logger.info(f"📝 Original S-Expression: {sexpr}")
                logger.info("🔧 Generated SPARQL Query:")
                logger.info(sparql)
                logger.info("=" * 80)
                if function_state:
                    logger.info("📋 Function State Context:")
                    for idx, func in enumerate(function_state, 1):
                        logger.info(f"  {idx}. {func}")
                    logger.info("=" * 80)
            # ===== END COMPARISON DEBUG BREAKPOINT =====

            # ===== TIME CONSTRAINT (TC) DEBUG BREAKPOINT =====
            # Check if this is a TC operation
            if ('TC(' in sexpr or 'TC ' in sexpr):
                logger.info("=" * 80)
                logger.info("⏱️  TIME CONSTRAINT OPERATION SPARQL CONVERSION DEBUG BREAKPOINT (TC)")
                logger.info("=" * 80)
                logger.info(f"📝 Original S-Expression: {sexpr}")
                logger.info("🔧 Generated SPARQL Query:")
                logger.info(sparql)
                logger.info("=" * 80)
                if function_state:
                    logger.info("📋 Function State Context:")
                    for idx, func in enumerate(function_state, 1):
                        logger.info(f"  {idx}. {func}")
                    logger.info("=" * 80)
            # ===== END TIME CONSTRAINT DEBUG BREAKPOINT =====
            
            # Execute SPARQL query
            if self.sparql_manager:
                # Persist context before handing off to SPARQL manager (ODBC layer)
                if not self.disable_disk_logs:
                    try:
                        os.makedirs("debug_logs", exist_ok=True)
                        ctx = {
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
                            "phase": "before_sparql_manager_execute",
                            "sexpr": sexpr,
                            "sparql": sparql,
                            "function_state": function_state or [],
                        }
                        with open(os.path.join("debug_logs", "last_sexpr_context.json"), "w", encoding="utf-8") as f:
                            json.dump(ctx, f, ensure_ascii=False, indent=2)
                        with open(os.path.join("debug_logs", "sexpr_context.jsonl"), "a", encoding="utf-8") as f:
                            f.write(json.dumps(ctx, ensure_ascii=False) + "\n")
                    except Exception:
                        pass
                # Prepare context information for slow query debugging
                context = {
                    'input': self._current_input_text,
                    'response': self._current_prediction_text,
                    'sexpr': sexpr,
                    'function_state': function_state or [],
                    'sample_id': getattr(self, '_current_sample_id', None)
                }
                
                execution_result = self.sparql_manager.execute_batch([sparql], [context])
                
                if "results" in execution_result and execution_result["results"]:
                    result = execution_result["results"][0]
                    
                    if isinstance(result, dict) and "error" in result:
                        # Log detailed context for SPARQL execution errors
                        logger.error("[SPARQL-ERROR] SPARQL execution failed")
                        logger.error(f"[SPARQL-ERROR] Error: {result['error']}")
                        logger.error(f"[SPARQL-ERROR] S-Expression: {sexpr}")
                        logger.error("[SPARQL-ERROR] Generated SPARQL query:")
                        logger.error(sparql)
                        if function_state:
                            logger.error("[SPARQL-ERROR] Function state (sexpr_list):")
                            for idx, func in enumerate(function_state, 1):
                                logger.error(f"  {idx}. {func}")
                        if not self.disable_disk_logs:
                            # Persist error context to file
                            self._save_sparql_error_context(
                                error_message=result['error'],
                                sexpr=sexpr,
                                sparql=sparql,
                                function_state=function_state,
                            )
                            # Also save a lightweight last context snapshot for quick inspection
                            try:
                                with open(os.path.join("debug_logs", "last_error_context.json"), "w", encoding="utf-8") as f:
                                    json.dump({
                                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
                                        "sexpr": sexpr,
                                        "sparql": sparql,
                                        "error": result['error'],
                                        "function_state": function_state or [],
                                    }, f, ensure_ascii=False, indent=2)
                            except Exception:
                                pass
                        
                        # Log execution results for debugging (SPARQL execution error case)
                        self._log_execution_results(
                            sexpr=sexpr,
                            sparql=sparql,
                            results=[],
                            function_state=function_state,
                            is_successful=False,
                            error_message=result["error"]
                        )
                        
                        return ExecutionResult(
                            sexpr=sexpr,
                            sparql=sparql,
                            results=[],
                            is_successful=False,
                            error_message=result["error"]
                        )
                    
                    # Extract results
                    if isinstance(result, dict) and "results" in result:
                        results = result["results"]
                    else:
                        results = [result] if result else []
                    
                    # Log execution results for debugging
                    self._log_execution_results(
                        sexpr=sexpr,
                        sparql=sparql,
                        results=results,
                        function_state=function_state,
                        is_successful=True
                    )
                    
                    return ExecutionResult(
                        sexpr=sexpr,
                        sparql=sparql,
                        results=results,
                        is_successful=True
                    )
                else:
                    # Log execution results for debugging (failure case)
                    self._log_execution_results(
                        sexpr=sexpr,
                        sparql=sparql,
                        results=[],
                        function_state=function_state,
                        is_successful=False,
                        error_message="No results returned from SPARQL execution"
                    )
                    
                    return ExecutionResult(
                        sexpr=sexpr,
                        sparql=sparql,
                        results=[],
                        is_successful=False,
                        error_message="No results returned from SPARQL execution"
                    )
            else:
                # No SPARQL manager configured, return SPARQL only
                return ExecutionResult(
                    sexpr=sexpr,
                    sparql=sparql,
                    results=[],
                    is_successful=True,
                    error_message="SPARQL generated but not executed (no manager configured)"
                )
                
        except RuntimeError as e:
            # Re-raise dangerous query errors for debugging
            if "Dangerous timeout query detected" in str(e):
                raise  # Let the RuntimeError propagate for debugging
            else:
                # Handle other RuntimeErrors normally
                logger.error(f"Runtime error executing s-expression '{sexpr}': {e}")
                if not self.disable_disk_logs:
                    # Persist error context to file
                    self._save_sparql_error_context(
                        error_message=str(e),
                        sexpr=sexpr,
                        sparql=locals().get('sparql'),
                        function_state=function_state,
                    )
                
                # Log execution results for debugging (RuntimeError case)
                self._log_execution_results(
                    sexpr=sexpr,
                    sparql=locals().get('sparql'),
                    results=[],
                    function_state=function_state,
                    is_successful=False,
                    error_message=str(e)
                )
                
                return ExecutionResult(
                    sexpr=sexpr,
                    sparql=None,
                    results=[],
                    is_successful=False,
                    error_message=str(e)
                )
        except Exception as e:
            logger.error(f"Error executing s-expression '{sexpr}': {e}")
            if not self.disable_disk_logs:
                # Persist error context to file
                self._save_sparql_error_context(
                    error_message=str(e),
                    sexpr=sexpr,
                    sparql=locals().get('sparql'),
                    function_state=function_state,
                )
            
            # Log execution results for debugging (Exception case)
            self._log_execution_results(
                sexpr=sexpr,
                sparql=locals().get('sparql'),
                results=[],
                function_state=function_state,
                is_successful=False,
                error_message=str(e)
            )
            
            return ExecutionResult(
                sexpr=sexpr,
                sparql=None,
                results=[],
                is_successful=False,
                error_message=str(e)
            )

    def _save_sparql_error_context(
        self,
        error_message: str,
        sexpr: Optional[str] = None,
        sparql: Optional[str] = None,
        function_state: Optional[List[str]] = None,
        output_dir: str = "debug_logs",
        output_file: str = "sparql_errors.jsonl",
    ) -> None:
        """Append SPARQL error context to a JSONL file for offline analysis."""
        if self.disable_disk_logs:
            return
        os.makedirs(output_dir, exist_ok=True)
        record = {
            "timestamp": datetime.now().isoformat(),
            "dataset_type": getattr(self, "dataset_type", None),
            "error": error_message,
            "sexpr": sexpr,
            "sparql": sparql,
            "function_state": function_state or [],
            "input": getattr(self, "_current_input_text", None),
            "pred": getattr(self, "_current_prediction_text", None),
        }
        file_path = os.path.join(output_dir, output_file)
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _log_execution_results(
        self,
        sexpr: str,
        sparql: str,
        results: List[Any],
        function_state: Optional[List[str]] = None,
        is_successful: bool = True,
        error_message: Optional[str] = None,
    ) -> None:
        """Log execution results for special operations with detailed context."""
        # Check if this is a special operation that needs detailed logging
        is_special_operation = (
            'AND(' in sexpr or 'AND ' in sexpr or  # MERGE operation
            'ARGMIN(' in sexpr or 'ARGMIN ' in sexpr or 'ARGMAX(' in sexpr or 'ARGMAX ' in sexpr or  # ARG operation
            'COUNT(' in sexpr or 'COUNT ' in sexpr or  # COUNT operation
            'CMP(' in sexpr or 'CMP ' in sexpr or  # COMPARISON operation
            ' le(' in sexpr or sexpr.startswith('le(') or ' le ' in sexpr or
            ' ge(' in sexpr or sexpr.startswith('ge(') or ' ge ' in sexpr or
            ' lt(' in sexpr or sexpr.startswith('lt(') or ' lt ' in sexpr or
            ' gt(' in sexpr or sexpr.startswith('gt(') or ' gt ' in sexpr or
            'TC(' in sexpr or 'TC ' in sexpr  # TIME CONSTRAINT operation
        )
        
        if not is_special_operation:
            return
        
        # Determine operation type for logging
        operation_type = "UNKNOWN"
        if 'AND(' in sexpr or 'AND ' in sexpr:
            operation_type = "MERGE"
        elif 'ARGMIN(' in sexpr or 'ARGMIN ' in sexpr or 'ARGMAX(' in sexpr or 'ARGMAX ' in sexpr:
            operation_type = "ARG"
        elif 'COUNT(' in sexpr or 'COUNT ' in sexpr:
            operation_type = "COUNT"
        elif ('CMP(' in sexpr or 'CMP ' in sexpr or
              ' le(' in sexpr or sexpr.startswith('le(') or ' le ' in sexpr or
              ' ge(' in sexpr or sexpr.startswith('ge(') or ' ge ' in sexpr or
              ' lt(' in sexpr or sexpr.startswith('lt(') or ' lt ' in sexpr or
              ' gt(' in sexpr or sexpr.startswith('gt(') or ' gt ' in sexpr):
            operation_type = "COMPARISON"
        elif 'TC(' in sexpr or 'TC ' in sexpr:
            operation_type = "TIME_CONSTRAINT"
        
        # Log execution results
        logger.info("=" * 80)
        logger.info(f"📊 {operation_type} OPERATION EXECUTION RESULTS")
        logger.info("=" * 80)
        logger.info(f"📝 S-Expression: {sexpr}")
        logger.info(f"🔧 Generated SPARQL: {sparql}")
        logger.info(f"✅ Execution Status: {'SUCCESS' if is_successful else 'FAILED'}")
        
        if is_successful:
            logger.info(f"📈 Results Count: {len(results)}")
            if results:
                logger.info("📋 Execution Results:")
                for idx, result in enumerate(results[:10], 1):  # Show first 10 results
                    logger.info(f"  {idx}. {result}")
                if len(results) > 10:
                    logger.info(f"  ... and {len(results) - 10} more results")
            else:
                logger.info("📋 Execution Results: No results returned")
        else:
            logger.info(f"❌ Error Message: {error_message}")
        
        if function_state:
            logger.info("📋 Function State Context:")
            for idx, func in enumerate(function_state, 1):
                logger.info(f"  {idx}. {func}")
        
        logger.info("=" * 80)
        
        # Also save to file for offline analysis
        if not self.disable_disk_logs:
            try:
                os.makedirs("debug_logs", exist_ok=True)
                record = {
                    "timestamp": datetime.now().isoformat(),
                    "operation_type": operation_type,
                    "sexpr": sexpr,
                    "sparql": sparql,
                    "is_successful": is_successful,
                    "results_count": len(results) if is_successful else 0,
                    "results": results[:50] if is_successful and results else [],  # Limit to first 50 results
                    "error_message": error_message,
                    "function_state": function_state or [],
                    "input": getattr(self, "_current_input_text", None),
                    "pred": getattr(self, "_current_prediction_text", None),
                    "sample_id": getattr(self, "_current_sample_id", None),
                }
                with open(os.path.join("debug_logs", "execution_results.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.warning(f"Failed to save execution results to file: {e}")

    
    def sexpr_to_sparql(self, sexpr: str, use_complete_converter: bool = True) -> Optional[str]:
        """
        Convert s-expression to SPARQL query
        Now uses complete KBQA-o1 integration for full functionality
        
        Args:
            sexpr: S-expression to convert
            use_complete_converter: Whether to use the complete converter (recommended)
            
        Returns:
            SPARQL query string or None if conversion fails
        """
        #FIXME, there's error here
        try:
            if use_complete_converter:
                # Use complete converter with all KBQA-o1 functionality
                return self.sparql_converter.lisp_to_sparql(sexpr)
            else:
                # Fall back to simple converter for backward compatibility 
                return self._simple_sexpr_to_sparql(sexpr) #TODO need check here
            
        except Exception as e:
            logger.error(f"Error converting s-expression to SPARQL: {e}")
            return None
    
    def _simple_sexpr_to_sparql(self, sexpr: str) -> Optional[str]:
        """
        Simple s-expression to SPARQL conversion (original implementation)
        Kept for backward compatibility
        """
        try:
            # Parse s-expression to nested structure
            nested_expr = self._parse_sexpr_to_nested(sexpr)
            
            if not nested_expr:
                return None
            
            # Convert nested structure to SPARQL
            sparql_body = self._nested_to_sparql(nested_expr)
            
            if not sparql_body:
                return None
            
            # Construct complete SPARQL query with LIMIT clause
            sparql = self.freebase_prefix + "SELECT DISTINCT ?x ?name WHERE {\n" + sparql_body + "\nOPTIONAL { ?x rdfs:label ?name . FILTER (langMatches(lang(?name), 'en')) }\n} LIMIT 1000"
            
            return sparql
            
        except Exception as e:
            logger.error(f"Error in simple conversion: {e}")
            return None
    
    def _parse_sexpr_to_nested(self, sexpr: str) -> Optional[List]:
        """
        Parse s-expression string to nested list structure
        Simplified version of KBQA-o1's lisp_to_nested_expression
        
        Args:
            sexpr: S-expression string
            
        Returns:
            Nested list representation or None if parsing fails
        """
        try:
            # Remove extra whitespace
            sexpr = sexpr.strip()
            
            # If it's a simple entity, return as is
            if not sexpr.startswith('('):
                return sexpr
            
            # Parse nested structure
            tokens = self._tokenize_sexpr(sexpr)
            result = self._parse_tokens(tokens)
            
            return result[0] if result else None
            
        except Exception as e:
            logger.error(f"Error parsing s-expression: {e}")
            return None
    
    def validate_sexpr_executability(self, sexpr: str) -> Dict[str, Any]:
        """Validate s-expression executability"""
        result = {
            'is_executable': False,
            'syntax_valid': False,
            'sparql_valid': False,
            'errors': [],
            'warnings': []
        }
        
        # Syntax validation
        validation_result = self.validator.validate(sexpr, ValidationLevel.STANDARD)
        result['syntax_valid'] = validation_result.is_valid
        result['errors'].extend(validation_result.errors)
        result['warnings'].extend(validation_result.warnings)
        
        # SPARQL conversion validation
        try:
            sparql = self.sparql_converter.lisp_to_sparql(sexpr)
            result['sparql_valid'] = sparql is not None
        except Exception as e:
            result['errors'].append(f"SPARQL conversion error: {e}")
        
        result['is_executable'] = result['syntax_valid'] and result['sparql_valid'] and len(result['errors']) == 0
        return result[0] if result else None

    
    # def execute_with_enhanced_features(self, sexpr: str) -> ExecutionResult:
    #     """
    #     Execute s-expression using enhanced features from KBQA-o1 integration
    #     """
    #     if not self.enhanced_executor:
    #         return self.execute_sexpr(sexpr, function_state=None)
        
    #     try:
    #         sparql = self.sparql_converter.lisp_to_sparql(sexpr)
    #         if not sparql:
    #             return ExecutionResult(
    #                 sexpr=sexpr, sparql=None, results=[], is_successful=False,
    #                 error_message="Failed to convert s-expression with complete converter"
    #             )
            
    #         results = self.enhanced_executor.execute_query_with_odbc(sparql)
    #         return ExecutionResult(sexpr=sexpr, sparql=sparql, results=results, is_successful=True)
            
    #     except Exception as e:
    #         logger.error(f"Enhanced execution failed: {e}")
    #         return ExecutionResult(sexpr=sexpr, sparql=None, results=[], is_successful=False, error_message=str(e))
    
    def _tokenize_sexpr(self, sexpr: str) -> List[str]:
        """Tokenize s-expression string"""
        # Add spaces around parentheses for easier splitting
        sexpr = re.sub(r'([()])', r' \1 ', sexpr)
        
        # Split and filter empty tokens
        tokens = [token.strip() for token in sexpr.split() if token.strip()]
        
        return tokens
    
    def _parse_tokens(self, tokens: List[str]) -> List:
        """Parse tokenized s-expression into nested structure"""
        stack = []
        current = []
        
        for token in tokens:
            if token == '(':
                # Start new nested expression
                stack.append(current)
                current = []
            elif token == ')':
                # End current nested expression
                if stack:
                    parent = stack.pop()
                    parent.append(current)
                    current = parent
            else:
                # Add token to current expression
                current.append(token)
        
        return current
    
    def _nested_to_sparql(self, nested_expr: Union[str, List], var_counter: int = 0) -> str:
        """
        Convert nested expression to SPARQL clauses
        Simplified version based on KBQA-o1's logic
        
        Args:
            nested_expr: Nested expression structure
            var_counter: Variable counter for generating unique variables
            
        Returns:
            SPARQL clauses string
        """
        if isinstance(nested_expr, str):
            # Simple entity or value
            if nested_expr.startswith('m.') or nested_expr.startswith('g.'):
                return f"ns:{nested_expr}"
            else:
                return nested_expr
        
        if not isinstance(nested_expr, list) or len(nested_expr) == 0:
            return ""
        
        operator = nested_expr[0]
        
        if operator == "JOIN":
            # JOIN relation expression
            if len(nested_expr) < 3:
                return ""
            
            relation = nested_expr[1]
            expression = nested_expr[2]
            
            # Handle reverse relations
            if relation.startswith('(R ') and relation.endswith(')'):
                relation_name = relation[3:-1]
                # Reverse JOIN: expression relation ?x
                expr_sparql = self._nested_to_sparql(expression, var_counter)
                return f"{expr_sparql} ns:{relation_name} ?x{var_counter} ."
            else:
                # Forward JOIN: ?x relation expression  
                expr_sparql = self._nested_to_sparql(expression, var_counter)
                return f"?x{var_counter} ns:{relation} {expr_sparql} ."
        
        elif operator == "AND":
            # AND expression1 expression2
            if len(nested_expr) < 3:
                return ""
            
            expr1_sparql = self._nested_to_sparql(nested_expr[1], var_counter)
            expr2_sparql = self._nested_to_sparql(nested_expr[2], var_counter + 1)
            
            return f"{expr1_sparql}\n{expr2_sparql}"
        
        elif operator in ["ARGMAX", "ARGMIN"]:
            # ARGMAX/ARGMIN expression relation
            if len(nested_expr) < 3:
                return ""
            
            expression = nested_expr[1]
            relation = nested_expr[2]
            
            expr_sparql = self._nested_to_sparql(expression, var_counter)
            order_clause = "ORDER BY DESC" if operator == "ARGMAX" else "ORDER BY ASC"
            
            return f"{expr_sparql}\n?x{var_counter} ns:{relation} ?value{var_counter} .\n}} {order_clause}(?value{var_counter}) LIMIT 1"
        
        elif operator == "COUNT":
            # COUNT expression
            if len(nested_expr) < 2:
                return ""
            
            expr_sparql = self._nested_to_sparql(nested_expr[1], var_counter)
            return f"SELECT (COUNT(*) as ?count) WHERE {{\n{expr_sparql}\n}}"
        
        elif operator in ["le", "ge", "lt", "gt"]:
            # Comparison operators
            if len(nested_expr) < 3:
                return ""
            
            relation = nested_expr[1]
            value = nested_expr[2]
            expression = nested_expr[3] if len(nested_expr) > 3 else "?x"
            
            operator_symbol = self.operator_mappings.get(operator, operator)
            expr_sparql = self._nested_to_sparql(expression, var_counter)
            
            return f"{expr_sparql}\n?x{var_counter} ns:{relation} ?value{var_counter} .\nFILTER(?value{var_counter} {operator_symbol} {value})"
        
        elif operator == "TC":
            # Time constraint
            if len(nested_expr) < 4:
                return ""
            
            expression = nested_expr[1]
            relation = nested_expr[2]
            time_value = nested_expr[3]
            
            expr_sparql = self._nested_to_sparql(expression, var_counter)
            
            return f"{expr_sparql}\n?x{var_counter} ns:{relation} ?time{var_counter} .\nFILTER(?time{var_counter} = \"{time_value}\")"
        
        else:
            logger.warning(f"Unknown operator in s-expression: {operator}")
            return ""
    
    # def batch_execute_sexprs(self, sexprs: List[str], validate: bool = True) -> List[ExecutionResult]:
    #     """
    #     Execute multiple s-expressions in batch
        
    #     Args:
    #         sexprs: List of s-expressions to execute
    #         validate: Whether to validate s-expressions first
            
    #     Returns:
    #         List of ExecutionResult objects
    #     """
    #     results = []
        
    #     for sexpr in sexprs:
    #         result = self.execute_sexpr(sexpr, validate, function_state=None)
    #         results.append(result)
        
    #     return result
    
    def _is_dangerous_query(self, sparql: str) -> bool:
        """
        Detect dangerous SPARQL queries that could cause timeouts due to full table scans
        
        Args:
            sparql: SPARQL query string
            
        Returns:
            True if the query is dangerous and should be blocked
        """
        if not sparql:
            return False
            
        # Normalize the query for analysis
        normalized = ' '.join(sparql.split()).upper()
        
        # Check for dangerous patterns:
        # 1. Query with only FILTER clauses and no actual constraints
        if 'WHERE {' in normalized:
            # Extract WHERE content more carefully - find the last closing brace
            where_start = normalized.find('WHERE {') + 7  # Skip "WHERE {"
            where_end = normalized.rfind('}')  # Last closing brace
            if where_end > where_start:
                where_part = normalized[where_start:where_end]
            else:
                where_part = normalized.split('WHERE {')[1].split('}')[0]
            
            # Remove FILTER and OPTIONAL clauses to see if there are actual constraints
            import re

            # Remove OPTIONAL and FILTER clauses with improved matching
            # First remove OPTIONAL blocks (may contain nested FILTER)
            def remove_optional_blocks(text):
                result = text
                while 'OPTIONAL' in result:
                    optional_pos = result.find('OPTIONAL')
                    if optional_pos == -1:
                        break
                    
                    # Find the opening brace
                    brace_start = result.find('{', optional_pos)
                    if brace_start == -1:
                        break
                    
                    # Find the matching closing brace
                    brace_count = 0
                    brace_end = -1
                    for i in range(brace_start, len(result)):
                        if result[i] == '{':
                            brace_count += 1
                        elif result[i] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                brace_end = i
                                break
                    
                    if brace_end != -1:
                        # Remove the entire OPTIONAL block
                        result = result[:optional_pos] + result[brace_end+1:]
                    else:
                        # Fallback: remove just the word OPTIONAL
                        result = result.replace('OPTIONAL', '', 1)
                
                return result
            
            where_cleaned = remove_optional_blocks(where_part)
            
            # Remove FILTER clauses with better parentheses matching
            # This handles nested parentheses in FILTER expressions
            def remove_filters(text):
                result = text
                while 'FILTER' in result:
                    # Find FILTER and match balanced parentheses
                    filter_pos = result.find('FILTER')
                    if filter_pos == -1:
                        break
                    
                    # Find the opening parenthesis
                    paren_start = result.find('(', filter_pos)
                    if paren_start == -1:
                        break
                    
                    # Find the matching closing parenthesis
                    paren_count = 0
                    paren_end = -1
                    for i in range(paren_start, len(result)):
                        if result[i] == '(':
                            paren_count += 1
                        elif result[i] == ')':
                            paren_count -= 1
                            if paren_count == 0:
                                paren_end = i
                                break
                    
                    if paren_end != -1:
                        # Remove the entire FILTER clause
                        result = result[:filter_pos] + result[paren_end+1:]
                    else:
                        # Fallback: remove just the word FILTER
                        result = result.replace('FILTER', '', 1)
                
                return result
            
            where_cleaned = remove_filters(where_cleaned)
            
            where_without_filters = where_cleaned
            
            # If after removing filters and optional, there's no substantial content
            substantial_content = where_without_filters.strip()
            logger.debug(f"[DANGEROUS-QUERY-DEBUG] After cleanup: '{substantial_content}' (length: {len(substantial_content)})")
            
            if not substantial_content or len(substantial_content) < 10:
                logger.warning("[DANGEROUS-QUERY] Query contains only filters/optional without constraints")
                #  
                return True
            
            # Additional check: if WHERE clause only contains FILTER and OPTIONAL without any triple patterns
            # Look for actual triple patterns (subject predicate object)
            # Remove all FILTER and OPTIONAL clauses
            cleaned_where = re.sub(r'FILTER\s*\([^)]*\)', '', where_part)
            cleaned_where = re.sub(r'OPTIONAL\s*\{[^}]*\}', '', cleaned_where)
            cleaned_where = cleaned_where.strip()
            
            # Check if there are any actual triple patterns left
            # A triple pattern should have at least one variable or entity reference
            if cleaned_where and not any(pattern in cleaned_where for pattern in ['?', 'NS:', 'HTTP:', '.']):
                logger.warning("[DANGEROUS-QUERY] Query contains no triple patterns, only filters")
                return True
                
            # Check if the query only contains language filters without entity constraints
            if 'ISLITERAL' in substantial_content and 'LANGMATCHES' in substantial_content:
                # Remove common filter patterns
                cleaned = re.sub(r'!\s*ISLITERAL\s*\(\s*\?\w+\s*\)', '', substantial_content)
                cleaned = re.sub(r'LANG\s*\(\s*\?\w+\s*\)\s*=\s*\'\'\'', '', cleaned)
                cleaned = re.sub(r'LANGMATCHES\s*\([^)]*\)', '', cleaned)
                
                if not cleaned.strip():
                    logger.warning("[DANGEROUS-QUERY] Query contains only language filters")
                    return True
        
        # 2. SELECT without substantial WHERE constraints
        if 'SELECT' in normalized and 'WHERE' in normalized:
            # Check if WHERE clause is essentially empty
            where_start = normalized.find('WHERE {')
            if where_start != -1:
                where_end = normalized.find('}', where_start)
                if where_end != -1:
                    where_content = normalized[where_start+7:where_end]  # Skip "WHERE {"
                    # Remove common patterns that don't provide real constraints
                    cleaned_where = re.sub(r'FILTER\s*\([^)]*\)', '', where_content)
                    cleaned_where = re.sub(r'OPTIONAL\s*\{[^}]*\}', '', cleaned_where)
                    cleaned_where = cleaned_where.strip()
                    
                    if len(cleaned_where) < 5:  # Very short WHERE clause
                        logger.warning("[DANGEROUS-QUERY] Query has insufficient WHERE constraints")
                        return True
        
        return False
    
    def validate_sexpr_executability(self, sexpr: str) -> Dict[str, Any]:
        """Validate s-expression executability"""
        result = {
            'is_executable': False,
            'syntax_valid': False,
            'sparql_valid': False,
            'errors': [],
            'warnings': []
        }
        
        # Syntax validation
        validation_result = self.validator.validate(sexpr, ValidationLevel.STANDARD)
        result['syntax_valid'] = validation_result.is_valid
        result['errors'].extend(validation_result.errors)
        result['warnings'].extend(validation_result.warnings)
        
        # SPARQL conversion validation
        try:
            sparql = self.sparql_converter.lisp_to_sparql(sexpr)
            result['sparql_valid'] = sparql is not None
        except Exception as e:
            result['errors'].append(f"SPARQL conversion error: {e}")
        
        result['is_executable'] = result['syntax_valid'] and result['sparql_valid'] and len(result['errors']) == 0
        return result
    
    # def execute_with_enhanced_features(self, sexpr: str) -> ExecutionResult:
    #     """
    #     Execute s-expression using enhanced features from KBQA-o1 integration
    #     """
    #     if not self.enhanced_executor:
    #         return self.execute_sexpr(sexpr, function_state=None)
        
    #     try:
    #         sparql = self.sparql_converter.lisp_to_sparql(sexpr)
    #         if not sparql:
    #             return ExecutionResult(
    #                 sexpr=sexpr, sparql=None, results=[], is_successful=False,
    #                 error_message="Failed to convert s-expression with complete converter"
    #             )
            
    #         results = self.enhanced_executor.execute_query_with_odbc(sparql)
    #         return ExecutionResult(sexpr=sexpr, sparql=sparql, results=results, is_successful=True)
            
    #     except Exception as e:
    #         logger.error(f"Enhanced execution failed: {e}")
    #         return ExecutionResult(sexpr=sexpr, sparql=None, results=[], is_successful=False, error_message=str(e))
    
    def get_sparql_only(self, sexpr: str) -> Optional[str]:
        """
        Get SPARQL conversion without executing
        
        Args:
            sexpr: S-expression to convert
            
        Returns:
            SPARQL query string or None
        """
        return self.sexpr_to_sparql(sexpr)
    
    def test_sexpr_conversion(self, sexpr: str) -> Dict[str, Any]:
        """
        Test s-expression conversion for debugging
        
        Args:
            sexpr: S-expression to test
            
        Returns:
            Dictionary with conversion details
        """
        result = {
            "input_sexpr": sexpr,
            "validation": None,
            "nested_structure": None,
            "sparql": None,
            "execution_result": None
        }
        
        # Validate
        validation_result = self.validator.validate(sexpr, ValidationLevel.STANDARD)
        result["validation"] = {
            "is_valid": validation_result.is_valid,
            "errors": validation_result.errors,
            "warnings": validation_result.warnings
        }
        
        # Parse to nested structure
        try:
            nested = self._parse_sexpr_to_nested(sexpr)
            result["nested_structure"] = nested
        except Exception as e:
            result["nested_structure"] = f"Error: {e}"
        
        # Convert to SPARQL
        try:
            sparql = self.sexpr_to_sparql(sexpr)
            result["sparql"] = sparql
        except Exception as e:
            result["sparql"] = f"Error: {e}"
        
        # Execute if possible
        if self.sparql_manager and result["sparql"]:
            try:
                exec_result = self.execute_sexpr(sexpr, validate=False, function_state=None)
                result["execution_result"] = {
                    "is_successful": exec_result.is_successful,
                    "results": exec_result.results,
                    "error": exec_result.error_message
                }
            except Exception as e:
                result["execution_result"] = f"Error: {e}"
        
        return result
    
    def validate_sexpr_executability(self, sexpr: str) -> Dict[str, Any]:
        """Validate s-expression executability"""
        result = {
            'is_executable': False,
            'syntax_valid': False,
            'sparql_valid': False,
            'errors': [],
            'warnings': []
        }
        
        # Syntax validation
        validation_result = self.validator.validate(sexpr, ValidationLevel.STANDARD)
        result['syntax_valid'] = validation_result.is_valid
        result['errors'].extend(validation_result.errors)
        result['warnings'].extend(validation_result.warnings)
        
        # SPARQL conversion validation
        try:
            sparql = self.sparql_converter.lisp_to_sparql(sexpr)
            result['sparql_valid'] = sparql is not None
        except Exception as e:
            result['errors'].append(f"SPARQL conversion error: {e}")
        
        result['is_executable'] = result['syntax_valid'] and result['sparql_valid'] and len(result['errors']) == 0
        return result
    
    # def execute_with_enhanced_features(self, sexpr: str) -> ExecutionResult:
    #     """
    #     Execute s-expression using enhanced features from KBQA-o1 integration
    #     """
    #     if not self.enhanced_executor:
    #         return self.execute_sexpr(sexpr, function_state=None)
        
    #     try:
    #         sparql = self.sparql_converter.lisp_to_sparql(sexpr)
    #         if not sparql:
    #             return ExecutionResult(
    #                 sexpr=sexpr, sparql=None, results=[], is_successful=False,
    #                 error_message="Failed to convert s-expression with complete converter"
    #             )
            
    #         results = self.enhanced_executor.execute_query_with_odbc(sparql)
    #         return ExecutionResult(sexpr=sexpr, sparql=sparql, results=results, is_successful=True)
            
    #     except Exception as e:
    #         logger.error(f"Enhanced execution failed: {e}")
    #         return ExecutionResult(sexpr=sexpr, sparql=None, results=[], is_successful=False, error_message=str(e))
