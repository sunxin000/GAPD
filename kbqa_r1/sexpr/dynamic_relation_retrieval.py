"""
Dynamic Relation Retrieval for S-Expression based KBQA-R1
Implements KBQA-o1's get_next_relations and get_next_r_relations mechanisms
"""

import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from ..sparql.sparql_manager import SPARQLConfig, SPARQLExecutionManager
from .sexpr_executor import SExprExecutor
# Import existing components
from .sexpr_generator import SExprGenerator

logger = logging.getLogger(__name__)

@dataclass
class RelationQueryResult:
    """Result of dynamic relation query"""
    relations: List[str]
    sparql_query: Optional[str] = None
    is_successful: bool = True
    error_message: Optional[str] = None

class DynamicRelationRetrieval:
    """
    Implements KBQA-o1's dynamic relation retrieval mechanism
    Dynamically queries knowledge base for available relations based on current state
    """
    
    def __init__(self, sparql_config: SPARQLConfig = None, dataset: str = "webqsp"):
        self.dataset = dataset
        self.sexpr_generator = SExprGenerator()
        self.sexpr_executor = SExprExecutor(sparql_config, dataset_type=dataset) if sparql_config else None
        self.sparql_manager = SPARQLExecutionManager(sparql_config) if sparql_config else None
        
        # Cache for relation queries
        self.relation_cache = {}
        self.cache_enabled = True
    
    def get_next_relations(self, function_list: List[str]) -> RelationQueryResult:
        """
        Get next possible forward relations from knowledge base
        Implements KBQA-o1's get_next_relations logic
        
        Args:
            function_list: Current function sequence
            
        Returns:
            RelationQueryResult with available forward relations
        """
        return self._get_relations(function_list, relation_type="forward")
    
    def get_next_r_relations(self, function_list: List[str]) -> RelationQueryResult:
        """
        Get next possible reverse relations from knowledge base
        Implements KBQA-o1's get_next_r_relations logic
        
        Args:
            function_list: Current function sequence
            
        Returns:
            RelationQueryResult with available reverse relations
        """
        return self._get_relations(function_list, relation_type="reverse")
    
    def _get_relations(self, function_list: List[str], relation_type: str) -> RelationQueryResult:
        """
        Core relation retrieval logic implementing KBQA-o1's pattern
        
        Args:
            function_list: Current function sequence
            relation_type: "forward" or "reverse"
            
        Returns:
            RelationQueryResult with relations
        """
        if not function_list:
            return RelationQueryResult(
                relations=[],
                is_successful=False,
                error_message="Empty function list"
            )
        
        # Create cache key
        cache_key = None
        if self.cache_enabled:
            cache_key = (tuple(function_list), relation_type)
            if cache_key in self.relation_cache:
                return self.relation_cache[cache_key]
        
        try:
            # Step 1: Extract current expression ID (KBQA-o1 pattern)
            id_now = self._extract_expression_id(function_list[-1])
            
            # Step 2: Prepare function list with STOP (KBQA-o1 pattern)
            func_list_copy = function_list.copy()
            func_list_copy.append(f'expression{id_now} = STOP(expression{id_now})')
            
            # Step 3: Generate S-Expression
            sexpr_result = self.sexpr_generator.generate_sexpr_from_strings(
                func_list_copy, f'expression{id_now}'
            )
            
            if not sexpr_result.is_valid:
                result = RelationQueryResult(
                    relations=[],
                    is_successful=False,
                    error_message=f"S-Expression generation failed: {sexpr_result.error_message}"
                )
                if cache_key:
                    self.relation_cache[cache_key] = result
                return result
            
            sexpr = sexpr_result.sexpr
            
            # Step 4: Generate SPARQL query based on S-Expression type (KBQA-o1 logic)
            sparql_query = self._generate_relation_query(sexpr, relation_type, id_now, func_list_copy)
            
            if not sparql_query:
                result = RelationQueryResult(
                    relations=[],
                    is_successful=False,
                    error_message="Failed to generate relation query"
                )
                if cache_key:
                    self.relation_cache[cache_key] = result
                return result
            
            # Step 5: Execute SPARQL query
            relations = self._execute_relation_query(sparql_query)
            
            result = RelationQueryResult(
                relations=relations,
                sparql_query=sparql_query,
                is_successful=True
            )
            
            if cache_key:
                self.relation_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error in dynamic relation retrieval: {e}")
            result = RelationQueryResult(
                relations=[],
                is_successful=False,
                error_message=str(e)
            )
            if cache_key:
                self.relation_cache[cache_key] = result
            return result
    
    def _extract_expression_id(self, function_string: str) -> str:
        """Extract expression ID from function string"""
        match = re.search(r"expression(\d*)", function_string)
        return match.group(1) if match else "1"
    
    def _detect_entity_type(self, entity: str) -> str:
        """Detect entity type (implements KBQA-o1's ent_type function)"""
        if not entity:
            return 'unknown'
        
        # Freebase entity IDs
        if entity.startswith('m.') or entity.startswith('g.'):
            return 'entity'
        
        # Check if it's a number
        try:
            float(entity)
            return 'int' if '.' not in entity else 'literal'
        except ValueError:
            pass
        
        # Check if it's a date
        if re.match(r'\d{4}(-\d{2})?(-\d{2})?', entity):
            return 'literal'
        
        # Check if it's a URL
        if entity.startswith('http'):
            return 'url'
        
        # Default classification
        return 'name'
    
    def _generate_relation_query(self, sexpr: str, relation_type: str, 
                                expr_id: str, func_list: List[str]) -> Optional[str]:
        """
        Generate SPARQL query for relation retrieval based on S-Expression
        Implements KBQA-o1's complex query generation logic
        """
        #  
        try:
            if not sexpr.startswith('('):
                # Simple entity case
                entity_type = self._detect_entity_type(sexpr)
                
                if entity_type in ['entity', 'onto']:
                    # Simple entity query
                    if relation_type == "reverse":
                        # ?y ?relation ns:entity
                        return f"""PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?relation
WHERE {{
    ns:{sexpr} ?relation ?y .
    FILTER (?y != ns:{sexpr})
}}"""
                    else:
                        # ns:entity ?relation ?y  
                        return f"""PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?relation
WHERE {{
    ?y ?relation ns:{sexpr} .
    FILTER (?y != ns:{sexpr})
}}"""
                
                elif entity_type in ['int', 'url']:
                    # Complex type query using template substitution
                    return self._generate_template_query(func_list, expr_id, relation_type)
            
            else:
                # Complex S-Expression case
                return self._generate_complex_query(sexpr, relation_type)
                
        except Exception as e:
            logger.error(f"Error generating relation query: {e}")
            return None
    
    def _generate_template_query(self, func_list: List[str], expr_id: str, relation_type: str) -> Optional[str]:
        """Generate query using template substitution for complex types"""
        try:
            # Create template function with placeholder relation
            if relation_type == "reverse":
                template_func = f"expression{expr_id} = JOIN('(R RELATION)', expression{expr_id})"
            else:
                template_func = f"expression{expr_id} = JOIN('RELATION', expression{expr_id})"
            
            # Insert template into function list
            func_list_template = func_list[:-1] + [template_func] + [func_list[-1]]
            
            # Generate S-Expression from template
            template_sexpr_result = self.sexpr_generator.generate_sexpr_from_strings(
                func_list_template, f'expression{expr_id}'
            )
            
            if not template_sexpr_result.is_valid:
                return None
            
            # Convert to SPARQL and replace placeholder
            if self.sexpr_executor:
                template_sparql = self.sexpr_executor.sexpr_to_sparql(template_sexpr_result.sexpr)
                if template_sparql:
                    return template_sparql.replace('ns:RELATION', '?relation').replace(
                        'SELECT DISTINCT ?x', 'SELECT DISTINCT ?relation'
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in template query generation: {e}")
            return None
    
    def _generate_complex_query(self, sexpr: str, relation_type: str) -> Optional[str]:
        """Generate query for complex S-Expression"""

        if self.sexpr_executor:
            original_sparql = self.sexpr_executor.sexpr_to_sparql(sexpr)
            
            if original_sparql:
                # 1) Drop all PREFIX lines (including rdfs, ns, etc.)
                cleaned = []
                for line in original_sparql.splitlines():
                    if line.strip().upper().startswith('PREFIX '):
                        continue
                    cleaned.append(line)
                cleaned_sparql = "\n".join(cleaned)

                # 2) Remove OPTIONAL label clauses (rdfs:label), if present
                cleaned_sparql = re.sub(r"OPTIONAL\s*\{[^}]*rdfs:label[^}]*\}\s*", "", cleaned_sparql, flags=re.IGNORECASE)

                # 3) Remove LIMIT clauses
                cleaned_sparql = re.sub(r"\bLIMIT\s+\d+\s*;?", "", cleaned_sparql, flags=re.IGNORECASE)

                # Add protective caps: simplify inner SELECT to only ?x and add inner LIMIT
                # Normalize SELECT projection to SELECT DISTINCT ?x
                cleaned_sparql = re.sub(r"^\s*SELECT\s+DISTINCT\s+\?x\s+\?name", "SELECT DISTINCT ?x", cleaned_sparql, flags=re.IGNORECASE | re.MULTILINE)
                cleaned_sparql = re.sub(r"^\s*SELECT\s+\*", "SELECT DISTINCT ?x", cleaned_sparql, flags=re.IGNORECASE | re.MULTILINE)
                cleaned_sparql = re.sub(r"^\s*SELECT\s+DISTINCT\s+\?x(\s+.*)?\bWHERE", "SELECT DISTINCT ?x WHERE", cleaned_sparql, flags=re.IGNORECASE | re.MULTILINE)

                inner_limit = int(os.getenv('KBQA_INNER_LIMIT', '500'))
                # Append inner LIMIT at the end of subquery
                cleaned_sparql = cleaned_sparql.rstrip() + f"\nLIMIT {inner_limit}\n"

                # # 4) Remove the SELECT line (keep only WHERE body)
                # cleaned_sparql = re.sub(r"^\s*SELECT[\s\S]*?WHERE\s*\{", "WHERE {", cleaned_sparql, flags=re.IGNORECASE | re.MULTILINE)

                # Indent cleaned query for embedding readability (use full cleaned query as subquery)
                body_indented = "\n".join(["    " + ln for ln in cleaned_sparql.strip().splitlines()])

                # Create nested query with clean body
                if relation_type == "reverse":
                    # Find relations where current results are objects
                    outer_limit = int(os.getenv('KBQA_OUTER_LIMIT', '200'))
                    return f"""PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?relation
WHERE {{
{{
{body_indented}
}}

?x ?relation ?y .
}}
LIMIT {outer_limit}"""
                else:
                    # Find relations where current results are subjects
                    outer_limit = int(os.getenv('KBQA_OUTER_LIMIT', '200'))
                    return f"""PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?relation
WHERE {{
{{
{body_indented}
}}

?y ?relation ?x .
}}
LIMIT {outer_limit}"""
        
        return None
            

    
    def _execute_relation_query(self, sparql_query: str) -> List[str]:
        """Execute SPARQL query and extract relations"""
        try:
            if not self.sparql_manager:
                logger.warning("No SPARQL manager available, returning empty relations")
                return []
            
            # Execute query
            result = self.sparql_manager.execute_batch([sparql_query])
            
            if "results" in result and result["results"]:
                query_result = result["results"][0]
                
                if isinstance(query_result, dict) and "error" in query_result:
                    logger.warning(f"SPARQL query error: {query_result['error']}")
                    return []
                
                # Extract relations from results
                relations = []
                if isinstance(query_result, dict) and "results" in query_result:
                    results_list = query_result["results"]
                elif isinstance(query_result, list):
                    results_list = query_result
                else:
                    results_list = [query_result]
                
                # Process results to extract relation names
                for res in results_list:
                    if isinstance(res, str):
                        # Clean relation name
                        clean_relation = res.replace("http://rdf.freebase.com/ns/", "")
                        if clean_relation and clean_relation != "None":
                            relations.append(clean_relation)
                    elif isinstance(res, dict) and "relation" in res:
                        clean_relation = str(res["relation"]).replace("http://rdf.freebase.com/ns/", "")
                        if clean_relation and clean_relation != "None":
                            relations.append(clean_relation)
                
                return list(set(relations))  # Remove duplicates
            
            return []
            
        except Exception as e:
            logger.error(f"Error executing relation query: {e}")
            return []
    
    def get_all_relations(self, function_list: List[str]) -> Dict[str, List[str]]:
        """Get both forward and reverse relations"""
        forward_result = self.get_next_relations(function_list)
        reverse_result = self.get_next_r_relations(function_list)
        
        return {
            "forward": forward_result.relations,
            "reverse": reverse_result.relations,
            "forward_success": forward_result.is_successful,
            "reverse_success": reverse_result.is_successful,
            "forward_query": forward_result.sparql_query,
            "reverse_query": reverse_result.sparql_query
        }
    
    def clear_cache(self):
        """Clear relation cache"""
        self.relation_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            "cache_size": len(self.relation_cache),
            "cache_enabled": self.cache_enabled
        }
    
    def enable_cache(self, enabled: bool = True):
        """Enable or disable caching"""
        self.cache_enabled = enabled
        if not enabled:
            self.clear_cache()


