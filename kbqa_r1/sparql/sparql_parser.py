"""
SPARQL Parser integrating KBQA-o1's parsing functionality
Provides universal SPARQL to S-Expression conversion
"""

import re
import logging
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class DatasetType(Enum):
    """Supported dataset types"""
    WEBQSP = "webqsp"
    GRAILQA = "grailqa"
    GRAPHQ = "graphq"
    UNIVERSAL = "universal"

@dataclass
class ParseResult:
    """Result of SPARQL parsing"""
    sexpr: str
    is_successful: bool
    dataset_type: DatasetType
    error_message: Optional[str] = None
    original_sparql: Optional[str] = None

class UniversalSPARQLParser:
    """
    Universal SPARQL parser that combines functionality from KBQA-o1's
    parse_sparql_webqsp.py, parse_sparql_grailqa.py, and parse_sparql_graphq.py
    """
    
    def __init__(self):
        self.parser_webqsp = WebQSPParser()
        self.parser_grailqa = GrailQAParser()
        self.parser_graphq = GraphQParser()
        
    def parse_sparql(self, sparql: str, dataset_type: DatasetType = DatasetType.UNIVERSAL,
                    mid_list: Optional[List[str]] = None) -> ParseResult:
        """
        Parse SPARQL query to S-Expression
        
        Args:
            sparql: SPARQL query string
            dataset_type: Type of dataset (auto-detect if UNIVERSAL)
            mid_list: List of entity MIDs in the query
            
        Returns:
            ParseResult with S-Expression
        """
        if mid_list is None:
            mid_list = self._extract_mids_from_sparql(sparql)
        
        try:
            if dataset_type == DatasetType.UNIVERSAL:
                # Try to auto-detect dataset type and parse
                return self._parse_with_auto_detection(sparql, mid_list)
            elif dataset_type == DatasetType.WEBQSP:
                sexpr = self.parser_webqsp.parse_query_webqsp(sparql, mid_list)
            elif dataset_type == DatasetType.GRAILQA:
                sexpr = self.parser_grailqa.parse_query_cwq(sparql, mid_list)
            elif dataset_type == DatasetType.GRAPHQ:
                sexpr = self.parser_graphq.parse_query_cwq(sparql, mid_list)
            else:
                return ParseResult(
                    sexpr="",
                    is_successful=False,
                    dataset_type=dataset_type,
                    error_message=f"Unsupported dataset type: {dataset_type}",
                    original_sparql=sparql
                )
            
            return ParseResult(
                sexpr=sexpr,
                is_successful=sexpr != 'null' and sexpr != '',
                dataset_type=dataset_type,
                original_sparql=sparql
            )
            
        except Exception as e:
            logger.error(f"Error parsing SPARQL: {e}")
            return ParseResult(
                sexpr="",
                is_successful=False,
                dataset_type=dataset_type,
                error_message=str(e),
                original_sparql=sparql
            )
    
    def _parse_with_auto_detection(self, sparql: str, mid_list: List[str]) -> ParseResult:
        """Try parsing with different dataset-specific parsers"""
        parsers = [
            (DatasetType.WEBQSP, self.parser_webqsp.parse_query_webqsp),
            (DatasetType.GRAILQA, self.parser_grailqa.parse_query_cwq),
            (DatasetType.GRAPHQ, self.parser_graphq.parse_query_cwq)
        ]
        
        for dataset_type, parser_func in parsers:
            try:
                sexpr = parser_func(sparql, mid_list)
                if sexpr and sexpr != 'null':
                    return ParseResult(
                        sexpr=sexpr,
                        is_successful=True,
                        dataset_type=dataset_type,
                        original_sparql=sparql
                    )
            except:
                continue
        
        return ParseResult(
            sexpr="",
            is_successful=False,
            dataset_type=DatasetType.UNIVERSAL,
            error_message="Failed to parse with any dataset-specific parser",
            original_sparql=sparql
        )
    
    def _extract_mids_from_sparql(self, sparql: str) -> List[str]:
        """Extract Freebase MIDs from SPARQL query"""
        # Pattern for Freebase entities
        mid_pattern = r'ns:([mg]\.\w+)'
        mids = re.findall(mid_pattern, sparql)
        return [f'ns:{mid}' for mid in mids]

class BaseParser:
    """Base class for dataset-specific parsers"""
    
    def __init__(self):
        pass
    
    def parse_assert(self, condition):
        """Assert condition for parsing validation"""
        if not condition:
            raise ParseError("Parse assertion failed")
    
    def normalize_body_lines(self, lines, filter_string_flag=False):
        """
        Normalize SPARQL body lines
        Common functionality extracted from KBQA-o1 parsers
        """
        spec_condition = []
        
        # Handle string filters
        if filter_string_flag:
            filter_lines = [x.strip() for x in lines if 'FILTER (str' in x]
            lines = [x.strip() for x in lines if 'FILTER (str' not in x]
        else:
            lines = [x.strip() for x in lines]
            filter_lines = None
        
        # Handle comparison operators
        if len(lines) >= 2 and (
            re.match(r'FILTER \(\?\w* (>|<|>=|<=) .*', lines[-2]) or
            re.match(r'FILTER \(xsd:integer\(\?\w*\) (>|<|>=|<=) .*', lines[-2])
        ):
            compare_line = lines.pop(-2)
            compare_var = re.findall(r'\?\w*', compare_line)[0]
            compare_operator = re.findall(r'(>|>=|<|<=)', compare_line)[0]
            operator_mapper = {'<': 'lt', '<=': 'le', '>': 'gt', ">=": "ge"}
            
            if "^^xsd:dateTime" in compare_line:
                compare_value = re.findall(r'".*"\^\^xsd:dateTime', compare_line)[0]
            else:
                compare_value = compare_line.replace(") .", "").split(" ")[-1]
            
            compare_value = compare_value.replace('"', '')
            compare_condition = ['COMPARATIVE', operator_mapper[compare_operator], compare_var, compare_value]
            spec_condition.append(compare_condition)
        
        # Handle superlative (ORDER BY ... LIMIT 1)
        body_lines = []
        if lines[-1] == 'LIMIT 1':
            order_line = lines[-2]
            direction = 'argmax' if 'DESC(' in order_line else 'argmin'
            compare_var = re.findall(r'\?\w*', order_line)[0]
            
            _tmp_body_lines = lines[1:-3]
            hit = False
            for l in _tmp_body_lines:
                if compare_var in l:
                    self.parse_assert(l.endswith(compare_var + " ") and not hit)
                    hit = True
                    arg_var, arg_r = l.split(' ')[0], l.split(' ')[1]
                    arg_r = arg_r[3:]  # remove ns:
                else:
                    body_lines.append(l)
            
            superlative_cond = ['SUPERLATIVE', direction, arg_var, arg_r]
            spec_condition.append(superlative_cond)
        
        # Handle time constraints (range filters)
        if not body_lines:
            body_lines = lines[1:-1]
        
        return body_lines, spec_condition, filter_lines

class WebQSPParser(BaseParser):
    """WebQSP-specific SPARQL parser"""
    
    def parse_query_webqsp(self, query: str, mid_list: List[str]) -> str:
        """
        Parse WebQSP SPARQL query to S-Expression
        Adapted from KBQA-o1's parse_sparql_webqsp.py
        """
        lines = query.split('\n')
        lines = [x for x in lines if x]
        
        if lines[0] == '#MANUAL SPARQL':
            return 'null'
        
        # Skip PREFIX statements
        line_num = 0
        while line_num < len(lines) and lines[line_num].startswith('PREFIX'):
            line_num += 1
        
        # Validate SELECT and WHERE
        if line_num >= len(lines) or not lines[line_num].startswith('SELECT DISTINCT ?x'):
            return 'null'
        line_num += 1
        
        if line_num >= len(lines) or lines[line_num] != 'WHERE {':
            return 'null'
        
        # Handle ORDER BY and LIMIT
        if re.match(r'ORDER BY .*\?\w*.* LIMIT 1', lines[-1]):
            lines[-1] = lines[-1].replace('LIMIT 1', '').strip()
            lines.append('LIMIT 1')
        
        if re.match(r'LIMIT \d*', lines[-1]):
            lines[-1] = 'LIMIT 1'
        
        if lines[-1].startswith('OFFSET'):
            lines.pop(-1)
        
        if lines[-1] not in ['}', 'LIMIT 1']:
            return 'null'
        
        lines = lines[line_num:]
        filter_string_flag = not all(['FILTER (str' not in x for x in lines])
        
        # Normalize body lines
        body_lines, spec_condition, filter_lines = self.normalize_body_lines(lines, filter_string_flag)
        body_lines = [x.strip() for x in body_lines]
        
        # Handle predefined filters
        if body_lines and body_lines[0].startswith('FILTER'):
            if len(body_lines) < 2:
                return 'null'
            
            predefined_filter0 = body_lines[0]
            predefined_filter1 = body_lines[1]
            
            # Validate filters
            filter_0_valid = (predefined_filter0 == 'FILTER (?x != ?c)')
            if not filter_0_valid:
                for mid in mid_list:
                    if predefined_filter0 == f'FILTER (?x != {mid})':
                        filter_0_valid = True
                        break
            
            if not filter_0_valid:
                return 'null'
            
            expected_filter1 = "FILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))"
            if predefined_filter1 != expected_filter1:
                return 'null'
            
            body_lines = body_lines[2:]
        
        # Validate body line format
        if not all([(x.startswith('?') or x.startswith('ns:')) for x in body_lines]):
            return 'null'
        
        # Parse to dependency graph and convert to S-Expression
        try:
            var_dep_list = self.parse_naive_body(body_lines, filter_lines, '?x', spec_condition)
            s_expr = self.dep_graph_to_s_expr(var_dep_list, '?x', spec_condition)
            return s_expr
        except:
            return 'null'
    
    def parse_naive_body(self, body_lines, filter_lines, ret_var, spec_condition=None):
        """Parse body lines to variable dependency list"""
        # Simplified implementation - would need full logic from KBQA-o1
        # This is a placeholder that would need the complete implementation
        var_dep_list = []
        
        # Process triplets
        triplets = []
        for line in body_lines:
            if line.endswith('.'):
                parts = line[:-1].split(' ')
                if len(parts) >= 3:
                    triplets.append(parts)
        
        # Build dependency graph (simplified)
        for triplet in triplets:
            var_dep_list.append((ret_var, [triplet]))
        
        return var_dep_list
    
    def dep_graph_to_s_expr(self, var_dep_list, ret_var, spec_condition=None):
        """Convert dependency graph to S-Expression"""
        # Simplified implementation - would need full logic from KBQA-o1
        if not var_dep_list:
            return 'null'
        
        # Basic S-Expression construction
        clauses = []
        for var, deps in var_dep_list:
            for dep in deps:
                if len(dep) >= 3:
                    if dep[1].startswith('ns:'):
                        relation = dep[1][3:]
                        if dep[2].startswith('ns:'):
                            entity = dep[2][3:]
                            clauses.append(f'(JOIN {relation} {entity})')
                        else:
                            clauses.append(f'(JOIN {relation} {dep[2]})')
        
        # Handle special conditions
        if spec_condition:
            for cond in spec_condition:
                if cond[0] == 'SUPERLATIVE':
                    direction, arg_var, arg_r = cond[1], cond[2], cond[3]
                    if clauses:
                        clauses[0] = f'({direction.upper()} {clauses[0]} {arg_r})'
        
        return clauses[0] if clauses else 'null'

class GrailQAParser(BaseParser):
    """GrailQA-specific SPARQL parser"""
    
    def parse_query_cwq(self, query: str, mid_list: List[str]) -> str:
        """
        Parse GrailQA SPARQL query to S-Expression
        Adapted from KBQA-o1's parse_sparql_grailqa.py
        """
        # Similar structure to WebQSP but with GrailQA-specific logic
        # This would be the full implementation from the original file
        return self._parse_cwq_internal(query, mid_list)
    
    def _parse_cwq_internal(self, query: str, mid_list: List[str]) -> str:
        """Internal CWQ parsing logic"""
        # Placeholder for full GrailQA parsing implementation
        # Would include the complete logic from parse_sparql_grailqa.py
        return 'null'

class GraphQParser(BaseParser):
    """GraphQ-specific SPARQL parser"""
    
    def parse_query_cwq(self, query: str, mid_list: List[str]) -> str:
        """
        Parse GraphQ SPARQL query to S-Expression
        Adapted from KBQA-o1's parse_sparql_graphq.py
        """
        # Similar structure with GraphQ-specific logic
        return self._parse_graphq_internal(query, mid_list)
    
    def _parse_graphq_internal(self, query: str, mid_list: List[str]) -> str:
        """Internal GraphQ parsing logic"""
        # Placeholder for full GraphQ parsing implementation
        # Would include the complete logic from parse_sparql_graphq.py
        return 'null'

class ParseError(Exception):
    """Exception raised during SPARQL parsing"""
    pass
