"""
Complete SPARQL Converter integrating KBQA-o1's logic_form_util.py functionality
Provides full S-Expression to SPARQL conversion with all advanced features
"""

import logging
import re
from typing import Dict, List

logger = logging.getLogger(__name__)

class SPARQLConverter:
    """
    WebQSP/GraphQ S-Expression to SPARQL converter
    Integrates core functionality from KBQA-o1's logic_form_util.py
    Optimized for WebQSP and GraphQ datasets with advanced features
    """
    
    def __init__(self):
        self.function_map = {'le': '<=', 'ge': '>=', 'lt': '<', 'gt': '>'}
        self.freebase_prefix = "PREFIX ns: <http://rdf.freebase.com/ns/>\nPREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
        
    def lisp_to_sparql(self, lisp_program: str) -> str:
        """
        Convert S-Expression to SPARQL query for WebQSP/GraphQ
        Main conversion function adapted from KBQA-o1's logic_form_util.py
        
        Args:
            lisp_program: S-expression string
            
        Returns:
            Complete SPARQL query string optimized for WebQSP/GraphQ
        """
        try:
            clauses = []
            order_clauses = []
            entities = set()  # collect entities for filtering
            identical_variables_r = {}  # variable merging mapping
            
            expression = self.lisp_to_nested_expression(lisp_program)
            superlative = False
            
            # Check for SUPERLATIVE operations
            if expression[0] in ['ARGMAX', 'ARGMIN']:
                superlative = True
                # Handle n-hop relations in superlative
                if isinstance(expression[2], list):
                    relations = self._retrieve_relations(expression[2])
                    expression = expression[:2]
                    expression.extend(relations)
            
            # Linearize nested expressions
            sub_programs = self._linearize_lisp_expression(expression, [0])
            question_var = len(sub_programs) - 1
            count = False
            
            def get_root(var: int):
                """Get root variable for merging"""
                while var in identical_variables_r:
                    var = identical_variables_r[var]
                return var
            
            # Process each sub-program
            for i, subp in enumerate(sub_programs):
                i_str = str(i)
                
                if subp[0] == 'JOIN':
                    self._process_join(subp, i_str, clauses, entities)
                    
                elif subp[0] == 'AND':
                    self._process_and(subp, i, i_str, clauses, identical_variables_r, get_root)
                    
                elif subp[0] in ['le', 'lt', 'ge', 'gt']:
                    self._process_comparison(subp, i, i_str, clauses, sub_programs)
                    
                elif subp[0] == 'TC':
                    self._process_time_constraint(subp, i, i_str, clauses, identical_variables_r, get_root)
                    
                elif subp[0] in ["ARGMIN", "ARGMAX"]:
                    self._process_superlative(subp, i, i_str, clauses, order_clauses, identical_variables_r, get_root)
                    superlative = True
                    
                elif subp[0] == 'COUNT':
                    if subp[1].startswith('#'):  # Variable reference
                        var = int(subp[1][1:])
                        root_var = get_root(var)
                        identical_variables_r[int(i)] = root_var
                    else:  # Direct class name like 'president'
                        # Add type constraint for the count target
                        clauses.append(f"?x{i_str} ns:type.object.type ns:{subp[1]} .")
                    count = True
            
            # Merge identical variables
            for i in range(len(clauses)):
                for k in identical_variables_r:
                    clauses[i] = clauses[i].replace(f'?x{k} ', f'?x{get_root(k)} ')
            
            question_var = get_root(question_var)
            
            # Replace question variable with ?x
            for i in range(len(clauses)):
                clauses[i] = clauses[i].replace(f'?x{question_var} ', '?x ')
            
            # Add entity and variable filters
            self._add_filters(clauses, entities)
            
            # Add WebQSP/GraphQ string matching filters
            self._add_string_filters(clauses)
            
            # Construct final SPARQL query
            return self._construct_sparql(clauses, order_clauses, count, superlative)
            
        except Exception as e:
            # Trigger breakpoint specifically for invalid int() literal errors to aid debugging
            try:
                if isinstance(e, ValueError) and 'invalid literal for int()' in str(e):
                    #  
                    pass
            except Exception:
                pass
            logger.error(f"Error converting S-expression to SPARQL: {e}")
            raise
    
    def lisp_to_nested_expression(self, lisp_string: str) -> List:
        """
        Parse S-expression string to nested list structure
        Adapted from KBQA-o1's lisp_to_nested_expression
        """
        stack: List = []
        current_expression: List = []
        tokens = lisp_string.split()
        
        for token in tokens:
            while token[0] == '(':
                nested_expression: List = []
                current_expression.append(nested_expression)
                stack.append(current_expression)
                current_expression = nested_expression
                token = token[1:]
            current_expression.append(token.replace(')', ''))
            while token[-1] == ')':
                current_expression = stack.pop()
                token = token[:-1]
        
        return current_expression[0]
    
    def _retrieve_relations(self, exp: list) -> List:
        """Retrieve relations from nested JOIN expressions"""
        rtn = []
        for element in exp:
            if element == 'JOIN':
                continue
            elif isinstance(element, str):
                rtn.append(element)
            elif isinstance(element, list) and element[0] == 'R':
                rtn.append(element)
            elif isinstance(element, list) and element[0] == 'JOIN':
                rtn.extend(self._retrieve_relations(element))
        return rtn
    
    def _linearize_lisp_expression(self, expression: list, sub_formula_id: List[int]) -> List:
        """Linearize nested expressions for processing"""
        sub_formulas = []
        for i, e in enumerate(expression):
            if isinstance(e, list) and e[0] != 'R':
                sub_formulas.extend(self._linearize_lisp_expression(e, sub_formula_id))
                expression[i] = '#' + str(sub_formula_id[0] - 1)
        
        sub_formulas.append(expression)
        sub_formula_id[0] += 1
        return sub_formulas
    
    def _process_join(self, subp: List, i: str, clauses: List[str], entities: set):
        """Process JOIN operations"""
        if isinstance(subp[1], list):  # Reverse relation
            if subp[2][:2] in ["m.", "g."]:  # entity
                clauses.append(f"ns:{subp[2]} ns:{subp[1][1]} ?x{i} .")
                entities.add(subp[2])
            elif subp[2][0] == '#':  # variable
                clauses.append(f"?x{subp[2][1:]} ns:{subp[1][1]} ?x{i} .")
            else:  # literal
                self._handle_literal_join(subp, i, clauses, reverse=True)
        else:  # Forward relation
            if subp[2][:2] in ["m.", "g."]:  # entity
                clauses.append(f"?x{i} ns:{subp[1]} ns:{subp[2]} .")
                entities.add(subp[2])
            elif subp[2][0] == '#':  # variable
                clauses.append(f"?x{i} ns:{subp[1]} ?x{subp[2][1:]} .")
            elif re.match(r'[\w_]*\.[\w_]*\.[\w_]*', subp[2]):  # 2-hop relation
                # Skip 2-hop relations - they will be handled by comparison operations
                pass
            else:  # literal
                self._handle_literal_join(subp, i, clauses, reverse=False)
    
    def _handle_literal_join(self, subp: List, i: str, clauses: List[str], reverse: bool):
        """Handle literal values in JOIN operations"""
        if re.match(r'[\w_]*\.[\w_]*\.[\w_]*', subp[2]):
            # 2-hop relation - skip for now, will be handled by comparison operations
            pass
        else:
            # Handle literals with datatypes
            if subp[2].__contains__('^^'):
                subp[2] = self._format_literal(subp[2])
            elif re.match(r"[a-zA-Z_]*\.[a-zA-Z_]*", subp[2]):  # type
                subp[2] = 'ns:' + subp[2]
            elif len(subp) > 3:  # error splitting
                subp[2] = " ".join(subp[2:])
            
            if reverse:
                clauses.append(f"{subp[2]} ns:{subp[1][1]} ?x{i} .")
            else:
                clauses.append(f"?x{i} ns:{subp[1]} {subp[2]} .")
    
    def _format_literal(self, literal: str) -> str:
        """Format literal values with proper datatypes"""
        if literal.__contains__('^^'):  # Check for any literal with datatype
            data_type_string = literal.split("^^")[1]
            if '#' in data_type_string:
                data_type = data_type_string.split("#")[1]
            elif 'xsd:' in data_type_string:
                data_type = data_type_string.split('xsd:')[1]
            else:
                data_type = 'dateTime'
            
            # Align with KBQA-o1: add timezone only for non-numeric, non-dateTime types
            # i.e., date/gYear/time/gYearMonth → add -08:00; dateTime → keep as is
            if data_type in ['date', 'gYear', 'gYearMonth']:
                value = literal.split("^^")[0]
                # If value already contains timezone (e.g., Z or ±HH:MM), don't append
                has_tz = value.endswith('Z') or bool(re.search(r"[\+\-]\d{2}:\d{2}$", value))
                if has_tz:
                    return f'"{value}"^^<{literal.split("^^")[1]}>'
                return f'"{value + "-08:00"}"^^<{literal.split("^^")[1]}>'
            else:
                return f'"{literal.split("^^")[0]}"^^<{literal.split("^^")[1]}>'
        return literal
    
    def _process_and(self, subp: List, i: int, i_str: str, clauses: List[str], 
                     identical_variables_r: Dict, get_root):
         """Process AND operations"""
         # Normalize order so that a variable reference is always in subp[2]
         # Accept both (AND class #k) and (AND #k class). If neither is a variable, skip.
         if isinstance(subp[2], str) and subp[2].startswith('#'):
             pass
         elif isinstance(subp[1], str) and subp[1].startswith('#'):
             subp[1], subp[2] = subp[2], subp[1]
         else:
             # Invalid/unsupported AND pattern; avoid ValueError
             return
         var1 = int(subp[2][1:])
         rooti = get_root(int(i))
         root1 = get_root(var1)
         
         if rooti > root1:
             identical_variables_r[rooti] = root1
         else:
             identical_variables_r[root1] = rooti
             root1 = rooti
         
         if subp[1][0] == "#":
             var2 = int(subp[1][1:])
             root2 = get_root(var2)
             if root1 > root2:
                 identical_variables_r[root1] = root2
             else:
                 identical_variables_r[root2] = root1
         else:  # 2nd argument is a class
             clauses.append(f"?x{i_str} ns:type.object.type ns:{subp[1]} .")
    
    def _process_comparison(self, subp: List, i: int, i_str: str, clauses: List[str], sub_programs: List):
        """Process comparison operations (le, lt, ge, gt)"""
        if subp[1].startswith('#'):  # Reference to previous operation
            line_num = int(subp[1].replace('#', ''))
            referenced_op = sub_programs[line_num]
            
            # Check if this is a 2-hop constraint (JOIN operation)
            if referenced_op[0] == 'JOIN' and len(referenced_op) >= 3:
                first_relation = referenced_op[1]
                second_relation = referenced_op[2]
                
                if isinstance(first_relation, list):  # first relation is reversed
                    clauses.append(f"?cvt ns:{first_relation[1]} ?x{i_str} .")
                else:
                    clauses.append(f"?x{i_str} ns:{first_relation} ?cvt .")
                
                if isinstance(second_relation, list):  # second relation is reversed
                    clauses.append(f"?y{i_str} ns:{second_relation[1]} ?cvt .")
                else:
                    clauses.append(f"?cvt ns:{second_relation} ?y{i_str} .")
            else:
                # For other operations like COUNT, the comparison applies to the result
                # This is handled differently - we don't add clauses here
                pass
        else:
            clauses.append(f"?x{i_str} ns:{subp[1]} ?y{i_str} .")
        
        # Add comparison filter
        if subp[0] == 'le':
            op = "<="
        elif subp[0] == 'lt':
            op = "<"
        elif subp[0] == 'ge':
            op = ">="
        else:  # gt
            op = ">"
        
        # Format literal value
        if subp[2].__contains__('^^'):
            value = self._format_literal(subp[2])
        else:
            value = subp[2]
        
        if re.match(r'\d+', subp[2]) or re.match(r'"\d+"^^xsd:integer', subp[2]):
            clauses.append(f"FILTER (xsd:integer(?y{i_str}) {op} {value})")
        else:
            clauses.append(f"FILTER (?y{i_str} {op} {value})")
    
    def _process_time_constraint(self, subp: List, i: int, i_str: str, clauses: List[str],
                               identical_variables_r: Dict, get_root):
        """Process time constraint (TC) operations"""
        var = int(subp[1][1:])
        rooti = get_root(int(i))
        root_var = get_root(var)
        
        if rooti > root_var:
            identical_variables_r[rooti] = root_var
        else:
            identical_variables_r[root_var] = rooti
        
        year = subp[3]
        if year == 'NOW' or year == 'now':
            from_para = '"2015-08-10"^^xsd:dateTime'
            to_para = '"2015-08-10"^^xsd:dateTime'
        else:
            if "^^" in year:
                year = year.split("^^")[0]
            from_para = f'"{year}-12-31"^^xsd:dateTime'
            to_para = f'"{year}-01-01"^^xsd:dateTime'
        
        # Handle reversed relation: subp[2] could be ['R', 'relation'] or just 'relation'
        is_reversed = isinstance(subp[2], list) and subp[2][0] == 'R'
        if is_reversed:
            relation_str = subp[2][1]
        else:
            relation_str = subp[2]
        
        # Get relation properties
        rel_from_property = relation_str.split('.')[-1]
        if rel_from_property == 'from':
            rel_to_property = 'to'
        elif rel_from_property == 'end_date':
            relation_str = relation_str.replace('end_date', 'start_date')
            rel_from_property = 'start_date'
            rel_to_property = 'end_date'
        else:
            rel_to_property = 'to_date'
        
        opposite_rel = relation_str.replace(rel_from_property, rel_to_property)
        
        # Add time constraint filters (use relation_str for SPARQL, not the list form)
        clauses.append(f'FILTER(NOT EXISTS {{?x{i_str} ns:{relation_str} ?sk0}} || ')
        clauses.append(f'EXISTS {{?x{i_str} ns:{relation_str} ?sk1 . ')
        clauses.append(f'FILTER(xsd:datetime(?sk1) <= {from_para}) }})')
        
        clauses.append(f'FILTER(NOT EXISTS {{?x{i_str} ns:{opposite_rel} ?sk2}} || ')
        clauses.append(f'EXISTS {{?x{i_str} ns:{opposite_rel} ?sk3 . ')
        clauses.append(f'FILTER(xsd:datetime(?sk3) >= {to_para}) }})')
    
    def _process_superlative(self, subp: List, i: int, i_str: str, clauses: List[str], order_clauses: List[str], identical_variables_r: Dict, get_root):
        """Process ARGMAX/ARGMIN operations"""
        if subp[1][0] == '#':
            # Merge current superlative variable root with the referenced expression root
            var = int(subp[1][1:])
            rooti = get_root(int(i))
            root_var = get_root(var)
            if rooti > root_var:
                identical_variables_r[rooti] = root_var
            else:
                identical_variables_r[root_var] = rooti
        else:
            clauses.append(f'?x{i_str} ns:type.object.type ns:{subp[1]} .')
        
        if len(subp) == 3:  # 1-hop relation
            clauses.append(f'?x{i_str} ns:{subp[2]} ?arg0 .')
        elif len(subp) > 3:  # multi-hop relations
            for j, relation in enumerate(subp[2:-1]):
                var0 = f'x{i_str}' if j == 0 else f'c{j - 1}'
                var1 = f'c{j}'
                
                if isinstance(relation, list) and relation[0] == 'R':
                    clauses.append(f'?{var1} ns:{relation[1]} ?{var0} .')
                else:
                    clauses.append(f'?{var0} ns:{relation} ?{var1} .')
            
            clauses.append(f'?c{j} ns:{subp[-1]} ?arg0 .')
        
        # Add ordering
        if subp[0] == 'ARGMIN':
            order_clauses.append("ORDER BY ?arg0")
        else:
            order_clauses.append("ORDER BY DESC(?arg0)")
        order_clauses.append("LIMIT 1")
    
    def _add_filters(self, clauses: List[str], entities: set):
        """Add entity and variable filters"""
        filter_variables = []
        for clause in clauses:
            variables = re.findall(r"\?\w*", clause)
            if variables:
                for var in variables:
                    var = var.strip()
                    if var not in filter_variables and var != '?x' and not var.startswith('?sk'):
                        filter_variables.append(var)
        
        # Add entity filters
        for entity in entities:
            clauses.append(f'FILTER (?x != ns:{entity})')
        
        # Add variable filters
        for var in filter_variables:
            clauses.append(f"FILTER (?x != {var})")
        
        # Add language filter
        clauses.insert(0, "FILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))")
    
    def _add_string_filters(self, clauses: List[str]):
        """Add string matching filters for partial matches (WebQSP/GraphQ specific)"""
        num = 0
        sentences = [s for s in clauses]
        
        for c, sentence in enumerate(sentences):
            if len(sentence.split(' ')) == 4 and sentence.split(' ')[-1] == '.':
                if sentence.split(' ')[-2].startswith('"') and sentence.split(' ')[-2].endswith('"'):
                    name = sentence.split(' ')[-2]
                    clauses[c] = clauses[c].replace(name, f'?st{num}')
                    clauses.append(f"FILTER (SUBSTR(STR(?st{num}), 1, STRLEN({name})) = {name})")
                    num += 1
                elif sentence.split(' ')[-2].endswith('"^^<http://www.w3.org/2001/XMLSchema#dateTime>'):
                    name = sentence.split(' ')[-2].replace("^^<http://www.w3.org/2001/XMLSchema#dateTime>", "")
                    clauses[c] = clauses[c].replace(sentence.split(' ')[-2], f'?st{num}')
                    clauses.append(f"FILTER (SUBSTR(STR(?st{num}), 1, STRLEN({name})) = {name})")
                    num += 1
    
    def _construct_sparql(self, clauses: List[str], order_clauses: List[str], 
                         count: bool, superlative: bool) -> str:
        """Construct final SPARQL query for WebQSP/GraphQ"""
        # Add WHERE clause
        clauses.insert(0, "WHERE {")
        
        # Add SELECT clause
        if count:
            clauses.insert(0, "SELECT COUNT DISTINCT ?x")
        else:
            clauses.insert(0, "SELECT DISTINCT ?x ?name")
        
        # Add PREFIX
        clauses.insert(0, self.freebase_prefix.strip())
        
        # Add name retrieval
        clauses.append("OPTIONAL { ?x rdfs:label ?name . FILTER (langMatches(lang(?name), 'en')) }")
        
        # Close WHERE clause
        clauses.append('}')
        
        # Add ordering clauses
        clauses.extend(order_clauses)
        
        # Add LIMIT clause to prevent excessive results (except for superlative queries which already have LIMIT 1)
        if not superlative:
            clauses.append("LIMIT 1000")
        
        return '\n'.join(clauses)
