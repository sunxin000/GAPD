"""
GrailQA-specific SPARQL Converter based on logic_form_util_grailqa.py
Simplified and optimized for GrailQA dataset characteristics
"""

import logging
import re
from typing import Dict, List

logger = logging.getLogger(__name__)

class SPARQLConverterGrailQA:
    """
    GrailQA-specific S-Expression to SPARQL converter
    Based on KBQA-o1's logic_form_util_grailqa.py
    Optimized for GrailQA's simplified logic and timezone requirements
    """
    
    def __init__(self):
        self.function_map = {'le': '<=', 'ge': '>=', 'lt': '<', 'gt': '>'}
        self.freebase_prefix = "PREFIX ns: <http://rdf.freebase.com/ns/>\nPREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
        
    def lisp_to_sparql(self, lisp_program: str) -> str:
        """
        Convert S-Expression to SPARQL query optimized for GrailQA
        Based on logic_form_util_grailqa.py
        """
        clauses = []
        order_clauses = []
        entities = set()
        identical_variables_r = {}
        
        expression = self.lisp_to_nested_expression(lisp_program)
        superlative = False
        
        # Check for SUPERLATIVE
        if expression[0] in ['ARGMAX', 'ARGMIN']:
            superlative = True
            if isinstance(expression[2], list):
                relations = self._retrieve_relations(expression[2])
                expression = expression[:2]
                expression.extend(relations)
        
        sub_programs = self._linearize_lisp_expression(expression, [0])
        question_var = len(sub_programs) - 1
        count = False
        
        def get_root(var: int):
            while var in identical_variables_r:
                var = identical_variables_r[var]
            return var
        
        # Process each sub-program (GrailQA simplified logic)
        for i, subp in enumerate(sub_programs):
            i_str = str(i)
            
            if subp[0] == 'JOIN':
                self._process_join_grailqa(subp, i_str, clauses, entities)
                
            elif subp[0] == 'AND':
                self._process_and_grailqa(subp, i, i_str, clauses, identical_variables_r, get_root)
                
            elif subp[0] in ['le', 'lt', 'ge', 'gt']:
                self._process_comparison_grailqa(subp, i, i_str, clauses, sub_programs)
                
            elif subp[0] == 'TC':
                self._process_time_constraint_grailqa(subp, i, i_str, clauses, identical_variables_r, get_root)
                
            elif subp[0] in ["ARGMIN", "ARGMAX"]:
                self._process_superlative_grailqa(subp, i, i_str, clauses, order_clauses, identical_variables_r, get_root, sub_programs)
                superlative = True
                
            elif subp[0] == 'COUNT':
                root_var = get_root(int(subp[1][1:]))
                identical_variables_r[int(i)] = root_var
                count = True
        
        # Merge identical variables
        for i in range(len(clauses)):
            for k in identical_variables_r:
                clauses[i] = clauses[i].replace(f'?x{k} ', f'?x{get_root(k)} ')
        
        question_var = get_root(question_var)
        
        # Replace question variable
        for i in range(len(clauses)):
            clauses[i] = clauses[i].replace(f'?x{question_var} ', '?x ')
        
        # Handle superlative queries like the original logic_form_util_grailqa.py
        if superlative:
            arg_clauses = clauses[:]  # Copy all clauses for inner query

        # Add entity filters
        for entity in entities:
            clauses.append(f'FILTER (?x != ns:{entity})')

        # Add standard filters
        clauses.insert(0, "FILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))")
        clauses.insert(0, "WHERE {")

        # Count: no label needed
        if count:
            clauses.insert(0, "SELECT COUNT DISTINCT ?x")
            clauses.insert(0, self.freebase_prefix)
            clauses.append('}')
            clauses.extend(order_clauses)
            return '\n'.join(clauses)

        # Superlative: original nested structure (LIMIT 1 keeps OPTIONAL fast)
        if superlative:
            clauses.insert(0, "{SELECT ?sk0")
            clauses = arg_clauses + clauses
            clauses.insert(0, "WHERE {")
            clauses.insert(0, "SELECT DISTINCT ?x ?name")
            clauses.insert(0, self.freebase_prefix)
            clauses.append('}')
            clauses.extend(order_clauses)
            clauses.append('}')
            clauses.append("OPTIONAL { ?x rdfs:label ?name . FILTER (langMatches(lang(?name), 'en')) }")
            clauses.append('}')
            return '\n'.join(clauses)

        # Normal: subquery approach to avoid slow OPTIONAL join plan
        clauses.insert(0, "SELECT DISTINCT ?x")
        clauses.append('}')
        clauses.extend(order_clauses)
        clauses.append("LIMIT 200")
        inner = '\n'.join(clauses)

        return (
            f"{self.freebase_prefix}"
            f"SELECT ?x ?name WHERE {{\n"
            f"{{ {inner} }}\n"
            f"OPTIONAL {{ ?x rdfs:label ?name . FILTER (langMatches(lang(?name), 'en')) }}\n"
            f"}}"
        )
            

    
    def lisp_to_nested_expression(self, lisp_string: str) -> List:
        """Parse S-expression string to nested list structure"""
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
        """Retrieve relations from nested expressions"""
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
        """Linearize nested expressions"""
        sub_formulas = []
        for i, e in enumerate(expression):
            if isinstance(e, list) and e[0] != 'R':
                sub_formulas.extend(self._linearize_lisp_expression(e, sub_formula_id))
                expression[i] = '#' + str(sub_formula_id[0] - 1)
        
        sub_formulas.append(expression)
        sub_formula_id[0] += 1
        return sub_formulas
    
    def _process_join_grailqa(self, subp: List, i: str, clauses: List[str], entities: set):
        """Process JOIN operations (GrailQA simplified version)"""
        # Handle JOIN with only 2 parameters (relation only, no object)
        if len(subp) == 2:
            # This is a relation-only JOIN, skip for now as it will be handled by ARG operations
            pass
        elif isinstance(subp[1], list):  # R relation
            if len(subp) > 2 and subp[2][:2] in ["m.", "g."]:  # entity
                clauses.append(f"ns:{subp[2]} ns:{subp[1][1]} ?x{i} .")
                entities.add(subp[2])
            elif len(subp) > 2 and subp[2][0] == '#':  # variable
                clauses.append(f"?x{subp[2][1:]} ns:{subp[1][1]} ?x{i} .")
            elif len(subp) > 2:  # literal
                if subp[2].__contains__('^^'):
                    subp[2] = self._format_literal_grailqa(subp[2])
                clauses.append(f"{subp[2]} ns:{subp[1][1]} ?x{i} .")
        else:
            if len(subp) > 2 and subp[2][:2] in ["m.", "g."]:  # entity
                clauses.append(f"?x{i} ns:{subp[1]} ns:{subp[2]} .")
                entities.add(subp[2])
            elif len(subp) > 2 and subp[2][0] == '#':  # variable
                clauses.append(f"?x{i} ns:{subp[1]} ?x{subp[2][1:]} .")
            elif len(subp) > 2 and re.match(r'[\w_]*\.[\w_]*\.[\w_]*', subp[2]):  # 2-hop relation
                # Skip 2-hop relations - they will be handled by comparison operations
                pass
            elif len(subp) > 2:  # literal
                if subp[2].__contains__('^^'):
                    subp[2] = self._format_literal_grailqa(subp[2])
                clauses.append(f"?x{i} ns:{subp[1]} {subp[2]} .")
    
    def _format_literal_grailqa(self, literal: str) -> str:
        """Format literal with XSD standard timezone handling"""
        if literal.__contains__('^^'):  # Check for any literal with datatype
            data_type_string = literal.split("^^")[1]
            if '#' in data_type_string:
                data_type = data_type_string.split("#")[1].rstrip('>')
            elif 'xsd:' in data_type_string:
                data_type = data_type_string.split('xsd:')[1]
            else:
                data_type = 'dateTime'

            # Align with KBQA-o1: add timezone only for non-numeric, non-dateTime types
            # i.e., date/time/gYear/gYearMonth → add -08:00; dateTime → keep as is
            if data_type in ['date', 'gYear', 'gYearMonth']:
                value = literal.split("^^")[0]
                # 若已包含时区（Z 或 ±HH:MM），则不重复追加
                has_tz = value.endswith('Z') or bool(re.search(r"[\+\-]\d{2}:\d{2}$", value))
                if has_tz:
                    return f'"{value}"^^<{literal.split("^^")[1]}>'
                return f'"{value + "-08:00"}"^^<{literal.split("^^")[1]}>'
            else:
                return f'"{literal.split("^^")[0]}"^^<{literal.split("^^")[1]}>'
        return literal
    
    def _process_and_grailqa(self, subp: List, i: int, i_str: str, clauses: List[str], 
                             identical_variables_r: Dict, get_root):
         """Process AND operations (GrailQA version)"""
         # Normalize argument order: ensure variable is in position 2 (subp[2])
         # Accept both (AND class #k) and (AND #k class)
         if isinstance(subp[2], str) and subp[2].startswith('#'):
             pass
         elif isinstance(subp[1], str) and subp[1].startswith('#'):
             subp[1], subp[2] = subp[2], subp[1]
         else:
             # Neither argument is a variable; treat as invalid pattern and skip gracefully
             # This prevents ValueError: invalid literal for int() when given a relation/class
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
    
    def _process_comparison_grailqa(self, subp: List, i: int, i_str: str, clauses: List[str], sub_programs: List):
        """Process comparison with support for 2-hop JOIN via #k reference"""
        if isinstance(subp[1], str) and subp[1].startswith('#'):
            line_num = int(subp[1].replace('#', ''))
            referenced_op = sub_programs[line_num]
            if referenced_op and referenced_op[0] == 'JOIN' and len(referenced_op) >= 3:
                first_relation = referenced_op[1]
                second_relation = referenced_op[2]
                # First hop
                if isinstance(first_relation, list):  # reversed
                    clauses.append(f"?cvt ns:{first_relation[1]} ?x{i_str} .")
                else:
                    clauses.append(f"?x{i_str} ns:{first_relation} ?cvt .")
                # Second hop
                if isinstance(second_relation, list):  # reversed
                    clauses.append(f"?y{i_str} ns:{second_relation[1]} ?cvt .")
                else:
                    clauses.append(f"?cvt ns:{second_relation} ?y{i_str} .")
            else:
                # Fallback to 1-hop if not a JOIN ref
                clauses.append(f"?x{i_str} ns:{subp[1]} ?y{i_str} .")
        else:
            # 1-hop comparison
            clauses.append(f"?x{i_str} ns:{subp[1]} ?y{i_str} .")
        
        if subp[0] == 'le':
            op = "<="
        elif subp[0] == 'lt':
            op = "<"
        elif subp[0] == 'ge':
            op = ">="
        else:
            op = ">"
        
        if subp[2].__contains__('^^'):
            subp[2] = self._format_literal_grailqa(subp[2])
        
        # Enhanced numeric handling: if the value looks like a number, cast the variable to xsd:double
        # This mimics WebQSP's behavior (which uses xsd:integer) but is more precise for floats
        # and handles cases where the DB value is stored as string but represents a number.
        # Check if value is a number (integer or float), ignoring quotes and datatype
        raw_val = subp[2].split('^^')[0].strip('"')
        # Use regex to check for number format (integer or float), but exclude dates (which contain - or :)
        if re.match(r'^-?\d+(\.\d+)?$', raw_val):
            clauses.append(f"FILTER (xsd:double(?y{i_str}) {op} {subp[2]})")
        else:
            clauses.append(f"FILTER (?y{i_str} {op} {subp[2]})")
    
    def _process_time_constraint_grailqa(self, subp: List, i: int, i_str: str, clauses: List[str],
                                        identical_variables_r: Dict, get_root):
        """Process time constraints (GrailQA simplified version)"""
        var = int(subp[1][1:])
        rooti = get_root(int(i))
        root_var = get_root(var)
        
        if rooti > root_var:
            identical_variables_r[rooti] = root_var
        else:
            identical_variables_r[root_var] = rooti
        
        year = subp[3]
        # GrailQA adds timezone processing for datetime values
        if year == 'NOW':
            from_para = '"2015-08-10-08:00"^^xsd:dateTime'  # Add timezone for GrailQA
            to_para = '"2015-08-10-08:00"^^xsd:dateTime'
        else:
            # Extract year if it has datatype annotation
            if "^^" in year:
                year = year.split("^^")[0].strip('"')
            else:
                year = year.strip('"')
            from_para = f'"{year}-12-31-08:00"^^xsd:dateTime'  # Add timezone for GrailQA
            to_para = f'"{year}-01-01-08:00"^^xsd:dateTime'
        
        # GrailQA's simplified relation handling
        clauses.append(f'FILTER(NOT EXISTS {{?x{i_str} ns:{subp[2]} ?sk0}} || ')
        clauses.append(f'EXISTS {{?x{i_str} ns:{subp[2]} ?sk1 . ')
        clauses.append(f'FILTER(xsd:datetime(?sk1) <= {from_para}) }})')
        
        if subp[2][-4:] == "from":
            opposite_rel = subp[2][:-4] + "to"
        else:  # from_date -> to_date
            opposite_rel = subp[2][:-9] + "to_date"
        
        clauses.append(f'FILTER(NOT EXISTS {{?x{i_str} ns:{opposite_rel} ?sk2}} || ')
        clauses.append(f'EXISTS {{?x{i_str} ns:{opposite_rel} ?sk3 . ')
        clauses.append(f'FILTER(xsd:datetime(?sk3) >= {to_para}) }})')
    
    def _process_superlative_grailqa(self, subp: List, i: int, i_str: str, clauses: List[str], order_clauses: List[str], identical_variables_r: Dict, get_root, sub_programs: List):
        """Process superlative operations (GrailQA version with ?sk0) - exactly like original logic_form_util_grailqa.py"""
        if subp[1][0] == '#':
            # Merge variable roots exactly like the original
            var = int(subp[1][1:])
            rooti = get_root(int(i))
            root_var = get_root(var)
            if rooti > root_var:
                identical_variables_r[rooti] = root_var
            else:
                identical_variables_r[root_var] = rooti
        else:  # arg1 is class
            clauses.append(f'?x{i_str} ns:type.object.type ns:{subp[1]} .')
        
        if len(subp) == 3:
            clauses.append(f'?x{i_str} ns:{subp[2]} ?sk0 .')
        elif len(subp) > 3:
            for j, relation in enumerate(subp[2:-1]):
                if j == 0:
                    var0 = f'x{i_str}'
                else:
                    var0 = f'c{j - 1}'
                var1 = f'c{j}'
                if isinstance(relation, list) and relation[0] == 'R':
                    clauses.append(f'?{var1} ns:{relation[1]} ?{var0} .')
                else:
                    clauses.append(f'?{var0} ns:{relation} ?{var1} .')
            
            clauses.append(f'?c{j} ns:{subp[-1]} ?sk0 .')
        
        # Add ordering exactly like the original
        if subp[0] == 'ARGMIN':
            order_clauses.append("ORDER BY ?sk0")
        elif subp[0] == 'ARGMAX':
            order_clauses.append("ORDER BY DESC(?sk0)")
        order_clauses.append("LIMIT 1")
