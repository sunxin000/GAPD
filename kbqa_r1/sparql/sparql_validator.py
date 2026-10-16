"""
SPARQL Query Validator to detect and reject potentially harmful queries.
"""
import re

class SPARQLValidator:
    """
    Validates SPARQL queries to detect patterns that can cause issues,
    such as fully unrestricted triple patterns.
    """

    def __init__(self):
        # Pattern for a variable in SPARQL (?var or $var)
        var_pattern = r'[\?\$]\w+'
        
        # This regex looks for a triple pattern composed of three variables.
        # CRITICAL: Updated to catch patterns with OR without spaces between variables
        # Examples: 
        # - ?s ?p ?o .        (with spaces)
        # - ?player?rel?team  (without spaces - BYPASS ATTEMPT)
        # - ?entity?property?value }
        # - ?x?y?z ;
        # The pattern matches three variables (with optional spaces) followed by optional whitespace
        # and then either: dot, closing brace, semicolon, whitespace, or end of string
        self.unrestricted_triple_pattern = re.compile(
            rf'({var_pattern}\s*{var_pattern}\s*{var_pattern})\s*(?:\.|}}|;|\s|$)',
            re.IGNORECASE
        )
        
        # Pattern for dangerous rdf:type sugar syntax: ?var a ?var
        # CRITICAL: Updated to catch patterns with OR without spaces
        # The 'a' is SPARQL syntax sugar for rdf:type
        # Examples that should be caught:
        # - ?x a ?z .         (with spaces)
        # - ?xa?z             (without spaces - BYPASS ATTEMPT)
        # - ?entity a ?type }
        # This pattern is dangerous because it matches almost all entities in the knowledge base
        self.rdf_type_unrestricted_pattern = re.compile(
            rf'({var_pattern}\s*a\s*{var_pattern})\s*(?:\.|}}|;|\s|$)',
            re.IGNORECASE
        )
        
        # Pattern for dangerous full rdf:type syntax: ?var rdf:type ?var
        # CRITICAL: Updated to catch patterns with OR without spaces
        # This is the complete form equivalent to the 'a' syntax sugar
        # Examples that should be caught:
        # - ?x rdf:type ?z .     (with spaces)
        # - ?xrdf:type?z         (without spaces - BYPASS ATTEMPT)
        # - ?entity rdf:type ?type }
        # This pattern is equally dangerous as the sugar syntax and matches almost all entities
        self.rdf_type_full_pattern = re.compile(
            rf'({var_pattern}\s*rdf:type\s*{var_pattern})\s*(?:\.|}}|;|\s|$)',
            re.IGNORECASE
        )
        
        # List of dangerous common properties that can cause performance issues
        # when used in unrestricted patterns (?var property ?var)
        dangerous_properties = [
            "rdfs:label",      # Resource labels - extremely common
            "rdfs:comment",    # Resource comments - very common  
            "foaf:name",       # FOAF names - common in many datasets
            "dc:title",        # Dublin Core titles - common
            "skos:prefLabel",  # SKOS preferred labels - common in vocabularies
            "owl:sameAs",      # Equivalence relations - can be numerous
            # "ns:type.object.name", 
        ]
        
        # Create a pattern that matches any of the dangerous properties
        # CRITICAL: Updated to catch patterns with OR without spaces to prevent bypass
        # Examples that should be caught:
        # - ?x rdfs:label ?name .     (with spaces)
        # - ?xrdfs:label?name         (without spaces - BYPASS ATTEMPT)
        # - ?entity rdfs:comment ?desc }
        # - ?item foaf:name ?n ;
        dangerous_props_pattern = '|'.join(re.escape(prop) for prop in dangerous_properties)
        self.common_property_unrestricted_pattern = re.compile(
            rf'({var_pattern}\s*(?:{dangerous_props_pattern})\s*{var_pattern})\s*(?:\.|}}|;|\s|$)',
            re.IGNORECASE
        )
        
        # Regex to find the content of the WHERE clause
        self.where_clause_pattern = re.compile(r'WHERE\s*\{([\s\S]*)\}', re.IGNORECASE)
        
        # Pattern for invalid syntax (attempts to bypass validation)


    def is_invalid(self, query: str) -> bool:
        """
        Checks if a SPARQL query contains dangerous unrestricted patterns.
        Returns True if the query is invalid, False otherwise.
        
        Detects four main types of dangerous patterns:
        1. Three-variable patterns: ?s ?p ?o (completely unrestricted triple)
        2. rdf:type sugar patterns: ?x a ?z (finds all typed entities - very broad)
        3. rdf:type full patterns: ?x rdf:type ?z (equivalent to sugar syntax)
        4. Common property patterns: ?x rdfs:label ?y (finds all entities with common properties)
        """
        # First check the entire query for dangerous patterns (including SELECT clause)
        if self.unrestricted_triple_pattern.search(query):
            return True
        
        # Then check WHERE clause for other patterns
        where_match = self.where_clause_pattern.search(query)
        if not where_match:
            # If there's no WHERE clause, we consider it valid for this check's purpose.
            return False
        
        where_content = where_match.group(1)
        
        # Block queries that have no Freebase triple patterns at all (degenerate: only FILTER/OPTIONAL)
        # Heuristic: if WHERE clause contains no 'ns:' occurrences, there are no core triples.
        if 'ns:' not in where_content:
            return True


        
        # Check for dangerous rdf:type patterns (?var a ?var)
        if self.rdf_type_unrestricted_pattern.search(where_content):
            return True
        
        # Check for dangerous full rdf:type patterns (?var rdf:type ?var)
        if self.rdf_type_full_pattern.search(where_content):
            return True
        
        # Check for dangerous common property patterns (?var property ?var)
        prop_matches = self.common_property_unrestricted_pattern.findall(where_content)
        if prop_matches:
            # 只有完全无限制且没有其他约束的查询才算危险
            has_constraints = (
                'rdf:type' in where_content.lower() or
                ' a ' in where_content or
                'filter' in where_content.lower() or
                len(where_content.split('.')) > 2
            )
            
            if not has_constraints:
                return True
        
        return False
    
    def get_violation_message(self, query: str) -> str:
        """
        Returns specific error message based on which pattern was violated.
        """
        # First check the entire query for dangerous patterns (including SELECT clause)
        if self.unrestricted_triple_pattern.search(query):
            return "Blocked: Unrestricted triple pattern detected (e.g., ?s?p?o or ?var1?var2?var3)"
        
        # Then check WHERE clause for other patterns
        where_match = self.where_clause_pattern.search(query)
        if not where_match:
            return "Valid query"
        
        where_content = where_match.group(1)
    
        # Message for degenerate queries without core triples
        if 'ns:' not in where_content:
            return "Blocked: No Freebase triple patterns in WHERE (degenerate query)"

        
        # Check for dangerous rdf:type patterns (?var a ?var)
        if self.rdf_type_unrestricted_pattern.search(where_content):
            return "Blocked: Unrestricted rdf:type pattern detected (?x a ?y)"
        
        # Check for dangerous full rdf:type patterns (?var rdf:type ?var)
        if self.rdf_type_full_pattern.search(where_content):
            return "Blocked: Unrestricted rdf:type pattern detected (?x rdf:type ?y)"
        
        # Check for dangerous common property patterns (?var property ?var)
        prop_matches = self.common_property_unrestricted_pattern.findall(where_content)
        if prop_matches:
            # 只有完全无限制且没有其他约束的查询才算危险（与is_invalid保持一致）
            has_constraints = (
                'rdf:type' in where_content.lower() or
                ' a ' in where_content or
                'filter' in where_content.lower() or
                len(where_content.split('.')) > 2
            )
            
            if not has_constraints:
                # Check which specific dangerous property was used
                dangerous_properties = [
                    "rdfs:label",      # Resource labels - extremely common
                    "rdfs:comment",    # Resource comments - very common  
                    "foaf:name",       # FOAF names - common in many datasets
                    "dc:title",        # Dublin Core titles - common
                    "skos:prefLabel",  # SKOS preferred labels - common in vocabularies
                    "owl:sameAs",      # Equivalence relations - can be numerous
                ]
                
                property_errors = {
                    "rdfs:label": "Blocked: Unrestricted rdfs:label pattern (?x rdfs:label ?y) - matches all labeled resources",
                    "rdfs:comment": "Blocked: Unrestricted rdfs:comment pattern (?x rdfs:comment ?y) - matches all commented resources", 
                    "foaf:name": "Blocked: Unrestricted foaf:name pattern (?x foaf:name ?y) - matches all named entities",
                    "dc:title": "Blocked: Unrestricted dc:title pattern (?x dc:title ?y) - matches all titled resources",
                    "skos:prefLabel": "Blocked: Unrestricted skos:prefLabel pattern (?x skos:prefLabel ?y) - matches all vocabulary labels",
                    "owl:sameAs": "Blocked: Unrestricted owl:sameAs pattern (?x owl:sameAs ?y) - matches all equivalent entities"
                }
                
                # Find which specific property was detected
                for prop in dangerous_properties:
                    if prop in where_content:
                        return property_errors.get(prop, f"Blocked: Unrestricted {prop} pattern detected")
                
                # Fallback if no specific property found
                return "Blocked: Unrestricted property pattern detected (e.g., ?x rdfs:label ?y)"
        
        return "Valid query" 