"""
Multi-result formatter for S-Expression generation
Handles formatting of multiple FIND_RELATION results with optimization
"""

import re
from typing import List


class SExprMultiResultFormatter:
    """
    Handles formatting of multiple FIND_RELATION results with optimization
    to reduce redundancy by sharing function_state while keeping individual result_mid_list
    """
    
    def __init__(self, result_processor):
        self.result_processor = result_processor
    
    def format_multi_result(self, results: List[str], is_final_turn: bool = False) -> str:
        """Format multiple result strings into a single information block with one guidance prompt.
        Optimized to reduce redundancy by sharing function_state only, while keeping individual result_mid_list.
        """
        if not results:
            return "<information>\nNo results.\n</information>"
        
        # Parse each result to extract components
        parsed_results = []
        shared_function_state = None
        
        for r in results:
            s = r if isinstance(r, str) else str(r)
            s = s.strip()
            start = s.find("<information>")
            end = s.find("</information>")
            if start != -1 and end != -1 and end > start:
                inner = s[start + len("<information>"):end].strip()
                parsed = self._parse_information_block(inner)
                parsed_results.append(parsed)
                
                # Extract shared function_state only (result_mid_list should remain individual)
                if parsed.get('function_state') and not shared_function_state:
                    shared_function_state = parsed['function_state']
            else:
                # Fallback: keep as-is (no tags found)
                parsed_results.append({'raw_content': s})
        
        # Build optimized information block
        info_lines = []
        
        # Add shared function_state once
        if shared_function_state:
            info_lines.append("functions:")
            for func in shared_function_state:
                info_lines.append(func)
        
        # Add individual FIND_RELATION results (content without functions)
        for i, parsed in enumerate(parsed_results):
            if parsed.get('content_without_functions'):
                # Extract expression ID that corresponds to this specific FIND_RELATION operation
                expression_id = self._extract_expression_id_for_find_relation(shared_function_state or [], i)
                info_lines.append(f"\n--- {expression_id} ---")
                info_lines.append(parsed['content_without_functions'])
            elif parsed.get('raw_content'):
                info_lines.append(f"\n--- FIND_RELATION {i+1} ---")
                info_lines.append(parsed['raw_content'])
        
        body = "\n".join(info_lines)
        aggregated = f"<information>\nMultiple FIND_RELATION results:\n\n{body}\n</information>"
        # Append a single guidance prompt using the same helper to keep behavior consistent
        return self.result_processor._add_guidance_prompt(aggregated, is_final_turn)
    
    def _parse_information_block(self, content: str) -> dict:
        """Parse an information block to extract functions and keep other content as-is."""
        parsed = {}
        lines = content.split('\n')
        
        current_section = None
        function_state = []
        content_without_functions = []
        
        for line in lines:
            original_line = line
            line = line.strip()
            
            if line == 'functions:':
                current_section = 'functions'
                continue
            elif current_section == 'functions' and line.startswith('expression'):
                function_state.append(line)
                continue
            elif current_section == 'functions' and line and not line.startswith('expression'):
                current_section = None
                # Include this line in content_without_functions
                content_without_functions.append(original_line)
            elif current_section != 'functions':
                content_without_functions.append(original_line)
        
        if function_state:
            parsed['function_state'] = function_state
        parsed['content_without_functions'] = '\n'.join(content_without_functions)
            
        return parsed
    
    def _extract_expression_id_for_find_relation(self, function_state: List[str], find_relation_index: int) -> str:
        """Extract the expression ID that corresponds to a specific FIND_RELATION operation."""
        if not function_state:
            return "expression"
        
        # Find all START functions to identify the expression sequence
        start_functions = []
        for func in function_state:
            if 'START(' in func:
                match = re.search(r'expression(\d+)\s*=\s*START\(', func)
                if match:
                    start_functions.append(int(match.group(1)))
        
        if not start_functions:
            return "expression"
        
        # Sort to get the order of expressions
        start_functions.sort()
        
        # Check if we have chain operations (more FIND_RELATION operations than START functions)
        if find_relation_index < len(start_functions):
            # Independent operations: each FIND_RELATION corresponds to a different expression
            return f"expression{start_functions[find_relation_index]}"
        else:
            # Chain operations: multiple FIND_RELATION operations on the same expression
            # Use the last expression with a suffix to indicate the operation order
            base_expression = start_functions[-1]
            operation_suffix = find_relation_index - len(start_functions) + 1
            return f"expression{base_expression}_op{operation_suffix}"
