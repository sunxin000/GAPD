"""
Utility functions for S-Expression generation
Handles various helper functions and utilities
"""

import re
import torch
from typing import List, Tuple, Dict


class SExprUtils:
    """
    Utility functions for S-Expression generation
    """
    
    @staticmethod
    def ensure_leading_newline_info(text: str) -> str:
        """Ensure observations starting with <information> break line after </action>."""
        if isinstance(text, str) and text.startswith("<information>"):
            return "\n" + text
        return text
    
    @staticmethod
    def batch_tokenize(tokenizer, responses: List[str]) -> torch.Tensor:
        """Tokenize batch of responses (same as original)"""
        return tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']
    
    @staticmethod
    def postprocess_responses(tokenizer, responses: torch.Tensor, no_think_rl: bool = False) -> Tuple[torch.Tensor, List[str]]:
        """Process responses to stop at search operation or answer operation."""
        responses_str = tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        responses_str = [resp.split('</action>')[0] + '</action>'
                 if '</action>' in resp 
                 else resp.split('</answer>')[0] + '</answer>'
                 if '</answer>' in resp 
                 else resp
                 for resp in responses_str]

        if no_think_rl:
            raise ValueError('stop')
            # if no_think_rl is enabled, only keep action in the str
            # actions, _ = self.env.postprocess_predictions(responses_str)
            # responses_str=[f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in enumerate(actions)]
            # print("RESPONSES:", responses_str)
        responses = SExprUtils.batch_tokenize(tokenizer, responses_str)
        return responses, responses_str
    
    @staticmethod
    def detect_math_expression(expression: str) -> bool:
        """
        Detect if an expression is a mathematical calculation expression
        Returns True if it's a calculation expression that should be rejected
        """
        # Skip detection for literal types (contain ^^)
        if '^^' in expression:
            return False
            
        # Check for mathematical operations, but be more specific about ^
        # ^ should only be considered as math operator if it's not part of ^^ (literal type)
        has_math_ops = False
        
        # Check for ^ operator (but not ^^)
        if '^' in expression and '^^' not in expression:
            has_math_ops = True
        # Check for other math operators
        elif any(op in expression for op in ['**', '*']):
            has_math_ops = True
            
        return has_math_ops
    
    @staticmethod
    def is_empty_result(result_string: str) -> bool:
        """
        Check if the result string indicates empty results
        """
        # Check for common empty result patterns
        empty_patterns = [
            "No results found",
            "Found 0 results",
            "Execution failed",
            "No valid actions found",
            "Failed to process actions",
            "Failed to build function sequence",
            "S-Expression generation failed",
            "S-Expression validation failed"
        ]
        
        for pattern in empty_patterns:
            if pattern in result_string:
                # Special case: if the message is very long (>200 chars), it might contain useful debugging info
                # even if it contains "No results found"
                if pattern == "No results found" and len(result_string) > 200:
                    # Check if there are MIDs in the long message
                    mid_pattern = re.compile(r'[mg]\.[A-Za-z0-9_]+')
                    mids = mid_pattern.findall(result_string)
                    if mids:
                        return False  # Long message with MIDs is not empty
                    # For long messages without MIDs, check if they contain useful debugging info
                    useful_patterns = [
                        "debugging", "information", "useful", "details", "context", "background"
                    ]
                    for useful_pattern in useful_patterns:
                        if useful_pattern.lower() in result_string.lower():
                            return False  # Long message with useful debugging info is not empty
                return True
        
        # Check if result contains actual MID results
        mid_pattern = re.compile(r'[mg]\.[A-Za-z0-9_]+')
        mids = mid_pattern.findall(result_string)
        
        # If MIDs found, definitely not empty
        if mids:
            return False
        
        # If no MIDs found and result is short, consider it empty
        # But only if it doesn't contain useful error information
        if len(result_string) < 200:
            # Check if it contains any useful error patterns that suggest it's not empty
            useful_patterns = [
                "error",
                "failed",
                "invalid",
                "syntax",
                "connection",
                "timeout"
            ]
            
            # If it contains useful error information, don't consider it empty
            for pattern in useful_patterns:
                if pattern.lower() in result_string.lower():
                    return False
            
            return True
        
        return False
    
    @staticmethod
    def initialize_candidate_entities_from_prompts(tokenizer, state_manager, initial_input_ids: torch.Tensor, batch_size: int):
        """Decode prompts to extract Candidate Entities and store per-sample in state manager.

        Expected prompt fragment example:
          Candidate Entities: ['Barack Obama' (m.02mjmr), 'Chicago' (m.01xxx)]
        We parse entries of the form 'name' (id).
        """
        for i in range(batch_size):
            text = tokenizer.decode(initial_input_ids[i], skip_special_tokens=False)
            # Persist the full decoded prompt for error context
            try:
                state_manager.set_sample_prompt(i, text)
            except Exception:
                pass

            if not text:
                continue
            # Prefer "Candidate Entities:", fallback to "Entities:" to be robust to earlier prompts
            m = re.search(r"(Candidate Entities|Entities):\s*\[(.*?)\]", text, re.DOTALL)
            if not m:
                continue
            inner = m.group(2)
            pairs = re.findall(r"'([^']+)'\s*\(([^)]+)\)", inner)
            if not pairs:
                continue
            entities: List[Tuple[str, str]] = []
            for name, ent_id in pairs:
                entities.append((name, ent_id))
            if entities:
                state_manager.set_sample_entities(i, entities)
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"[SEXPR] Sample {i}: Initialized {len(entities)} candidate entities from prompt")
