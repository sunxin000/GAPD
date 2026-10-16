"""
S-Expression Action Processing Module
Handles different action types and their processing logic
"""

import logging
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

from ..sexpr.action_parser import ActionResult, ActionType
from ..sexpr.constants import COMPARISON_MODE_MAPPING
from ..sexpr.limit import name_relation_list, tc_time_list

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class SExprActionProcessor:
    """
    Processes different action types in S-Expression generation
    """
    _GRAILQA_GRAPHQ_BLOCKED_RELATIONS = {
        "common.topic.notable_types",
        "kg.object_profile.prominent_type",
        "freebase.unit_profile.dimension",
        "type.object.name",
        "common.topic.notable_for",
    }
    
    def __init__(self, relation_retrieval, state_manager):
        self.relation_retrieval = relation_retrieval
        self.state_manager = state_manager

        dataset_attr = getattr(self.relation_retrieval, 'dataset', None)
        if hasattr(dataset_attr, 'value'):
            dataset_attr = dataset_attr.value
        if dataset_attr is None:
            env_dataset = os.getenv("DATASET_TYPE")
            if env_dataset:
                dataset_attr = env_dataset
            else:
                raise AttributeError(
                    "relation_retrieval.dataset is required (or DATASET_TYPE env must be set) for relation filtering"
                )
        dataset_str = str(dataset_attr).strip().lower()
        self._dataset_type = dataset_str
        self._filter_relations_for_context = dataset_str in {"grailqa", "graphq"}
        
        # Stats for threshold-not-met cases
        self._relation_threshold_not_met_counts = Counter()
        
        # Per-step special action metrics (attempts and successes)
        self._special_action_names = {
            ActionType.MERGE: 'Merge',
            ActionType.ORDER: 'Order',
            ActionType.COMPARE: 'Compare',
            ActionType.TIME_CONSTRAINT: 'Time_constraint',
            ActionType.COUNT: 'Count',
        }
        self._action_metrics = defaultdict(lambda: {'attempts': 0, 'successes': 0})
        
        # Candidate length tracking
        self._cand_all_sum = 0
        self._cand_all_count = 0
        self._cand_notmet_sum = 0
        self._cand_notmet_count = 0

        # Per-step TOP-1 similarity scores for select_best_relations
        # Structure: {step_number: [top1_scores...]}
        # Only records the selected (best) relation's score, not all candidates
        self._per_step_relation_similarity = defaultdict(list)

        # ACTION-BASED STATE MANAGEMENT: Use sample+action_index indexed dictionaries
        # This prevents race conditions in parallel execution AND conflicts within the same turn
        # Structure: {sample_id: {action_index: state}}
        self._sample_action_selected_relations: Dict[int, Dict[int, Any]] = {}  # {sample_id: {action_index: selected_relation}}
        self._sample_action_candidate_relations: Dict[int, Dict[int, List]] = {}  # {sample_id: {action_index: candidate_relations}}
        self._sample_action_ranked_top5: Dict[int, Dict[int, List]] = {}  # {sample_id: {action_index: ranked_top5}}
        self._sample_action_states: Dict[int, Dict[int, Dict]] = {}  # {sample_id: {action_index: state}}
        
        # Sample-level state (non-legacy)
        self._sample_selected_relations: Dict[int, Any] = {}
        self._sample_candidate_relations: Dict[int, List] = {}
        self._sample_ranked_top5: Dict[int, List] = {}
        self._sample_per_action_states: Dict[int, Dict] = {}

        # Optional debug breakpoint configuration
        try:
            self._debug_sample_idx = int(os.getenv("DEBUG_SAMPLE_INDEX", "-1"))
        except ValueError:
            self._debug_sample_idx = -1
        self._debug_action_type = (os.getenv("DEBUG_BREAK_ACTION") or "").strip().lower()
        
        # Cache for efficient literal relation remapping: token -> full literal expression containing it
        self._literal_contains_map = None

    def _filter_prompt_relations(self, relation_names: List[str]) -> List[str]:
        if not relation_names:
            return relation_names or []
        if not self._filter_relations_for_context:
            return relation_names
        blocked = self._GRAILQA_GRAPHQ_BLOCKED_RELATIONS
        return [name for name in relation_names if name not in blocked]

    def _maybe_debug_break(self, sample_id, action_name: str):
        try:
            target_idx = int(self._debug_sample_idx)
        except Exception:
            target_idx = -1
        if target_idx < 0 or sample_id is None or sample_id != target_idx:
            return
        action_key = (action_name or "").strip().lower()
        if self._debug_action_type and self._debug_action_type != action_key:
            return
        print(f"[DEBUG] Breaking before {action_name} for sample {sample_id}")
        breakpoint()
    
    def _truncate_function_state_to_expression(self, function_state: List[str], expression_ref: str) -> List[str]:
        """
        Truncate function_state up to and including the assignment of the referenced expression.
        This ensures dynamic relation retrieval is executed in the correct historical context
        when the user asks Find_relation on an earlier expression (not the latest one).

        Example: expression_ref='expression2' -> keep up to the line starting with 'expression2 = '.
        If not found, return the original function_state.
        """
        try:
            if not function_state or not isinstance(expression_ref, str):
                return function_state
            prefix = f"{expression_ref.strip()} = "
            # Use the LAST assignment to the referenced expression to preserve
            # the most recent context (e.g., after JOIN/AND/TC on the same id)
            cut_idx = -1
            for idx, stmt in enumerate(function_state):
                if isinstance(stmt, str) and stmt.strip().startswith(prefix):
                    cut_idx = idx
                    # do not break; keep scanning to capture the last occurrence
            if cut_idx >= 0:
                return function_state[:cut_idx + 1]
            return function_state
        except Exception:
            return function_state

    def _get_action_relation_state(self, sample_id: int, action_index: int):
        """Get relation state for a specific sample and action (thread-safe and action-isolated)"""
        return (
            self._sample_action_selected_relations.get(sample_id, {}).get(action_index),
            self._sample_action_candidate_relations.get(sample_id, {}).get(action_index, []),
            self._sample_action_ranked_top5.get(sample_id, {}).get(action_index, []),
            self._sample_action_states.get(sample_id, {}).get(action_index, {})
        )
    
    def _set_action_relation_state(self, sample_id: int, action_index: int, selected_relation=None, candidate_relations=None, ranked_top5=None, state_dict=None):
        """Set relation state for a specific sample and action (thread-safe and action-isolated)"""
        # Initialize nested dictionaries if needed
        if sample_id not in self._sample_action_selected_relations:
            self._sample_action_selected_relations[sample_id] = {}
        if sample_id not in self._sample_action_candidate_relations:
            self._sample_action_candidate_relations[sample_id] = {}
        if sample_id not in self._sample_action_ranked_top5:
            self._sample_action_ranked_top5[sample_id] = {}
        if sample_id not in self._sample_action_states:
            self._sample_action_states[sample_id] = {}
        
        # Normalize selected_relation to a minimal dict if it's a plain string to keep downstream consumers uniform
        if selected_relation is not None:
            if isinstance(selected_relation, str):
                try:
                    # Wrap string relation id into a dict with relation_name for compatibility
                    selected_relation = {'relation_name': selected_relation}
                except Exception:
                    # fallback: leave as-is
                    pass
            self._sample_action_selected_relations[sample_id][action_index] = selected_relation
        if candidate_relations is not None:
            self._sample_action_candidate_relations[sample_id][action_index] = candidate_relations
        if ranked_top5 is not None:
            self._sample_action_ranked_top5[sample_id][action_index] = ranked_top5
        if state_dict is not None:
            # If provided state_dict contains a plain-string selected_relation, normalize it as well
            try:
                if isinstance(state_dict, dict) and 'selected_relation' in state_dict and isinstance(state_dict.get('selected_relation'), str):
                    state_dict = dict(state_dict)  # shallow copy
                    state_dict['selected_relation'] = {'relation_name': state_dict['selected_relation']}
            except Exception:
                pass
            self._sample_action_states[sample_id][action_index] = state_dict
    
    def _get_sample_relation_state(self, sample_id: int):
        """Get relation state for a specific sample (thread-safe) - LEGACY"""
        return (
            self._sample_selected_relations.get(sample_id),
            self._sample_candidate_relations.get(sample_id, []),
            self._sample_ranked_top5.get(sample_id, []),
            self._sample_per_action_states.get(sample_id, {})
        )
    
    def _set_sample_relation_state(self, sample_id: int, selected_relation=None, candidate_relations=None, ranked_top5=None, per_action_states=None):
        """Set relation state for a specific sample (thread-safe)"""
        if selected_relation is not None:
            self._sample_selected_relations[sample_id] = selected_relation
        if candidate_relations is not None:
            self._sample_candidate_relations[sample_id] = candidate_relations
        if ranked_top5 is not None:
            self._sample_ranked_top5[sample_id] = ranked_top5
        if per_action_states is not None:
            self._sample_per_action_states[sample_id] = per_action_states
    
    def _update_sample_per_action_state(self, sample_id: int, action_index: int, state_dict: Dict):
        """Update per-action state for a specific sample"""
        if sample_id not in self._sample_per_action_states:
            self._sample_per_action_states[sample_id] = {}
        self._sample_per_action_states[sample_id][action_index] = state_dict
    
    def _get_sample_per_action_state(self, sample_id: int, action_index: int):
        """Get per-action state for a specific sample"""
        return self._sample_per_action_states.get(sample_id, {}).get(action_index)
    
    def clear_sample_action_states(self, sample_id: int):
        """Clear all action states for a specific sample (called at the start of each turn)"""
        if sample_id in self._sample_action_selected_relations:
            del self._sample_action_selected_relations[sample_id]
        if sample_id in self._sample_action_candidate_relations:
            del self._sample_action_candidate_relations[sample_id]
        if sample_id in self._sample_action_ranked_top5:
            del self._sample_action_ranked_top5[sample_id]
        if sample_id in self._sample_action_states:
            del self._sample_action_states[sample_id]
        logger.debug(f"[SEXPR] Cleared action states for sample {sample_id} (new turn)")
    
    def initialize_batch_action_states(self, batch_size: int):
        """Initialize action states for a batch of samples (called at the start of each turn)"""
        for i in range(batch_size):
            self.clear_sample_action_states(i)
        logger.debug(f"[SEXPR] Initialized action states for batch of {batch_size} samples (new turn)")
    
    def reset_candidate_stats(self):
        """Reset candidate statistics for new batch"""
        self._cand_all_sum = 0
        self._cand_all_count = 0
        self._cand_notmet_sum = 0
        self._cand_notmet_count = 0
        # Also reset per-step relation similarity for new batch
        self._per_step_relation_similarity.clear()
    
    def get_candidate_stats(self):
        """Get candidate statistics"""
        return {
            'cand_all_sum': self._cand_all_sum,
            'cand_all_count': self._cand_all_count,
            'cand_notmet_sum': self._cand_notmet_sum,
            'cand_notmet_count': self._cand_notmet_count,
            'relation_threshold_not_met_counts': dict(self._relation_threshold_not_met_counts),
            'per_step_relation_similarity': {int(k): list(v) for k, v in self._per_step_relation_similarity.items()}
        }
    
    def clear_threshold_not_met_counts(self):
        """Clear threshold not met counts after reporting"""
        self._relation_threshold_not_met_counts.clear()
        # Also clear per-step similarity scores to prevent data contamination
        self._per_step_relation_similarity.clear()
    
    def get_action_metrics(self):
        """Get action metrics and clear them"""
        metrics = dict(self._action_metrics)
        self._action_metrics.clear()
        return metrics
    
    def is_valid_mid_format(self, entity_arg: str) -> bool:
        """
        Check if entity argument is in valid MID format (m.XXXX or g.XXXX)
        """
        if not entity_arg or not isinstance(entity_arg, str):
            return False
        
        # MID format: m.XXXX or g.XXXX where XXXX can be alphanumeric and underscore
        mid_pattern = re.compile(r"^[mg]\.[A-Za-z0-9_]+$")
        return bool(mid_pattern.match(entity_arg.strip()))
    
    def _extract_entity_from_function_state(self, function_state: List[str]) -> str:
        """
        Extract entity MID from function_state for error reporting.
        Returns the last entity found in the function state or 'unknown entity'.
        """
        if not function_state:
            return "unknown entity"
        
        # Look for START() or JOIN() calls to extract entity MIDs
        for func_call in reversed(function_state):  # Start from the most recent
            # Pattern for START('m.xxxxx') or JOIN('relation', 'm.xxxxx')
            start_match = re.search(r"START\('([mg]\.[A-Za-z0-9_]+)'\)", func_call)
            if start_match:
                return start_match.group(1)
            
            # Pattern for JOIN calls - extract the second parameter if it's an entity
            join_match = re.search(r"JOIN\('[^']+',\s*'([mg]\.[A-Za-z0-9_]+)'\)", func_call)
            if join_match:
                return join_match.group(1)
        
        return "unknown entity"
    
    def process_actions_with_retrieval(self, actions: List[ActionResult], function_state: List[str] = None, 
                                     candidate_entities: List[Tuple[str, str]] = None, sample_id: int = None) -> List[ActionResult]:
        """
        Process actions with relation/entity retrieval (implements KBQA-o1's core logic)
        Now with persistent function_state management
        """
        processed_actions = []
        current_function_state = function_state.copy() if function_state else []        
        for action_idx, action in enumerate(actions):
            # Set action index for per-action state tracking
            if not hasattr(action, 'action_index'):
                action.action_index = action_idx
            # Count attempts for MERGE at parse-time as well (fine-grained)
            if action.action_type in self._special_action_names:
                name = self._special_action_names[action.action_type]
                # Do not double count attempts here if already counted per prediction; keeping both provides two views.
                self._action_metrics[name]['attempts'] += 0  # no-op placeholder to ensure key exists
                
            # if action.action_type == ActionType.EXTRACT_ENTITY:
            #     # Process Extract_entity with candidate entity selection
            #     processed_action = self.process_extract_entity_action(action, candidate_entities)
            #     processed_actions.append(processed_action)
                
            #     # Update function state
            #     if processed_action.is_valid:
            #         entity_id = processed_action.arguments[0]  # This would be the selected entity ID
            #         # Use proper expression ID management
            #         if sample_id is not None:
            #             expr_id = self.state_manager.get_next_expression_id(sample_id)
            #             function_string = f"expression{expr_id} = START('{entity_id}')"
            #             self.state_manager.update_sample_function_state(sample_id, function_string)
            #             current_function_state.append(function_string)
            #         else:
            #             # Fallback to hardcoded expression1 if no sample_id
            #             current_function_state.append(f"expression1 = START('{entity_id}')")
                    
            if action.action_type == ActionType.FIND_RELATION:
                # Process Find_relation with new format: [entity | relation]
                processed_action = self.process_find_relation_action(action, current_function_state, sample_id)
                processed_actions.append(processed_action)
                
                # Update function state
                if processed_action.is_valid and len(processed_action.arguments) == 2:
                    # New format: [entity | relation]
                    entity_id, relation_id = processed_action.arguments[0], processed_action.arguments[1]
                    if sample_id is not None:
                        # 隐式 START：若 entity_id 不是 expressionX，则先创建起点并自增一次
                        if entity_id.startswith('expression'):
                            # 链式：复用该 expression 的 id 作为当前 id
                            try:
                                current_expr_id = int(entity_id.replace('expression',''))
                            except Exception:
                                current_expr_id = int(self.state_manager.get_current_expression_id(sample_id))
                        else:
                            # 新起点：为实体创建 START，自增一次，并作为当前 id
                            current_expr_id = int(self.state_manager.get_next_expression_id(sample_id))
                            start_function = f"expression{current_expr_id} = START('{entity_id}')"
                            self.state_manager.update_sample_function_state(sample_id, start_function)
                            current_function_state.append(start_function)

                        # JOIN 不自增，复用 current_expr_id
                        function_string = f"expression{current_expr_id} = JOIN('{relation_id}', expression{current_expr_id})" if entity_id.startswith('expression') else f"expression{current_expr_id} = JOIN('{relation_id}', expression{current_expr_id})"
                        self.state_manager.update_sample_function_state(sample_id, function_string)
                        current_function_state.append(function_string)
                    else:
                        # Fallback to hardcoded expression1 if no sample_id
                        if entity_id.startswith('expression'):
                            current_function_state.append(f"expression1 = JOIN('{relation_id}', {entity_id})")
                        else:
                            current_function_state.append(f"expression1 = JOIN('{relation_id}', '{entity_id}')")
                    
            elif action.action_type == ActionType.MERGE:
                # Process Merge action (AND operation)
                processed_action = self.process_merge_action(action, current_function_state, sample_id)
                processed_actions.append(processed_action)
                
                # Update function state
                if processed_action.is_valid:
                    expr1, expr2 = processed_action.arguments[0], processed_action.arguments[1]
                    if sample_id is not None:
                        # MERGE 写回 prev，并回退 current 指针
                        try:
                            curr = int(self.state_manager.get_current_expression_id(sample_id))
                            prev = curr - 1
                        except Exception:
                            prev = -1
                        if prev < 1:
                            # 无法回写，标记为无效
                            processed_action.is_valid = False
                            processed_action.error_message = "Merge requires at least one existing expression (expression1) to merge with"
                            logger.warning(f"[MERGE] Sample {sample_id}: Cannot merge when prev < 1 (curr={curr if 'curr' in locals() else 'unknown'})")
                        else:
                            function_string = f"expression{prev} = AND({expr1}, {expr2})" #TODO
                            self.state_manager.update_sample_function_state(sample_id, function_string)
                            self.state_manager.set_current_expression_id(sample_id, prev)
                            current_function_state.append(function_string)
                    else:
                        # Fallback to hardcoded expression1 if no sample_id
                        current_function_state.append(f"expression1 = AND({expr1}, {expr2})")
                    
            elif action.action_type == ActionType.COMPARE:
                # Process Compare action with relation selection
                processed_action = self.process_compare_action(action, sample_id)
                processed_actions.append(processed_action)
                
                # Update function state for COMPARE actions
                if processed_action.is_valid:
                    if len(processed_action.arguments) == 3:
                        # New format: [operator | relation | number]
                        mode, relation, number = processed_action.arguments
                        if sample_id is not None:
                            # 隐式 START 数字：自增一次
                            expr_id = self.state_manager.get_next_expression_id(sample_id)
                            start_function = f"expression{expr_id} = START('{number}')"
                            self.state_manager.update_sample_function_state(sample_id, start_function)
                            current_function_state.append(start_function)

                            # CMP 复用同一 id，不再自增
                            standardized_mode = COMPARISON_MODE_MAPPING.get(mode, mode.lower())
                            cmp_function = f"expression{expr_id} = CMP('{standardized_mode}', '{relation}', expression{expr_id})"
                            self.state_manager.update_sample_function_state(sample_id, cmp_function)
                            current_function_state.append(cmp_function)
                        else:
                            # Fallback to hardcoded expression IDs if no sample_id
                            start_function = f"expression1 = START('{number}')"
                            current_function_state.append(start_function)
                            # Use common comparison mode mapping
                            standardized_mode = COMPARISON_MODE_MAPPING.get(mode, mode.lower())
                            cmp_function = f"expression2 = CMP('{standardized_mode}', '{relation}', expression1)"
                            current_function_state.append(cmp_function)
                
            elif action.action_type == ActionType.TIME_CONSTRAINT:
                # Validate Time_constraint requires an existing path/expression
                processed_action = self.process_time_constraint_action(action, current_function_state)
                processed_actions.append(processed_action)
                
                # Update function state for TIME_CONSTRAINT actions
                if processed_action.is_valid:
                    if len(processed_action.arguments) == 2:
                        relation, time = processed_action.arguments
                        # Normalize relation like "/a/b.c" -> "a.b.c"
                        # if isinstance(relation, str):
                        #     norm_rel = relation.lstrip('/')
                        #     norm_rel = norm_rel.replace('/', '.')
                        # else:
                        #     norm_rel = relation
                        # # Validate relation suffix
                        # import re as _re
                        # if not _re.search(r"\.(from|to|end_date)$", str(norm_rel)):
                        #     processed_action.is_valid = False
                        #     processed_action.error_message = f"Time_constraint relation must end with .from/.to/.end_date, got '{relation}'"
                        #     return processed_action
                        # # Validate time


                        if sample_id is not None:
                            # TC 复用当前 id，不自增
                            current_expr_id = self.state_manager.get_current_expression_id(sample_id)
                            function_string = f"expression{current_expr_id} = TC(expression{current_expr_id}, '{relation}', '{time}')"
                            self.state_manager.update_sample_function_state(sample_id, function_string)
                            current_function_state.append(function_string)
                        else:
                            # Fallback to hardcoded expression IDs if no sample_id
                            current_function_state.append(f"expression2 = TC(expression1, '{relation}', '{time}')")
                
            elif action.action_type == ActionType.ORDER:
                # Process Order action (ARG operation)
                processed_action = self.process_order_action(action, current_function_state, sample_id)
                processed_actions.append(processed_action)
                
                # Update function state for ORDER actions (执行阶段仅落地，不再重复校验)
                if processed_action.is_valid:
                    mode, expr_token, relation = processed_action.arguments
                    if sample_id is not None:
                        expr_token = expr_token.strip()
                        if expr_token.startswith('expression'):
                            try:
                                expr_id = int(expr_token.replace('expression',''))
                            except Exception:
                                expr_id = int(self.state_manager.get_current_expression_id(sample_id))
                        else:
                            # expr_token 为已在校验阶段确认过的 ontology type：隐式 START 一次
                            new_id = self.state_manager.get_next_expression_id(sample_id)
                            start_fn = f"expression{new_id} = START('{expr_token}')"
                            self.state_manager.update_sample_function_state(sample_id, start_fn)
                            current_function_state.append(start_fn)
                            expr_id = new_id
                        function_string = f"expression{expr_id} = ARG('{mode}', expression{expr_id}, '{relation}')"
                        self.state_manager.update_sample_function_state(sample_id, function_string)
                        current_function_state.append(function_string)
                    else:
                        current_function_state.append(f"expression2 = ARG('{mode}', expression1, '{relation}')")
                    

                
            elif action.action_type == ActionType.COUNT:
                # Process Count action
                processed_action = self.process_count_action(action, current_function_state, sample_id)
                processed_actions.append(processed_action)
                
                # Update function state for COUNT actions
                if processed_action.is_valid:
                    if len(processed_action.arguments) == 1:
                        expression = processed_action.arguments[0]
                        if sample_id is not None:
                            # COUNT 复用当前 id，不自增
                            current_expr_id = self.state_manager.get_current_expression_id(sample_id)
                            function_string = f"expression{current_expr_id} = COUNT({expression})"
                            self.state_manager.update_sample_function_state(sample_id, function_string)
                            current_function_state.append(function_string)
                        else:
                            # Fallback to hardcoded expression IDs if no sample_id
                            current_function_state.append(f"expression2 = COUNT({expression})")
                
                
            else:
                # Other actions pass through unchanged
                action.is_valid = True
                processed_actions.append(action)
        
        return processed_actions
    
    # def process_extract_entity_action(self, action: ActionResult, candidate_entities: List[Tuple[str, str]] = None) -> ActionResult:
    #     """
    #     Process Extract_entity action with entity selection (implements KBQA-o1's entity selection)
    #     """
    #     if not action.arguments:
    #         return action
        
    #     gen_argument = action.arguments[0]  # LLM generated entity description
        
    #     # Get candidate entities
    #     if not candidate_entities:
    #         candidate_entities = self.relation_retrieval.get_candidate_entities()
        
    #     # Select best entity using similarity
    #     selected_entity = self.relation_retrieval.select_best_entity(gen_argument, candidate_entities)
        
    #     if selected_entity:
    #         entity_name, entity_id = selected_entity
    #         # Update action with selected entity ID
    #         new_action = ActionResult(
    #             action_type=action.action_type,
    #             arguments=[entity_id],  # Use entity ID instead of name
    #             raw_text=action.raw_text,
    #             step_number=action.step_number,
    #             is_valid=True
    #         )
    #         logger.info(f"Selected entity: {entity_name} ({entity_id}) for '{gen_argument}'")
    #         return new_action
    #     else:
    #         # No suitable entity found
    #         logger.warning(f"No suitable entity found for '{gen_argument}'")
    #         action.is_valid = False
    #         action.error_message = f"No suitable entity found for '{gen_argument}'"
    #         return action
    


    def _is_ontology_entity(self, value: str) -> bool:
        """Return True if value looks like an ontology identifier.

        Notes:
        - Must NOT be MID, URL, quoted string, or xsd literal (with ^^)
        - Must match domain.type, or the deeper base.*.* namespace used by GraphQ.
        """
        if not value or not isinstance(value, str):
            return False
        s = value.strip()
        if s.startswith(('m.', 'g.', 'http', '"')) or ('^^' in s):
            return False
        pattern = re.compile(
            r"^(?:"
            r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*"
            r"|base(?:\.[A-Za-z_][A-Za-z0-9_]*){2,}"
            r")$"
        )
        return bool(pattern.fullmatch(s))
    
    def process_merge_action(self, action: ActionResult, function_state: List[str], sample_id: int = None) -> ActionResult:
        """
        Process Merge action (AND operation) with support for ontology entities
        """
        self._maybe_debug_break(sample_id, "merge")
        # Enhanced debug: print sample_id, function_state, and merge action details
        fs_len = len(function_state) if function_state is not None else 0
        logger.info(f"[MERGE-DEBUG] Sample {sample_id}: Processing Merge action")
        logger.info(f"[MERGE-DEBUG] Sample {sample_id}: Raw action text: {action.raw_text}")
        logger.info(f"[MERGE-DEBUG] Sample {sample_id}: Action arguments: {action.arguments}")
        logger.info(f"[MERGE-DEBUG] Sample {sample_id}: Persistent function_state length: {fs_len}")
        if function_state:
            for idx, fs in enumerate(function_state, 1):
                logger.info(f"[MERGE-DEBUG] Sample {sample_id} #{idx}: {fs}")

        if not action.arguments or len(action.arguments) != 2:
            action.is_valid = False
            action.error_message = "Merge action requires exactly 2 arguments"
            return action
        
        expr1, expr2 = action.arguments[0], action.arguments[1]
        # Literal-type constraints for Merge
        is_onto1 = self._is_ontology_entity(expr1)
        is_onto2 = self._is_ontology_entity(expr2)
        # 1) Do not allow merging two literal types
        if is_onto1 and is_onto2:
            action.is_valid = False
            action.error_message = f"Invalid Merge between two ontology types: {expr1} and {expr2}"
            return action
        # 2) If a literal_type appears, it must exactly match one in candidate_entities
        if (is_onto1 or is_onto2) and sample_id is not None:
            #  
            candidate_entities = self.state_manager.get_sample_entities(sample_id) or []
            candidate_type_ids = {ent_id for _, ent_id in candidate_entities}
            lit_to_check = expr1 if is_onto1 else expr2
            if lit_to_check not in candidate_type_ids:
                action.is_valid = False
                action.error_message = f"Merge blocked: ontology entity '{lit_to_check}' not in candidate entities"
                return action
        
        if sample_id is not None:
            current_function_state = self.state_manager.get_sample_function_state(sample_id)
            
            # Require both arguments to be expression references like expression1, expression
            # Check if expressions exist or are ontology entities
            
            # Helper to check expression status based on its LAST assignment
            # This fixes a bug where an expression that started as START but was updated (e.g. via JOIN)
            # was still considered a START operation, blocking valid merges.
            def get_expression_status(expr_name, func_state):
                # Returns (exists, is_start)
                if not func_state:
                    return False, False
                for func in reversed(func_state):
                    clean_func = func.strip()
                    if clean_func.startswith(f"{expr_name} = "):
                        return True, clean_func.startswith(f"{expr_name} = START('")
                return False, False

            expr1_exists, expr1_is_start = get_expression_status(expr1, current_function_state)
            expr2_exists, expr2_is_start = get_expression_status(expr2, current_function_state)
            
            # Handle ontology entities by creating START operations
            if not expr1_exists and self._is_ontology_entity(expr1):
                if expr2_is_start:
                    action.is_valid = False
                    action.error_message = f"{expr2} is a START operation, which is not allowed to be merged with {expr1}"
                    return action
                # Create START operation for ontology entity
                if expr2_exists:
                    expr_id = self.state_manager.get_next_expression_id(sample_id)
                    start_function = f"expression{expr_id} = START('{expr1}')"
                    self.state_manager.update_sample_function_state(sample_id, start_function)
                    # Update the argument to reference the new expression
                    action.arguments[0] = f"expression{expr_id}"
                    logger.info(f"Created START operation for ontology entity: {expr1}")
                else:
                    action.is_valid = False
                    action.error_message = f"Expression not found and not an ontology entity: {expr1}"
                    return action
                
            elif not expr1_exists:
                action.is_valid = False
                action.error_message = f"Expression not found and not an ontology entity: {expr1}"
                return action
            
            if not expr2_exists and self._is_ontology_entity(expr2):
                if expr1_is_start:
                    action.is_valid = False
                    action.error_message = f"{expr1} is a START operation, which is not allowed to be merged with {expr2}"
                    return action
                # Create START operation for ontology entity
                if expr1_exists:
                    expr_id = self.state_manager.get_next_expression_id(sample_id)
                    start_function = f"expression{expr_id} = START('{expr2}')"
                    self.state_manager.update_sample_function_state(sample_id, start_function)
                    # Update the argument to reference the new expression
                    action.arguments[1] = f"expression{expr_id}"
                    logger.info(f"Created START operation for ontology entity: {expr2}")
                else:
                    action.is_valid = False
                    action.error_message = f"Expression not found and not an ontology entity: {expr2}"
                    return action
            elif not expr2_exists:
                action.is_valid = False
                action.error_message = f"Expression not found and not an ontology entity: {expr2}"
                return action
        
        # Merge action is valid
        action.is_valid = True
        return action

    def process_time_constraint_action(self, action: ActionResult, function_state: List[str]) -> ActionResult:
        """
        Process Time_constraint action with validation:
        - Requires at least one existing expression path in function_state
        - If absent, mark action invalid and DO NOT attempt execution
        """
        # Debug: print full persistent function_state for TC
        fs_len = len(function_state) if function_state is not None else 0
        logger.info(f"[TC-DEBUG] Persistent function_state length: {fs_len}")
        if function_state:
            for idx, fs in enumerate(function_state, 1):
                logger.info(f"[TC-DEBUG] #{idx}: {fs}")

        # Detect if there is at least one existing expression= assignment
        has_expression = any(isinstance(s, str) and s.strip().startswith("expression") and " = " in s for s in function_state)
        if not has_expression:
            action.is_valid = False
            action.error_message = "Time_constraint requires an existing path (e.g., a prior JOIN). Please add a relation first."
            action.raw_text = (action.raw_text or '') + "\n" + action.error_message
            logger.warning(action.error_message)
            return action

    # Dynamic relation retrieval for TC: prefer time-bound relations from current context
        if action.arguments and len(action.arguments) == 2:
            input_relation, input_time = action.arguments
            current_relation = input_relation

            # Build candidate relations from dynamic retrieval and filter to time-like relations (heuristic)
            try:
                candidates_all = self.relation_retrieval.get_candidate_relations(function_state, allow_literal_relations=True)
            except Exception:
                candidates_all = []
            

            # Prefer last-token whitelist based on observed stats; then fall back to looser time keywords
            allowed_last_tokens = [
                "from", "end_date", "from_date", "start_date", "to", "date_written", "year"
            ]


            def _is_time_relation_name(rel_name: str) -> bool:
                if not isinstance(rel_name, str) or not rel_name:
                    return False
                name = rel_name.lower()
                last_tok = name.split('.')[-1]
                if last_tok in allowed_last_tokens:
                    return True
                return False

            time_relations = [(rel, rel_id) for (rel, rel_id) in (candidates_all or []) if _is_time_relation_name(rel)]


            # If provided relation is not a proper time relation, try similarity selection
            if time_relations:
                # If user passed only a hint like 'from', bias towards that token first
                # hint = None
                # if isinstance(current_relation, str) and '.' not in current_relation:
                #     hint = current_relation.strip().lower()
                # biased = time_relations
                # if hint:
                #     subset = [c for c in time_relations if isinstance(c[0], str) and c[0].split('.')[-1].lower() == hint]
                #     if subset:
                #         biased = subset
                ranked = self.relation_retrieval.select_best_relations(str(current_relation), time_relations, source='literal_plain')
                if ranked:
                    best = ranked[0]
                    # Update to selected relation id
                    action.arguments[0] = best.relation_id
                else:
                    # Provide candidates to caller
                    cand_names = self._filter_prompt_relations([r for r, _ in time_relations][:20])
                    info = "\n".join(["No suitable time relation found; choose one:", *cand_names])
                    action.is_valid = False
                    action.error_message = info
                    action.raw_text = (action.raw_text or '') + "\n" + info
                    return action
            else:
                # No time-like relations available in current context; do not proceed silently
                if not candidates_all:
                    info = "No candidate relations available for Time_constraint in current context; please add a relation (e.g., JOIN) first or adjust context."
                else:
                    cand_names = self._filter_prompt_relations([r for r, _ in time_relations][:20])
                    info = "\n".join(["No time-like relations (from/to/end_date/...) found for Time_constraint in current context; please choose one relation from candidates below by quoting the exact relation name:", *cand_names, "If you don't find the relation you want, you can choose another relation or refine the path."])
                logger.warning(info)
                action.is_valid = False
                action.error_message = info
                action.raw_text = (action.raw_text or '') + "\n" + info
                return action

            # Validate/assist time input using tc_time_list
            def _valid_time(t: str) -> bool:
                if t == 'NOW':
                    return True
                if t.strip() not in tc_time_list:
                    return 
                return True

            if not _valid_time(input_time):
                # Offer suggestions from tc_time_list
                suggest = ", ".join(tc_time_list[:10])
                info = f"Invalid time '{input_time}'. Use NOW or YYYY[-MM[-DD]]. Examples: {suggest}"
                action.is_valid = False
                action.error_message = info
                action.raw_text = (action.raw_text or '') + "\n" + info


        action.is_valid = True
        return action

    def process_compare_action(self, action: ActionResult, sample_id: int = None) -> ActionResult:
        """
        Process Compare action with relation selection from literal_relation_list
        """
        if not action.arguments or len(action.arguments) != 3:
            action.is_valid = False
            action.error_message = "Compare action requires exactly 3 arguments: [operator | relation | number]"
            return action
        
        mode, relation, number = action.arguments
        
        # Validate mode: only support normalized forms {le, ge, lt, gt}
        allowed_normalized_modes = {"le", "ge", "lt", "gt"}
        is_supported = str(mode).lower() in allowed_normalized_modes
        if not is_supported:
            msg = f"Unsupported CMP mode: '{mode}'. Supported modes: le, ge, lt, gt"
            logger.warning(msg)
            action.is_valid = False
            action.error_message = msg
            action.raw_text = (action.raw_text or '') + "\n" + msg
            return action
        
        # Validate value for CMP: allow numeric and date-like literals
        # Supported typed literals: integer/int/float/double/decimal/date/dateTime/gYear/gYearMonth
        # Also allow untyped numeric and date patterns (YYYY[-MM[-DD]] or ISO datetime)
        def _is_supported_cmp_value(val: str) -> bool:
            if not isinstance(val, str) or not val.strip():
                return False
            s = val.strip()
            # Disallow Freebase MIDs
            if s.startswith('m.') or s.startswith('g.'):
                return False
            # Typed literal: ensure numeric xsd datatype
            if '^^' in s:
                try:
                    val_part, dtype = s.split('^^', 1)
                    # Accept both full IRI and xsd: prefixed forms
                    if '#' in dtype:
                        dtype = dtype.split('#', 1)[1].rstrip('>')
                    elif 'xsd:' in dtype:
                        dtype = dtype.split('xsd:', 1)[1]
                    dtype = dtype.lower()
                    # Normalize value part (strip quotes if present)
                    val_part = val_part.strip().strip('"')
                    numeric_allowed = {'integer', 'int', 'float', 'double', 'decimal'}
                    date_allowed = {'date', 'datetime', 'gyear', 'gyearmonth'}
                    if dtype in numeric_allowed:
                        import re as _re
                        num_pattern = _re.compile(r"^[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$")
                        return bool(num_pattern.fullmatch(val_part))
                    if dtype in date_allowed:
                        import re as _re

                        # YYYY or YYYY-MM or YYYY-MM-DD
                        date_basic = _re.compile(r"^\d{4}(?:-(?:0[1-9]|1[0-2]))?(?:-(?:0[1-9]|[12]\d|3[01]))?$")
                        # ISO datetime: YYYY-MM-DDTHH:MM[:SS]
                        date_time = _re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?$")
                        return bool(date_basic.fullmatch(val_part)) or bool(date_time.fullmatch(val_part))
                    return False
                except Exception:
                    return False
            # Plain number: allow int/float/scientific
            import re as _re
            num_pattern = _re.compile(r"^[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$")
            if bool(num_pattern.fullmatch(s)):
                return True
            # Also accept plain date-like values without explicit xsd type
            date_basic = _re.compile(r"^\d{4}(?:-(?:0[1-9]|1[0-2]))?(?:-(?:0[1-9]|[12]\d|3[01]))?$")
            date_time = _re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?$")
            return bool(date_basic.fullmatch(s)) or bool(date_time.fullmatch(s))

        if not _is_supported_cmp_value(str(number)):
            info = (
                f"Invalid CMP value '{number}'. The third argument must be a numeric or date literal"
            )
            logger.warning(info)
            action.is_valid = False
            action.error_message = info
            action.raw_text = (action.raw_text or '') + "\n" + info
            return action

        # Use relation retrieval to find the best matching literal relation
        best_relation = self.relation_retrieval.select_best_relation_for_cmp(relation, mode)

        # Build scored candidate list from last_similarity_scores for UI/guidance
        scored_candidates = []
        if hasattr(self.relation_retrieval, 'last_similarity_scores') and self.relation_retrieval.last_similarity_scores:
            literal_relations = list(self.relation_retrieval.literal_relation_list)
            for rel_name, score in zip(literal_relations, self.relation_retrieval.last_similarity_scores):
                scored_candidates.append((rel_name, score))
            scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # If threshold not met (<0.5), do NOT attempt execution. Show top-20.
        if not best_relation:
            ranked = [rel for rel, _ in scored_candidates[:20]] if scored_candidates else []
            ranked = self._filter_prompt_relations(ranked)
            if ranked:
                cand_lines = [f"{rel_name}" for rel_name in ranked]
                info = "\n".join([f"No suitable relation found for '{relation}' in CMP (threshold not met); please choose one relation from candidates below by quoting the exact relation name:", *cand_lines])
            else:
                info = f"CMP threshold not met; no candidates available for relation '{relation}'"
            logger.warning(info)

            action.is_valid = False
            action.error_message = info
            action.raw_text = (action.raw_text or '') + "\n" + info

            # Clear selection context and store per-action state for UI
            action_index = getattr(action, 'action_index')
            
            # Use sample-based state management if sample_id is provided
            if sample_id is not None:
                self._set_action_relation_state(
                    sample_id=sample_id,
                    action_index=action_index,
                    selected_relation=None,
                    candidate_relations=[rel for rel, _ in scored_candidates],
                    ranked_top5=[],
                    state_dict={
                        'action_type': action.action_type.name if hasattr(action.action_type, 'name') else str(action.action_type),
                        'raw_text': getattr(action, 'raw_text', None),
                        'arguments': list(getattr(action, 'arguments', [])),
                        'writer': 'process_compare_action',
                        'selected_relation': None,
                        'candidate_relations': [rel for rel, _ in scored_candidates],
                        'threshold_not_met': True,
                        'error_message': info,
                        'entity_argument': number,
                        'relation_prompt': relation,
                    }
                )
            else:
                # Fallback to legacy state management
                self._last_selected_relation = None
                self._per_action_relation_states[action_index] = {
                    'selected_relation': None,
                    'candidate_relations': [rel for rel, _ in scored_candidates],
                    'threshold_not_met': True,
                    'error_message': info
                }
            return action

        # If we have a best relation, update and proceed
        if best_relation:
            # Support both RelationCandidate-like objects (have relation_name/score)
            # and legacy string returns. Prefer the relation_name when available.
            try:
                if hasattr(best_relation, 'relation_name'):
                    best_relation_name = getattr(best_relation, 'relation_name')
                    best_relation_score = getattr(best_relation, 'score', None)
                else:
                    best_relation_name = str(best_relation)
                    best_relation_score = None
            except Exception:
                best_relation_name = str(best_relation)
                best_relation_score = None

            if best_relation_name != relation:
                if best_relation_score is not None:
                    logger.info(f"Improved relation for CMP: '{relation}' -> '{best_relation_name}' (score: {best_relation_score:.3f})")
                else:
                    logger.info(f"Improved relation for CMP: '{relation}' -> '{best_relation_name}'")
                action.arguments[1] = best_relation_name

        # Show top-5 only when 0.5 <= best_score < 0.9
        best_score = scored_candidates[0][1] if scored_candidates else 0.0
        ranked_top5 = []
        ranked_top5 = [rel for rel, _ in scored_candidates[:6]]
        
        # Use sample-based state management (consistent with Find_relation)
        # Try to use action-based state if action_index is available
        action_index = getattr(action, 'action_index', None)
        if action_index is not None:
            self._set_action_relation_state(
                sample_id=sample_id,
                action_index=action_index,
                selected_relation=best_relation,
                candidate_relations=[rel for rel, _ in scored_candidates],
                ranked_top5=ranked_top5,
                state_dict={
                    'action_type': action.action_type.name if hasattr(action.action_type, 'name') else str(action.action_type),
                    'raw_text': getattr(action, 'raw_text', None),
                    'arguments': list(getattr(action, 'arguments', [])),
                    'writer': 'process_compare_action',
                    'selected_relation': best_relation,
                    'ranked_top5': ranked_top5,
                    'candidate_relations': [rel for rel, _ in scored_candidates],
                    'entity_argument': number,
                    'relation_prompt': relation,
                }
            )
        else:
            # Fallback to legacy state management
            self._last_ranked_top5 = ranked_top5



        action.is_valid = True
        return action

    def _get_candidate_relations_by_entity_type(self, entity_id: str, sample_id: int = None) -> List[Tuple[str, str]]:
        """Get candidate relations based on entity type, following KBQA-o1's logic
        
        KBQA-o1 uses name_relation_list for non-entity types: ['name', 'literal', 'int', 'url', 'onto']
        This is needed for WebQSP cases where START argument is a string literal like "Firearms"
        
        NOTE: 'name' type handling is ONLY applied to WebQSP dataset, not GrailQA/GraphQ
        """
        entity_type = self._get_entity_type(entity_id)
        
        if entity_type == 'name':
            # String literals like "Firearms" or "Firearms"@en
            # IMPORTANT: Only apply to WebQSP dataset (not GrailQA/GraphQ)
            if self._dataset_type == 'webqsp':
                # Use predefined name_relation_list (same as KBQA-o1 logic)
                logger.info(f"[WebQSP] Entity type 'name' detected for '{entity_id}', using name_relation_list")
                result_candidates = self._process_predefined_relations(name_relation_list)
            else:
                # For non-WebQSP datasets, 'name' type is not supported
                logger.info(f"[{self._dataset_type}] Entity type 'name' not supported for Find_relation, returning empty candidates")
                result_candidates = []
            
        elif entity_type in ['literal', 'int']:
            # Only allow specific literal datatypes to proceed for Find_relation
            # Allowed: float, integer, date, dateTime, gYear, gYearMonth
            allowed_types = {"float", "integer", "date", "dateTime", "gYear", "gYearMonth"}

            def _infer_literal_type(lit: str) -> str:
                # Examples:
                #   "12"^^http://www.w3.org/2001/XMLSchema#integer
                #   "12.3"^^xsd:float
                #   "2001-01-01"^^xsd:date
                #   "2001"^^xsd:gYear
                #   "2001-05"^^xsd:gYearMonth
                if '^^' not in lit:
                    return "string"  # treat as plain string literal
                dtype = lit.split('^^', 1)[1]
                if '#"' in dtype:
                    # unlikely malformed, fallback
                    dtype = dtype.split('#"', 1)[-1]
                if '#' in dtype:
                    dtype = dtype.split('#', 1)[1]
                elif 'xsd:' in dtype:
                    dtype = dtype.split('xsd:', 1)[1]
                return dtype.strip().rstrip('>')

            literal_type = _infer_literal_type(entity_id)
            if literal_type not in allowed_types:
                # Disallow string-like literals for Find_relation to avoid empty queries
                logger.info(f"Disallowed literal type for Find_relation: {literal_type}; returning no candidate relations")
                return []

            # For allowed literal types, use name_relation_list to align with KBQA-o1's get_candidate_relations logic
            # KBQA-o1 uses name_relation_list for ['name', 'literal', 'int', 'url', 'onto']
            result_candidates = self._process_predefined_relations(name_relation_list)
            
        elif entity_type in ['entity']:
            # Use SPARQL to find adjacent relations
            result_candidates = self._get_sparql_relations(entity_id)
            # Ensure result_candidates is never None
            if result_candidates is None:
                result_candidates = []
            
        elif entity_type == 'expression':
            # For expression references, use the general relation list as fallback
            # The actual SPARQL-based relation retrieval will be handled in process_find_relation_action
            # Note: function_state is not available in this context, so we use fallback
            function_state = self.state_manager.get_sample_function_state(sample_id)     
            result_candidates = self._get_sparql_relations(entity_id, function_state)
            # Ensure result_candidates is never None
            if result_candidates is None:
                result_candidates = []
            
        else:
            # Fallback for unknown entity types
            result_candidates = []        
        logger.info(f"Entity type '{entity_type}' -> {len(result_candidates)} candidate relations")
        return result_candidates
    
    def _process_predefined_relations(self, relation_list: List[str]) -> List[Tuple[str, str]]:
        """Process predefined relation list with KBQA-o1's JOIN and reverse relation logic"""
        import re
        forward_relations = []
        reverse_relations = []
        
        for rel in relation_list:
            if 'JOIN' not in rel:
                if '(R ' in rel:
                    # Extract reverse relation: "(R some.relation)" -> "some.relation"
                    r_rel = re.findall(r"\(R (.*?)\)", rel)[0]
                    reverse_relations.append(r_rel)
                else:
                    forward_relations.append(rel)
        
        # Combine forward and reverse relations
        result = [(rel, rel) for rel in forward_relations]
        result.extend([(rel, f'(R {rel})') for rel in reverse_relations])
        return result

    def _ensure_literal_contains_map(self):
        """Build a token -> ordered literal expression map for fast remapping.

        Each token (a dotted relation segment detected inside a literal relation string,
        e.g., law.us_patent.patent_office) may appear in multiple literal expressions. We
        keep all of them, preferring ones that include JOIN when ordering the list so that
        downstream consumers can try the best-matching literal first while still retaining
        the rest as fallbacks.
        """
        if self._literal_contains_map is not None:
            return
        mapping = {}
        try:
            literal_list = getattr(self.relation_retrieval, 'literal_relation_list', None) or []
            import re as _re
            token_pattern = _re.compile(r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z0-9_\.]+")
            for lit in literal_list:
                if not isinstance(lit, str):
                    continue
                # Extract all dotted tokens inside the literal expression
                for token in set(token_pattern.findall(lit)):
                    entries = mapping.setdefault(token, [])
                    if lit in entries:
                        continue
                    if '(JOIN ' in lit:
                        # Keep JOIN variants at the front so they are preferred
                        entries.insert(0, lit)
                    else:
                        entries.append(lit)
        except Exception:
            mapping = {}
        self._literal_contains_map = mapping
    
    def _get_sparql_relations(self, entity_id: str, function_state: List[str] = None) -> List[Tuple[str, str]]:
        """
        Get relations via SPARQL query with fallback to general relation list
        Supports both entity IDs and expression references
        """
        if entity_id.startswith('expression'):
            # For expression references, use the provided function_state
            if function_state:
                # Truncate function_state to the referenced expression context
                fs_scoped = self._truncate_function_state_to_expression(function_state, entity_id)
                result = self.relation_retrieval.get_candidate_relations(fs_scoped)
                if result:
                    logger.info(f"Found {len(result)} relations via SPARQL for expression {entity_id}")
                    return result
                else:
                    logger.warning(f"SPARQL query returned no results for expression {entity_id}, falling back to relation_list")
                    return []
            else:
                logger.warning(f"No function_state provided for expression {entity_id}, falling back to relation_list")
                return []
        else:
            # For entity IDs, create a simple function state
            # Normalize typed literals to avoid duplicated quotes in downstream SPARQL
            normalized_entity = self._normalize_typed_literal(entity_id)
            entity_function_state = [f"expression1 = START('{normalized_entity}')"]
            result = self.relation_retrieval.get_candidate_relations(entity_function_state)
            
            if result:
                logger.info(f"Found {len(result)} relations via SPARQL for entity {entity_id}")
                return result
            else:
                logger.warning(f"SPARQL query returned no results for {entity_id}, falling back to relation_list")
                return []
    
    def _get_entity_type(self, entity_id: str) -> str:
        """Get entity type for relation selection, following KBQA-o1's ent_type logic
        
        Entity types:
        - 'entity': MID format (m.xxx or g.xxx)
        - 'name': Quoted string like "Firearms" or "New York"@en
        - 'literal': Typed literal with ^^ (e.g., "2001"^^xsd:gYear)
        - 'url': URL starting with http or <http
        - 'onto': Ontology entity with dots (e.g., type.object.name)
        - 'int': Pure integer (e.g., 123, -456)
        - 'expression': Expression reference (expressionN)
        """
        if entity_id.startswith('m.') or entity_id.startswith('g.'):
            return 'entity'
        elif entity_id.startswith('"') and (entity_id.endswith('"') or entity_id.endswith('"@en')):
            return 'name'
        elif '^^' in entity_id:
            return 'literal'
        elif entity_id.isdigit() or (entity_id.startswith('-') and entity_id[1:].isdigit()):
            return 'int'
        elif entity_id.startswith('http'):
            return 'url'
        elif self._is_ontology_entity(entity_id):
            return 'onto'
        elif entity_id.startswith('expression'):
            return 'expression'
        else:
            return 'unknown'

    def _normalize_typed_literal(self, value: str) -> str:
        """Normalize a possibly quoted typed literal into the canonical form
        "value"^^http://www.w3.org/2001/XMLSchema#type (without angle brackets).

        Examples:
        - '"80"^^http://...#integer' -> '"80"^^http://...#integer'
        - '80^^http://...#integer'     -> '"80"^^http://...#integer'
        - '"80"'                       -> '"80"' (unchanged)
        - '80'                          -> '80' (unchanged)
        """
        try:
            if not isinstance(value, str):
                return value
            s = value.strip()
            if '^^' not in s:
                return s
            val_part, dtype_part = s.split('^^', 1)
            val_part = val_part.strip().strip('"')
            dtype_part = dtype_part.strip().strip('<>').strip('"')
            return f'{val_part}^^{dtype_part}'
        except Exception:
            return value

    def _is_valid_literal_for_find_relation(self, literal: str) -> bool:
        """Strict validation for typed literals used as Find_relation entity.

        Rules:
        - Must include ^^ and datatype in {integer,int,float,double,decimal,date,dateTime,gYear,gYearMonth}
        - Left part must be a valid numeric/date value; reject unquoted path-like strings (contains '/')
        - For date types, allow YYYY, YYYY-MM, YYYY-MM-DD, or ISO datetime (YYYY-MM-DDTHH:MM[:SS])
        """
        try:
            if not isinstance(literal, str):
                return False
            s = literal.strip()
            if '^^' not in s:
                return False
            left_raw, dtype_part = s.split('^^', 1)
            # Reject path-like strings on the left (e.g., /type/object/...)
            if '/' in left_raw and not (left_raw.strip().startswith('"') and left_raw.strip().endswith('"')):
                return False
            # Normalize dtype
            dtype = dtype_part.strip()
            if '#' in dtype:
                dtype = dtype.split('#', 1)[1].rstrip('>')
            elif 'xsd:' in dtype:
                dtype = dtype.split('xsd:', 1)[1]
            dtype = dtype.lower()
            numeric_allowed = {'integer', 'int', 'float', 'double', 'decimal'}
            date_allowed = {'date', 'datetime', 'gyear', 'gyearmonth'}
            left_val = left_raw.strip().strip('"')
            import re as _re
            if dtype in numeric_allowed:
                num_pattern = _re.compile(r"^[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$")
                return bool(num_pattern.fullmatch(left_val))
            if dtype in date_allowed:
                date_basic = _re.compile(r"^\d{4}(?:-(?:0[1-9]|1[0-2]))?(?:-(?:0[1-9]|[12]\d|3[01]))?$")
                date_time = _re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?$")
                return bool(date_basic.fullmatch(left_val)) or bool(date_time.fullmatch(left_val))
            return False
        except Exception:
            return False

    def process_find_relation_action(self, action: ActionResult, function_state: List[str], sample_id: int = None) -> ActionResult:
        """
        Process Find_relation action with new format: [entity | relation]
        Keeps the similarity-based relation selection mechanism
        """
        self._maybe_debug_break(sample_id, "find_relation")
        if not action.arguments:
            return action
        
        if len(action.arguments) == 2:
            # New format: [entity | relation_description]
            entity_arg, relation_description = action.arguments
            
            # Check entity format based on entity type
            entity_type = self._get_entity_type(entity_arg)
            if entity_type == 'entity' and not self.is_valid_mid_format(entity_arg):
                logger.warning(f"Entity argument '{entity_arg}' is not in valid MID format (expected m.XXXX or g.XXXX)")
                action.is_valid = False
                action.error_message = f"Entity argument '{entity_arg}' is not in valid MID format. Expected format: m.XXXX or g.XXXX"
                return action
            elif entity_type == 'literal':
                # Require well-formed typed literal and valid value
                if '^^' not in entity_arg or not self._is_valid_literal_for_find_relation(entity_arg):
                    logger.warning(f"Invalid literal for Find_relation: '{entity_arg}'")
                    action.is_valid = False
                    action.error_message = (
                        f"Invalid literal '{entity_arg}'. Use numeric/date typed literals, e.g. "
                        f"\"2001\"^^xsd:gYear, \"2001-05\"^^xsd:gYearMonth, \"2001-01-01\"^^xsd:date, 12^^xsd:integer"
                    )
                    return action
            elif entity_type == 'name':
                # 'name' type is ONLY supported for WebQSP dataset
                if self._dataset_type != 'webqsp':
                    logger.warning(f"[{self._dataset_type}] Entity type 'name' is not supported for Find_relation (WebQSP only)")
                    action.is_valid = False
                    action.error_message = f"Entity type 'name' ('{entity_arg}') is only supported for WebQSP dataset. Current dataset: {self._dataset_type}"
                    return action
                # WebQSP: validate name format
                if not (entity_arg.startswith('"') and (entity_arg.endswith('"') or entity_arg.endswith('"@en'))):
                    logger.warning(f"Name argument '{entity_arg}' should be quoted string (e.g., \"New York\"@en)")
                    action.is_valid = False
                    action.error_message = f"Name argument '{entity_arg}' should be quoted string. Expected format: \"Name\"@en"
                    return action
            elif entity_type == 'expression':
                # Validate that the expression exists in function_state
                if sample_id is not None:
                    current_function_state = self.state_manager.get_sample_function_state(sample_id)
                    expr_exists = any(func.strip().startswith(f"{entity_arg} = ") for func in current_function_state)
                    if not expr_exists:
                        action.is_valid = False
                        action.error_message = f"Expression '{entity_arg}' not found in function state. Available expressions: {[f for f in current_function_state if f.strip().startswith('expression') and ' = ' in f]}"
                        return action
                # Expression validation passed, continue with processing
            elif entity_type in ['int', 'onto', 'url']:
                logger.warning(f"Entity type '{entity_type}' is not supported for Find_relation. Expected formats: m.XXXX (entity), \"value\"^^type (literal), \"Name\"@en (name), expressionN (expression)")
                action.is_valid = False
                action.error_message = f"{entity_arg} is an unsupported entity arg for Find_relation. Expected formats: m.XXXX (entity), \"value\"^^type (literal), \"Name\"@en (name), expressionN (expression)"
                return action
            elif entity_type == 'unknown':
                logger.warning(f"Unknown entity type for '{entity_arg}'. Expected formats: m.XXXX (entity), \"value\"^^type (literal), \"Name\"@en (name), expressionN (expression)")
                action.is_valid = False
                action.error_message = f"Unknown entity type for '{entity_arg}'. Expected formats: m.XXXX (entity), \"value\"^^type (literal), \"Name\"@en (name), expressionN (expression)"
                return action
            
            # Get candidate relations based on entity type (KBQA-o1 style)
            candidate_relations = self._get_candidate_relations_by_entity_type(entity_arg, sample_id)
            
            # Store last candidate names for guidance
            candidate_relations_list = []
            try:
                candidate_relations_list = [rel for rel, _ in candidate_relations]
            except Exception:
                candidate_relations_list = []
            
            action_index = getattr(action, 'action_index', None)
            if sample_id is not None and action_index is not None:
                self._set_action_relation_state(
                    sample_id=sample_id,
                    action_index=action_index,
                    candidate_relations=candidate_relations_list,
                    state_dict={
                        'action_type': action.action_type.name if hasattr(action.action_type, 'name') else str(action.action_type),
                        'raw_text': getattr(action, 'raw_text', None),
                        'arguments': list(getattr(action, 'arguments', [])),
                        'writer': 'process_find_relation_action',
                        'candidate_relations': candidate_relations_list,
                        'entity_argument': entity_arg,
                        'relation_prompt': relation_description,
                    }
                )

            try:
                self._cand_all_sum += len(candidate_relations)
                self._cand_all_count += 1
            except Exception:
                pass
            
            if not candidate_relations:
                logger.warning(f"No candidate relations found for entity {entity_arg}")
                action.is_valid = False
                action.error_message = f"No candidate relations available for entity {entity_arg}"
                return action
            
            # Select best relations using similarity (keep the original mechanism)
            # 对 name / literal 实体，提示后端选择对应的小 vocab index
            source_hint = None
            if entity_type in ['literal', 'int', 'name']:
                source_hint = 'name'

            selected_relations = self.relation_retrieval.select_best_relations(
                relation_description,
                candidate_relations,
                source=source_hint,
            )
            
            # Reuse similarity scores from select_best_relations to get top-6 for UI
            try:
                if hasattr(self.relation_retrieval, 'last_similarity_scores') and self.relation_retrieval.last_similarity_scores:
                    # Create candidates with scores and sort by score
                    scored_candidates = []
                    for (rel_name, rel_id), score in zip(candidate_relations, self.relation_retrieval.last_similarity_scores):
                        scored_candidates.append((rel_name, score))
                    scored_candidates.sort(key=lambda x: x[1], reverse=True)
                    # Take top-6 to ensure we have 5 after removing chosen
                    ranked_top5 = [rel_name for rel_name, _ in scored_candidates[:6]]
                else:
                    ranked_top5 = []
            except Exception:
                ranked_top5 = []
            
        # # Use sample-based state management if sample_id is provided
        #     if sample_id is not None and action_index is not None:
        #         self._set_action_relation_state(
        #             sample_id=sample_id,
        #             action_index=action_index,
        #             ranked_top5=ranked_top5
        #         )
            
            if selected_relations:
                # Use the best relation
                best_relation = selected_relations[0]
                
                # Record only TOP-1 score for this step (not all candidate scores)
                # This gives meaningful distribution of selected relation quality
                try:
                    if action.step_number is not None and best_relation.score is not None:
                        self._per_step_relation_similarity[action.step_number].append(float(best_relation.score))
                except Exception:
                    pass
                
                # Store per-action state for multiple Find_relation support
                action_index = getattr(action, 'action_index')
                
                # Use action-based state management (sample_id is always provided in current system)
                self._set_action_relation_state(
                    sample_id=sample_id,
                    action_index=action_index,
                    selected_relation=best_relation,
                    candidate_relations=list(candidate_relations_list) if candidate_relations_list else [],
                    ranked_top5=list(ranked_top5) if ranked_top5 else [],
                    state_dict={
                        'action_type': action.action_type.name if hasattr(action.action_type, 'name') else str(action.action_type),
                        'raw_text': getattr(action, 'raw_text', None),
                        'arguments': list(getattr(action, 'arguments', [])),
                        'writer': 'process_find_relation_action',
                        'selected_relation': best_relation,
                        'ranked_top5': list(ranked_top5) if ranked_top5 else [],
                        'candidate_relations': list(candidate_relations_list) if candidate_relations_list else [],
                        'entity_argument': entity_arg,
                        'relation_prompt': relation_description,
                    }
                )
                
                logger.info(f"Selected relation: {best_relation.relation_name} (score: {best_relation.score:.3f}) for '{relation_description}'")
                
                # Update action with entity and selected relation
                # Normalize typed literal entity to avoid nested quotes later
                normalized_entity_arg = self._normalize_typed_literal(entity_arg)
                new_action = ActionResult(
                    action_type=action.action_type,
                    arguments=[normalized_entity_arg, best_relation.relation_id],  # Keep entity, replace relation description with selected relation
                    raw_text=action.raw_text,
                    step_number=action.step_number,
                    is_valid=True
                )
                return new_action
            else:
                # Threshold not met: provide candidate list for LLM selection (cap at 20)
                ranked = self.relation_retrieval.rank_relations_no_threshold(relation_description, candidate_relations, topk=20)
                if ranked:
                    cand_lines = [f"{c.relation_name}" for c in ranked if getattr(c, 'relation_name', None)]
                    cand_lines = self._filter_prompt_relations(cand_lines)
                    info = "\n".join([f"No suitable relation found for '{relation_description}' (threshold not met); please choose one relation from candidates below by quoting the exact relation name:", *cand_lines])
                else:
                    info = f"threshold not met; no candidates available for entity {entity_arg}"

                logger.warning(f"No suitable relation found for '{relation_description}' (threshold not met). Providing {len(ranked)} candidates to LLM.")
                try:
                    # record the original available candidates length (pre-truncation)
                    self._relation_threshold_not_met_counts[len(candidate_relations)] += 1
                    self._cand_notmet_sum += len(candidate_relations)
                    self._cand_notmet_count += 1
                except Exception:
                    pass
                # Attach info in raw_text so caller can surface it in observation
                action.is_valid = False
                action.error_message = info
                action.raw_text = (action.raw_text or '') + "\n" + info
                # Clear selection context so top-5 block is not shown when threshold not met
                action_index = getattr(action, 'action_index', None) or getattr(action, 'step_number', 0)
                

                # if sample_id is not None:
                #     self._set_action_relation_state(
                #         sample_id=sample_id,
                #         action_index=action_index,
                #         selected_relation=None,
                #         ranked_top5=[],
                #         state_dict={
                #             'selected_relation': None,
                #             'ranked_top5': [],
                #             'candidate_relations': candidate_relations_list,
                #             'threshold_not_met': True,
                #             'error_message': info
                #         }
                #     )
                
                return action
        else:
            # Explicitly invalidate old 1-arg format

            # Only support new format: require exactly 2 arguments
            action.is_valid = False
            action.error_message = "Find_relation requires exactly 2 arguments: [entity | relation]"
            return action

    def process_order_action(self, action: ActionResult, function_state: List[str], sample_id: int = None) -> ActionResult:
        """
        Process Order action (ARG operation) with validation
        Based on KBQA-o1's Order action processing
        """
        self._maybe_debug_break(sample_id, "order")
        if not action.arguments or len(action.arguments) != 3:
            action.is_valid = False
            action.error_message = "Order action requires  3 arguments: [mode | entity_id | relation]"
            return action
        
        action_index = getattr(action, 'action_index', None)
        mode, expr_token, relation = action.arguments

        
        # Validate mode - KBQA-o1 uses ARGMAX/ARGMIN
        valid_modes = {"ARGMAX", "ARGMIN", "MAX", "MIN", "maximum", "minimum", "max", "min"}
        if mode.upper() not in valid_modes:
            action.is_valid = False
            action.error_message = f"Invalid mode '{mode}' for Order action. Valid modes: {', '.join(valid_modes)}"
            return action
        
        # Standardize mode to ARGMAX/ARGMIN format (KBQA-o1 standard)
        mode_mapping = {
            "maximum": "ARGMAX",
            "max": "ARGMAX", 
            "MAX": "ARGMAX",
            "minimum": "ARGMIN",
            "min": "ARGMIN",
            "MIN": "ARGMIN",
            "ARGMAX": "ARGMAX",
            "ARGMIN": "ARGMIN"
        }
        standardized_mode = mode_mapping.get(mode.upper(), mode.upper())
        action.arguments[0] = standardized_mode
        
        # Validate that we have an existing expression to operate on
        if sample_id is not None:
            current_function_state = self.state_manager.get_sample_function_state(sample_id)
            if expr_token:
                expr_token = expr_token.strip()
                if expr_token.startswith('expression'):
                    expr_exists = any(func.strip().startswith(f"{expr_token} = ") for func in current_function_state)
                    if not expr_exists:
                        action.is_valid = False
                        action.error_message = f"Referenced expression '{expr_token}' not found for Order action"
                        return action
                    # Build candidate relations by scoping function_state to referenced expression
                    fs_scoped = self._truncate_function_state_to_expression(current_function_state, expr_token)
                    candidate_relations = self.relation_retrieval.get_candidate_relations(
                        fs_scoped,
                        allow_literal_relations=True,
                    )
                    # Efficient remap using cached token->literal map
                    self._ensure_literal_contains_map()
                    if self._literal_contains_map:
                        mapped = []
                        for rel_name, rel_id in (candidate_relations or []):
                            replacements = None
                            if isinstance(rel_id, str):
                                replacements = self._literal_contains_map.get(rel_id)
                            if (not replacements) and isinstance(rel_name, str):
                                replacements = self._literal_contains_map.get(rel_name)
                            if replacements:
                                if isinstance(replacements, str):
                                    replacements = [replacements]
                                for replacement in replacements:
                                    # replace only id, keep name unchanged; allow 1->many mappings
                                    mapped.append((replacement, replacement))
                        # Strict mode: drop any candidate not present in the mapping
                        candidate_relations = mapped

                    # Selection: choose best relation by similarity against provided relation text
                    if not candidate_relations:
                        action.is_valid = False
                        action.error_message = f"No candidate relations available for {expr_token} in Order"
                        return action
                    selected = self.relation_retrieval.select_best_relations(
                        relation,
                        candidate_relations,
                        source='literal_plain',
                    )
                    if selected:
                        best_relation = selected[0]
                        #  
                        # Update action to carry resolved relation id
                        action.arguments = [action.arguments[0], expr_token, best_relation.relation_id]
                        # Cache candidate names/top5 for UI
                        cand_names = [rel for rel, _ in candidate_relations]
                        
                        ranked_top5 = []
                        if hasattr(self.relation_retrieval, 'last_similarity_scores') and self.relation_retrieval.last_similarity_scores:
                            scored = list(zip(cand_names, self.relation_retrieval.last_similarity_scores))
                            scored.sort(key=lambda x: x[1], reverse=True)
                            ranked_top5 = [name for name, _ in scored[:6]]

                        if sample_id is not None and action_index is not None:
                            self._set_action_relation_state(
                                sample_id=sample_id, 
                                action_index=action_index, 
                                selected_relation=best_relation,
                                candidate_relations=cand_names,
                                ranked_top5=ranked_top5,
                                state_dict={
                                    'action_type': action.action_type.name if hasattr(action.action_type, 'name') else str(action.action_type),
                                    'raw_text': getattr(action, 'raw_text', None),
                                    'arguments': list(getattr(action, 'arguments', [])),
                                    'writer': 'process_order_action',
                                    'selected_relation': best_relation,
                                    'ranked_top5': ranked_top5,
                                    'candidate_relations': cand_names,
                                    'entity_argument': expr_token,
                                    'relation_prompt': relation,
                                }
                            )
                    else:
                        # Threshold not met: provide ranked candidates
                        try:
                            ranked = self.relation_retrieval.rank_relations_no_threshold(relation, candidate_relations, topk=20)
                            cand_lines = [f"{c.relation_name}" for c in ranked] if ranked else []
                            info = "\n".join([f"No suitable relation found for '{relation}' (threshold not met); please choose from candidates:", *cand_lines]) if cand_lines else "threshold not met; no candidates available"
                        except Exception:
                            info = "threshold not met; no candidates available"
                        action.is_valid = False
                        action.error_message = info
                        action.raw_text = (action.raw_text or '') + "\n" + info
                        return action
                else:
                    # 允许直接传入 ontology type（如 astronomy.planet）
                    if not self._is_ontology_entity(expr_token):
                        action.is_valid = False
                        action.error_message = f"Invalid middle argument '{expr_token}'. Expect expressionX or ontology_type"
                        return action
                    # 一致性校验：与候选本体集合/唯一本体一致
                    candidates = self.state_manager.get_sample_entities(sample_id) or []
                    onto_ids = [ent_id for _, ent_id in candidates if self._is_ontology_entity(ent_id)]
                    onto_set = set(onto_ids)
                    if onto_set and expr_token not in onto_set:
                        action.is_valid = False
                        action.error_message = f"Ontology '{expr_token}' not in candidate ontology types"
                        return action
                    # For ontology type, use literal_relation_list as candidate relations (KBQA-o1 style)
                    try:
                        candidate_relations = [(rel, rel) for rel in (getattr(self.relation_retrieval, 'literal_relation_list', None) or [])]
                    except Exception:
                        candidate_relations = []
                    if not candidate_relations:
                        action.is_valid = False
                        action.error_message = f"No literal candidate relations available for ontology '{expr_token}'"
                        return action
                    selected = self.relation_retrieval.select_best_relations(
                        relation,
                        candidate_relations,
                        source='literal_plain',  # Order + ontology 使用裸 literal index
                    )
                    if selected:
                        best_relation = selected[0]
                        action.arguments = [action.arguments[0], expr_token, best_relation.relation_id]
                        # Cache candidate names/top5 for UI
                        try:
                            cand_names = [rel for rel, _ in candidate_relations]
                            
                            ranked_top5 = []
                            if hasattr(self.relation_retrieval, 'last_similarity_scores') and self.relation_retrieval.last_similarity_scores:
                                scored = list(zip(cand_names, self.relation_retrieval.last_similarity_scores))
                                scored.sort(key=lambda x: x[1], reverse=True)
                                ranked_top5 = [name for name, _ in scored[:6]]

                            if sample_id is not None and action_index is not None:
                                self._set_action_relation_state(
                                    sample_id=sample_id, 
                                    action_index=action_index, 
                                    selected_relation=best_relation,
                                    candidate_relations=cand_names,
                                    ranked_top5=ranked_top5,
                                    state_dict={
                                        'action_type': action.action_type.name if hasattr(action.action_type, 'name') else str(action.action_type),
                                        'raw_text': getattr(action, 'raw_text', None),
                                        'arguments': list(getattr(action, 'arguments', [])),
                                        'writer': 'process_order_action',
                                        'selected_relation': best_relation,
                                        'ranked_top5': ranked_top5,
                                            'candidate_relations': cand_names,
                                            'entity_argument': expr_token,
                                            'relation_prompt': relation,
                                    }
                                )
                        except Exception:
                            pass
                    else:
                        try:
                            ranked = self.relation_retrieval.rank_relations_no_threshold(relation, candidate_relations, topk=20)
                            cand_lines = [f"{c.relation_name}" for c in ranked] if ranked else []
                            info = "\n".join([f"No suitable relation found for '{relation}' (threshold not met); please choose from candidates:", *cand_lines]) if cand_lines else "threshold not met; no candidates available"
                        except Exception:
                            info = "threshold not met; no candidates available"
                        action.is_valid = False
                        action.error_message = info
                        action.raw_text = (action.raw_text or '') + "\n" + info
                        return action
            else:
                # no expr_token provided (should not happen under current 3-arg requirement)
                action.is_valid = False
                action.error_message = "Order requires [mode | expression_id_or_ontology | relation]"
                return action
        
        action.is_valid = True
        return action
    
    def process_count_action(self, action: ActionResult, function_state: List[str], sample_id: int = None) -> ActionResult:
        """
        Process Count action with validation
        Based on KBQA-o1's Count action processing
        """
        if not action.arguments or len(action.arguments) != 1:
            action.is_valid = False
            action.error_message = "Count action requires exactly 1 argument: [expression]"
            return action
        
        expression = action.arguments[0]
        
        # Validate that expression exists in function_state
        if sample_id is not None:
            current_function_state = self.state_manager.get_sample_function_state(sample_id)
            
            # Helper to check expression status based on its LAST assignment
            def get_expression_status(expr_name, func_state):
                # Returns (exists, is_start)
                if not func_state:
                    return False, False
                for func in reversed(func_state):
                    clean_func = func.strip()
                    if clean_func.startswith(f"{expr_name} = "):
                        return True, clean_func.startswith(f"{expr_name} = START('")
                return False, False

            expr_exists, expr_is_start = get_expression_status(expression, current_function_state)
            
            # Check if expression is a START operation
            if expr_is_start:
                action.is_valid = False
                action.error_message = f"{expression} is a START operation, which is not allowed in Count action"
                return action
            
            if not expr_exists:
                action.is_valid = False
                action.error_message = f"Expression not found: {expression}"
                return action
        
        action.is_valid = True
        return action
