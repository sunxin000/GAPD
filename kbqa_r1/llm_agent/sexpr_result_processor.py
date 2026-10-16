"""
S-Expression Result Processing Module
Handles execution results processing and output formatting
"""

import logging
import os
import re
from typing import Any, List, Optional, Tuple

from ..sparql.sparql_manager import SPARQLExecutionManager

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class SExprResultProcessor:
    """
    Processes S-Expression execution results and formats output
    """
    
    def __init__(self, config, relation_retrieval=None, model_name: Optional[str] = None, tokenizer: Optional[Any] = None):
        self.config = config
        # Support for multiple Find_relation actions per turn (action-indexed state only)
        self._per_action_relation_states = {}
        self.relation_retrieval = relation_retrieval
        # Try to infer model name for formatting decisions (e.g., Qwen chat template)
        inferred_name = None
        if model_name:
            inferred_name = str(model_name)
        elif tokenizer is not None and hasattr(tokenizer, 'name_or_path'):
            inferred_name = str(getattr(tokenizer, 'name_or_path'))
        elif hasattr(config, 'experiment_name') and getattr(config, 'experiment_name'):
            inferred_name = str(getattr(config, 'experiment_name'))
        self._model_name_or_path = (inferred_name or "").lower()
    
    def _add_guidance_prompt(self, info_block: str, is_final_turn: bool = False) -> str:
        """Add guidance prompt after information block.
        For Qwen models, prefer ChatML assistant prefix; otherwise keep Llama-style header ids.
        On final turn, require the LLM to provide an answer.
        """
        if is_final_turn:
            guidance_body = ("\n\nBecause this is the final turn, you must provide the final answer (only MID / number / date) after thinking.\n")
        else:
            guidance_body = ("\n\n"
                             "Use think tags to contain your findings. If you need more information, "
                             "use action tags to take further actions, otherwise provide the answer (only MID / number / date).\n")

        def _is_qwen(name: str) -> bool:
            name = (name or "").lower()
            return any(k in name for k in ["qwen", "qwen2", "qwen2.5", "qwen3", "coder"])  # coder variants

        if _is_qwen(self._model_name_or_path):
            assistant_prefix = "<|im_start|>assistant\n"
        else:
            assistant_prefix = "<|start_header_id|>assistant<|end_header_id|>"

        return "\n" + info_block + guidance_body + assistant_prefix
    
    # Legacy setters removed: selected relation/top5 are accessed from action-based state
    
    def set_per_action_relation_states(self, states_dict):
        """Set per-action relation states for multiple Find_relation support"""
        self._per_action_relation_states = states_dict or {}
    
    def truncate_result(self, result_string: str, max_chars: int = None) -> str:
        """Truncate results to prevent overflow"""
        if max_chars is None:
            max_chars = self.config.max_obs_length * 3
        
        if len(result_string) <= max_chars:
            return result_string
        
        lines = result_string.split('\n')
        truncated_lines = []
        current_length = 0
        
        for line in lines:
            if current_length + len(line) + 1 > max_chars - 100:
                break
            truncated_lines.append(line)
            current_length += len(line) + 1
        
        if len(lines) > len(truncated_lines):
            remaining_lines = len(lines) - len(truncated_lines)
            truncated_lines.append(f"... ({remaining_lines} more lines truncated)")
            logger.warning(f"Result truncated: {len(lines)} lines -> {len(truncated_lines)-1} lines")
        
        return '\n'.join(truncated_lines)
    
    def extract_mid_list(self, results: Any) -> List[str]:
        """Extract MID list (e.g., m.XXXX, g.XXXX) from execution results.
        Supports both key-prefixed entries like "x: m.XXXX" and plain values like "m.XXXX".
        Now also handles name information when available.
        Also handles non-MID values (like counts) by including them if no MIDs are found.
        """
        mids: List[str] = []
        var_mid_pattern = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*):\s*([mg]\.[A-Za-z0-9_]+)")
        mid_only_pattern = re.compile(r"([mg]\.[A-Za-z0-9_]+)")

        def add_mid(maybe_var: str, mid: str):
            if maybe_var:
                mids.append(f"{maybe_var}:{mid}")
            else:
                mids.append(mid)

        if isinstance(results, list):
            for item in results:
                if isinstance(item, dict):
                    # Check if we have both x and name variables
                    x_value = item.get('x')
                    name_value = item.get('name')
                    
                    if x_value and isinstance(x_value, str):
                        s = x_value.strip()
                        # exact value is a MID
                        if mid_only_pattern.fullmatch(s):
                            # If we have a name, format as "MID (Name)"
                            if name_value and isinstance(name_value, str) and name_value.strip():
                                add_mid('x', f"{s} ({name_value.strip()})")
                            else:
                                add_mid('x', s)
                        else:
                            # Check for embedded MIDs first
                            has_embedded_mids = False
                            for var, mid in var_mid_pattern.findall(s):
                                add_mid(var, mid)
                                has_embedded_mids = True
                            for mid in mid_only_pattern.findall(s):
                                add_mid('x', mid)
                                has_embedded_mids = True
                            
                            # If no embedded MIDs found, add the whole value (e.g., dates, numbers)
                            if not has_embedded_mids and s:
                                add_mid('x', s)
                    else:
                        # Fallback to original logic for other variables
                        for k, v in item.items():
                            if isinstance(v, str):
                                s = v.strip()
                                # exact value is a MID
                                if mid_only_pattern.fullmatch(s):
                                    add_mid(k, s)
                                else:
                                    # Check for embedded MIDs first
                                    has_embedded_mids = False
                                    for var, mid in var_mid_pattern.findall(s):
                                        add_mid(var, mid)
                                        has_embedded_mids = True
                                    for mid in mid_only_pattern.findall(s):
                                        add_mid(k, mid)
                                        has_embedded_mids = True
                                    
                                    # If no embedded MIDs found, add the whole value (e.g., dates, numbers)
                                    if not has_embedded_mids and s:
                                        add_mid(k, s)
                            else:
                                s = str(v)
                                has_embedded_mids = False
                                for var, mid in var_mid_pattern.findall(s):
                                    add_mid(var, mid)
                                    has_embedded_mids = True
                                for mid in mid_only_pattern.findall(s):
                                    add_mid(k, mid)
                                    has_embedded_mids = True
                                
                                # If no embedded MIDs found, add the whole value (e.g., counts, numbers)
                                if not has_embedded_mids and s:
                                    add_mid(k, s)
                else:
                    s = str(item)
                    has_embedded_mids = False
                    for var, mid in var_mid_pattern.findall(s):
                        add_mid(var, mid)
                        has_embedded_mids = True
                    for mid in mid_only_pattern.findall(s):
                        add_mid(None, mid)
                        has_embedded_mids = True
                    
                    if not has_embedded_mids and s:
                        add_mid(None, s)

        elif isinstance(results, dict):
            for k, v in results.items():
                if isinstance(v, list):
                    for sub in v:
                        s = str(sub)
                        has_embedded_mids = False
                        for var, mid in var_mid_pattern.findall(s):
                            add_mid(var, mid)
                            has_embedded_mids = True
                        for mid in mid_only_pattern.findall(s):
                            add_mid(k, mid)
                            has_embedded_mids = True
                        
                        if not has_embedded_mids and s:
                            add_mid(k, s)
                else:
                    s = str(v)
                    has_embedded_mids = False
                    for var, mid in var_mid_pattern.findall(s):
                        add_mid(var, mid)
                        has_embedded_mids = True
                    for mid in mid_only_pattern.findall(s):
                        add_mid(k, mid)
                        has_embedded_mids = True
                    
                    if not has_embedded_mids and s:
                        add_mid(k, s)
        else:
            s = str(results)
            has_embedded_mids = False
            for var, mid in var_mid_pattern.findall(s):
                add_mid(var, mid)
                has_embedded_mids = True
            for mid in mid_only_pattern.findall(s):
                add_mid(None, mid)
                has_embedded_mids = True
            
            if not has_embedded_mids and s:
                add_mid(None, s)

        # deduplicate preserving order
        seen = set()
        unique = []
        for m in mids:
            if m not in seen:
                seen.add(m)
                unique.append(m)
        # 不在提取阶段做数量限制，完整返回，由显示层控制截断/分组
        return unique
    
    def format_mid_list_for_display(self, mid_list: List[str], max_display: int = 50) -> str:
        """格式化MID列表用于显示。当 mid_list 长度>50 时启用分组/截断显示；否则直接完整展示。
        - 若全部来自同一变量（如全是 x: m.***），则去掉前缀仅显示 MID。
        - 现在支持显示名称信息，格式为 "MID (Name)"。
        """
        if not mid_list:
            return "[]"

        # 小规模结果（<=100）：直接完整展示
        if len(mid_list) <= 200:
            # 若全部都有相同的变量名前缀，则去掉前缀
            var_names = set()
            mids_no_prefix: List[str] = []
            all_have_prefix = True
            for mid in mid_list:
                if ':' in mid and not mid.startswith('m.') and not mid.startswith('g.'):
                    var_name, mid_value = mid.split(':', 1)
                    var_names.add(var_name)
                    mids_no_prefix.append(mid_value)
                else:
                    all_have_prefix = False
                    break
            if all_have_prefix and len(var_names) == 1:
                return ", ".join(mids_no_prefix)
            # 否则按原样连接
            return ", ".join(mid_list)

        # 大规模结果（>100）：启用分组/截断显示
        # 分离带变量名前缀和不带前缀的MID
        var_prefixed_mids = [mid for mid in mid_list if ':' in mid and not mid.startswith('m.') and not mid.startswith('g.')]
        regular_mids = [mid for mid in mid_list if ':' not in mid or mid.startswith('m.') or mid.startswith('g.')]

        formatted_parts = []

        # 按变量名分组处理带前缀的MID
        var_groups = {}
        for mid in var_prefixed_mids:
            if ':' in mid:
                var_name, mid_value = mid.split(':', 1)
                if var_name not in var_groups:
                    var_groups[var_name] = []
                var_groups[var_name].append(mid_value)

        # 处理每个变量组
        for var_name, mid_values in var_groups.items():
            if len(mid_values) <= max_display:
                var_list = f"{var_name}: [{', '.join(mid_values)}]"
            else:
                displayed_mids = mid_values[:max_display]
                var_list = f"{var_name}: [{', '.join(displayed_mids)}, ...] (total {len(mid_values)} items)"
            formatted_parts.append(var_list)

        # 处理不带变量名前缀的MID
        if regular_mids:
            if len(regular_mids) <= max_display:
                regular_list = ", ".join(regular_mids)
            else:
                displayed_regular = regular_mids[:max_display]
                regular_list = ", ".join(displayed_regular) + f", ... (total {len(regular_mids)} items)"
            formatted_parts.append(regular_list)

        return "; ".join(formatted_parts)
    
    def format_function_with_truncated_mids(self, function_str: str) -> str:
        # 匹配JOIN命令中的MID列表模式
        pattern = r"(expression\d+ = JOIN\('[^']+', ')([^']+)('\))"
        
        def replace_mid_list(match):
            prefix = match.group(1)
            mid_list_str = match.group(2)
            suffix = match.group(3)
            
            # 检查是否包含MID列表（多个MID用逗号分隔）
            if ',' in mid_list_str and ('m.' in mid_list_str or 'g.' in mid_list_str):
                mids = [mid.strip() for mid in mid_list_str.split(',')]
                if len(mids) > 3:
                    # 只保留前3个MID，其余用省略号表示
                    truncated_mids = mids[:3]
                    return f"{prefix}{', '.join(truncated_mids)}, ...{suffix}"
            
            return match.group(0)
        
        return re.sub(pattern, replace_mid_list, function_str)
    
    def build_enriched_info_block(self, execution_result, sexpr_result, mid_list: List[str], action_index: int = None, sample_id: int = None, is_final_turn: bool = False) -> str:
        """Build enriched information block for successful execution"""
        info_lines = []
        
        # Determine which relation state to use
        selected_relation = None
        ranked_top5 = None

        # 从 action_processor 中按 (sample_id, action_index) 取出 per-action 关系状态
        action_selected_relation, action_candidate_relations, action_ranked_top5, action_state = self.action_processor._get_action_relation_state(sample_id, action_index)

        # Use action-based state
        selected_relation = action_selected_relation
        ranked_top5 = action_ranked_top5

        # selected_best_relation
        # 这里加入精简日志，帮助定位是哪类 action 把 selected_relation 写成了 str/dict 等非 RelationCandidate 类型
        if selected_relation is not None and not hasattr(selected_relation, "relation_name"):
            raise ValueError(
                f"[ENRICHED-INFO] Non-RelationCandidate selected_relation detected | "
                f"type={type(selected_relation)}, value={selected_relation!r}, "
                f"sample_id={sample_id}, action_index={action_index}, state={action_state}"
            )

        if selected_relation is not None:
            rel = selected_relation

            # CONDITIONAL BREAKPOINT: Check if selected relation is not in function_state
            # Get function_state from sexpr_result if available
            # function_state = getattr(sexpr_result, 'function_sequence', None)
            # if function_state:
            #     function_state_str = ' '.join(function_state)
            #     selected_relation_name = getattr(rel, 'relation_name', str(rel))
                
                # Check if selected relation name is NOT in function_state
                # if selected_relation_name not in function_state_str:
                #     import threading
                #     current_thread = threading.current_thread()
                    
                #     print("\n🚨 CONDITIONAL BREAKPOINT TRIGGERED 🚨")
                #     print(f"Selected relation '{selected_relation_name}' NOT found in function_state!")
                #     print(f"Function state: {function_state}")
                #     print(f"Action index: {action_index}")
                #     print(f"Sample ID: {sample_id}")
                #     print(f"Selected relation: {rel}")
                #     print(f"Current thread: {current_thread.name} (ID: {current_thread.ident})")
                #     print(f"Per-action states: {self._per_action_relation_states}")
                #     print(f"Last selected relation: {self._last_selected_relation}")
                    
                #     # Check action-based state (should always be available in current system)
                #     if (hasattr(self, 'action_processor') and
                #         hasattr(self.action_processor, '_get_action_relation_state')):
                #         action_selected_relation, action_candidate_relations, action_ranked_top5, action_state = self.action_processor._get_action_relation_state(sample_id, action_index)
                #         print(f"Action-based selected relation: {action_selected_relation}")
                #         print(f"Action-based state: {action_state}")
                #         print(f"Action-based candidate relations: {action_candidate_relations}")
                #         print(f"Action-based ranked top5: {action_ranked_top5}")
                #     else:
                #         print("Action-based state not available (this should not happen)")
                    
                #     # Additional debugging: check what relations ARE in function_state
                #     print("\n🔍 DEBUGGING INFO:")
                #     print("Relations found in function_state:")
                #     for i, func in enumerate(function_state):
                #         print(f"  {i}: {func}")
                    
                #     # Check if any relation from function_state matches selected relation
                #     found_match = False
                #     for func in function_state:
                #         if selected_relation_name in func:
                #             print(f"  ✅ Found match: '{selected_relation_name}' in '{func}'")
                #             found_match = True
                    
                #     if not found_match:
                #         print(f"  ❌ No match found for '{selected_relation_name}'")
                    
                #     print("=" * 80)
                #      
            
            # 兼容 RelationCandidate / dict / str 等多种类型，避免 AttributeError
            rel_name = None
            rel_score = None

            # 提取关系名
            if hasattr(rel, "relation_name"):
                rel_name = rel.relation_name
            elif isinstance(rel, dict) and "relation_name" in rel:
                rel_name = rel.get("relation_name")
            else:
                rel_name = str(rel)

            # 提取得分
            if hasattr(rel, "score"):
                rel_score = getattr(rel, "score", None)
            elif isinstance(rel, dict) and "score" in rel:
                rel_score = rel.get("score")

            entity_hint = action_state.get('entity_argument')
            relation_hint = action_state.get('relation_prompt')
            if float(rel_score) < 0.98:
                info_lines.append(f"No relation matched {relation_hint} connected to {entity_hint}; Selected the most similar relation {rel_name} to execute, score={float(rel_score):.3f}")

        # Show top-5 most similar relations and guidance (only when threshold met, a relation was selected,
        # and the best score is not already very high)
        show_top5 = False
        if selected_relation is not None and ranked_top5 is not None:
            best_score = getattr(selected_relation, 'score', None)
            # Only show top-5 when score is not strong enough (<= 0.95)
            show_top5 = (best_score is None) or (best_score <= 0.95)

        if show_top5:
            chosen = None
            if selected_relation is not None:
                chosen = getattr(selected_relation, 'relation_name', str(selected_relation))
            top5_list = [r for r in (ranked_top5 or []) if r and r != chosen]
            
            # Filter relations if action_processor is available
            top5_list = self.action_processor._filter_prompt_relations(top5_list)
            
            # Use min to avoid index out of range, show up to 5 alternatives
            display_count = min(5, len(top5_list))
            top5_display = ", ".join(top5_list[:display_count]) if top5_list else ""
            chosen_display = f"{chosen}" if chosen else "(not selected)"
            _msg = (
                f"Based on your relation_arg, the most similar adjacent relation \"{chosen_display}\" was selected for execution. "
                f"The other top-5 most similar relations are [ {top5_display} ]. "
                "If the current best match does not satisfy your intent, please choose one from the top-5 list by quoting its exact relation id. "
                "If none of the top-5 fit, change your relation_arg to trigger a new similarity match and obtain a different candidate list."
            )
            info_lines.append(_msg)
        
        # executed definitions (do not assume expression id); include actual function strings if available
        if getattr(sexpr_result, "function_sequence", None):
            info_lines.append("functions:")
            for fs in sexpr_result.function_sequence:
                # 格式化函数字符串，截断JOIN命令中的MID列表
                formatted_fs = self.format_function_with_truncated_mids(fs)
                info_lines.append(formatted_fs)
        else:
            # fallback: show the final s-expression only
            info_lines.append(f"sexpr: {sexpr_result.sexpr}")
        
        # MID list - 优先显示精简的MID列表；若为空再显示原始详细结果
        if mid_list:
            info_lines.append("result_mid_list: [ " + self.format_mid_list_for_display(mid_list) + " ]")
        else:
            info_lines.append("result_mid_list: []")
            # 仅在没有MID列表时，补充显示原始结果，避免信息重复
            results_str = SPARQLExecutionManager.results_to_string(execution_result.results)
            truncated_result = self.truncate_result(results_str)
            info_lines.append(truncated_result)
        
        # Detect CVT entities and add information if found
        cvt_mids = self.detect_cvt_mids(mid_list)
        if cvt_mids:
            cvt_info = self._build_cvt_info(cvt_mids)
            if cvt_info:
                info_lines.append("\n" + cvt_info)
        
        # 构建信息块并添加引导提示
        info_block = "<information>\n" + "\n".join(info_lines) + "\n</information>"
        
        # 使用辅助函数添加引导提示
        return self._add_guidance_prompt(info_block, is_final_turn=is_final_turn)
    
    def detect_cvt_mids(self, mid_list: List[str]) -> List[str]:
        """Detect CVT entities and return pure MID list. Returns empty list if not CVT."""
        if not mid_list:
            return []
            
        # Check first entry - if it lacks name, entire list is CVT
        first_entry = mid_list[0]
        
        # Parse the first entry to extract the MID part
        if ':' in first_entry and not first_entry.startswith('m.') and not first_entry.startswith('g.'):
            # Format: "x:m.043zb7r" or "x:m.043zb7r (Name)"
            _, mid_part = first_entry.split(':', 1)
            mid_part = mid_part.strip()
        else:
            # Format: "m.043zb7r" or "m.043zb7r (Name)"
            mid_part = first_entry.strip()
            
        # If first entry has no name (no parentheses), entire list is CVT
        if not ('(' in mid_part and ')' in mid_part):
            # Extract all pure MIDs from the list
            cvt_mids = []
            mid_pattern = re.compile(r'([mg]\.[A-Za-z0-9_]+)')
            for entry in mid_list:
                match = mid_pattern.search(entry)
                if match:
                    cvt_mids.append(match.group(1))
            return list(set(cvt_mids))  # Remove duplicates
        
        return []  # Not a CVT list
    
    def _query_entity_type_and_relations(self, entity_mid: str) -> Tuple[Optional[str], List[str]]:
        """Query entity type and candidate relations"""
        entity_type = None
        candidate_relations = []
        
        if self.relation_retrieval and hasattr(self.relation_retrieval, 'sparql_manager'):
            # 查询实体类型
            type_query = f"""
            PREFIX ns: <http://rdf.freebase.com/ns/>
            SELECT DISTINCT ?type WHERE {{
                ns:{entity_mid} ns:type.object.type ?type .
            }}
            """
            
            type_results = self.relation_retrieval.sparql_manager.execute_batch([type_query])
            if type_results and "results" in type_results and type_results["results"]:
                result = type_results["results"][0]
                if isinstance(result, dict) and "results" in result and result["results"]:
                    for type_item in result["results"]:
                        if isinstance(type_item, dict) and 'type' in type_item:
                            type_value = type_item['type']
                            if isinstance(type_value, str) and not type_value.startswith('type.'):
                                entity_type = type_value
                                break
            
            # 获取候选关系
            function_state = [f"expression1 = START('{entity_mid}')"]
            relations = self.relation_retrieval.get_candidate_relations(function_state)
            candidate_relations = [rel_name for rel_name, rel_id in relations[:10]]  # 取前10个关系
        
        return entity_type, candidate_relations
    
    def _build_cvt_info(self, cvt_mids: List[str]) -> str:
        """Build CVT entity information prompt in English"""
        if not cvt_mids or not self.relation_retrieval:
            return ""
        
        cvt_info_lines = []
        
        # Only query the first CVT entity since all CVT entities from the same find_relation
        # query will have the same type and relations
        first_entity_mid = cvt_mids[0]
        entity_type, relations = self._query_entity_type_and_relations(first_entity_mid)
        
        # Build summary message
        total_entities = len(cvt_mids)
        cvt_info_lines.append(f"{total_entities} entities have no name attribute and may represent CVT entities. These entities do not represent a topic, so they are generally not used as answers. You should continue querying from these entities.")
        
        # Add type and relation information (applies to all CVT entities)
        type_display = f"({entity_type})" if entity_type else "(unknown type)"
        
        if total_entities == 1:
            # Single entity - show full details
            if relations:
                relation_list = ", ".join(relations)
                cvt_info_lines.append(f"{first_entity_mid} {type_display} available relations: [{relation_list}]")
            else:
                cvt_info_lines.append(f"{first_entity_mid} {type_display} available relations: [unable to retrieve relation information]")
        else:
            # Multiple entities - show summary with sample entities
            sample_entities = ", ".join(cvt_mids[:3])  # Show first 3 as examples
            if total_entities > 3:
                sample_entities += f", ...{total_entities-3} more"
            
            if relations:
                relation_list = ", ".join(relations)
                cvt_info_lines.append(f"{sample_entities} {type_display} available relations: [{relation_list}]")
            else:
                cvt_info_lines.append(f"{sample_entities} {type_display} available relations: [unable to retrieve relation information]")
        
        return "\n".join(cvt_info_lines)
