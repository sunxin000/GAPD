"""
S-Expression State Management Module
Manages persistent state for samples during S-Expression processing
"""

import logging
import os
import re
from typing import Dict, List, Optional, Tuple

from ..sexpr.limit import duplicated_relation_list, ping_pong_relations

# 允许在ping-pong结果后使用的JOIN关系列表
PING_PONG_CONTINUATION_ALLOWED_RELATIONS = {
    "amusement_parks.ride.designer",
    "fictional_universe.fictional_character.quotations", 
    "internet.website_owner.websites_owned",
    "people.profession.part_of_professional_field",
    "religion.religion.practices",
    "spaceflight.bipropellant_rocket_engine.engine_cycle",
    "spaceflight.rocket_engine_fuel.rocket_engines"
}

PING_PONG_FILTER_ENABLED = os.getenv("ENABLE_PING_PONG_FILTER", "0").lower() in ("1", "true", "yes")

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "DEBUG"))  # Set to DEBUG to show ping-pong detection


class SExprStateManager:
    """
    Manages persistent state for samples during S-Expression processing
    Handles function states, expression counters, and entity information
    """
    
    def __init__(self):
        # 持久化function_state管理 - 参考KBQA-o1的AgentState机制
        self._sample_function_states: Dict[int, List[str]] = {}  # 存储每个样本的function_state
        self._sample_expression_counters: Dict[int, int] = {}  # 存储每个样本的expression计数器
        self._sample_entities: Dict[int, List[Tuple[str, str]]] = {}  # 存储每个样本的实体信息
        # 存储每个样本的原始输入/提示文本（用于错误上下文）
        self._sample_prompts: Dict[int, str] = {}
    
    def get_sample_function_state(self, sample_id: int) -> List[str]:
        """获取样本的持久化function_state"""
        return self._sample_function_states.get(sample_id, [])
    
    def update_sample_function_state(self, sample_id: int, function_string: str):
        """更新样本的function_state"""
        if sample_id not in self._sample_function_states:
            self._sample_function_states[sample_id] = []
        
        # 检查join链深度限制
        if self._is_join_operation(function_string):
            current_join_depth = self._calculate_join_chain_depth(sample_id, function_string)
            if current_join_depth > 3:
                logger.debug(f"[SEXPR] Skip join operation - depth {current_join_depth} exceeds limit of 3 for sample {sample_id}: {function_string}")
                return
            
            # 检查是否为ping-pong结果的后续JOIN操作
            if PING_PONG_FILTER_ENABLED and self._is_ping_pong_result(sample_id, function_string):
                curr_rel_raw = self._extract_join_relation(function_string)
                if not self._is_relation_allowed_for_ping_pong_continuation(curr_rel_raw):
                    logger.debug(f"[SEXPR] Skip ping-pong continuation - relation {curr_rel_raw} not in allowed list for sample {sample_id}: {function_string}")
                    return
                else:
                    logger.debug(f"[SEXPR] Allow ping-pong continuation - relation {curr_rel_raw} in allowed list for sample {sample_id}: {function_string}")
        
        # 去重策略：检查引用的expression是否与当前JOIN操作构成重复关系
        if self._sample_function_states[sample_id]:
            # 提取当前JOIN操作的信息
            curr_rel_raw = self._extract_join_relation(function_string)
            curr_base, curr_rev = self._normalize_relation_and_dir(curr_rel_raw) if curr_rel_raw else (None, None)
            curr_entity = self._extract_join_entity(function_string)
            
            # 如果是JOIN操作，检查与引用的expression的重复关系
            if curr_base and curr_entity:
                # 获取被引用的function
                referenced_function = self._get_referenced_function(sample_id, curr_entity)
                
                # 如果找到了被引用的function，检查是否构成重复关系
                if referenced_function and self._is_join_operation(referenced_function):
                    ref_rel_raw = self._extract_join_relation(referenced_function)
                    ref_base, ref_rev = self._normalize_relation_and_dir(ref_rel_raw) if ref_rel_raw else (None, None)
                    
                    # 检查是否使用相同的基础关系
                    if ref_base == curr_base:
                        if curr_rev != ref_rev:
                            # 方向相反 -> 仅当基础关系在 ping_pong_relations 中允许
                            if PING_PONG_FILTER_ENABLED and curr_base not in ping_pong_relations:
                                logger.debug(f"[SEXPR] Skip ping-pong duplicate for sample {sample_id}: {function_string}")
                                return
                        else:
                            # 同向同关系 -> 需要检查是否在白名单中
                            if not self._is_relation_allowed_duplicate(curr_rel_raw):
                                logger.debug(f"[SEXPR] Skip same-direction duplicate for sample {sample_id}: {function_string}")
                                return
            
            # 检查RHS重复（非JOIN操作或不同expression变量的情况）
            referenced_step = self._sample_function_states[sample_id][-1]
            if self._is_duplicate_rhs(referenced_step, function_string):
                # 非JOIN操作直接去重
                if curr_base is None:
                    logger.debug(f"[SEXPR] Skip non-JOIN duplicate for sample {sample_id}: {function_string}")
                    return
                # JOIN操作检查是否在白名单中
                if not self._is_relation_allowed_duplicate(curr_rel_raw):
                    logger.debug(f"[SEXPR] Skip JOIN duplicate (not whitelisted) for sample {sample_id}: {function_string}")
                    return

        self._sample_function_states[sample_id].append(function_string)
        logger.debug(f"[SEXPR] Updated function_state for sample {sample_id}: {function_string}")

    def _extract_join_relation(self, function_string: str) -> Optional[str]:
        """从函数步骤字符串中提取 JOIN 的 relation 字符串；若不是 JOIN（FIND_RELATION）步骤则返回 None。
        支持形如：
          "expression = JOIN('relation.name', expression)"
          "expression = JOIN('(R relation.name)', 'm.xxx')"
        """
        if not isinstance(function_string, str):
            return None
        # 仅当包含 JOIN( 才尝试匹配
        if 'JOIN(' not in function_string:
            return None
        # 提取 JOIN('...'
        match = re.search(r"JOIN\(\s*([\'\"])\s*([^'\"]+)\s*\1\s*,", function_string)
        if not match:
            return None
        relation = match.group(2).strip()
        return relation if relation else None

    def _extract_join_entity(self, function_string: str) -> Optional[str]:
        """从函数步骤字符串中提取 JOIN 的 entity 参数；若不是 JOIN（FIND_RELATION）步骤则返回 None。
        支持形如：
          "expression = JOIN('relation.name', 'm.xxx')"
          "expression = JOIN('(R relation.name)', expression1)"
        """
        if not isinstance(function_string, str):
            return None
        # 仅当包含 JOIN( 才尝试匹配
        if 'JOIN(' not in function_string:
            return None
        # 提取 JOIN 的第二个参数
        match = re.search(r"JOIN\(\s*[\'\"]\s*[^\'\"]+\s*[\'\"]\s*,\s*([^)]+)\)", function_string)
        if not match:
            return None
        entity = match.group(1).strip()
        return entity if entity else None

    def _rhs_of(self, step: str) -> str:
        """返回步骤等号右侧的字符串（去首尾空白）。若格式异常则返回原串的去空白版本。"""
        if not isinstance(step, str):
            return ""
        parts = step.split('=', 1)
        if len(parts) == 2:
            return parts[1].strip()
        return step.strip()

    def _normalize_relation_and_dir(self, relation: str) -> Tuple[str, bool]:
        """规格化关系为 (基础关系名, 是否反向)。
        例如："media_common.media_genre.parent_genre" -> ("media_common.media_genre.parent_genre", False)
              "(R media_common.media_genre.parent_genre)" -> ("media_common.media_genre.parent_genre", True)
        """
        rel = relation.strip()
        if rel.startswith('(R ') and rel.endswith(')'):
            return rel[3:-1].strip(), True
        return rel, False

    def _is_same_or_reverse_relation(self, rel_a: str, rel_b: str) -> bool:
        base_a, _ = self._normalize_relation_and_dir(rel_a)
        base_b, _ = self._normalize_relation_and_dir(rel_b)
        return base_a == base_b
    
    def _is_duplicate_rhs(self, prev_step: str, curr_step: str) -> bool:
        prev_rhs = self._rhs_of(prev_step)
        curr_rhs = self._rhs_of(curr_step)
        return bool(prev_rhs and curr_rhs and prev_rhs == curr_rhs)

    def _is_relation_allowed_duplicate(self, relation_raw: Optional[str]) -> bool:
        """检查 JOIN 重复是否允许：relation 在 duplicated_relation_list 白名单中。
        同时兼容传入 '(R rel)' 和基础 rel 两种形式。
        """
        if not relation_raw:
            return False
        base, _ = self._normalize_relation_and_dir(relation_raw)
        return (relation_raw in duplicated_relation_list) or (base in duplicated_relation_list)
    
    def _is_join_operation(self, function_string: str) -> bool:
        """检查函数字符串是否为JOIN操作"""
        if not isinstance(function_string, str):
            return False
        return 'JOIN(' in function_string
    
    def _calculate_join_chain_depth(self, sample_id: int, function_string: str) -> int:
        """计算当前JOIN操作在链中的深度
        
        根据用户示例：
        "expression = START('m.0pm2fgf')",
        "expression = JOIN('(R opera.opera_production.designers)', expression)",
        "expression = JOIN('(R opera.opera_designer_gig.design_role)', expression)",
        
        路径长度为2，因为有两个连续的JOIN操作都使用了前面的expression结果。
        
        Args:
            sample_id: 样本ID
            function_string: 当前函数字符串
            
        Returns:
            JOIN链的深度
        """
        if sample_id not in self._sample_function_states:
            return 0
        
        function_states = self._sample_function_states[sample_id]
        if not function_states:
            return 0
        
        # 检查当前函数是否为JOIN操作且使用expression变量
        if not self._is_join_operation(function_string):
            return 0
        
        # 提取当前函数使用的expression变量（支持expression, expression1, expression2等）
        current_expr = self._extract_expression_variable(function_string)
        if not current_expr:
            return 0
        
        # 计算连续的JOIN操作链深度
        # 从最新的function开始向前查找连续的JOIN操作
        consecutive_joins = 0
        
        # 遍历现有的function_states（从最新到最旧）
        for func in reversed(function_states):
            if self._is_join_operation(func):
                # 检查这个函数是否使用相同的expression变量
                func_expr = self._extract_expression_variable(func)
                if func_expr == current_expr:
                    # 这是一个使用相同expression变量的JOIN操作
                    consecutive_joins += 1
                else:
                    # 使用不同的expression变量，停止计数
                    break
            else:
                # 遇到非JOIN操作，停止计数
                break
        
        # 加上当前操作
        return consecutive_joins + 1
    
    def _extract_expression_variable(self, function_string: str) -> Optional[str]:
        """从函数字符串中提取expression变量名
        
        例如：
        "expression = JOIN('relation', expression)" -> "expression"
        "expression2 = JOIN('relation', expression2)" -> "expression2"
        "expression1 = JOIN('relation', expression1)" -> "expression1"
        
        Args:
            function_string: 函数字符串
            
        Returns:
            expression变量名，如果未找到则返回None
        """
        if not isinstance(function_string, str):
            return None
        
        # 使用正则表达式匹配expression变量
        import re
        # 匹配形如 "expression = JOIN('...', expression)" 或 "expression2 = JOIN('...', expression2)"
        match = re.search(r'expression(\d*)\s*=\s*JOIN\([^,]+,\s*(expression\d*)\)', function_string)
        if match:
            return match.group(2)  # 返回第二个expression变量（参数中的）
        
        return None
    

    
    def _get_referenced_function(self, sample_id: int, expression_id: str) -> Optional[str]:
        """获取被引用的expression ID出现的最后一行function
        
        Args:
            sample_id: 样本ID
            expression_id: 要查找的expression ID
            
        Returns:
            被引用的function字符串，如果未找到则返回None
        """
        if sample_id not in self._sample_function_states:
            return None
            
        # 从最新的function开始向前查找，找到该expression ID出现的最后一行
        for hist_func in reversed(self._sample_function_states[sample_id]):
            if (hist_func.startswith(expression_id + ' = ')):
                return hist_func
        
        return None
    
    def _is_ping_pong_result(self, sample_id: int, function_string: str) -> bool:
        """检测当前JOIN操作是否使用了ping-pong结果
        
        检查当前JOIN操作的输入表达式是否来自任何ping-pong操作的结果。
        
        Args:
            sample_id: 样本ID
            function_string: 当前函数字符串
            
        Returns:
            True如果当前JOIN使用了ping-pong结果，False否则
        """
        if not self._is_join_operation(function_string):
            return False
            
        if sample_id not in self._sample_function_states:
            return False
            
        function_states = self._sample_function_states[sample_id]
        if len(function_states) < 2:
            return False
            
        # 获取当前JOIN操作的输入表达式
        current_input_expr = self._extract_join_entity(function_string)
        if not current_input_expr:
            return False
            
        # 获取被引用的function
        referenced_function = self._get_referenced_function(sample_id, current_input_expr)
        
        # 如果找到了被引用的function且它是JOIN操作，检查是否构成ping-pong
        if referenced_function and self._is_join_operation(referenced_function):
            # 检查被引用的function是否来自ping-pong操作
            # 递归检查被引用function的输入是否来自ping-pong
            ref_input_expr = self._extract_join_entity(referenced_function)
            if ref_input_expr:
                # 获取被引用function的输入expression的定义
                ref_referenced_function = self._get_referenced_function(sample_id, ref_input_expr)
                
                # 如果被引用function的输入也是JOIN操作，检查是否构成ping-pong
                if (ref_referenced_function and 
                    self._is_join_operation(ref_referenced_function) and
                    self._is_join_operation(referenced_function)):
                    
                    # 检查是否构成ping-pong（相同基础关系，方向相反）
                    ref_rel = self._extract_join_relation(referenced_function)
                    ref_ref_rel = self._extract_join_relation(ref_referenced_function)
                    
                    if ref_rel and ref_ref_rel:
                        ref_base, ref_rev = self._normalize_relation_and_dir(ref_rel)
                        ref_ref_base, ref_ref_rev = self._normalize_relation_and_dir(ref_ref_rel)
                        
                        # 检查是否为基础关系相同但方向相反的ping-pong
                        if ref_base == ref_ref_base and ref_rev != ref_ref_rev:
                            return True
                    
        return False
    
    def _is_relation_allowed_for_ping_pong_continuation(self, relation_raw: Optional[str]) -> bool:
        """检查关系是否允许用于ping-pong结果的后续JOIN操作
        
        Args:
            relation_raw: 关系字符串，可能包含反向标记如"(R relation.name)"
            
        Returns:
            True如果关系在允许列表中，False否则
        """
        if not relation_raw:
            return False
            
        # 提取基础关系名
        base_relation, _ = self._normalize_relation_and_dir(relation_raw)
        
        # 检查是否在允许列表中
        return base_relation in PING_PONG_CONTINUATION_ALLOWED_RELATIONS
    
    def get_next_expression_id(self, sample_id: int) -> str:
        """获取样本的下一个expression ID"""
        if sample_id not in self._sample_expression_counters:
            self._sample_expression_counters[sample_id] = 0
        self._sample_expression_counters[sample_id] += 1
        return str(self._sample_expression_counters[sample_id])
    
    def get_current_expression_id(self, sample_id: int) -> str:
        """获取样本的当前expression ID"""
        if sample_id not in self._sample_expression_counters:
            return "1"
        return str(self._sample_expression_counters[sample_id])

    def set_current_expression_id(self, sample_id: int, expression_id: int | str):
        """显式设置样本当前的 expression ID（用于 MERGE 回退或对齐外部引用）。"""
        try:
            eid = int(expression_id)
        except Exception:
            # 空字符串或异常则不更新
            return
        if eid < 1:
            return
        self._sample_expression_counters[sample_id] = eid
    
    def set_sample_entities(self, sample_id: int, entities: List[Tuple[str, str]]):
        """设置样本的实体信息"""
        self._sample_entities[sample_id] = entities
    
    def get_sample_entities(self, sample_id: int) -> List[Tuple[str, str]]:
        """获取样本的实体信息"""
        return self._sample_entities.get(sample_id, [])
    
    def clear_sample_state(self, sample_id: int):
        """清除样本的状态（用于重置或完成时）"""
        if sample_id in self._sample_function_states:
            del self._sample_function_states[sample_id]
        if sample_id in self._sample_expression_counters:
            del self._sample_expression_counters[sample_id]
        if sample_id in self._sample_entities:
            del self._sample_entities[sample_id]
        if sample_id in self._sample_prompts:
            del self._sample_prompts[sample_id]
        logger.debug(f"[SEXPR] Cleared state for sample {sample_id}")
    
    def initialize_batch_states(self, batch_size: int):
        """初始化一批样本的状态"""
        for i in range(batch_size):
            self.clear_sample_state(i)
            logger.debug(f"[SEXPR] Initialized persistent state for sample {i}")
    
    def get_all_function_states(self) -> Dict[int, List[str]]:
        """获取所有样本的function_state（用于调试）"""
        return self._sample_function_states.copy()
    
    def get_all_expression_counters(self) -> Dict[int, int]:
        """获取所有样本的expression计数器（用于调试）"""
        return self._sample_expression_counters.copy()
    
    def rollback_sample_function_state(self, sample_id: int, num_functions: int = 1):
        """回滚样本的function_state（移除最近的num_functions个functions）"""
        if sample_id in self._sample_function_states and self._sample_function_states[sample_id]:
            functions_to_remove = min(num_functions, len(self._sample_function_states[sample_id]))
            removed_functions = []
            for _ in range(functions_to_remove):
                if self._sample_function_states[sample_id]:
                    removed_functions.append(self._sample_function_states[sample_id].pop())
            
            # Also rollback expression counter
            if sample_id in self._sample_expression_counters:
                self._sample_expression_counters[sample_id] = max(0, 
                    self._sample_expression_counters[sample_id] - functions_to_remove)
            
            logger.debug(f"[SEXPR] Rolled back {functions_to_remove} functions for sample {sample_id}: {removed_functions}")
            return removed_functions
        return []

    # ===== Prompt text storage (for error context) =====
    def set_sample_prompt(self, sample_id: int, prompt_text: str):
        """设置样本的原始输入/提示文本（用于错误上下文持久化）。"""
        if isinstance(prompt_text, str) and prompt_text:
            self._sample_prompts[sample_id] = prompt_text

    def get_sample_prompt(self, sample_id: int) -> str:
        """获取样本的原始输入/提示文本。"""
        return self._sample_prompts.get(sample_id, "")