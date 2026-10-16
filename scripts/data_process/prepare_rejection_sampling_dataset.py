#!/usr/bin/env python3
"""
Prepare dataset for rejection sampling with action hints
为拒绝采样准备数据集，添加 action hint 作为参考信息

核心思路：
1. 读取原始 RL 数据集（包含 function_list 和 ground truth）
2. 从 function_list 提取 action sequence 作为 hint
3. 将 hint 插入到 prompt 中（entities 之后，question 之前）
4. 生成新的数据集用于 rejection sampling

Hint 位置：
```
Available Actions: ...

Candidate Entities: [...]

**Reference Action Sequence (for guidance):**
Action1: Find_relation [...]
Action2: Merge [...]
...

Question: What is ...?
```
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# 这些依赖在测试 extract_action_hint_from_response 时不需要，做延迟/容错导入
try:
    import datasets  # type: ignore
except Exception:  # pragma: no cover
    datasets = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):  # type: ignore
        return x

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


def extract_action_hint_from_response(response: str, extra_info: dict = None) -> str:
    """从 response 中提取 <action> 标签内的 action 序列，按严格的 expression 管理器规则进行转换。

    规则（符合用户期望）：
    - 仅 Extract_entity 会使 expression_id 递增（expression1, expression2, ...）。
    - Merge 的结果表达式编号为其两个输入表达式编号中的较小者（等价于指针 -1，在常见相邻合并场景）。
    - 其他操作不会改变当前 expression 指针。
    - 当 Find_relation/Order 等目标缺省时：
      - 若紧随 Extract_entity，则用该实体 MID 作为左参数；
      - 否则使用当前 expression{pointer} 作为左参数。
    - 删除 Extract_entity 和 Finish，不出现在输出序列中。

    Args:
        response: 字段值，内部包含 <action>...<\action>
        extra_info: 需要包含 extracted_entities 或 candidate_entities，形如 [[name, mid], ...]

    Returns:
        转换后的动作序列（多行字符串）。
    """
    # 提取 <action> 内容
    action_match = re.search(r'<action>\s*(.*?)\s*</action>', response, re.DOTALL | re.IGNORECASE)
    if not action_match:
        return ""

    raw_lines = [ln.strip() for ln in action_match.group(1).strip().split('\n') if ln.strip()]

    # 实体来源：extra_info.extracted_entities 或 candidate_entities（按 Extract_entity 顺序消费）
    # 注意：parquet 解析后字段可能是 numpy array，禁止用 "or" 连接以免触发歧义布尔判断
    candidate_entities = []
    if isinstance(extra_info, dict):
        cand1 = extra_info.get('extracted_entities', None)
        cand2 = extra_info.get('candidate_entities', None)

        # 提前转换为 Python list（若支持 tolist）
        if cand1 is not None and hasattr(cand1, 'tolist'):
            try:
                cand1 = cand1.tolist()
            except Exception:
                pass
        if cand2 is not None and hasattr(cand2, 'tolist'):
            try:
                cand2 = cand2.tolist()
            except Exception:
                pass

        def is_non_empty(seq):
            return seq is not None and (not hasattr(seq, '__len__') or len(seq) > 0)

        if is_non_empty(cand1):
            candidate_entities = cand1
        elif is_non_empty(cand2):
            candidate_entities = cand2
        else:
            candidate_entities = []

    def get_entity_mid_by_index(idx: int, fallback_name: str) -> str:
        if 0 <= idx < len(candidate_entities):
            tup = candidate_entities[idx]
            if hasattr(tup, 'tolist'):
                tup = tup.tolist()
            if isinstance(tup, (list, tuple)):
                if len(tup) >= 2 and tup[1]:
                    return tup[1]
                if len(tup) >= 1 and tup[0]:
                    return tup[0]
        return fallback_name

    # 状态：expression 管理器
    expr_counter = 0  # 仅在 Extract_entity 时 +1
    last_was_extract = False
    extract_index = 0  # 用于从 candidate_entities 顺序取 MID
    expr_to_entity = {}  # expressionN -> entity_mid

    # 解析辅助
    def parse_bracket_content(s: str) -> List[str]:
        # 拆分 [ ... ] 内 "a | b | c" 的片段，保持简单分割
        parts = [p.strip() for p in s.split('|')]
        return parts

    def normalize_expr_token(tok: str) -> str:
        """将 'expression' 替换为当前 expression 指针，'expressionN' 保持原样。其他原样返回。"""
        if re.fullmatch(r'expression', tok):
            if expr_counter <= 0:
                return 'expression1'  # 安全兜底
            return f'expression{expr_counter}'
        m = re.fullmatch(r'expression(\d+)', tok)
        if m:
            return f'expression{int(m.group(1))}'
        return tok

    def expr_id_number(tok: str) -> int:
        m = re.fullmatch(r'expression(\d+)', tok)
        return int(m.group(1)) if m else -1

    def is_mid(s: str) -> bool:
        return isinstance(s, str) and re.fullmatch(r"[mg]\.[A-Za-z0-9_]+", s or "") is not None

    def is_literal_value(value: str) -> bool:
        if not isinstance(value, str):
            return False
        v = value.strip()
        if not v:
            return False
        if '^^' in v or '@' in v:
            return True
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            return True
        return re.fullmatch(r'-?\d+(?:\.\d+)?', v) is not None

    def is_ontology_identifier(value: str) -> bool:
        if not isinstance(value, str):
            return False
        v = value.strip()
        if not v:
            return False
        return re.fullmatch(
            r"(?:[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*|base(?:\.[A-Za-z_][A-Za-z0-9_]*){2,})",
            v,
        ) is not None

    def get_expr_display_and_id(
        token: str,
        inline_literals: bool = True,
        inline_ontology: bool = True,
    ) -> Tuple[str, int]:
        """给定表达式 token，返回用于展示的字符串与其表达式 id。
        规则：
        - token == 'expression' -> id = expr_counter；若允许且映射为字面量/ontology，则内联。
        - token == 'expressionN' -> id = N；若允许且映射为字面量/ontology，则内联。
        - 其他（非表达式）-> 原样展示，id = -1。
        通过 inline_literals / inline_ontology 控制可被展开的类型。
        """
        token = token.strip()
        if token == 'expression':
            eid = expr_counter if expr_counter > 0 else 1
            mapped = expr_to_entity.get(f'expression{eid}')
            if mapped and not is_mid(mapped):
                if inline_literals and is_literal_value(mapped):
                    return mapped, eid
                if inline_ontology and is_ontology_identifier(mapped):
                    return mapped, eid
            return f'expression{eid}', eid
        m = re.fullmatch(r'expression(\d+)', token)
        if m:
            eid = int(m.group(1))
            mapped = expr_to_entity.get(f'expression{eid}')
            if mapped and not is_mid(mapped):
                if inline_literals and is_literal_value(mapped):
                    return mapped, eid
                if inline_ontology and is_ontology_identifier(mapped):
                    return mapped, eid
            return f'expression{eid}', eid
        return token, -1

    converted: List[str] = []

    # 比较运算符规范化映射
    _CMP_MAP = {
        # less-than family
        'less than': 'lt',
        'lessthan': 'lt',
        'lt': 'lt',
        # "LESS EQUAL" / "LESS THAN OR EQUAL" 等统一到 le
        'less equal': 'le',
        'lessequal': 'le',
        'less than or equal': 'le',
        'less than or equal to': 'le',
        'lessthanorequal': 'le',
        'le': 'le',
        # greater-than family
        'greater than': 'gt',
        'greaterthan': 'gt',
        'greater than ': 'gt',
        'gt': 'gt',
        # "GREATER EQUAL" / "GREATER THAN OR EQUAL" 等统一到 ge
        'greater equal': 'ge',
        'greaterequal': 'ge',
        'greater than or equal': 'ge',
        'greater than or equal to': 'ge',
        'greaterthanorequal': 'ge',
        'ge': 'ge',
        # equality family
        '=': 'eq',
        'equal': 'eq',
        'equals': 'eq',
        'equal to': 'eq',
        # not-equal family
        'neq': 'ne',
        'not equal': 'ne',
        'not equal to': 'ne',
        '!=': 'ne',
    }

    def normalize_cmp_op(op: str) -> str:
        key = (op or '').strip().lower().replace('_', ' ')
        key = re.sub(r'\s+', ' ', key)
        return _CMP_MAP.get(key, op.lower())

    # 排序操作符规范化映射（ARGMIN/ARGMAX -> MIN/MAX）
    _ORDER_MAP = {
        'argmin': 'MIN',
        'arg max': 'MAX',
        'argmax': 'MAX',
        'min': 'MIN',
        'max': 'MAX',
    }

    def normalize_order_mode(mode: str) -> str:
        key = (mode or '').strip().lower()
        key = re.sub(r'\s+', ' ', key)
        return _ORDER_MAP.get(key, mode.upper())

    for line in raw_lines:
        # 跳过 Finish
        if re.search(r'Action\d+:\s*Finish\s*\[', line):
            last_was_extract = False
            continue

        # 提取动作名和方括号内容
        m = re.match(r'Action(\d+):\s*([A-Za-z_]+)\s*\[\s*(.*?)\s*\]$', line)
        if not m:
            # 不符合格式，忽略此行
            last_was_extract = False
            continue

        action_name = m.group(2)
        content = m.group(3)

        if action_name == 'Extract_entity':
            # 仅更新 expression 管理器，不输出
            entity_name = content.strip()
            expr_counter += 1
            mid = get_entity_mid_by_index(extract_index, entity_name)
            extract_index += 1
            expr_to_entity[f'expression{expr_counter}'] = mid
            last_was_extract = True
            continue

        if action_name == 'Find_relation':
            # 两种形态：
            # 1) 仅 relation
            # 2) entity_or_expr | relation
            parts = parse_bracket_content(content)
            if len(parts) == 1:
                relation = parts[0]
                if last_was_extract and expr_counter > 0 and f'expression{expr_counter}' in expr_to_entity:
                    left = expr_to_entity[f'expression{expr_counter}']
                else:
                    left = f'expression{expr_counter if expr_counter>0 else 1}'
                converted.append(f"Find_relation [ {left} | {relation} ]")
            else:
                left = normalize_expr_token(parts[0])
                relation = parts[1]
                # 如果 left 是 expression，使用规范化结果；如果是 entity/mid 就直接用
                converted.append(f"Find_relation [ {left} | {relation} ]")
            last_was_extract = False
            continue

        if action_name == 'Order':
            parts = parse_bracket_content(content)
            # 形态：mode | relation 或 mode | target | relation
            if len(parts) == 2:
                mode, relation = normalize_order_mode(parts[0]), parts[1]
                if last_was_extract and expr_counter > 0 and f'expression{expr_counter}' in expr_to_entity:
                    target = expr_to_entity[f'expression{expr_counter}']
                else:
                    target = f'expression{expr_counter if expr_counter>0 else 1}'
                converted.append(f"Order [ {mode} | {target} | {relation} ]")
            elif len(parts) == 3:
                mode, target, relation = normalize_order_mode(parts[0]), normalize_expr_token(parts[1]), parts[2]
                converted.append(f"Order [ {mode} | {target} | {relation} ]")
            else:
                # 保底：原样塞回
                converted.append(f"Order [ {content} ]")
            last_was_extract = False
            continue

        if action_name == 'Merge':
            parts = parse_bracket_content(content)
            if len(parts) >= 2:
                # 针对 Merge，保留表达式语义但在可用时内联 ontology 类型（非 MID）。
                # 同时，为了与既有测试对齐：当两个都是 expression token 时，较大 id 在前。
                l_disp, l_id = get_expr_display_and_id(parts[0], inline_literals=False, inline_ontology=True)
                r_disp, r_id = get_expr_display_and_id(parts[1], inline_literals=False, inline_ontology=True)

                # 仅当两侧都是表达式（有 id）时才进行重排；若有一侧是文字/ontology 则保持原顺序
                if l_id != -1 and r_id != -1 and l_disp.startswith('expression') and r_disp.startswith('expression'):
                    if l_id < r_id:
                        l_disp, r_disp = r_disp, l_disp
                        l_id, r_id = r_id, l_id
                # 指针回退到较小的表达式 id（如有两侧 id）
                if l_id != -1 and r_id != -1:
                    expr_counter = min(l_id, r_id)
                elif l_id != -1:
                    expr_counter = l_id
                elif r_id != -1:
                    expr_counter = r_id

                converted.append(f"Merge [ {l_disp} | {r_disp} ]")
            else:
                converted.append(f"Merge [ {content} ]")
            last_was_extract = False
            continue

        if action_name == 'Compare':
            # Compare [ OP | relation | value ]
            parts = parse_bracket_content(content)
            if len(parts) >= 3:
                op_raw, relation, value_raw = parts[0], parts[1], parts[2]
                op = normalize_cmp_op(op_raw)
                # 如果 value 是表达式，尝试内联其文字值（若为非 MID 的字面量），并在使用后取消其内联映射
                value_tok = value_raw.strip()
                value = normalize_expr_token(value_tok)
                # 若明确是 expressionN 或 expression 裸
                vid = -1
                if value_tok == 'expression':
                    vid = expr_counter if expr_counter > 0 else 1
                else:
                    m_expr = re.fullmatch(r'expression(\d+)', value_tok)
                    if m_expr:
                        vid = int(m_expr.group(1))
                if vid != -1:
                    mapped = expr_to_entity.get(f'expression{vid}')
                    if mapped and not is_mid(mapped):
                        value = mapped
                        # 使用后，防止后续 Merge 将其继续当作字面量内联
                        expr_to_entity.pop(f'expression{vid}', None)
                converted.append(f"Compare [ {op} | {relation} | {value} ]")
            elif len(parts) == 2:
                # 常见两段式：前一步 Extract_entity 给出数值，这里补上 value
                op = normalize_cmp_op(parts[0])
                relation = parts[1]
                value = None
                if last_was_extract and expr_counter > 0:
                    maybe = expr_to_entity.get(f'expression{expr_counter}')
                    if maybe is not None:
                        value = maybe
                        # 使用后取消内联，避免 Merge 再次把该 expression 渲染为字面量
                        expr_to_entity.pop(f'expression{expr_counter}', None)
                if value is None:
                    # 兜底：使用当前表达式引用
                    value = f'expression{expr_counter if expr_counter>0 else 1}'
                converted.append(f"Compare [ {op} | {relation} | {value} ]")
            else:
                converted.append(f"Compare [ {content} ]")
            last_was_extract = False
            continue

        if action_name == 'Time_constraint':
            # 不改变 expression 指针，原样透传内容
            converted.append(f"Time_constraint [ {content} ]")
            last_was_extract = False
            continue

        # 其他动作：仅做 expression token 规范化
        def replace_expr_token(mo):
            tok = mo.group(0)
            return normalize_expr_token(tok)

        content_norm = re.sub(r'\bexpression(?:\d+)?\b', replace_expr_token, content)
        converted.append(f"{action_name} [ {content_norm} ]")
        last_was_extract = False

    return '\n'.join(converted)


def create_hint_enhanced_prompt(
    original_prompt: str,
    action_hint: str,
    hint_style: str = "reference",
) -> str:
    """在原始 prompt 中插入 action hint
    
    Args:
        original_prompt: 原始 prompt（包含 entities 和 question）
        action_hint: Action sequence hint
        hint_style: Hint 风格
            - "reference": 作为参考信息（推荐）
            - "hidden": 在 system message 中隐藏
            - "example": 作为示例
    
    Returns:
        增强后的 prompt
    """
    
    # 找到 "Question:" 的位置
    question_match = re.search(r'\n(Question:.*?)$', original_prompt, re.DOTALL)
    if not question_match:
        # 如果找不到 Question，直接附加在最后
        print("Warning: Could not find 'Question:' in prompt, appending hint at the end")
        return f"{original_prompt}\n\n**Reference Action Sequence:**\n{action_hint}\n"
    
    question_part = question_match.group(1)
    before_question = original_prompt[:question_match.start()]
    
    # 根据不同风格插入 hint
    if hint_style == "reference":
        hint_block = f"""
**Reference Action Sequence (for guidance):**
The following is a reference action sequence that can solve this problem. 

IMPORTANT INSTRUCTIONS:
1. Generate actions ONE AT A TIME - do NOT generate all actions at once.
2. Pretend you CANNOT see this reference sequence - generate your own natural reasoning path as if solving from scratch. Your reasoning should be natural and step-by-step.

Reference sequence:
{action_hint}

---

"""
    elif hint_style == "hidden":
        # 这种风格需要在 system message 中处理，这里只是标记
        hint_block = f"\n[HINT_PLACEHOLDER: {action_hint}]\n\n"
    
    elif hint_style == "example":
        hint_block = f"""
**Example Solution Approach:**

{action_hint}

Now, solve the following question using a similar approach:

"""
    else:
        raise ValueError(f"Unknown hint_style: {hint_style}")

    after_question = "\nThe reference already provides a complete action sequence to solve the problem. You only need to provide concise, effective reasoning. Immediately after <think>, issue the next step using <action> so the environment can execute it. Keep your reasoning brief—just explain why you will take the next action."
    
    # 组合：原始内容（到 entities 为止）+ hint + Question
    enhanced_prompt = f"{before_question}\n{hint_block}{question_part}{after_question}"

    return enhanced_prompt


def prepare_rejection_sampling_dataset(
    input_path: str,
    output_path: str,
    hint_style: str = "reference",
    num_samples: int = None,
    skip_no_function_list: bool = False,
) -> Dict[str, Any]:
    """准备用于 rejection sampling 的数据集
    
    Args:
        input_path: 输入的 RL 数据集路径（parquet）
        output_path: 输出路径（parquet）
        hint_style: Hint 风格
        num_samples: 处理样本数（None = 全部）
        skip_no_function_list: 是否跳过没有 function_list 的样本（默认 False，保留用于 RL 训练）
    
    Returns:
        统计信息
    """
    
    print(f"Loading dataset from {input_path}...")
    df = pd.read_parquet(input_path)
    data = df.to_dict('records')
    
    # 限制样本数
    if num_samples:
        data = data[:num_samples]
        print(f"Processing first {num_samples} samples")
    
    processed_data = []
    # 收集 action 转换前后内容，便于调试与审计
    action_conversion_records: List[Dict[str, Any]] = []
    skipped_count = 0
    
    stats = {
        'total': len(data),
        'processed': 0,
        'skipped': 0,
        'with_hint': 0,
        'without_hint': 0,
    }
    
    for item in tqdm(data, desc="Adding action hints"):
        # 反序列化嵌套的 JSON 字段（parquet 会将 dict/list 序列化为字符串）
        if isinstance(item.get('reward_model'), str):
            try:
                item['reward_model'] = json.loads(item['reward_model'])
            except (json.JSONDecodeError, TypeError):
                pass
        
        if isinstance(item.get('extra_info'), str):
            try:
                item['extra_info'] = json.loads(item['extra_info'])
            except (json.JSONDecodeError, TypeError):
                pass
        
        if isinstance(item.get('prompt'), str):
            try:
                item['prompt'] = json.loads(item['prompt'])
            except (json.JSONDecodeError, TypeError):
                pass
        
        # 提取必要信息
        reward_model = item.get('reward_model', {})
        ground_truth = reward_model.get('ground_truth', {})
        function_list = ground_truth.get('function_list', [])
        extra_info = item.get('extra_info', {})
        
        # 跳过没有 function_list 的样本（安全检查：处理 None、空列表、空数组）
        if function_list is None or (hasattr(function_list, '__len__') and len(function_list) == 0):
            if skip_no_function_list:
                skipped_count += 1
                stats['skipped'] += 1
                continue
            else:
                stats['without_hint'] += 1
                processed_data.append(item)
                continue
        
        # 获取原始 prompt 和 response
        prompt_list = item.get('prompt', [])
        
        # 将 numpy array 转换为 list（parquet 读取后可能是 numpy array）
        if hasattr(prompt_list, 'tolist'):
            prompt_list = prompt_list.tolist()
        
        # 检查是否为有效的 list 格式
        if not isinstance(prompt_list, list) or len(prompt_list) == 0:
            print(f"Warning: Invalid prompt format, skipping")
            skipped_count += 1
            stats['skipped'] += 1
            continue
        
        # 确保第一个元素是 dict
        if not isinstance(prompt_list[0], dict):
            print(f"Warning: First prompt element is not a dict, skipping")
            skipped_count += 1
            stats['skipped'] += 1
            continue
        
        original_prompt = prompt_list[0].get('content', '')
        if not original_prompt:
            print(f"Warning: Empty prompt content, skipping")
            skipped_count += 1
            stats['skipped'] += 1
            continue
        
        # 从 response 中提取 action hint（而不是从 function_list 重新生成）
        response = item.get('response', '')
        if not response:
            print(f"Warning: Empty response, skipping")
            skipped_count += 1
            stats['skipped'] += 1
            continue
        
        # 抓取原始 <action> 内容作为 before 记录
        action_block_match = re.search(r'<action>\s*(.*?)\s*</action>', response, re.DOTALL | re.IGNORECASE)
        before_actions = action_block_match.group(1).strip() if action_block_match else ""

        try:
            action_hint = extract_action_hint_from_response(response, extra_info)
        except Exception as e:
            print(f"Error extracting action hint from response: {e}")
            # 也记录失败样本的 before，以便排查
            action_conversion_records.append({
                'index': stats['processed'] + stats['skipped'],
                'before': before_actions,
                'after': "",
                'error': str(e),
            })
            skipped_count += 1
            stats['skipped'] += 1
            continue
        
        # 记录本样本的转换前后（即使 after 为空也记录，便于定位）
        action_conversion_records.append({
            'index': stats['processed'] + stats['skipped'],
            'before': before_actions,
            'after': action_hint,
        })

        if not action_hint:
            print(f"Warning: Empty action hint for question: {extra_info.get('original_question', '')[:50]}...")
            stats['without_hint'] += 1
            processed_data.append(item)
            continue
        
        # 创建 hint-enhanced prompt
        try:
            enhanced_prompt = create_hint_enhanced_prompt(
                original_prompt=original_prompt,
                action_hint=action_hint,
                hint_style=hint_style,
            )
        except Exception as e:
            print(f"Error creating enhanced prompt: {e}")
            skipped_count += 1
            stats['skipped'] += 1
            continue
        
        # 创建新的数据项
        new_item = item.copy()
        new_item['prompt'] = [{"role": "user", "content": enhanced_prompt}]
        
        # 在 extra_info 中记录 hint 信息
        if 'extra_info' not in new_item:
            new_item['extra_info'] = {}
        new_item['extra_info']['has_action_hint'] = True
        new_item['extra_info']['action_hint'] = action_hint
        new_item['extra_info']['hint_style'] = hint_style
        
        processed_data.append(new_item)
        stats['processed'] += 1
        stats['with_hint'] += 1
    
    # 保存结果
    print(f"\nSaving {len(processed_data)} samples to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    dataset = datasets.Dataset.from_list(processed_data)
    dataset.to_parquet(output_path)
    
    print(f"Saved to {output_path}")

    # 额外保存 action 转换前后对照（JSONL），路径与 output_path 同目录，仅更改文件名
    try:
        out_path_obj = Path(output_path)
        stem = out_path_obj.stem  # e.g., out from out.parquet
        conversion_file = out_path_obj.with_name(f"{stem}.action_conversions.jsonl")
        with open(conversion_file, 'w', encoding='utf-8') as f:
            for rec in action_conversion_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Action conversions saved to {conversion_file}")
        stats['action_conversions_path'] = str(conversion_file)
        stats['action_conversions_count'] = len(action_conversion_records)
    except Exception as e:
        print(f"Warning: Failed to save action conversions file: {e}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Prepare dataset for rejection sampling with action hints'
    )
    parser.add_argument(
        '--input_path',
        type=str,
        required=True,
        help='Path to input RL dataset (parquet file)'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='Path to output dataset with hints (parquet file)'
    )
    parser.add_argument(
        '--hint_style',
        type=str,
        choices=['reference', 'hidden', 'example'],
        default='reference',
        help='Style of hint presentation'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=None,
        help='Number of samples to process (default: all)'
    )
    parser.add_argument(
        '--skip_no_hint',
        action='store_true',
        help='Skip samples without function_list (default: keep them for RL training)'
    )
    
    args = parser.parse_args()
    
    # 准备数据集
    stats = prepare_rejection_sampling_dataset(
        input_path=args.input_path,
        output_path=args.output_path,
        hint_style=args.hint_style,
        num_samples=args.num_samples,
        skip_no_function_list=args.skip_no_hint,
    )
    
    # 打印统计信息
    print("\n" + "="*60)
    print("Dataset Preparation Statistics:")
    print("="*60)
    print(f"Total samples:        {stats['total']}")
    print(f"Processed:            {stats['processed']} ({stats['processed']/stats['total']*100:.1f}%)")
    print(f"  - With hint:        {stats['with_hint']}")
    print(f"  - Without hint:     {stats['without_hint']}")
    print(f"Skipped:              {stats['skipped']} ({stats['skipped']/stats['total']*100:.1f}%)")
    print("="*60)


if __name__ == '__main__':
    main()
