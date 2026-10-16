# Modified 2026-10-16 for the GAPD release candidate; see CHANGELOG.md.
import argparse
import json
import logging
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import datasets
import pandas as pd
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from scripts.data_process.prepare_rejection_sampling_dataset import extract_action_hint_from_response

# Import necessary functions
try:
    from .webqsp_sparql import get_label_with_odbc_safe
except ImportError:
    # Handle case when run as script
    import os
    import sys
    sys.path.append(os.path.dirname(__file__))
    from webqsp_sparql import get_label_with_odbc_safe

logger = logging.getLogger(__name__)

def ent_type(ent_id: str) -> str:
    """Determine the type of entity ID"""
    if ent_id.startswith('m.') or ent_id.startswith('g.'):
        return 'entity'
    elif ent_id.startswith('"') and (ent_id.endswith('"') or ent_id.endswith('"@en')):
        return 'name'
    elif '^^' in ent_id:
        return 'literal'
    elif '<http:' in ent_id:
        return 'url'
    elif ent_id.isdigit() or (ent_id[0] == '-' and ent_id[1:].isdigit()):
        return 'int'
    elif '.' in ent_id:
        return 'onto'
    else:
        return 'unknown'

def load_existing_entities_cache(output_path: str) -> Dict[str, List[Tuple[str, str]]]:
    """Load existing dataset and extract entity mappings to avoid re-using ODBC"""
    entity_cache = {}    
    if not os.path.exists(output_path):
        print(f"Output file {output_path} does not exist, will use ODBC if needed.")
        return entity_cache
        
    try:
        print(f"Loading existing file {output_path} to check for cached entities...")
        df = pd.read_parquet(output_path)
        
        cached_count = 0
        for _, row in df.iterrows():
            extra_info = row.get('extra_info', {})
            if isinstance(extra_info, str):
                try:
                    extra_info = json.loads(extra_info)
                except (json.JSONDecodeError, ValueError):
                    continue
            
            # Use question or ID as cache key
            question = extra_info.get('original_question', '')
            item_id = extra_info.get('id', '')
            extracted_entities = extra_info.get('extracted_entities', [])
            
            if extracted_entities is not None:
                # Use question as primary key, ID as fallback
                cache_key = question if question else item_id
                if cache_key:
                    entity_cache[cache_key] = extracted_entities
                    cached_count += 1
        
        print(f"Loaded {cached_count} cached entity mappings from existing file.")
        return entity_cache
        
    except Exception as e:
        print(f"Error loading existing file {output_path}: {e}")
        print("Will proceed without entity cache.")
        return {}

# Import prompt_function from webqsp_sparql.py to extract entities
def prompt_function(function_list: List[str], use_odbc: bool = False, entity_cache: Dict[str, List[Tuple[str, str]]] = None, cache_key: str = None): #! very important function
    scratchpad = ''
    entities = []
    
    # Check if we can use cached entities
    if entity_cache and cache_key and cache_key in entity_cache:
        print(f"Using cached entities for key: {cache_key[:50]}...")
        cached_entities = entity_cache[cache_key]
        use_odbc = False  # Force disable ODBC when using cache
    else:
        cached_entities = None
    
    for step_n, func in enumerate(function_list):
        step_n += 1
        if "START" in func:
            argument = re.findall(r"expression(.*?) = START\('(.*?)'\)", func)[0]
            plan = f'Thought{step_n}: At this step, we should identify a topic entity from the question to start a new expression.\nAction{step_n}: Extract_entity '
            
            entity_id = argument[1]
            # Try to get entity label from cache first
            if cached_entities is not None:
                # Find entity by ID in cached entities
                entity_label = None
                for cached_name, cached_id in cached_entities:
                    if cached_id == entity_id:
                        entity_label = cached_name
                        break
                
                if entity_label:
                    entity = entity_label
                else:
                    # Fallback: handle different entity types
                    entity_type = ent_type(entity_id)
                    if entity_type == 'onto':
                        # Convert ontology types to readable format
                        entity = entity_id.replace('.', ' ').replace('_', ' ').title()
                    else:
                        entity = entity_id
            else:
                # Use original ODBC logic with proper entity type handling
                entity_type = ent_type(entity_id)
                if entity_type == 'entity':
                    # Regular entities (m.*, g.*)
                    entity = get_label_with_odbc_safe(entity_id, use_odbc)
                elif entity_type == 'onto':
                    # Ontology types (e.g., measurement_unit.conductance_unit)
                    # Convert dot notation to more readable format
                    entity = entity_id.replace('.', ' ').replace('_', ' ').title()
                else:
                    # Other types (literals, names, etc.)
                    entity = entity_id
            
            action = f'[ {entity} ]\n'
            observation = f"Observation{step_n}: expression{argument[0]} = START('{entity_id}')\n" # | Start Excuted Answers: {entity} (total 1 answers)
            scratchpad += f'{plan}{action}{observation}'
            entities.append((entity, entity_id))
        elif "JOIN(" in func:
            argument = re.findall(r"expression(.*?) = JOIN\('(.*?)', expression(.*?)\)", func)[0]
            plan = f'Thought{step_n}: At this step, we should find the one-hop relation that is connected to the current expression.\nAction{step_n}: Find_relation '
            relation = argument[1]
            action = f'[ {relation} ]\n'
            observation = f"Observation{step_n}: expression{argument[0]} = JOIN('{argument[1]}', expression{argument[2]})\n" # | Intermidiate Excuted Answers: {prompt_excuted_answers} (total {len(excuted_answers)} answers)
            scratchpad += f'{plan}{action}{observation}'
        elif "AND" in func:
            argument = re.findall(r"expression(.*?) = AND\(expression(.*?), expression(.*?)\)", func)[0]
            plan = f'Thought{step_n}: At this step, we should merge these two expressions.\nAction{step_n}: Merge '
            expression1, expression2 = f'expression{argument[1]}', f'expression{argument[2]}'
            action = f'[ {expression1} | {expression2} ]\n'
            observation = f"Observation{step_n}: expression{argument[0]} = AND({expression1}, {expression2})\n" # | Intermidiate Excuted Answers: {prompt_excuted_answers} (total {len(excuted_answers)} answers)
            scratchpad += f'{plan}{action}{observation}'
        elif "ARG" in func:
            argument = re.findall(r"expression(.*?) = ARG\('(.*?)', expression(.*?), '(.*?)'\)", func)[0]
            plan = f'Thought{step_n}: At this step, we should perform a sorting operation and impose a constraint to output either the maximum or minimum value.\nAction{step_n}: Order '
            mode, relation = argument[1], argument[3]
            action = f'[ {mode} | {relation} ]\n'
            observation = f"Observation{step_n}: expression{argument[0]} = ARG('{mode}', expression{argument[2]}, '{relation}')\n" # | Intermidiate Excuted Answers: {prompt_excuted_answers} (total {len(excuted_answers)} answers)
            scratchpad += f'{plan}{action}{observation}'                
        elif "CMP" in func:
            argument = re.findall(r"expression(.*?) = CMP\('(.*?)', '(.*?)', expression(.*?)\)", func)[0]
            plan = f'Thought{step_n}: At this step, we should perform a numerical comparison to determine the range.\nAction{step_n}: Compare '
            mode_dict = {'le':'LESS EQUAL', 'ge':'GREATER EQUAL', 'lt':'LESS THAN', 'gt':'GREATER THAN'}
            mode, relation = mode_dict[argument[1]], argument[2]
            action = f'[ {mode} | {relation} ]\n'
            observation = f"Observation{step_n}: expression{argument[0]} = CMP('{argument[1]}', '{relation}', expression{argument[3]})\n" # | Intermidiate Excuted Answers: {prompt_excuted_answers} (total {len(excuted_answers)} answers)
            scratchpad += f'{plan}{action}{observation}'
        elif "TC" in func:
            argument = re.findall(r"expression(.*?) = TC\(expression(.*?), '(.*?)', '(.*?)'\)", func)[0]
            plan = f'Thought{step_n}: At this step, we should add a time constraint.\nAction{step_n}: Time_constraint '
            relation, time = argument[2], argument[3]
            action = f'[ {relation} | {time} ]\n'
            observation = f"Observation{step_n}: expression{argument[0]} = TC(expression{argument[1]}, '{relation}', '{time}')\n" # | Intermidiate Excuted Answers: {prompt_excuted_answers} (total {len(excuted_answers)} answers)
            scratchpad += f'{plan}{action}{observation}'
            if time != 'NOW':
                entities.append((time, argument[3]))
        elif "COUNT" in func:
            argument = re.findall(r"expression(.*?) = COUNT\(expression(.*?)\)", func)[0]
            plan = f'Thought{step_n}: At this step, we should perform a counting operation to determine the number of answers.\nAction{step_n}: Count '
            expression = f'expression{argument[1]}'
            action = f'[ {expression} ]\n'
            observation = f"Observation{step_n}: expression{argument[0]} = COUNT({expression})\n" # | Intermidiate Excuted Answers: {prompt_excuted_answers} (total {len(excuted_answers)} answers)
            scratchpad += f'{plan}{action}{observation}'
        elif "STOP" in func:
            argument = re.findall(r"expression(.*?) = STOP\(expression(.*?)\)", func)[0]
            plan = f'Thought{step_n}: At this step, we conclude that it is appropriate to end and output the expression.\nAction{step_n}: Finish '
            expression = f'expression{argument[1]}'
            action = f'[ {expression} ]\n'
            observation = f"Observation{step_n}: expression{argument[0]} = STOP({expression})\n" # | Final Excuted Answers: {prompt_excuted_answers} (total {len(excuted_answers)} answers)
            scratchpad += f'{plan}{action}{observation}'
        else:
            pass
    # print(scratchpad)
    return scratchpad, entities



def create_action_training_prompt(question: str, entities: List[Tuple[str, str]] = None, dataset_name: str = "webqsp") -> str:
    """Create prompt for Action-based reasoning model training following KBQA-o1 format

    Dataset-specific constraints:
    - webqsp: exclude Compare and Count actions
    - grailqa/graphq: exclude Time_constraint action
    """
    
    entities_str = ""
    if entities:
        entities_parts = [f"'{name}' ({mid})" for name, mid in entities[:10]]
        entities_display = ", ".join(entities_parts)
        if len(entities) > 50:
            entities_display += ", ..."
        entities_str = f"Candidate Entities: [{entities_display}]\n"
    else:
        entities_str = "Entities: []\n"
    
    # Assemble dataset-specific Available Actions
    actions_sections: List[str] = []

    # Find_relation (always)
    actions_sections.append(
        """* Find_relation [entity_id | relation]: Find entities connected to the given entity through the specified relation. The entity can be:
  - A MID (e.g., m.02mjmr) for entity types
  - A literal value (e.g., 2011-01-17^^http://www.w3.org/2001/XMLSchema#date, 62.0^^http://www.w3.org/2001/XMLSchema#float) for literal types
  - An expression reference for chaining operations. Use it when the referenced expression has many results.
  The relation should be a Freebase relation (e.g., people.person.place_of_birth, government.election_poll.poll_end_date). This will return a list of entities connected by the relation. 
  Examples: 
  - Find_relation [m.02mjmr | people.person.place_of_birth]
  - Find_relation [62.0^^http://www.w3.org/2001/XMLSchema#float | architecture.lighthouse.focal_height_of_light]
  - Find_relation [expression_id | media_common.quotation.author] (for chaining operations). 
  Chain example: for the query "Who authored quotations about Barack Obama?", you can chain operations: Find_relation [m.02mjmr | media_common.quotation.subjects] to get quotations about Obama, then Find_relation [expression1 | media_common.quotation.author] to get the authors of those quotations."""
    )

    # Merge (always)
    actions_sections.append(
        """* Merge [expression_id1 | expression_id2]: Merge two expressions using logical AND. This can be used in two scenarios:
  1. Merge two expressions from two early actions and get the intersection.(e.g., Merge [expression_id1 | expression_id2], expression_id1 and expression_id2 are the function_id of two previous actions)
  2. Merge an expression with an ontology type (e.g., Merge [expression_id | measurement_unit.conductance_unit])
Note: When using the merge operation, carefully verify the expression IDs to ensure you are merging the correct ones."""
    )

    # Order (always)
    actions_sections.append(
        """* Order [MAX/MIN | expression_id/ontology_type | relation]: Sort by relation and return max/min
Use example: 
  1. Order a existing expression: "When did Manchester United first win a trophy?", you should first get all of the team's championships with Find_relation [m.050fh | sports.sports_team.championships]. Then, apply the ordering constraint using Order [MIN | expression1 | time.event.end_date], which finds the championship with the earliest end date (i.e., the first trophy won).
  2. Order a ontology type: "which film director is the heaviest?", you should do the action  Order [MAX | film.director | people.person.weight_kg] to get the heaviest film director."""
    )

    is_webqsp = (dataset_name.lower() == "webqsp")
    is_grail_or_graphq = (dataset_name.lower() in ["grailqa", "graphq"])
    is_cwq = (dataset_name.lower() == "cwq")

    # Compare (exclude for webqsp; include for grailqa, graphq, cwq)
    if not is_webqsp:
        actions_sections.append(
            """* Compare [operator | relation | number]: Apply numerical comparison. Available mode: le, lt, ge, gt.
Use example: for the query "what unit of conductance has equal to or smaller than 0.001 conductance in siemens?", you should apply the numerical comparison using Compare [ le | measurement_unit.conductance_unit.conductance_in_siemens | 0.001], which filters entities to keep only those with conductance values ≤ 0.001. Then, merge with the ontology type using Merge [expression1 | measurement_unit.conductance_unit] to ensure results are of the correct conductance unit type."""
        )

    # Time_constraint (exclude for grailqa/graphq; include for webqsp and cwq)
    if not is_grail_or_graphq:
        actions_sections.append(
            """* Time_constraint [relation | time]: Filters a list of CVT entities, keeping only those where the <time> falls within the CVT's time range. Note that the relation for time_constraint should be of the type *.*.from, *.*.to, *.*.end_date. If the time constraint is now, use NOW as time parameter.
Use example: for the query "What team did Kaká play for in 2009?", you should first get all of the player's team memberships (CVTs) with Find_relation [m.04qv66 | sports.pro_athlete.teams]. Then, apply the time constraint using Time_constraint [sports.sports_team_roster.from | 2009], which returns a new expression representing the filtered results. Finally, use Find_relation on this expression to find the exact team he played for: Find_relation [expression2 | sports.sports_team_roster.team]."""
        )

    # Count (exclude for webqsp; include for grailqa, graphq, cwq)
    if not is_webqsp:
        actions_sections.append("""* Count [expression]: Count results"""
        )

    available_actions_block = "\n\n".join(actions_sections)

    # Use KBQA-o1 format for action-based reasoning
    prompt = f"""You are an expert assistant for querying the Freebase knowledge base using structured reasoning actions.

Answer the given question about Freebase knowledge base. \
You **must** conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, provide structured actions inside <action> and </action> tags that will be executed to query the knowledge base. \
The system will return query results between <information> and </information>. \
You can query as many times as you want. And you need to answer the question with the appropriate identifier or value (MID for entities like m.02mjmr for Barack Obama, or literal values like dates, numbers, etc.).\
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, e.g. <answer> m.*** </answer> or <answer> 2012-02-07 </answer>, without detailed illustrations. For multiple answers, use space to separate them, e.g. <answer> m.01428y m.04ygk0 </answer>.

Available Actions (use exact format):

{available_actions_block}

Begin from the candidate entities detected in the question (you must start from one of these entities):
{entities_str}


Question: {question}
"""
    
    return prompt


def create_action_response_from_function_list(function_list: List[str], sexpr: str, answer: List[str], use_odbc: bool = False, entity_cache: Dict[str, List[Tuple[str, str]]] = None, cache_key: str = None) -> str:
    """Create structured training response with Action sequences following KBQA-o1 format"""
    
    action_scratchpad, entities = prompt_function(function_list, use_odbc=use_odbc, entity_cache=entity_cache, cache_key=cache_key)
    
    # 提取纯Action序列（去掉Thought和Observation）
    action_lines = []
    for line in action_scratchpad.split('\n'):
        if line.strip().startswith('Action'):
            action_lines.append(line.strip())
    
    # Format answer
    if isinstance(answer, list):
        answer_str = answer[0] if len(answer) == 1 else ", ".join(answer)
    else:
        answer_str = str(answer)
    
    # Create structured response with <action> tags for easy parsing
    response = f"""<think>
Let me analyze this question step by step and determine the appropriate actions to find the answer.
</think>

<action>
{chr(10).join(action_lines)}
</action>

<answer>
{answer_str}
</answer>"""
    
    return response

def _load_with_hint_map(with_hint_path: str) -> Dict[str, str]:
    """Load action_hint mapping from an existing with-hint dataset.

    Supports .parquet, .jsonl, and .json formats.
    Returns a dict mapping sample ID -> action_hint string.
    """
    hint_map: Dict[str, str] = {}
    if not os.path.exists(with_hint_path):
        print(f"[WARN] with_hint_path not found: {with_hint_path}")
        return hint_map

    try:
        if with_hint_path.endswith('.parquet'):
            import pandas as pd
            df = pd.read_parquet(with_hint_path)
            for _, row in df.iterrows():
                extra = row.get('extra_info', {})
                if isinstance(extra, str):
                    try:
                        extra = json.loads(extra)
                    except (json.JSONDecodeError, TypeError):
                        extra = {}
                sample_id = extra.get('id', '') if isinstance(extra, dict) else ''
                action_hint = extra.get('action_hint', '') if isinstance(extra, dict) else ''
                if sample_id and action_hint:
                    hint_map[sample_id] = action_hint
        elif with_hint_path.endswith('.jsonl'):
            with open(with_hint_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    extra = record.get('extra_info', {})
                    sample_id = extra.get('id', '')
                    action_hint = extra.get('action_hint', '')
                    if sample_id and action_hint:
                        hint_map[sample_id] = action_hint
        else:
            with open(with_hint_path, 'r', encoding='utf-8') as f:
                records = json.load(f)
            if isinstance(records, dict):
                records = [records]
            for record in records:
                extra = record.get('extra_info', {})
                sample_id = extra.get('id', '')
                action_hint = extra.get('action_hint', '')
                if sample_id and action_hint:
                    hint_map[sample_id] = action_hint
    except Exception as e:
        print(f"[WARN] Failed to load with_hint file {with_hint_path}: {e}")

    print(f"Loaded {len(hint_map)} action hints from {with_hint_path}")
    return hint_map


def _load_gold_trace_map(gold_trace_path: str) -> Dict[str, List[dict]]:
    if not gold_trace_path:
        raise ValueError(
            "gold_trace_path is required. Refusing to prepare RL data without "
            "pre-computed gold traces. Build them first with "
            "scripts/data_process/build_gold_trace.py and pass --gold_trace_path."
        )
    if not os.path.exists(gold_trace_path):
        raise FileNotFoundError(f"gold_trace_path does not exist: {gold_trace_path}")
    with open(gold_trace_path, 'r', encoding='utf-8') as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"gold_trace_path must contain a JSON object mapping sample_id -> trace: {gold_trace_path}")
    print(f"Loaded {len(loaded)} pre-computed gold traces from {gold_trace_path}")
    return loaded


def _gold_trace_has_nonempty_mid_set(trace) -> bool:
    if not isinstance(trace, list) or not trace:
        return False
    for step in trace:
        if not isinstance(step, dict):
            continue
        next_mid_set = step.get("next_mid_set", [])
        if hasattr(next_mid_set, "tolist"):
            next_mid_set = next_mid_set.tolist()
        if isinstance(next_mid_set, (list, tuple, set)) and len(next_mid_set) > 0:
            return True
    return False


def _validate_processed_gold_trace_quality(
    processed_data: List[Dict],
    min_gold_trace_success_rate: float,
) -> None:
    total = len(processed_data)
    if total == 0:
        raise ValueError("No processed samples; refusing to write empty dataset.")

    present = 0
    samples_with_nonempty_step = 0
    steps_total = 0
    steps_nonempty = 0
    empty_examples = []
    missing_examples = []

    for item in processed_data:
        extra = item.get("extra_info", {}) or {}
        sample_id = extra.get("id", "")
        trace = extra.get("gold_trace", None)
        if isinstance(trace, list) and trace:
            present += 1
            steps_total += len(trace)
            for step in trace:
                if not isinstance(step, dict):
                    continue
                next_mid_set = step.get("next_mid_set", [])
                if hasattr(next_mid_set, "tolist"):
                    next_mid_set = next_mid_set.tolist()
                if isinstance(next_mid_set, (list, tuple, set)) and len(next_mid_set) > 0:
                    steps_nonempty += 1
            if _gold_trace_has_nonempty_mid_set(trace):
                samples_with_nonempty_step += 1
            elif len(empty_examples) < 5:
                empty_examples.append(sample_id)
        elif len(missing_examples) < 5:
            missing_examples.append(sample_id)

    present_rate = present / total
    sample_nonempty_rate = samples_with_nonempty_step / total
    step_nonempty_rate = steps_nonempty / steps_total if steps_total else 0.0
    print(
        "[GOLD-TRACE-CHECK] "
        f"total={total} present={present} ({present_rate:.2%}) "
        f"samples_with_nonempty_step={samples_with_nonempty_step} ({sample_nonempty_rate:.2%}) "
        f"steps_nonempty={steps_nonempty}/{steps_total} ({step_nonempty_rate:.2%}) "
        f"threshold={min_gold_trace_success_rate:.2%}"
    )

    if (
        present_rate < min_gold_trace_success_rate
        or sample_nonempty_rate < min_gold_trace_success_rate
        or step_nonempty_rate < min_gold_trace_success_rate
    ):
        raise ValueError(
            "Gold trace quality check failed; refusing to write parquet.\n"
            f"  present_rate={present_rate:.2%}, "
            f"sample_nonempty_mid_set_rate={sample_nonempty_rate:.2%}, "
            f"step_nonempty_mid_set_rate={step_nonempty_rate:.2%}, "
            f"required>={min_gold_trace_success_rate:.2%}\n"
            f"  missing_examples={missing_examples}\n"
            f"  empty_mid_set_examples={empty_examples}\n"
            "Use scripts/data_process/build_gold_trace.py to precompute relation-retrieval-based "
            "gold traces, then rerun prepare_rl_dataset.py with --gold_trace_path."
        )


def process_sexpr_data(data: List[Dict], output_dir: str, training_mode: str = "action", use_odbc: bool = False, dataset_name: str = "webqsp", entity_cache: Dict[str, List[Tuple[str, str]]] = None, with_hint_map: Optional[Dict[str, str]] = None, gold_trace_path: Optional[str] = None, min_gold_trace_success_rate: float = 0.9):
    """Process KBQA-o1 format data"""

    processed_data = []

    gold_trace_map = _load_gold_trace_map(gold_trace_path)

    for item in tqdm(data, desc="Processing KBQA-o1 data"):
        question = item.get('question', '')
        if question and not question.endswith('?'):
            question += '?'

        sexpr = item.get('sexpr', '')
        function_list = item.get('function_list', [])
        answer = item.get('answer', [])
        sample_id = item.get('ID', '')

        # Create cache key for entity lookup
        cache_key = question if question else sample_id

        # Extract entities using KBQA-o1's method with caching
        _, entities = prompt_function(function_list, use_odbc=use_odbc, entity_cache=entity_cache, cache_key=cache_key)

        if training_mode == "action":
            # Action-based reasoning training (following KBQA-o1 format)
            prompt_content = create_action_training_prompt(question, entities, dataset_name)
            response = create_action_response_from_function_list(function_list, sexpr, answer, use_odbc, entity_cache, cache_key)

            # Get action_hint: prefer pre-computed from with_hint_map (rejection-sampling
            # pipeline output), which has correct entity MID inlining. Fallback to
            # extract_action_hint_from_response only when with_hint is unavailable.
            extra_info_dict = {
                'extracted_entities': entities,
                'candidate_entities': entities,
            }
            if with_hint_map and sample_id in with_hint_map:
                action_hint = with_hint_map[sample_id]
            else:
                action_hint = extract_action_hint_from_response(response, extra_info=extra_info_dict) or ""

            # Get gold execution trace for state-anchored per-turn GAPD.
            # This script intentionally does not compute gold_trace inline:
            # precompute it via scripts/data_process/build_gold_trace.py so the
            # relation-retrieval execution path is explicit and auditable.
            gold_trace_serializable = None
            if str(sample_id) in gold_trace_map:
                gold_trace_serializable = gold_trace_map[str(sample_id)]

            extra_info = {
                'id': item.get('ID', ''),
                'function_list': function_list,
                'action_hint': action_hint,
                'extracted_entities': entities,
                'original_question': question,
                'level': item.get('level', None),  # GrailQA specific
                'dataset_type': dataset_name,
                'sparql_conversion_type': dataset_name if dataset_name != 'cwq' else 'webqsp'  # CWQ uses WebQSP SPARQL converter
            }
            if gold_trace_serializable is not None:
                extra_info["gold_trace"] = gold_trace_serializable

            data_item = {
                "data_source": dataset_name,
                "prompt": [{"role": "user", "content": prompt_content}],
                "response": response,  # Add expected response for SFT
                "ability": "kbqa-action-generation",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {
                        "sexpr": sexpr,
                        "target": answer,
                        "sparql": item.get('sparql', ''),
                        "function_list": function_list,
                        "action_hint": action_hint,
                        "candidate_entities": entities,
                        "gold_trace": gold_trace_serializable,
                    }
                },
                "extra_info": extra_info,
            }
        
        else:  # sparql mode (original)
            raise NotImplementedError("SPARQL mode is not implemented yet")
        processed_data.append(data_item)

    _validate_processed_gold_trace_quality(processed_data, min_gold_trace_success_rate)
    return processed_data

def process_original_webqsp_data(data: List[Dict], output_dir: str):
    """Process original WebQSP format data (existing functionality)"""
    # Use existing webqsp_sparql.py logic here
    # ... (existing implementation)
    pass

def main():
    parser = argparse.ArgumentParser(description='Process train.json files with enhanced entity extraction for KBQA-R1')
    parser.add_argument('--data_path', type=str, default='dataset/WebQSP/processed/WebQSP_train.json',
                        help='Path to input JSON data file (e.g., dataset/CWQ/processed/CWQ_train.json for CWQ)')
    parser.add_argument('--output_dir', type=str, default=None, 
                        help='Output directory for parquet files (auto-generated based on dataset if not specified)')
    parser.add_argument('--split', choices=['train', 'dev', 'test'], default=None,
                        help='Explicit output split; overrides inference from the input path.')
    parser.add_argument('--template_type', type=str, default='base', 
                        help='Template type for prompt generation')
    parser.add_argument('--use_odbc', action='store_true', 
                        help='Use ODBC to get entity labels (requires ODBC setup)')
    parser.add_argument('--dataset', type=str, choices=['webqsp', 'grailqa', 'graphq', 'cwq'], default='webqsp',
                        help='Dataset type (webqsp, grailqa, graphq, or cwq)')
    parser.add_argument('--original_data_path', type=str, default='dataset/WebQSP/origin/WebQSP.train.json', 
                        help='Path to original WebQSP data (deprecated)')
    parser.add_argument('--auto_detect', action='store_true', default=True,
                        help='Auto-detect data format')
    parser.add_argument('--compute_gold_trace', action='store_true',
                        help='Deprecated and disabled. Use scripts/data_process/build_gold_trace.py, then pass --gold_trace_path.')
    parser.add_argument('--with_hint_path', type=str, default=None,
                        help='Path to existing with-hint dataset (parquet/jsonl/json) to reuse pre-computed action_hint. '
                             'When provided, action_hint is read from this file instead of being re-extracted from response.')
    parser.add_argument('--gold_trace_path', type=str, default=None,
                        help='Path to pre-computed gold trace JSON (sample_id -> List[GoldTraceStep]). '
                             'Required: this script refuses to compute gold traces on-the-fly.')
    parser.add_argument('--min_gold_trace_success_rate', type=float, default=0.9,
                        help='Minimum required fraction of samples with present gold_trace and non-empty next_mid_set.')

    args = parser.parse_args()
    data_format = "sexpr"
    
    # Load data
    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        data = [data]

    if args.compute_gold_trace:
        raise ValueError(
            "--compute_gold_trace is disabled because the inline builder can diverge from "
            "the relation-retrieval action execution path. Run "
            "scripts/data_process/build_gold_trace.py first and pass --gold_trace_path."
        )
    
    # Detect format

    if args.dataset:
        dataset_name = args.dataset
    else:
        dataset_name = "webqsp"
    
    # Auto-generate output_dir based on dataset if not specified
    if args.output_dir is None:
        args.output_dir = f'data/{dataset_name}_rl_dataset'
        print(f"Output directory auto-generated: {args.output_dir}")
    
    print(f"Processing {dataset_name} dataset in {data_format} format")
    
    # Load entity cache from existing output file if available
    if args.split:
        split = args.split
    elif 'train' in args.data_path:
        split = 'train'
    elif 'dev' in args.data_path:
        split = 'dev'
    else:
        split = 'test'
    output_path = os.path.join(args.output_dir, f'{split}.parquet')
    entity_cache = load_existing_entities_cache(output_path)
    
    # If we have cached entities and not explicitly forcing ODBC, disable ODBC
    if entity_cache and not args.use_odbc:
        print("Found cached entities, will skip ODBC calls for cached items.")
        use_odbc_final = False
    else:
        use_odbc_final = args.use_odbc
        if args.use_odbc:
            print("ODBC explicitly enabled, will use ODBC for new/missing entities.")
    
    # Load with_hint mapping if provided
    with_hint_map = None
    if args.with_hint_path:
        with_hint_map = _load_with_hint_map(args.with_hint_path)

    # Process based on format
    if dataset_name in ["webqsp", "grailqa", "graphq", "cwq"]:
        processed_data = process_sexpr_data(
            data,
            args.output_dir,
            "action",
            use_odbc_final,
            dataset_name,
            entity_cache,
            with_hint_map=with_hint_map,
            gold_trace_path=args.gold_trace_path,
            min_gold_trace_success_rate=args.min_gold_trace_success_rate,
        )
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    dataset = datasets.Dataset.from_list(processed_data)
    
    dataset.to_parquet(output_path)
    
    print(f"Processed {len(processed_data)} items")
    print(f"Saved to {output_path}")

if __name__ == '__main__':
    main()
