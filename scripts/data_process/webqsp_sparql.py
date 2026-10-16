# Modified 2026-10-16 for the GAPD release candidate; see CHANGELOG.md.
"""
Preprocess the WebQSP dataset to parquet format for KBQA-R1
Enhanced with integrated entity extraction functionality
"""

import re
import os
import json
import datasets
from tqdm import tqdm
import argparse
from typing import List, Tuple, Optional

def get_label_with_odbc_safe(entity_id: str, use_odbc: bool = False) -> str:
    """
    Safely get entity label with optional ODBC support
    Falls back to entity_id if ODBC is not available or fails
    """
    if not use_odbc:
        return entity_id
    
    # Only try ODBC for Freebase entities
    if not (entity_id.startswith('m.') or entity_id.startswith('g.')):
        return entity_id

    from kbqa_r1.sparql.sparql_executor import get_label_with_odbc
    label = get_label_with_odbc(entity_id)
    return label if label is not None else entity_id

def extract_entities_from_function_list(function_list: List[str], use_odbc: bool = False) -> List[Tuple[str, str]]:
    """
    Enhanced entity extraction from function_list with better error handling
    Returns list of (entity_label, entity_id) tuples
    """
    entities = []
    
    for func in function_list:
        if "START" in func:
            # More robust regex pattern matching
            match = re.search(r"expression.*?\s*=\s*START\(\'([^\']+)\'\)", func)
            if match:
                entity_id = match.group(1)
                entity_label = get_label_with_odbc_safe(entity_id, use_odbc)
                entities.append((entity_label, entity_id))
            else:
                print(f"Warning: Could not extract entity from START function: {func}")
        elif "TC" in func:
            # Also extract time entities from TC functions
            tc_match = re.search(r"TC\(expression.*?, \'.*?\', \'([^\']+)\'\)", func)
            if tc_match:
                time_entity = tc_match.group(1)
                if time_entity != 'NOW':
                    entities.append((time_entity, time_entity))
    
    return entities

def prompt_function(function_list: List[str], use_odbc: bool = False): #! very important function
    scratchpad = ''
    entities = []
    for step_n, func in enumerate(function_list):
        step_n += 1
        if "START" in func:
            argument = re.findall("expression(.*?) = START\\('(.*?)'\\)", func)[0]
            plan = f'Thought{step_n}: At this step, we should identify a topic entity from the question to start a new expression.\nAction{step_n}: Extract_entity '
            entity = get_label_with_odbc_safe(argument[1], use_odbc) if argument[1].startswith('m.') or argument[1].startswith('g.') else argument[1]
            action = f'[ {entity} ]\n'
            observation = f"Observation{step_n}: expression{argument[0]} = START('{argument[1]}')\n" # | Start Excuted Answers: {entity} (total 1 answers)
            scratchpad += f'{plan}{action}{observation}'
            entities.append((entity,argument[1]))
        elif "JOIN(" in func:
            argument = re.findall("expression(.*?) = JOIN\\('(.*?)', expression(.*?)\\)", func)[0]
            plan = f'Thought{step_n}: At this step, we should find the one-hop relation that is connected to the current expression.\nAction{step_n}: Find_relation '
            relation = argument[1] if '(R ' not in argument[1] else re.findall('\\(R (.*?)\\)', argument[1])[0]
            action = f'[ {relation} ]\n'
            observation = f"Observation{step_n}: expression{argument[0]} = JOIN('{argument[1]}', expression{argument[2]})\n" # | Intermidiate Excuted Answers: {prompt_excuted_answers} (total {len(excuted_answers)} answers)
            scratchpad += f'{plan}{action}{observation}'
        elif "AND" in func:
            argument = re.findall('expression(.*?) = AND\\(expression(.*?), expression(.*?)\\)', func)[0]
            plan = f'Thought{step_n}: At this step, we should merge these two expressions.\nAction{step_n}: Merge '
            expression1, expression2 = f'expression{argument[1]}', f'expression{argument[2]}'
            action = f'[ {expression1} | {expression2} ]\n'
            observation = f"Observation{step_n}: expression{argument[0]} = AND({expression1}, {expression2})\n" # | Intermidiate Excuted Answers: {prompt_excuted_answers} (total {len(excuted_answers)} answers)
            scratchpad += f'{plan}{action}{observation}'
        elif "ARG" in func:
            argument = re.findall("expression(.*?) = ARG\\('(.*?)', expression(.*?), '(.*?)'\\)", func)[0]
            plan = f'Thought{step_n}: At this step, we should perform a sorting operation and impose a constraint to output either the maximum or minimum value.\nAction{step_n}: Order '
            mode, relation = argument[1], argument[3]
            action = f'[ {mode} | {relation} ]\n'
            observation = f"Observation{step_n}: expression{argument[0]} = ARG('{mode}', expression{argument[2]}, '{relation}')\n" # | Intermidiate Excuted Answers: {prompt_excuted_answers} (total {len(excuted_answers)} answers)
            scratchpad += f'{plan}{action}{observation}'                
        elif "CMP" in func:
            argument = re.findall("expression(.*?) = CMP\\('(.*?)', '(.*?)', expression(.*?)\\)", func)[0]
            plan = f'Thought{step_n}: At this step, we should perform a numerical comparison to determine the range.\nAction{step_n}: Compare '
            mode_dict = {'le':'LESS EQUAL', 'ge':'GREATER EQUAL', 'lt':'LESS THAN', 'gt':'GREATER THAN'}
            mode, relation = mode_dict[argument[1]], argument[2]
            action = f'[ {mode} | {relation} ]\n'
            observation = f"Observation{step_n}: expression{argument[0]} = CMP('{argument[1]}', '{relation}', expression{argument[3]})\n" # | Intermidiate Excuted Answers: {prompt_excuted_answers} (total {len(excuted_answers)} answers)
            scratchpad += f'{plan}{action}{observation}'
        elif "TC" in func:
            argument = re.findall("expression(.*?) = TC\\(expression(.*?), '(.*?)', '(.*?)'\\)", func)[0]
            plan = f'Thought{step_n}: At this step, we should add a time constraint.\nAction{step_n}: Time_constraint '
            relation, time = argument[2], argument[3]
            action = f'[ {relation} | {time} ]\n'
            observation = f"Observation{step_n}: expression{argument[0]} = TC(expression{argument[1]}, '{relation}', '{time}')\n" # | Intermidiate Excuted Answers: {prompt_excuted_answers} (total {len(excuted_answers)} answers)
            scratchpad += f'{plan}{action}{observation}'
            if time != 'NOW':
                entities.append((time,argument[3]))
        elif "COUNT" in func:
            argument = re.findall('expression(.*?) = COUNT\\(expression(.*?)\\)', func)[0]
            plan = f'Thought{step_n}: At this step, we should perform a counting operation to determine the number of answers.\nAction{step_n}: Count '
            expression = f'expression{argument[1]}'
            action = f'[ {expression} ]\n'
            observation = f"Observation{step_n}: expression{argument[0]} = COUNT({expression})\n" # | Intermidiate Excuted Answers: {prompt_excuted_answers} (total {len(excuted_answers)} answers)
            scratchpad += f'{plan}{action}{observation}'
        elif "STOP" in func:
            argument = re.findall('expression(.*?) = STOP\\(expression(.*?)\\)', func)[0]
            plan = f'Thought{step_n}: At this step, we conclude that it is appropriate to end and output the expression.\nAction{step_n}: Finish '
            expression = f'expression{argument[1]}'
            action = f'[ {expression} ]\n'
            observation = f"Observation{step_n}: expression{argument[0]} = STOP({expression})\n" # | Final Excuted Answers: {prompt_excuted_answers} (total {len(excuted_answers)} answers)
            scratchpad += f'{plan}{action}{observation}'
        else:
            pass
    # print(scratchpad)
    return scratchpad, entities

def make_prefix(dp, template_type):
    question = dp['question']
    entities = dp.get('candidate_entities', [])

    if entities:
        entities_str_parts = [f"'{name}' ({mid})" for name, mid in entities[:10]]
        entities_display = ", ".join(entities_str_parts)
        if len(entities) > 10:
            entities_display += ", ..."
        entities_prompt_part = f"Candidate Entities: [{entities_display}]\\n"
    else:
        entities_prompt_part = "Entities: []\\n"

    if template_type == 'base':
        """This works for any base model"""
#         prefix = f"""Answer the given question about Freebase knowledge base. \
# You must conduct reasoning inside <think> and </think> first every time you get new information. \
# After reasoning, if you find you need to query the knowledge base, you can write a SPARQL query inside <sparql> and </sparql> and it will return the query results between <information> and </information>. \
# You can query as many times as you want. And you need to answer the question with MID, e.g. m.02mjmr for Barack Obama.\
# If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> m.02mjmr </answer>. 

# {entities_prompt_part} 

# Question: {question}\n"""
#     else:
#         raise NotImplementedError
#     return prefix
        # Build the prefix template with proper string formatting
        examples_section = ""
        
        prefix = f"""You are an expert assistant for querying the Freebase knowledge base using SPARQL.

Answer the given question about Freebase knowledge base. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you need to query the knowledge base, you can write a SPARQL query inside <sparql> and </sparql> and it will return the query results between <information> and </information>. \
You can query as many times as you want. And you need to answer the question with MID, e.g. m.02mjmr for Barack Obama.\
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> m.02mjmr </answer>. 


When you are at Step 2 (Explore) and writing a query, you **must** adhere to the following guidelines for our Virtuoso database which contains **only Freebase data Your queries need to know about the Freebase (`ns:`) namespace. You **must** declare them at the start: PREFIX ns: <http://rdf.freebase.com/ns/>.
To help you start, here are some relevant entities identified from the question. Begin your exploration with one of these entities, but remember to explore their properties step-by-step rather than trying to construct a complete solution immediately.
{entities_prompt_part}

**Your task**: Start by exploring the properties of one of the provided entities. Do NOT try to answer the question in your first query - instead, discover what relationships and properties are available for further exploration.

Question: {question}
"""
    else:
        raise NotImplementedError
    return prefix

def process_webqsp_data(data_path, output_dir, template_type='base', use_odbc=False, data_source_name="webqsp"):
    """
    Process WebQSP data for KBQA-R1 with enhanced entity extraction.
    
    Args:
        data_path: Path to input JSON file
        output_dir: Output directory for parquet files
        template_type: Template type for prompt generation
        use_odbc: Whether to use ODBC for entity label retrieval
        data_source_name: Name for the data source
    """
    print(f"Processing {data_path} with ODBC={'enabled' if use_odbc else 'disabled'}")
    
    # Load WebQSP data
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        data = [data]
    
    processed_data = []
    all_entities = set()
    total_entity_mentions = 0
    items_with_entities = 0
    
    for item in tqdm(data, desc="Processing data items"):
        # Get basic info
        question = item.get('question', '')
        if question and not question.endswith('?'):
            question += '?'

        function_list = item.get('function_list', [])
        
        # Extract entities using enhanced function
        entities = extract_entities_from_function_list(function_list, use_odbc=use_odbc)
        
        # Also get entities from original prompt_function for backward compatibility
        scratchpad, legacy_entities = prompt_function(function_list, use_odbc=use_odbc)
        
        # Combine and deduplicate entities
        all_extracted_entities = entities.copy()
        for legacy_entity in legacy_entities:
            if legacy_entity not in all_extracted_entities:
                all_extracted_entities.append(legacy_entity)
        
        # Statistics tracking
        if all_extracted_entities:
            items_with_entities += 1
            for _, entity_id in all_extracted_entities:
                all_entities.add(entity_id)
                total_entity_mentions += 1
        
        prompt_input_data = {"question": question, "candidate_entities": all_extracted_entities}
        
        # Create prompt
        prompt = make_prefix(prompt_input_data, template_type=template_type)
        
        # Create solution
        solution = {
            "target": item.get('answer', []),
            "sparql": item.get('sparql', ''),
            "sexpr": item.get('sexpr', None)
        }
        
        # Create data item
        data_item = {
            "data_source": data_source_name,
            "prompt": [{
                "role": "user",
                "content": prompt,
            }],
            "ability": "kbqa-reasoning",
            "reward_model": {
                "style": "rule",
                "ground_truth": solution
            },
            "extra_info": {
                'split': 'train' if 'train' in data_path else 'test',
                'id': item.get('ID', item.get('id', '')),
                'function_list': function_list,
                'extracted_entities': all_extracted_entities,
                'original_question': question,
                'scratchpad': scratchpad,  # Include reasoning scratchpad
            }
        }
        
        processed_data.append(data_item)
    
    # Print statistics
    print(f"\n=== Entity Extraction Statistics ===")
    print(f"Total items processed: {len(processed_data)}")
    print(f"Items with entities: {items_with_entities}")
    print(f"Total entity mentions: {total_entity_mentions}")
    print(f"Unique entities: {len(all_entities)}")
    if len(processed_data) > 0:
        print(f"Average entities per item: {total_entity_mentions / len(processed_data):.2f}")
    
    # Convert to dataset
    dataset = datasets.Dataset.from_list(processed_data)
    
    # Save to parquet
    os.makedirs(output_dir, exist_ok=True)
    split = 'train' if 'train' in data_path else 'test'
    output_path = os.path.join(output_dir, f'{split}.parquet')
    dataset.to_parquet(output_path)
    
    print(f"\nProcessed {len(processed_data)} items from {data_path}")
    print(f"Saved to {output_path}")
    
    # Show some example entities
    if all_entities:
        print(f"\nSample entities (first 10):")
        for i, entity in enumerate(sorted(all_entities)[:10]):
            print(f"  {i+1}. {entity}")
        if len(all_entities) > 10:
            print(f"  ... and {len(all_entities) - 10} more")
    
    return processed_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process train.json files with enhanced entity extraction for KBQA-R1')
    parser.add_argument('--data_path', type=str, default='dataset/WebQSP/processed/WebQSP_train.json', 
                        help='Path to input JSON data file')
    parser.add_argument('--output_dir', type=str, default='data/webqsp_sparql', 
                        help='Output directory for parquet files')
    parser.add_argument('--template_type', type=str, default='base', 
                        help='Template type for prompt generation')
    parser.add_argument('--use_odbc', action='store_true', 
                        help='Use ODBC to get entity labels (requires ODBC setup)')
    parser.add_argument('--data_source_name', type=str, default='webqsp',
                        help='Name for the data source in output')
    parser.add_argument('--original_data_path', type=str, default='dataset/WebQSP/origin/WebQSP.train.json', 
                        help='Path to original WebQSP data (deprecated)')
    
    args = parser.parse_args()
    
    print(f"Configuration:")
    print(f"  Input path: {args.data_path}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Template type: {args.template_type}")
    print(f"  Use ODBC: {args.use_odbc}")
    print(f"  Data source name: {args.data_source_name}")
    print()
    
    try:
        result = process_webqsp_data(
            data_path=args.data_path,
            output_dir=args.output_dir,
            template_type=args.template_type,
            use_odbc=args.use_odbc,
            data_source_name=args.data_source_name
        )
        print(f"\n✓ Successfully processed {len(result)} items!")
        print("Dataset is ready for verl training.")
    except Exception as e:
        print(f"\n✗ Error processing data: {e}")
        import traceback
        traceback.print_exc() 