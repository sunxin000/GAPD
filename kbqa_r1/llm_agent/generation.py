# import torch
# import re
# from collections import defaultdict
# import os
# import logging
# from typing import List, Dict, Any, Tuple
# from dataclasses import dataclass
# from .tensor_helper import TensorHelper, TensorConfig
# from verl import DataProto
# import requests
# import json
# import ray
# import numpy as np
# import asyncio
# import aiohttp
# import time
# from tqdm.asyncio import tqdm as atqdm
# from tqdm import tqdm
# import random
# import datetime

# # Import SPARQL manager
# from ..sparql.sparql_manager import SPARQLExecutionManager, SPARQLConfig

# # Configure logger
# logger = logging.getLogger(__name__)

# @dataclass
# class GenerationConfig:
#     max_turns: int
#     max_start_length: int
#     max_prompt_length: int 
#     max_response_length: int
#     max_obs_length: int
#     num_gpus: int
#     no_think_rl: bool=False
#     sparql_url: str = None
#     sparql_batch_size: int = 128  # 分批大小
#     sparql_max_concurrent: int = 16  # 最大并发批次数
#     use_odbc: bool = True  # 是否使用ODBC直连模式
#     use_aioodbc: bool = True  # 是否使用异步aioodbc执行器
#     odbc_config: dict = None  # ODBC配置参数
    
#     # 日志记录配置参数
#     log_dir: str = "logs"  # 日志目录
#     log_filename: str = None  # 日志文件名，如果为None则自动生成
#     log_interval: int = 10  # 每多少个step记录一次
#     log_sample_size: int = 30  # 每次记录的样本数量
#     enable_logging: bool = True  # 是否启用日志记录
    
#     # 实验信息参数
#     experiment_name: str = None  # 实验名称，用于创建子文件夹
#     current_step: int = None  # 当前训练步数，用于文件名
    
#     def get_sparql_config(self) -> SPARQLConfig:
#         """Get SPARQL configuration from generation config."""
#         return SPARQLConfig(
#             sparql_url=self.sparql_url,
#             sparql_batch_size=self.sparql_batch_size,
#             sparql_max_concurrent=self.sparql_max_concurrent,
#             use_odbc=self.use_odbc,
#             use_aioodbc=self.use_aioodbc,
#             odbc_config=self.odbc_config
#         )

# class LLMGenerationManager:
#     # 类变量，用于跟踪调用次数
#     _call_counter = -1
#     # 类变量，用于跟踪调试保存次数（限制总共10次）
#     _debug_save_counter = 0
    
#     def __init__(
#         self,
#         tokenizer,
#         actor_rollout_wg,
#         config: GenerationConfig,
#         is_validation: bool = False,
#         sparql_config: dict = None,  # 新增可选参数用于传递完整的SPARQL配置
#     ):
#         self.tokenizer = tokenizer
#         self.actor_rollout_wg = actor_rollout_wg
#         self.config = config
#         self.is_validation = is_validation

#         self.tensor_fn = TensorHelper(TensorConfig(
#             pad_token_id=tokenizer.pad_token_id,
#             max_prompt_length=config.max_prompt_length,
#             max_obs_length=config.max_obs_length,
#             max_start_length=config.max_start_length
#         ))
        
#         # 优先使用传入的sparql_config，否则使用GenerationConfig中的默认配置
#         if sparql_config:
#             # 使用传入的完整SPARQL配置（包含ODBC参数）
#             sparql_config_obj = SPARQLConfig(
#                 sparql_url=sparql_config.get('sparql_url'),
#                 sparql_batch_size=sparql_config.get('sparql_batch_size', 128),
#                 sparql_max_concurrent=sparql_config.get('sparql_max_concurrent', 16),
#                 use_odbc=sparql_config.get('use_odbc', False),
#                 use_aioodbc=sparql_config.get('use_aioodbc', True),
#                 odbc_config=sparql_config.get('odbc_config', None)
#             )
#             self.sparql_manager = SPARQLExecutionManager(sparql_config_obj)
#         else:
#             # 向后兼容：使用GenerationConfig中的配置
#             self.sparql_manager = SPARQLExecutionManager(config.get_sparql_config())

#     def _save_dialogue_log(self, dialogue_data: List[Dict], sample_indices: List[int]):
#         """保存对话数据到日志文件"""
#         if not self.config.enable_logging:
#             return
        
#         try:
#             # 获取实验名称
#             experiment_name = getattr(self.config, 'experiment_name', None)
#             if not experiment_name:
#                 # 尝试从环境变量获取
#                 experiment_name = os.getenv('EXPERIMENT_NAME', 'unknown_experiment')
            
#             # 获取当前步数
#             current_step = getattr(self.config, 'current_step', None)
#             if not current_step:
#                 current_step = self._call_counter
            
#             # 创建实验特定的日志目录
#             if experiment_name and experiment_name != 'unknown_experiment':
#                 log_dir = os.path.join(self.config.log_dir, experiment_name)
#             else:
#                 log_dir = self.config.log_dir
            
#             if not os.path.exists(log_dir):
#                 os.makedirs(log_dir, exist_ok=True)
            
#             # 确定日志文件名
#             if self.config.log_filename:
#                 base_filename = self.config.log_filename
#             else:
#                 # 自动生成文件名
#                 timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#                 base_filename = f"generation_logs_{timestamp}"
            
#             # 添加步数信息到文件名
#             if current_step is not None:
#                 log_filename = f"{base_filename}_step_{current_step}.jsonl"
#             else:
#                 log_filename = f"{base_filename}.jsonl"
            
#             # 完整的日志文件路径
#             log_file_path = os.path.join(log_dir, log_filename)
            
#             # 准备日志条目
#             log_entry = {
#                 "timestamp": datetime.datetime.now().isoformat(),
#                 "call_counter": self._call_counter,
#                 "current_step": current_step,
#                 "experiment_name": experiment_name,
#                 "sample_count": len(dialogue_data),
#                 "sample_indices": sample_indices,
#                 "dialogues": dialogue_data
#             }
            
#             # 追加到JSONL文件
#             with open(log_file_path, 'a', encoding='utf-8') as f:
#                 f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
            
#             logger.info(f"[GENERATION-LOG] 🚀 EXPERIMENT: {experiment_name} - Step {current_step} - Saved {len(dialogue_data)} dialogues to {log_file_path}")
            
#         except Exception as e:
#             logger.error(f"Failed to save dialogue log: {e}")

#     def _should_log(self) -> bool:
#         """判断是否应该记录日志"""
#         return (self.config.enable_logging and 
#                 self._call_counter % self.config.log_interval == 0)

#     def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
#         """Tokenize a batch of responses."""
#         return self.tokenizer(
#             responses, 
#             add_special_tokens=False, 
#             return_tensors='pt', 
#             padding="longest"
#         )['input_ids']

#     def _postprocess_responses(self, responses: torch.Tensor) -> Tuple[torch.Tensor, List[str]]:
#         """Process responses to stop at SPARQL operation or answer operation."""
#         # 添加输入检查
#         #  
#         if responses is None or responses.shape[0] == 0:
#             # print("Warning: empty responses in _postprocess_responses")
#             return torch.zeros((0, 0), dtype=torch.long), []
        
#         responses_str = self.tokenizer.batch_decode(
#             responses, 
#             skip_special_tokens=True
#         )

#         responses_str = [resp.split('</sparql>')[0] + '</sparql>'
#                  if '</sparql>' in resp 
#                  else resp.split('</answer>')[0] + '</answer>'
#                  if '</answer>' in resp 
#                  else resp
#                  for resp in responses_str]

#         if self.config.no_think_rl:
#             raise ValueError('stop')
            
#         responses = self._batch_tokenize(responses_str)
#         return responses, responses_str

#     def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
#         """Process next observations from environment."""
        
#         next_obs_ids = self.tokenizer(
#             next_obs, 
#             padding='longest',
#             return_tensors='pt',
#             add_special_tokens=False,  # Prevents adding special tokens
#         )['input_ids']

#         # Safety check after tokenization
#         if next_obs_ids.shape[1] > self.config.max_obs_length:
#             logger.warning(f"Observation still too long after SPARQL truncation: {next_obs_ids.shape[1]} > {self.config.max_obs_length}, "
#                          f"consider increasing max_obs_length config or check _truncate_sparql_result method")
#             next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

#         return next_obs_ids

#     def _update_rolling_state(self, rollings, cur_responses: torch.Tensor, 
#                             next_obs_ids: torch.Tensor) -> Dict:
#         """Update rolling state with new responses and observations."""
#         # Concatenate and handle padding        
#         new_input_ids = self.tensor_fn.concatenate_with_padding([
#             rollings.batch['input_ids'],
#             cur_responses,
#             next_obs_ids
#         ])
        
#         # Create attention mask and position ids
#         new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
#         #  
#         new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

#         # Cut to appropriate length
#         effective_len = new_attention_mask.sum(dim=1).max()
#         max_len = min(self.config.max_prompt_length, effective_len)
        
#         return DataProto.from_dict({
#             'input_ids': new_input_ids[:, -max_len:],
#             'position_ids': new_position_ids[:, -max_len:],
#             'attention_mask': new_attention_mask[:, -max_len:]
#         })

#     def _update_right_side(self, right_side: Dict, 
#                           cur_responses: torch.Tensor,
#                           next_obs_ids: torch.Tensor = None) -> Dict:
#         """Update right side state."""
#         if next_obs_ids is not None:
#             responses = self.tensor_fn.concatenate_with_padding([
#                 right_side['responses'],
#                 cur_responses,
#                 next_obs_ids
#             ], pad_to_left=False)
#         else:
#             responses = self.tensor_fn.concatenate_with_padding([
#                 right_side['responses'],
#                 cur_responses,
#             ], pad_to_left=False)
        
#         effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
#         max_len = min(self.config.max_prompt_length, effective_len)
        
#         return {'responses': responses[:, :max_len]}

#     def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
#         """
#             Wrapper for generation that handles multi-GPU padding requirements.
#             if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
#             if active_batch size is not divisible by num_gpus, pad with first sequence
#             then remove padding from output
#         """
#         num_gpus = self.config.num_gpus
#         if num_gpus <= 1:
#             return self.actor_rollout_wg.generate_sequences(active_batch)
            
#         batch_size = active_batch.batch['input_ids'].shape[0]
#         remainder = batch_size % num_gpus
        
#         # Ensure all batch tensors are long type (align with Search-R1)
#         for key in active_batch.batch.keys():
#             active_batch.batch[key] = active_batch.batch[key].long()
        
#         if remainder == 0:
#             return self.actor_rollout_wg.generate_sequences(active_batch)
        
#         # Add padding sequences
#         padding_size = num_gpus - remainder
#         padded_batch = {}
        
#         for k, v in active_batch.batch.items():
#             # Use first sequence as padding template
#             pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
#             padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

#         padded_active_batch = DataProto.from_dict(padded_batch)
#         # Ensure padded batch tensors are also long type
#         for key in padded_active_batch.batch.keys():
#             padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

#         # Generate with padded batch
#         padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)
        
#         # Remove padding from output
#         trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
#         # Handle meta_info if present
#         if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
#             trimmed_meta = {}
#             for k, v in padded_output.meta_info.items():
#                 if isinstance(v, torch.Tensor):
#                     trimmed_meta[k] = v[:-padding_size]
#                 else:
#                     trimmed_meta[k] = v
#             padded_output.meta_info = trimmed_meta
            
#         padded_output.batch = trimmed_batch
#         return padded_output

#     def _compose_final_output(self, left_side: Dict,
#                             right_side: Dict,
#                             meta_info: Dict):
#         """Compose final generation output."""
#         final_output = right_side.copy()
#         final_output['prompts'] = left_side['input_ids']
        
#         # Combine input IDs
#         final_output['input_ids'] = torch.cat([
#             left_side['input_ids'],
#             right_side['responses']
#         ], dim=1)
        
#         # Create attention mask and position ids
#         final_output['attention_mask'] = torch.cat([
#             self.tensor_fn.create_attention_mask(left_side['input_ids']),
#             self.tensor_fn.create_attention_mask(final_output['responses'])
#         ], dim=1)
#         #  
        
#         final_output['position_ids'] = self.tensor_fn.create_position_ids(
#             final_output['attention_mask']
#         )

        
#         final_output = DataProto.from_dict(final_output)
#         final_output.meta_info.update(meta_info)
        
#         return final_output

#     def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor):
#         """Run main LLM generation loop."""
#         # 增加调用计数器
#         LLMGenerationManager._call_counter += 1
        
#         # 初始化对话数据收集
#         dialogue_data = []
        
#         # 使用 gen_batch 的形状确定批量大小
#         #  
#         batch_size = gen_batch.batch['input_ids'].shape[0]
        
#         # 打印初始形状信息
#         # print(f"Initial shapes - gen_batch: {gen_batch.batch['input_ids'].shape}, initial_input_ids: {initial_input_ids.shape}")
        
#         # 确保初始输入IDs的批次大小与gen_batch匹配
#         if initial_input_ids.shape[0] != batch_size:
#             print(f"Warning: initial_input_ids batch size ({initial_input_ids.shape[0]}) != gen_batch batch size ({batch_size})")
#             if initial_input_ids.shape[0] > batch_size:
#                 initial_input_ids = initial_input_ids[:batch_size]
#             else:
#                 # 如果initial_input_ids批次大小小于gen_batch，则用第一个示例填充
#                 pad_count = batch_size - initial_input_ids.shape[0]
#                 padding = initial_input_ids[0:1].repeat(pad_count, 1)
#                 initial_input_ids = torch.cat([initial_input_ids, padding], dim=0)
#             print(f"Adjusted initial_input_ids shape: {initial_input_ids.shape}")
        
#         original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
#         original_right_side = {'responses': initial_input_ids[:, []]}
        
#         # 用于跟踪活跃轨迹的数量
#         active_mask = torch.ones(batch_size, dtype=torch.bool)
#         active_num_list = [active_mask.sum().item()]
#         rollings = gen_batch
        
#         # 初始化元信息
#         meta_info = {
#             'done': torch.zeros(batch_size, dtype=torch.bool),
#             'turn': 0,
#         }
        
#         # 主生成循环
#         for turn in range(self.config.max_turns):
#             meta_info['turn'] = turn
            
#             # 如果所有样本都完成，则提前退出
#             if not active_mask.sum():
#                 break
                
#             print(f"Turn {turn}: active samples: {active_mask.sum().item()}/{batch_size}")
            
#             rollings.batch = self.tensor_fn.cut_to_effective_len(
#                 rollings.batch,
#                 keys=['input_ids', 'attention_mask', 'position_ids']
#             )
            
#             # 为活跃的批次创建数据
#             rollings_active = DataProto.from_dict(tensors={
#                 k: v[active_mask] for k, v in rollings.batch.items()
#             })
#             #  
#         # 生成响应
#             gen_output = self._generate_with_gpu_padding(rollings_active)
#             meta_info = gen_output.meta_info

#             # 检查响应是否有效
#             if 'responses' not in gen_output.batch or gen_output.batch['responses'].shape[0] == 0:
#                 print("Warning: Invalid response from _generate_with_gpu_padding")
#                 # 创建一个最小的有效响应
#                 active_size = active_mask.sum().item()
#                 dummy_responses = torch.ones((active_size, 1), dtype=torch.long) * self.tokenizer.pad_token_id
#                 gen_output.batch['responses'] = dummy_responses
                
#             # 处理响应
#             responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
#             #  
            
#             # 调试空响应问题（总共只保存10次）
#             if turn == 1 and "" in responses_str and LLMGenerationManager._debug_save_counter < 10:
#                 # 保存空响应的详细信息到文件
#                 import json
#                 import os
#                 from datetime import datetime
                
#                 # 增加保存计数器
#                 LLMGenerationManager._debug_save_counter += 1
                
#                 # 找到空字符串的索引
#                 empty_indices = [i for i, resp in enumerate(responses_str) if resp == ""]
                
#                 debug_info = {
#                     "timestamp": datetime.now().isoformat(),
#                     "turn": turn,
#                     "call_counter": LLMGenerationManager._call_counter,
#                     "debug_save_number": LLMGenerationManager._debug_save_counter,
#                     "empty_response_indices": empty_indices,
#                     "total_responses": len(responses_str),
#                     "active_mask": active_mask.tolist(),
#                     "batch_size": batch_size,
#                     "num_gpus": self.config.num_gpus,
#                     "remainder": rollings_active.batch['input_ids'].shape[0] % self.config.num_gpus
#                 }
                
#                 # 为每个空响应收集详细信息（每次只保存前3个）
#                 empty_samples = []
#                 for idx in empty_indices[:3]:  # 每次只保存前3个空响应
#                     if idx < len(rollings_active.batch['input_ids']):
#                         sample_info = {
#                             "sample_index": idx,
#                             "input_ids": rollings_active.batch['input_ids'][idx].tolist(),
#                             "input_text": self.tokenizer.decode(rollings_active.batch['input_ids'][idx], skip_special_tokens=False),
#                             "attention_mask": rollings_active.batch['attention_mask'][idx].tolist(),
#                             "position_ids": rollings_active.batch['position_ids'][idx].tolist(),
#                             "raw_gen_output": gen_output.batch['responses'][idx].tolist() if 'responses' in gen_output.batch and idx < gen_output.batch['responses'].shape[0] else None,
#                             "raw_gen_output_text": self.tokenizer.decode(gen_output.batch['responses'][idx], skip_special_tokens=False) if 'responses' in gen_output.batch and idx < gen_output.batch['responses'].shape[0] else None,
#                             "postprocessed_response": responses_str[idx],
#                             "postprocessed_ids": responses_ids[idx].tolist() if idx < len(responses_ids) else None
#                         }
#                         empty_samples.append(sample_info)
                
#                 debug_info["empty_samples"] = empty_samples
                
#                 # 保存到文件
#                 debug_dir = "debug_logs"
#                 os.makedirs(debug_dir, exist_ok=True)
#                 debug_file = os.path.join(debug_dir, f"empty_response_debug_generation_{LLMGenerationManager._debug_save_counter:02d}.json")
                
#                 with open(debug_file, 'w', encoding='utf-8') as f:
#                     json.dump(debug_info, f, indent=2, ensure_ascii=False)
                
#                 print(f"[DEBUG {LLMGenerationManager._debug_save_counter}/10] Found {len(empty_indices)} empty responses in turn 1. Saved {len(empty_samples)} samples to: {debug_file}")
#             elif turn == 1 and "" in responses_str:
#                 # 超过10次保存限制，只记录日志
#                 empty_count = len([i for i, resp in enumerate(responses_str) if resp == ""])
#                 print(f"[DEBUG SKIP] Found {empty_count} empty responses in turn 1, but debug save limit (10) reached")
            
#             # 检查响应是否为空
#             if len(responses_str) == 0 or all(not s for s in responses_str):
#                 print("Warning: Empty responses after postprocessing, marking all active samples as done")
#                 # 标记所有活跃样本为完成
#                 active_mask = torch.zeros_like(active_mask)
#                 meta_info['done'] = ~active_mask
#                 continue
            
#             # 应用示例级填充，将响应扩展回完整批次大小
#             responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)
            
#             # 执行预测
#             next_obs, dones = self.execute_predictions(responses_str, self.tokenizer.pad_token, active_mask)
            
#             # 处理观察
#             next_obs_ids = self._process_next_obs(next_obs) #TODO modify this for too long oberservation
            
#             # 更新活跃掩码
#             curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
#             active_mask = active_mask * curr_active_mask  # 保留乘法，效果相同
#             active_num_list.append(active_mask.sum().item())
            
#             # 更新状态
#             rollings = self._update_rolling_state(rollings, responses_ids, next_obs_ids)
#             original_right_side = self._update_right_side(original_right_side, responses_ids, next_obs_ids)
            
#             # 收集对话数据用于日志记录
#             if self.config.enable_logging:
#                 for i, (resp_str, obs_str) in enumerate(zip(responses_str, next_obs)):
#                     if i >= len(dialogue_data):
#                         dialogue_data.append({
#                             "sample_id": i,
#                             "turns": []
#                         })
                    
#                     # 直接记录原始响应和观察
#                     dialogue_data[i]["turns"].append({
#                         "turn": turn,
#                         "raw_response": resp_str,
#                         "raw_observation": obs_str
#                     })
            
#             # 更新完成状态
#             meta_info['done'] = ~active_mask

#         # final LLM rollout
#         if active_mask.sum():
#             rollings.batch = self.tensor_fn.cut_to_effective_len(
#                 rollings.batch,
#                 keys=['input_ids', 'attention_mask', 'position_ids']
#             )

#             rollings_active = DataProto.from_dict({
#                 k: v[active_mask] for k, v in rollings.batch.items()
#             })            
#             gen_output = self._generate_with_gpu_padding(rollings_active)

#             meta_info = gen_output.meta_info            
#             responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
#             responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

#             # Execute in environment and process observations
#             _, dones = self.execute_predictions(responses_str, self.tokenizer.pad_token, active_mask, do_sparql=False)

#             curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
#             active_mask = active_mask * curr_active_mask
#             active_num_list.append(active_mask.sum().item())

#             original_right_side = self._update_right_side(original_right_side, responses_ids)
        
#             # 收集最终turn的对话数据
#             if self.config.enable_logging:
#                 for i, resp_str in enumerate(responses_str):
#                     if i < len(dialogue_data):
#                         # 直接记录最终响应
#                         dialogue_data[i]["turns"].append({
#                             "turn": "final",
#                             "raw_response": resp_str,
#                             "raw_observation": ""
#                         })
        
#         print(f"Active trajectory numbers: {active_num_list}")
        
#         # 在方法结束时检查是否需要记录日志
#         if self._should_log() and dialogue_data:
#             # 从对话数据中随机采样指定数量的样本
#             sample_size = min(self.config.log_sample_size, len(dialogue_data))
#             if sample_size < len(dialogue_data):
#                 # 随机选择样本
#                 selected_indices = random.sample(range(len(dialogue_data)), sample_size)
#                 selected_dialogues = [dialogue_data[i] for i in selected_indices]
#             else:
#                 # 使用所有样本
#                 selected_indices = list(range(len(dialogue_data)))
#                 selected_dialogues = dialogue_data
            
#             # 保存对话日志
#             self._save_dialogue_log(selected_dialogues, selected_indices)
            
#             print(f"Logged {len(selected_dialogues)} dialogues (call #{self._call_counter})")
        
#         # 组合最终输出
#         return self._compose_final_output(original_left_side, original_right_side, meta_info)

#     def _truncate_sparql_result(self, result_string: str, max_chars: int = None) -> str:
#         """Intelligently truncate SPARQL results while preserving important information structure."""
#         if max_chars is None:
#             # Conservative estimate based on max_obs_length
#             max_chars = self.config.max_obs_length * 3  # Conservative estimate
        
#         if len(result_string) <= max_chars:
#             return result_string
        
#         # Try to preserve the beginning of results and overall structure
#         lines = result_string.split('\n')
#         truncated_lines = []
#         current_length = 0
        
#         # Keep the first few result lines to ensure users can see what the query returned
#         for line in lines:
#             if current_length + len(line) + 1 > max_chars - 100:  # Reserve space for truncation notice
#                 break
#             truncated_lines.append(line)
#             current_length += len(line) + 1
        
#         # Add truncation notice
#         if len(lines) > len(truncated_lines):
#             remaining_lines = len(lines) - len(truncated_lines)
#             truncated_lines.append(f"... ({remaining_lines} more lines truncated due to length limit)")
#             logger.warning(f"SPARQL result truncated: original {len(lines)} lines, kept {len(truncated_lines)-1} lines, "
#                          f"original length {len(result_string)} chars, truncated to ~{current_length} chars")
        
#         return '\n'.join(truncated_lines)

#     def execute_predictions(self, predictions: List[str], pad_token: str, active_mask=None, do_sparql=True) -> Tuple[List[str], List[bool]]:
#         """Execute predictions and return next observations."""
#         batch_size = len(predictions)
#         next_obs = [""] * batch_size
#         dones = [False] * batch_size
        
#         # 添加调试信息
#         print(f"execute_predictions: predictions length={len(predictions)}, active_mask shape={active_mask.shape if active_mask is not None else None}")
#         print(f"active_mask values: {active_mask}")
        
#         # If active_mask is None, all are active
#         if active_mask is None:
#             active_mask = torch.ones(batch_size, dtype=torch.bool)
        
#         # Extract SPARQL queries
#         sparql_queries = []
#         sparql_indices = []
        
#         # 遍历每个预测结果
#         for i, pred in enumerate(predictions):
#             if not active_mask[i]:
#                 dones[i] = True  # 非活跃样本标记为已完成
#                 continue
            
#             # Extract SPARQL query using regex
#             sparql_match = re.search(r'<sparql>(.*?)</sparql>', pred, re.DOTALL)
#             if sparql_match and do_sparql:
#                 sparql_query = sparql_match.group(1).strip()
#                 sparql_queries.append(sparql_query)
#                 sparql_indices.append(i)
            
#             # Check if prediction contains an answer
#             answer_match = re.search(r'<answer>(.*?)</answer>', pred, re.DOTALL)
#             if answer_match:
#                 dones[i] = True
        
#         # 添加调试信息
#         print(f"Found {len(sparql_queries)} SPARQL queries for execution")
#         #  
#         # if sparql_queries:
#         #     print(f"SPARQL query indices: {sparql_indices}")
        
#         # Execute SPARQL queries
#         if sparql_queries and do_sparql:
#             # 执行 SPARQL 查询
#             sparql_results = self.sparql_manager.execute_batch(sparql_queries)
            
#             # Process results
#             if "results" in sparql_results:
#                 results_list = sparql_results["results"]
#                 for idx, result_idx in enumerate(sparql_indices):
#                     if idx < len(results_list):
#                         result = results_list[idx]
                        
#                         # 处理不同格式的结果
#                         if isinstance(result, dict):
#                             if "error" in result:
#                                 # 处理错误情况
#                                 error_message = result["error"]
#                                 if "unrestricted triple pattern" in error_message:
#                                     logger.warning(f"Invalid SPARQL query rejected: {error_message} | Query: {result.get('query', 'N/A')}")
#                                 next_obs[result_idx] = f"<information>\nSPARQL Query Error: {error_message}\n</information>"
#                             elif "results" in result:
#                                 # 处理正常的结果，添加智能截断
#                                 result_string = SPARQLExecutionManager.results_to_string(result['results'])
#                                 truncated_result = self._truncate_sparql_result(result_string)
#                                 next_obs[result_idx] = f"<information>\n{truncated_result}\n</information>"
#                             else:
#                                 # 其他情况，添加智能截断
#                                 result_string = SPARQLExecutionManager.results_to_string(result)
#                                 truncated_result = self._truncate_sparql_result(result_string)
#                                 next_obs[result_idx] = f"<information>\n{truncated_result}\n</information>"
#                         elif isinstance(result, str):
#                             # 处理字符串结果（可能包含错误消息），添加智能截断
#                             truncated_result = self._truncate_sparql_result(result)
#                             next_obs[result_idx] = f"<information>\n{truncated_result}\n</information>"
#                         else:
#                             # 其他情况，添加智能截断
#                             result_string = SPARQLExecutionManager.results_to_string(result)
#                             truncated_result = self._truncate_sparql_result(result_string)
#                             next_obs[result_idx] = f"<information>\n{truncated_result}\n</information>"
#                     else:
#                         # 处理索引越界的情况
#                         print(f"Warning: index {idx} is out of range for SPARQL results")
#                         next_obs[result_idx] = "<information>\nNo results found.\n</information>"
#             else:
#                 print(f"Warning: No 'results' key in SPARQL response: {sparql_results}")
#                 for result_idx in sparql_indices:
#                     next_obs[result_idx] = "<information>\nError: Invalid SPARQL response format\n</information>"
        
#         return next_obs, dones

#     def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[bool]]:
#         """Postprocess predictions to determine active and done indices."""
#         active_indices = []
#         done_indices = []
        
#         for i, pred in enumerate(predictions):
#             # Check if prediction contains an answer
#             answer_match = re.search(r'<answer>(.*?)</answer>', pred, re.DOTALL)
#             if answer_match:
#                 done_indices.append(i)
#             else:
#                 active_indices.append(i)
        
#         return active_indices, done_indices

#     # SPARQL execution is now handled by sparql_manager
#     # Methods removed: batch_sparql, _batch_sparql, _async_batch_sparql, 
#     # _async_sparql_batch, _batch_sparql_odbc, _results2string 