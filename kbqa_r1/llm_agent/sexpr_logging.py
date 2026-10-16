"""
S-Expression Logging Module
Handles dialogue logging and long observation recording
"""

import datetime
import json
import logging
import os
import random
from typing import Dict, List

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))  # Default to INFO for S-Expression logs


class SExprLoggingManager:
    """
    Manages logging for S-Expression generation including dialogue logs and long observations
    """
    
    def __init__(self, config):
        self.config = config
        # 强制关闭文件日志保存（用户需求：NAS空间不足，仅保留checkpoint）
        # 可通过设置环境变量 ENABLE_SEXPR_FILE_LOGS=1 临时恢复
        if os.getenv("ENABLE_SEXPR_FILE_LOGS", "0").lower() not in ("1", "true", "yes"):
            self.config.enable_logging = False
    
    def save_dialogue_log(self, dialogue_data: List[Dict], sample_indices: List[int], call_counter: int):
        """Save dialogue data to log file with enhanced filtering and organization"""
        if not self.config.enable_logging:
            return
        
        try:
            # Get experiment name and step info from config or environment
            experiment_name = getattr(self.config, 'experiment_name', None)
            if not experiment_name:
                # Try to get from environment variable
                experiment_name = os.getenv('EXPERIMENT_NAME', 'unknown_experiment')
            
            # Get current step info
            current_step = getattr(self.config, 'current_step', None)
            if not current_step:
                # Try to get from call counter as fallback
                current_step = call_counter
            
            # Create experiment-specific directory
            if experiment_name:
                log_dir = os.path.join(self.config.log_dir, experiment_name)
            else:
                log_dir = self.config.log_dir
            
            if not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            # Filter dialogues to prioritize special operations
            # Expand and make case-insensitive; also match sexpr keywords
            special_actions = [
                'Merge', 'Order', 'Compare', 'Time_constraint', 'Count',
                'ARGMIN', 'ARGMAX', 'COUNT', 'CMP', 'TC', ' le', ' ge', ' lt', ' gt'
            ]
            prioritized_dialogues = []
            regular_dialogues = []
            
            for dialogue in dialogue_data:
                has_special_action = False
                for turn in dialogue.get('turns', []):
                    raw_response = str(turn.get('raw_response', '')).lower()
                    raw_observation = str(turn.get('raw_observation', '')).lower()
                    # Check if response or observation contains any special action
                    for action in special_actions:
                        if action.lower() in raw_response or action.lower() in raw_observation:
                            has_special_action = True
                            break
                    if has_special_action:
                        break
                
                if has_special_action:
                    prioritized_dialogues.append(dialogue)
                else:
                    regular_dialogues.append(dialogue)
            
            # Determine which dialogues to save
            dialogues_to_save = []
            if prioritized_dialogues:
                # Randomize order within prioritized and regular groups, then cap to size
                prio_shuffled = list(prioritized_dialogues)
                reg_shuffled = list(regular_dialogues)
                random.shuffle(prio_shuffled)
                random.shuffle(reg_shuffled)
                if len(prio_shuffled) >= self.config.log_sample_size:
                    dialogues_to_save = prio_shuffled[: self.config.log_sample_size]
                else:
                    dialogues_to_save = prio_shuffled
                    remaining_slots = self.config.log_sample_size - len(dialogues_to_save)
                    dialogues_to_save.extend(reg_shuffled[:remaining_slots])
            else:
                # No special actions found, randomize selection up to log_sample_size
                if len(dialogue_data) <= self.config.log_sample_size:
                    dialogues_to_save = list(dialogue_data)
                else:
                    # Shuffle and take head for efficiency and reproducibility control via seed
                    shuffled = list(dialogue_data)
                    random.shuffle(shuffled)
                    dialogues_to_save = shuffled[: self.config.log_sample_size]
            
            # Create filename with step information
            if self.config.log_filename:
                base_filename = self.config.log_filename
            else:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                base_filename = f"sexpr_generation_logs_{timestamp}"
            
            # Add step information to filename
            if current_step is not None:
                filename = f"{base_filename}_step_{current_step}.jsonl"
            else:
                filename = f"{base_filename}.jsonl"
            
            log_file_path = os.path.join(log_dir, filename)
            
            # Build indices of selected dialogues relative to input ordering
            # Create a robust mapping from selected dialogues back to their indices (handles potential duplicates)
            id_to_indices = {}
            for i, d in enumerate(dialogue_data):
                id_to_indices.setdefault(id(d), []).append(i)
            selected_indices = []
            for d in dialogues_to_save:
                lst = id_to_indices.get(id(d), [])
                if lst:
                    selected_indices.append(lst.pop(0))

            # Create log entry with enhanced metadata
            log_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "call_counter": call_counter,
                "current_step": current_step,
                "experiment_name": experiment_name,
                "sample_count": len(dialogues_to_save),
                "total_samples": len(dialogue_data),
                "prioritized_samples": len(prioritized_dialogues),
                "regular_samples": len(regular_dialogues),
                "sample_indices": selected_indices,
                "dialogues": dialogues_to_save,
                "mode": "sexpr" if self.config.enable_sexpr_mode else "sparql",
                "special_actions_detected": [action for action in special_actions 
                                           if any(action.lower() in str(dialogue).lower() for dialogue in dialogues_to_save)]
            }
            
            with open(log_file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
            
            # Enhanced logging with priority information
            priority_info = ""
            if prioritized_dialogues:
                priority_info = f" (prioritized {len(prioritized_dialogues)} with special actions)"
            
            logger.info(f"[SEXPR-LOG] 🚀 EXPERIMENT: {experiment_name} - Step {current_step} - Saved {len(dialogues_to_save)} S-Expression dialogues to {log_file_path}{priority_info}")
            logger.info(f"[SEXPR-LOG] 🚀 EXPERIMENT: {experiment_name} - Step {current_step} - Special actions detected: {log_entry['special_actions_detected']}")
            
        except Exception as e:
            logger.error(f"Failed to save dialogue log: {e}")
    
    def log_long_observations(self, next_obs: List[str], actual_length: int):
        """
        记录过长的observation到专门的日志文件（只记录最长的那个）
        """
        try:
            if not self.config.enable_logging:
                return
            if not next_obs:
                return
                
            # 找到最长的observation
            longest_obs = ""
            longest_index = -1
            longest_char_length = 0
            
            for i, obs in enumerate(next_obs):
                if obs and len(obs) > longest_char_length:
                    longest_obs = obs
                    longest_index = i
                    longest_char_length = len(obs)
            
            # 如果没有找到非空observation，直接返回
            if not longest_obs:
                return
            
            # 获取实验名称
            experiment_name = getattr(self.config, 'experiment_name', None)
            if not experiment_name:
                experiment_name = os.getenv('EXPERIMENT_NAME', 'unknown_experiment')
            
            # 获取当前步数
            current_step = getattr(self.config, 'current_step', None)
            if not current_step:
                current_step = 0
            
            # 创建实验特定的日志目录
            base_log_dir = getattr(self.config, 'log_dir', 'logs')
            if experiment_name and experiment_name != 'unknown_experiment':
                log_dir = os.path.join(base_log_dir, experiment_name)
            else:
                log_dir = base_log_dir
            
            os.makedirs(log_dir, exist_ok=True)
            
            # 创建专门的日志文件名
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"long_observations_step{current_step}_{timestamp}.log"
            log_path = os.path.join(log_dir, log_filename)
            
            # 写入最长的observation信息
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write("=== Longest Observation Log ===\n")
                f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
                f.write(f"Batch Max Token Length: {actual_length}\n")
                f.write(f"Max Length Limit: {self.config.max_obs_length}\n")
                f.write(f"Batch Size: {len(next_obs)}\n")
                f.write(f"Experiment: {experiment_name}\n")
                f.write(f"Step: {current_step}\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"--- Longest Sample (Index {longest_index}) ---\n")
                f.write(f"Character Length: {longest_char_length} characters\n")
                f.write(f"Estimated Token Length: {int(longest_char_length * 0.75)} tokens\n")
                f.write(f"Content:\n{longest_obs}\n")
                f.write("-" * 30 + "\n\n")
            
            logger.info(f"[SEXPR-LOG] 🚀 EXPERIMENT: {experiment_name} - Step {current_step} - Longest observation (Sample {longest_index}, {longest_char_length} chars) logged to: {log_path}")
            
        except Exception as e:
            logger.error(f"Failed to log long observations: {e}")
    
    def should_log(self, call_counter: int) -> bool:
        """Check if logging should be performed"""
        return (self.config.enable_logging and 
                call_counter % self.config.log_interval == 0)
    
    def save_threshold_not_met_stats(self, threshold_not_met_counts: Dict, call_counter: int):
        """保存阈值未满足的统计信息"""
        try:
            if not self.config.enable_logging:
                return
            if not threshold_not_met_counts:
                return
            
            # 获取实验名称
            experiment_name = getattr(self.config, 'experiment_name', None)
            if not experiment_name:
                experiment_name = os.getenv('EXPERIMENT_NAME', 'unknown_experiment')
            
            # 创建实验特定的统计目录
            base_log_dir = self.config.log_dir or 'logs'
            if experiment_name and experiment_name != 'unknown_experiment':
                stats_dir = os.path.join(base_log_dir, experiment_name)
            else:
                stats_dir = base_log_dir
            
            os.makedirs(stats_dir, exist_ok=True)
            stats_path = os.path.join(stats_dir, 'relation_threshold_not_met_stats.jsonl')
            payload = {
                'timestamp': datetime.datetime.now().isoformat(),
                'call_counter': call_counter,
                'counts': dict(threshold_not_met_counts)
            }
            with open(stats_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(payload, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.warning(f"Failed to write threshold-not-met stats: {e}")

    def save_relation_similarity_distributions(self, per_step_scores: Dict[int, List[float]], call_counter: int):
        """保存每个step的 select_best_relation 相似度分布"""
        try:
            if not self.config.enable_logging:
                return
            if not per_step_scores:
                return
            
            # 获取实验名称
            experiment_name = getattr(self.config, 'experiment_name', None)
            if not experiment_name:
                experiment_name = os.getenv('EXPERIMENT_NAME', 'unknown_experiment')
            
            # 创建实验特定的统计目录
            base_log_dir = self.config.log_dir or 'logs'
            if experiment_name and experiment_name != 'unknown_experiment':
                stats_dir = os.path.join(base_log_dir, experiment_name)
            else:
                stats_dir = base_log_dir
            
            os.makedirs(stats_dir, exist_ok=True)
            stats_path = os.path.join(stats_dir, 'relation_similarity_distributions.jsonl')
            payload = {
                'timestamp': datetime.datetime.now().isoformat(),
                'call_counter': call_counter,
                'per_step_scores': {str(k): [float(vv) for vv in v] for k, v in per_step_scores.items()}
            }
            with open(stats_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(payload, ensure_ascii=False) + '\n')
            # brief console summary
            logger.info(f"[SEXPR-LOG] Saved relation similarity distributions for {len(per_step_scores)} steps -> {stats_path}")
        except Exception as e:
            logger.warning(f"Failed to write relation similarity distributions: {e}")
