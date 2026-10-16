# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
KBQA-R1 entry point for PPO/DAPO training with GAPD support.
Based on verl 0.8.0's main_ppo.py TaskRunner pattern.
"""

import glob
import logging
import os
import re
import socket
from datetime import datetime

import hydra
import numpy as np
import ray
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.main_ppo import TaskRunner as BaseTaskRunner
from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler, run_ppo
from kbqa_r1.utils.reward_score import mid_reward, qa_em


def _select_rm_score_fn(data_source):
    """Select reward model score function based on data source."""
    if data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle']:
        return qa_em.compute_score_em
    elif data_source in ['webqsp', 'grailqa', 'graphq']:
        return mid_reward.compute_mid_reward
    else:
        raise NotImplementedError(f"Unsupported data source: {data_source}")


class KBQARewardManager:
    """Reward manager for KBQA tasks with error logging."""

    def __init__(self, tokenizer, num_examine, format_score=0.0, structure_format_score=0.0,
                 mid_f1_weight: float = 1.0) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.format_score = format_score
        self.structure_format_score = structure_format_score
        self.mid_f1_weight = mid_f1_weight
        self.rewards_history = []
        self.history_size = 100
        self.training_step = 0
        self._setup_error_logging()

    def _setup_error_logging(self):
        os.makedirs('logs', exist_ok=True)
        self.error_logger = logging.getLogger('test_set_errors')
        self.error_logger.setLevel(logging.INFO)
        for handler in self.error_logger.handlers[:]:
            self.error_logger.removeHandler(handler)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        error_log_file = f'logs/test_set_errors_{timestamp}.log'
        file_handler = logging.FileHandler(error_log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        self.error_logger.addHandler(file_handler)
        self.error_logger.propagate = False
        print(f"[REWARD-MANAGER] Test error logging initialized: {error_log_file}")

    def _log_test_error(self, sample_idx, data_source, sequences_str, ground_truth, base_score, error_type,
                        error_details=""):
        try:
            error_info = {
                'sample_index': sample_idx, 'data_source': data_source,
                'timestamp': datetime.now().isoformat(), 'training_step': self.training_step,
                'error_type': error_type, 'error_details': error_details, 'base_score': base_score,
                'ground_truth': ground_truth,
                'predicted_sequence': sequences_str[:500] + "..." if len(sequences_str) > 500 else sequences_str
            }
            self.error_logger.error(f"TEST_ERROR: {error_info}")
        except Exception as e:
            print(f"[REWARD-MANAGER] Failed to log test error: {e}")

    def _detect_error_patterns(self, sequences_str):
        error_patterns = []
        for tag in ['think', 'action', 'information', 'answer']:
            if len(re.findall(f'<{tag}>', sequences_str)) != len(re.findall(f'</{tag}>', sequences_str)):
                error_patterns.append(f"unbalanced_{tag}_tags")
        if '<answer></answer>' in sequences_str or '<answer> </answer>' in sequences_str:
            error_patterns.append("empty_answer")
        if sequences_str.strip().endswith('<') or sequences_str.strip().endswith('('):
            error_patterns.append("incomplete_sequence")
        if sequences_str.count('(') != sequences_str.count(')'):
            error_patterns.append("unbalanced_parentheses")
        if len(sequences_str.strip()) < 10:
            error_patterns.append("too_short_response")
        words = sequences_str.split()
        if len(words) > 5:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            if max(word_counts.values()) > len(words) * 0.3:
                error_patterns.append("excessive_repetition")
        return error_patterns

    def __call__(self, data: DataProto):
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        all_scores = []
        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            sequences_str = self.tokenizer.decode(valid_response_ids)
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']

            error_patterns = self._detect_error_patterns(sequences_str)
            if error_patterns:
                self._log_test_error(i, data_source, sequences_str, ground_truth, 0.0,
                                     "pattern_errors", f"Patterns: {error_patterns}")

            compute_score_fn = _select_rm_score_fn(data_source)
            score = compute_score_fn(sequences_str, ground_truth)
            all_scores.append(score)
            self.rewards_history.append(score)
            if len(self.rewards_history) > self.history_size:
                self.rewards_history.pop(0)

            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1

        if all_scores:
            print(f"[REWARDS] mean: {np.mean(all_scores):.4f}, max: {np.max(all_scores):.4f}, "
                  f"min: {np.min(all_scores):.4f}, std: {np.std(all_scores):.4f}")

        return reward_tensor

    def update_training_step(self, step):
        self.training_step = step


class KBQATaskRunner(BaseTaskRunner):
    """KBQA-specific TaskRunner that uses KBQARewardManager and custom trainer."""

    def run(self, config):
        from pprint import pprint
        from verl.utils.fs import copy_to_local
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.utils import hf_processor, hf_tokenizer
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer
        from verl.trainer.ppo.utils import need_critic, need_reference_policy
        from verl.utils.config import validate_config

        print(f"KBQATaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_resource_pool(config)
        self.add_teacher_model_resource_pool(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        resource_pool_manager = self.init_resource_pool_mgr(config)

        train_dataset = create_rl_dataset(
            config.data.train_files, config.data, tokenizer, processor,
            is_train=True, max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files, config.data, tokenizer, processor,
            is_train=False, max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        trainer.fit()


@hydra.main(config_path="../../verl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config):
    from verl.utils.device import auto_set_device
    from verl.experimental.reward_loop import migrate_legacy_reward_impl
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)

    task_runner_class = ray.remote(num_cpus=1)(KBQATaskRunner)
    run_ppo(config, task_runner_class=task_runner_class)


if __name__ == "__main__":
    main()
