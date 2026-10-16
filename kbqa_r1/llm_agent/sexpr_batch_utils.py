"""
Batch processing utilities for S-Expression generation
Handles GPU padding, tensor operations, and batch management
"""

import logging
from typing import Dict, List, Tuple

import torch

from verl import DataProto

from .tensor_helper import TensorHelper

# Initialize logger
logger = logging.getLogger(__name__)


class SExprBatchUtils:
    """
    Handles batch processing utilities for S-Expression generation
    """
    
    def __init__(self, tensor_helper: TensorHelper, config, actor_rollout_wg=None, tokenizer=None):
        self.tensor_fn = tensor_helper
        self.config = config
        self.actor_rollout_wg = actor_rollout_wg
        self.tokenizer = tokenizer
    
    def generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """Generate with GPU padding (same as original)"""
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        # Ensure all batch tensors are long type (align with Search-R1)
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        # Ensure padded batch tensors are also long type
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()
        
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)
        
        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output
    
    def update_rolling_state(self, rollings, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor, is_final_turn: bool = False) -> Dict:
        """Update rolling state (same as original)"""
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        result = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        
        # 只在最后一轮计算截断统计信息 (优化性能)
        if is_final_turn:
            batch_effective_lengths = new_attention_mask.sum(dim=1)  # (batch_size,)
            batch_is_truncated = batch_effective_lengths > self.config.max_prompt_length  # (batch_size,)
            batch_truncation_ratios = torch.where(
                batch_effective_lengths > 0,
                (batch_effective_lengths - max_len) / batch_effective_lengths,
                torch.zeros_like(batch_effective_lengths, dtype=torch.float)
            )  # (batch_size,)
            
            truncation_info = {
                'batch_effective_lengths': batch_effective_lengths,
                'batch_is_truncated': batch_is_truncated,
                'batch_truncation_ratios': batch_truncation_ratios,
                'max_effective_length': effective_len.item(),
                'max_prompt_length': self.config.max_prompt_length,
                'truncated_length': max_len,
                'batch_truncation_ratio': batch_is_truncated.float().mean().item(),  # batch级别的截断比例
                'batch_mean_effective_length': batch_effective_lengths.float().mean().item(),
                'batch_mean_truncation_ratio': batch_truncation_ratios.mean().item()
            }
            
            # 将截断信息添加到meta_info中
            if not hasattr(result, 'meta_info'):
                result.meta_info = {}
            result.meta_info.update(truncation_info)
        
        return result
    
    def record_final_truncation_info(self, rollings, meta_info: Dict, batch_size: int):
        """
        Record truncation information directly from current rollings state.
        This avoids using empty tensors which cause calculation issues.
        """
        if hasattr(rollings, 'batch') and 'attention_mask' in rollings.batch:
            attention_mask = rollings.batch['attention_mask']
            batch_effective_lengths = attention_mask.sum(dim=1)  # (batch_size,)
            max_effective_length = batch_effective_lengths.max().item()
            
            # Calculate truncation statistics
            batch_is_truncated = batch_effective_lengths > self.config.max_prompt_length
            batch_truncation_ratios = torch.where(
                batch_effective_lengths > 0,
                torch.clamp((batch_effective_lengths - self.config.max_prompt_length).float() / batch_effective_lengths, min=0.0),
                torch.zeros_like(batch_effective_lengths, dtype=torch.float)
            )
            
            # Calculate final truncated length (what's actually used)
            truncated_length = min(self.config.max_prompt_length, max_effective_length)
            
            # Record truncation metrics
            meta_info['turn_final/truncation/clip_ratio'] = batch_is_truncated.float().mean().item()
            meta_info['turn_final/truncation/mean_effective_length'] = batch_effective_lengths.float().mean().item()
            meta_info['turn_final/truncation/mean_truncation_ratio'] = batch_truncation_ratios.mean().item()
            meta_info['turn_final/truncation/max_effective_length'] = float(max_effective_length)
            meta_info['turn_final/truncation/max_prompt_length'] = self.config.max_prompt_length
            meta_info['turn_final/truncation/truncated_length'] = truncated_length
            
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"[SEXPR] Recorded final truncation info: clip_ratio={meta_info['turn_final/truncation/clip_ratio']:.4f}, "
                        f"mean_effective_len={meta_info['turn_final/truncation/mean_effective_length']:.1f}, "
                        f"max_effective_len={max_effective_length}")
        else:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("[SEXPR] Cannot record truncation info: no attention_mask in rollings")
            # Set default values to avoid missing metrics
            meta_info['turn_final/truncation/clip_ratio'] = 0.0
            meta_info['turn_final/truncation/mean_effective_length'] = 0.0
            meta_info['turn_final/truncation/mean_truncation_ratio'] = 0.0
            meta_info['turn_final/truncation/max_effective_length'] = 0.0
            meta_info['turn_final/truncation/max_prompt_length'] = self.config.max_prompt_length
            meta_info['turn_final/truncation/truncated_length'] = 0
    
    def info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                response: torch.Tensor, 
                info: torch.Tensor = None,
                pad_to_left: bool = True
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info

    def update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          cur_rollout_log_probs: torch.Tensor = None,
                          next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state.
        
        IMPORTANT: Also handles rollout_log_probs accumulation for mismatch metrics.
        rollout_log_probs should be updated in sync with responses.
        """
        if next_obs_ids is not None:
            responses, responses_with_info_mask = self.info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    next_obs_ids, 
                    pad_to_left=False
                )
        else:
            responses, responses_with_info_mask = self.info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False
                )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        result = {
            'responses': responses[:, :max_len], 
            'responses_with_info_mask': responses_with_info_mask[:, :max_len]
        }
        
        # CRITICAL FIX: Update rollout_log_probs in sync with responses
        # rollout_log_probs correspond to response tokens only (not observations)
        if cur_rollout_log_probs is not None:
            # For observations, we need to pad log_probs with zeros (no log_probs for info blocks)
            if next_obs_ids is not None:
                obs_length = next_obs_ids.shape[1]
                # Create zero padding for observation tokens
                zero_log_probs = torch.zeros(
                    cur_rollout_log_probs.shape[0], 
                    obs_length,
                    dtype=cur_rollout_log_probs.dtype,
                    device=cur_rollout_log_probs.device
                )
                # Concatenate: response_log_probs + zero_padding_for_obs
                cur_turn_log_probs = torch.cat([cur_rollout_log_probs, zero_log_probs], dim=1)
            else:
                cur_turn_log_probs = cur_rollout_log_probs
            
            # Concatenate with previous turns' rollout_log_probs
            if 'rollout_log_probs' not in right_side:
                # First turn: pad left with zeros to match prompt length (which is excluded)
                accumulated_log_probs = cur_turn_log_probs
            else:
                # Subsequent turns: concatenate
                accumulated_log_probs = torch.cat([
                    right_side['rollout_log_probs'],
                    cur_turn_log_probs
                ], dim=1)
            
            # Apply the same truncation as responses
            result['rollout_log_probs'] = accumulated_log_probs[:, :max_len]
            
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"[MISMATCH FIX] Updated rollout_log_probs in update_right_side, "
                       f"cur_turn shape: {cur_turn_log_probs.shape}, "
                       f"accumulated shape before truncate: {accumulated_log_probs.shape}, "
                       f"after truncate: {result['rollout_log_probs'].shape}")
        elif 'rollout_log_probs' in right_side:
            # No new log_probs this turn, but preserve existing ones (with truncation)
            result['rollout_log_probs'] = right_side['rollout_log_probs'][:, :max_len]
        
        return result

    def compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict):
        """Compose final output (same as original)
        
        IMPORTANT: Preserves rollout_log_probs from right_side for mismatch metrics computation.
        """
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        #  
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        
        # Add info_mask for information block masking
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(right_side['responses_with_info_mask'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        # CRITICAL FIX: Preserve rollout_log_probs if present in right_side
        # This is needed for mismatch metrics (rollout IS weights) computation
        # rollout_log_probs come from vLLM generate_sequences() during multi-turn loop
        # IMPORTANT: Must align rollout_log_probs length with responses length!
        if 'rollout_log_probs' in right_side:
            # Get final responses length
            response_length = right_side['responses'].shape[1]
            rollout_log_probs = right_side['rollout_log_probs']
            
            # Ensure rollout_log_probs matches responses length
            # This handles the case where previous turns used different max_len
            if rollout_log_probs.shape[1] != response_length:
                # Truncate or pad to match
                if rollout_log_probs.shape[1] > response_length:
                    # Truncate from the right (keep left side)
                    rollout_log_probs = rollout_log_probs[:, :response_length]
                else:
                    # Pad on the right with zeros
                    pad_length = response_length - rollout_log_probs.shape[1]
                    zero_pad = torch.zeros(
                        rollout_log_probs.shape[0],
                        pad_length,
                        dtype=rollout_log_probs.dtype,
                        device=rollout_log_probs.device
                    )
                    rollout_log_probs = torch.cat([rollout_log_probs, zero_pad], dim=1)
            
            final_output['rollout_log_probs'] = rollout_log_probs
            logger.info(f"[MISMATCH FIX] Aligned rollout_log_probs to responses length: {rollout_log_probs.shape} == responses: {right_side['responses'].shape}")
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        
        
        return final_output
