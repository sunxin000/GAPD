"""GAPD loss function: pg_coef * PPO_loss + jsd_coef * distill_loss.

Supports two modes for the distillation loss:
- Supervised (gapd_use_pg=False): Directly backpropagate mean KL(T||S) as a
  supervised loss. Simple but forces every token position to mimic teacher.
- Policy Gradient (gapd_use_pg=True): Use -KL_per_token.detach() as per-token
  advantages in a PPO-clipped policy gradient loss. The distillation signal
  acts precisely on each sampled token through the PG mechanism, providing
  finer-grained credit assignment and better compatibility with GRPO.

The KL distillation loss is computed inline during the forward pass (inside
prepare_model_outputs) to avoid CUDA driver errors with FSDP +
param_offload=True. This loss function reads the pre-computed
gapd_jsd_loss from model_output and combines it with the PG loss.
"""

import os
import random
from collections.abc import Sequence

import numpy as np
import torch
from tensordict import TensorDict

from verl.utils.metric import AggregationType, Metric
from verl.utils import tensordict_utils as tu
from verl.workers.config import ActorConfig
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import no_padding_2_padding


_GAPD_METRIC_DEFAULT_KEYS = (
    "actor/gapd/teacher_mass",
    "actor/gapd/student_mass",
    "actor/gapd/drop_answer_turns",
    "actor/gapd/drop_same_next_state_turns",
    "actor/gapd/drop_expression_id_risk_turns",
    "actor/gapd/drop_no_action_span_turns",
    "actor/gapd/drop_answer_turn1",
    "actor/gapd/drop_same_next_state_turn1",
    "actor/gapd/drop_expression_id_risk_turn1",
    "actor/gapd/drop_no_action_span_turn1",
    "actor/gapd/initial_supervised_turns",
    "actor/gapd/final_supervised_turns",
    "actor/gapd/drop_total_turns",
    "actor/gapd/raw/kl_mean",
    "actor/gapd/raw/kl_max",
    "actor/gapd/raw/kl_min",
    "actor/gapd/raw/kl_p50",
    "actor/gapd/raw/kl_p90",
    "actor/gapd/raw/kl_std",
    "actor/gapd/raw/kl_valid_tokens",
    "actor/gapd/clip_ratio",
    "actor/gapd/kl_after_clip_mean",
    "actor/gapd/token_clip",
    "actor/gapd/kl_mode",
    "actor/gapd/kl_nonzero_ratio",
    "actor/gapd/distill_adv_mean",
    "actor/gapd/distill_adv_std",
    "actor/gapd/distill_adv_min",
    "actor/gapd/distill_adv_max",
    "actor/gapd/per_sample_kl_mean",
    "actor/gapd/per_sample_kl_max",
    "actor/gapd/per_sample_kl_min",
    "actor/gapd/valid_distill_tokens",
    "actor/gapd_distill_actor/pg_clipfrac",
    "actor/gapd_distill_actor/ppo_kl",
    "actor/gapd_distill_actor/pg_clipfrac_lower",
    "actor/gapd_jsd_loss",
    "actor/gapd_distill_loss",
    "actor/gapd_pg_loss",
    "actor/gapd_total_loss",
    "actor/gapd_teacher_available",
    "actor/gapd_use_pg",
    "actor/gapd_pg_coef",
    "actor/gapd_jsd_coef",
)


def _ensure_gapd_metric_schema(metrics: dict, zero: torch.Tensor | float) -> None:
    """Keep the GAPD metric key set identical across DP ranks/micro-batches."""
    if not isinstance(zero, torch.Tensor):
        zero = torch.tensor(float(zero))
    for key in _GAPD_METRIC_DEFAULT_KEYS:
        if key not in metrics:
            metrics[key] = Metric(value=zero, aggregation=AggregationType.MEAN)


def _log_kl_stats(
    metrics: dict,
    kl_per_token: torch.Tensor,
    mask: torch.Tensor,
    prefix: str = "actor/gapd",
) -> None:
    """Log detailed KL distribution statistics to tensorboard."""
    if mask is None or mask.sum() == 0:
        return

    valid_kl = kl_per_token[mask].float()
    if valid_kl.numel() == 0:
        return

    metrics[f"{prefix}/kl_mean"] = Metric(value=valid_kl.mean(), aggregation=AggregationType.MEAN)
    metrics[f"{prefix}/kl_max"] = Metric(value=valid_kl.max(), aggregation=AggregationType.MEAN)
    metrics[f"{prefix}/kl_min"] = Metric(value=valid_kl.min(), aggregation=AggregationType.MEAN)
    metrics[f"{prefix}/kl_p50"] = Metric(value=valid_kl.median(), aggregation=AggregationType.MEAN)
    metrics[f"{prefix}/kl_p90"] = Metric(value=valid_kl.quantile(0.9), aggregation=AggregationType.MEAN)
    metrics[f"{prefix}/kl_std"] = Metric(value=valid_kl.std(unbiased=False), aggregation=AggregationType.MEAN)
    metrics[f"{prefix}/kl_valid_tokens"] = Metric(
        value=torch.tensor(float(valid_kl.numel())), aggregation=AggregationType.MEAN
    )


def _log_non_tensor_mean_metric(metrics: dict, data: TensorDict, name: str, metric_name: str) -> float | None:
    value = tu.get_non_tensor_data(data, name, default=None)
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        tensor = value.float()
    else:
        if isinstance(value, np.generic):
            values = [value.item()]
        elif isinstance(value, np.ndarray):
            values = value.reshape(-1).tolist()
        elif hasattr(value, "tolist"):
            listed = value.tolist()
            values = listed if isinstance(listed, list) else [listed]
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            values = list(value)
        else:
            values = [value]
        numeric_values = []
        for item in values:
            if hasattr(item, "data"):
                item = item.data
            if isinstance(item, np.generic):
                item = item.item()
            if item is None:
                continue
            numeric_values.append(float(item))
        if not numeric_values:
            return None
        tensor = torch.tensor(numeric_values, dtype=torch.float32)
    metric_value = tensor.mean()
    metrics[metric_name] = Metric(value=metric_value, aggregation=AggregationType.MEAN)
    return float(metric_value.detach().cpu())


def _maybe_dump_token_level_data(
    config,
    data: TensorDict,
    gapd_jsd_loss_raw: torch.Tensor,
    gapd_jsd_loss_clipped: torch.Tensor,
    distill_advantages: torch.Tensor,
    log_prob: torch.Tensor | None,
    old_log_prob: torch.Tensor | None,
    distill_mask: torch.Tensor,
):
    """Dump token-level GAPD data for offline analysis.

    Selects one representative sample from the batch and writes a .pt file with:
      - GRPO advantages (constant across a rollout)
      - distill advantages (token-level -KL)
      - raw / clipped KL
      - rollout response mask, GAPD distillation mask, gapd_turn_mask
      - log probs, response token ids

    Controlled by config:
        gapd_token_dump_dir: directory to write files
        gapd_token_dump_max_samples: max total dumps across training
        gapd_token_dump_prob: probability of dumping this batch (0.0-1.0)
        gapd_token_dump_strategy: "representative" (default) or "random"
        gapd_token_dump_min_valid_tokens: skip batch if no sample has this many
            supervised distillation tokens
    """
    dump_dir = getattr(config, "gapd_token_dump_dir", None)
    max_dumps = getattr(config, "gapd_token_dump_max_samples", 0)
    dump_prob = getattr(config, "gapd_token_dump_prob", 0.0)
    dump_strategy = getattr(config, "gapd_token_dump_strategy", "representative")
    min_valid_tokens = int(getattr(config, "gapd_token_dump_min_valid_tokens", 1))

    if not dump_dir or max_dumps <= 0 or dump_prob <= 0:
        return

    # Global counter (persisted on the function object)
    counter = getattr(_maybe_dump_token_level_data, "_counter", 0)
    if counter >= max_dumps:
        return
    if random.random() > dump_prob:
        return

    bsz = gapd_jsd_loss_raw.shape[0]

    # GRPO advantages (may be nested). Convert before selecting so the
    # representative sampler can prefer cases where GAPD and GRPO disagree.
    grpo_adv = data["advantages"]
    if grpo_adv.is_nested:
        grpo_adv = grpo_adv.to_padded_tensor(0.0)

    def _select_representative_sample() -> tuple[int, dict] | None:
        if dump_strategy == "random":
            idx = random.randint(0, bsz - 1)
            return idx, {"strategy": "random", "score": None}

        min_len = min(distill_advantages.size(1), distill_mask.size(1), grpo_adv.size(1))
        if min_len <= 0:
            return None

        mask = distill_mask[:, :min_len].bool()
        adv = distill_advantages[:, :min_len].float().masked_fill(~mask, 0.0)
        grpo = grpo_adv[:, :min_len].float().masked_fill(~mask, 0.0)

        valid_tokens = mask.float().sum(dim=1)
        eligible = valid_tokens >= max(min_valid_tokens, 1)
        if not eligible.any():
            return None

        abs_adv = adv.abs()
        mean_abs_adv = abs_adv.sum(dim=1) / valid_tokens.clamp(min=1)
        max_abs_adv = abs_adv.max(dim=1).values
        max_negative_adv = (-adv).clamp_min(0.0).max(dim=1).values
        max_positive_adv = adv.clamp_min(0.0).max(dim=1).values
        has_positive = ((adv > 0) & mask).any(dim=1).float()
        has_negative = ((adv < 0) & mask).any(dim=1).float()
        bidirectional = has_positive * has_negative

        grpo_sign_disagree = ((adv * grpo < 0) & mask).float().sum(dim=1) / valid_tokens.clamp(min=1)

        # High-value dumps should reveal actual token-level supervision, not
        # arbitrary rollouts. Prefer strong guide magnitudes, especially
        # suppression cases, plus examples where GAPD and GRPO push in different
        # directions or where both positive and negative token guides appear.
        score = (
            2.0 * max_abs_adv
            + mean_abs_adv
            + 0.75 * max_negative_adv
            + 0.5 * max_positive_adv
            + 0.5 * bidirectional
            + 0.5 * grpo_sign_disagree
            + 0.01 * valid_tokens.clamp(max=100.0)
        )
        score = score.masked_fill(~eligible, float("-inf"))
        sample_idx_tensor = score.argmax()
        sample_idx_int = int(sample_idx_tensor.item())

        if not torch.isfinite(score[sample_idx_tensor]):
            return None

        selection_debug = {
            "strategy": "representative",
            "score": float(score[sample_idx_tensor].detach().cpu()),
            "valid_distill_tokens": int(valid_tokens[sample_idx_tensor].detach().cpu().item()),
            "mean_abs_distill_adv": float(mean_abs_adv[sample_idx_tensor].detach().cpu()),
            "max_abs_distill_adv": float(max_abs_adv[sample_idx_tensor].detach().cpu()),
            "max_negative_distill_adv": float(max_negative_adv[sample_idx_tensor].detach().cpu()),
            "max_positive_distill_adv": float(max_positive_adv[sample_idx_tensor].detach().cpu()),
            "has_bidirectional_guides": bool(bidirectional[sample_idx_tensor].detach().cpu().item() > 0),
            "grpo_sign_disagree_ratio": float(grpo_sign_disagree[sample_idx_tensor].detach().cpu()),
            "eligible_samples": int(eligible.sum().detach().cpu().item()),
            "batch_size": int(bsz),
        }
        return sample_idx_int, selection_debug

    selected = _select_representative_sample()
    if selected is None:
        return
    sample_idx, dump_selection = selected

    def _optional_padded_tensor(name: str, fill_value):
        value = data.get(name, None)
        if value is not None and value.is_nested:
            value = value.to_padded_tensor(fill_value)
        return value

    def _sample_non_tensor(name: str):
        value = tu.get_non_tensor_data(data, name, default=None)
        if value is None:
            return None
        if isinstance(value, (str, bytes, int, float, bool)):
            return value
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return value.item()
            if value.shape[0] <= sample_idx:
                return None
            value = value[sample_idx]
            return value.item() if isinstance(value, np.generic) else value
        if hasattr(value, "tolist") and not isinstance(value, torch.Tensor):
            values = value.tolist()
            if isinstance(values, list):
                if len(values) <= sample_idx:
                    return None
                return values[sample_idx]
            return values
        if hasattr(value, "data"):
            value = value.data
        if isinstance(value, (str, bytes, int, float, bool)):
            return value
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return value.item()
            if value.shape[0] <= sample_idx:
                return None
            value = value[sample_idx]
            return value.item() if isinstance(value, np.generic) else value
        if hasattr(value, "tolist") and not isinstance(value, torch.Tensor):
            values = value.tolist()
            if isinstance(values, list):
                if len(values) <= sample_idx:
                    return None
                return values[sample_idx]
            return values
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return value.item()
            if value.shape[0] <= sample_idx:
                return None
            value = value[sample_idx]
            return value.item() if value.ndim == 0 else value.detach().cpu()
        if isinstance(value, dict):
            return value
        if isinstance(value, (str, bytes)):
            return value
        if isinstance(value, Sequence):
            if len(value) <= sample_idx:
                return None
            value = value[sample_idx]
            return value.data if hasattr(value, "data") else value
        # Some reward extras arrive as already-selected numpy/Python scalars.
        return value

    returns = _optional_padded_tensor("returns", 0.0)
    rm_scores = _optional_padded_tensor("rm_scores", 0.0)
    token_level_scores = _optional_padded_tensor("token_level_scores", 0.0)
    token_level_rewards = _optional_padded_tensor("token_level_rewards", 0.0)

    # Response ids (may be nested)
    responses = data.get("responses", None)
    if responses is not None and responses.is_nested:
        responses = responses.to_padded_tensor(0)

    prompts = data.get("prompts", None)
    if prompts is not None and prompts.is_nested:
        prompts = prompts.to_padded_tensor(0)

    # gapd_turn_mask (if available)
    gapd_turn_mask = data.get("gapd_turn_mask", None)
    if gapd_turn_mask is not None and gapd_turn_mask.is_nested:
        gapd_turn_mask = gapd_turn_mask.to_padded_tensor(False)

    # Original rollout response mask: 1 for LLM generated tokens, 0 for tool
    # observations/padding. Keep this separate from the GAPD distillation mask.
    response_mask = data.get("response_mask", None)
    if response_mask is not None and response_mask.is_nested:
        response_mask = response_mask.to_padded_tensor(False)

    gapd_turn_ids = data.get("gapd_turn_ids", None)
    if gapd_turn_ids is not None and gapd_turn_ids.is_nested:
        gapd_turn_ids = gapd_turn_ids.to_padded_tensor(0)

    # IS weights
    is_weights = data.get("rollout_is_weights", None)

    # Teacher top-k data (if available) — stored as nested tensors; pad before slicing
    teacher_logprobs = data.get("teacher_logprobs", None)
    teacher_ids = data.get("teacher_ids", None)
    if teacher_logprobs is not None:
        if hasattr(teacher_logprobs, "is_nested") and teacher_logprobs.is_nested:
            teacher_logprobs = teacher_logprobs.to_padded_tensor(-100.0)
    if teacher_ids is not None:
        if hasattr(teacher_ids, "is_nested") and teacher_ids.is_nested:
            teacher_ids = teacher_ids.to_padded_tensor(0)

    student_topk_logprobs = data.get("gapd_student_topk_logprobs", None)
    student_topk_ids = data.get("gapd_student_topk_ids", None)
    if student_topk_logprobs is not None:
        if hasattr(student_topk_logprobs, "is_nested") and student_topk_logprobs.is_nested:
            student_topk_logprobs = student_topk_logprobs.to_padded_tensor(-100.0)
    if student_topk_ids is not None:
        if hasattr(student_topk_ids, "is_nested") and student_topk_ids.is_nested:
            student_topk_ids = student_topk_ids.to_padded_tensor(0)

    # Teacher prompt ids come via non_tensor_batch (Python list), not tensor batch
    teacher_prompt_ids = _sample_non_tensor("teacher_prompt_ids")
    gapd_teacher_prompt_debug = _sample_non_tensor("gapd_teacher_prompt_debug")
    gapd_state_match_debug = _sample_non_tensor("gapd_state_match_debug")
    rollout_debug = _sample_non_tensor("rollout_debug")
    reward_debug = {}
    for name in (
        "uid",
        "data_source",
        "reward",
        "acc",
        "em",
        "f1",
        "mid_f1",
        "format_reward",
        "accuracy_reward",
        "reward_extra_info",
        "extra_info",
        "reward_model",
    ):
        value = _sample_non_tensor(name)
        if value is not None:
            reward_debug[name] = value

    dump = {
        "batch_idx": counter,
        "sample_idx": sample_idx,
        "dump_selection": dump_selection,
        "grpo_advantages": grpo_adv[sample_idx].cpu(),
        "returns": returns[sample_idx].cpu() if returns is not None else None,
        "rm_scores": rm_scores[sample_idx].cpu() if rm_scores is not None else None,
        "token_level_scores": token_level_scores[sample_idx].cpu() if token_level_scores is not None else None,
        "token_level_rewards": token_level_rewards[sample_idx].cpu() if token_level_rewards is not None else None,
        "distill_advantages": distill_advantages[sample_idx].cpu(),
        "kl_raw": gapd_jsd_loss_raw[sample_idx].cpu(),
        "kl_clipped": gapd_jsd_loss_clipped[sample_idx].cpu(),
        "response_mask": response_mask[sample_idx].cpu() if response_mask is not None else None,
        "distill_mask": distill_mask[sample_idx].cpu(),
        "gapd_turn_mask": gapd_turn_mask[sample_idx].cpu() if gapd_turn_mask is not None else None,
        "gapd_turn_ids": gapd_turn_ids[sample_idx].cpu() if gapd_turn_ids is not None else None,
        "log_probs": log_prob[sample_idx].cpu() if log_prob is not None else None,
        "old_log_probs": old_log_prob[sample_idx].cpu() if old_log_prob is not None else None,
        "prompt_ids": prompts[sample_idx].cpu() if prompts is not None else None,
        "response_ids": responses[sample_idx].cpu() if responses is not None else None,
        "rollout_is_weight": is_weights[sample_idx].cpu() if is_weights is not None else None,
        "teacher_logprobs": teacher_logprobs[sample_idx].cpu() if teacher_logprobs is not None else None,
        "teacher_ids": teacher_ids[sample_idx].cpu() if teacher_ids is not None else None,
        "student_topk_logprobs": student_topk_logprobs[sample_idx].cpu() if student_topk_logprobs is not None else None,
        "student_topk_ids": student_topk_ids[sample_idx].cpu() if student_topk_ids is not None else None,
        "teacher_prompt_ids": teacher_prompt_ids,
        "gapd_teacher_prompt_debug": gapd_teacher_prompt_debug,
        "gapd_state_match_debug": gapd_state_match_debug,
        "rollout_debug": rollout_debug,
        "reward_debug": reward_debug,
    }

    os.makedirs(dump_dir, exist_ok=True)
    rank = (
        torch.distributed.get_rank()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else 0
    )
    rank_dir = os.path.join(dump_dir, f"rank{rank}")
    os.makedirs(rank_dir, exist_ok=True)

    path = os.path.join(rank_dir, f"batch{counter:04d}_sample{sample_idx}.pt")
    torch.save(dump, path)

    _maybe_dump_token_level_data._counter = counter + 1


def _get_gapd_mask(data: TensorDict) -> torch.Tensor:
    """Get the distillation mask: gapd_turn_mask if available, else response_mask."""
    gapd_turn_mask = data.get("gapd_turn_mask", None)
    if gapd_turn_mask is not None:
        if gapd_turn_mask.is_nested:
            gapd_turn_mask = gapd_turn_mask.to_padded_tensor(False)
        return gapd_turn_mask.bool()

    response_mask = data.get("response_mask", None)
    if response_mask is not None:
        if response_mask.is_nested:
            response_mask = response_mask.to_padded_tensor(False)
        return response_mask.bool()

    return None


def gapd_ppo_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    """Combined GAPD loss: gapd_pg_coef * PG_loss + gapd_jsd_coef * distill_loss.

    When gapd_use_pg=False (supervised mode):
        total_loss = gapd_pg_coef * PG_loss + gapd_jsd_coef * mean_KL

    When gapd_use_pg=True (policy gradient mode):
        distill_advantages = -KL_per_token.detach()
        distill_pg_loss = PPO_clip(log_prob, old_log_prob, distill_advantages, mask)
        total_loss = gapd_pg_coef * PG_loss + gapd_jsd_coef * distill_pg_loss

    The PG mode uses per-token KL as advantages, so the distillation signal
    precisely acts on each sampled token through the policy gradient mechanism,
    with PPO clipping for stability.
    """
    from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla

    pg_coef = getattr(config, "gapd_pg_coef", 1.0)
    jsd_coef = getattr(config, "gapd_jsd_coef", 0.1)
    use_pg = getattr(config, "gapd_use_pg", False)
    token_clip = getattr(config, "gapd_jsd_token_clip", 10.0)
    kl_mode = getattr(config, "gapd_kl_mode", "forward_kl")

    # Step 1: Compute standard PG loss (GRPO/PPO with outcome rewards)
    pg_loss_val, pg_metrics = ppo_loss(config, model_output, data, dp_group)

    # Step 2: Read pre-computed KL loss from model_output
    gapd_jsd_loss = model_output.get("gapd_jsd_loss", None)

    # Log teacher/student probability mass metrics (stored as non_tensor_data
    # during forward pass by _compute_gapd_kl_inline, since they are scalars
    # and cannot go through the logits_processor_func return path).
    teacher_mass = tu.get_non_tensor_data(data, "gapd_teacher_mass", default=None)
    student_mass = tu.get_non_tensor_data(data, "gapd_student_mass", default=None)
    if teacher_mass is not None:
        pg_metrics["actor/gapd/teacher_mass"] = Metric(value=torch.tensor(float(teacher_mass)), aggregation=AggregationType.MEAN)
    if student_mass is not None:
        pg_metrics["actor/gapd/student_mass"] = Metric(value=torch.tensor(float(student_mass)), aggregation=AggregationType.MEAN)

    drop_answer = _log_non_tensor_mean_metric(
        pg_metrics, data, "gapd_drop_answer_turns", "actor/gapd/drop_answer_turns"
    )
    drop_same_next = _log_non_tensor_mean_metric(
        pg_metrics, data, "gapd_drop_same_next_state_turns", "actor/gapd/drop_same_next_state_turns"
    )
    drop_expression = _log_non_tensor_mean_metric(
        pg_metrics, data, "gapd_drop_expression_id_risk_turns", "actor/gapd/drop_expression_id_risk_turns"
    )
    drop_no_action_span = _log_non_tensor_mean_metric(
        pg_metrics, data, "gapd_drop_no_action_span_turns", "actor/gapd/drop_no_action_span_turns"
    )
    _log_non_tensor_mean_metric(
        pg_metrics, data, "gapd_skip_high_mid_f1_sample", "actor/gapd/skip_high_mid_f1_samples"
    )
    _log_non_tensor_mean_metric(
        pg_metrics, data, "gapd_reward_mid_f1", "actor/gapd/reward_mid_f1"
    )
    _log_non_tensor_mean_metric(
        pg_metrics, data, "gapd_skip_high_mid_f1_threshold", "actor/gapd/skip_high_mid_f1_threshold"
    )
    _log_non_tensor_mean_metric(
        pg_metrics, data, "gapd_drop_answer_turn1", "actor/gapd/drop_answer_turn1"
    )
    _log_non_tensor_mean_metric(
        pg_metrics, data, "gapd_drop_same_next_state_turn1", "actor/gapd/drop_same_next_state_turn1"
    )
    _log_non_tensor_mean_metric(
        pg_metrics, data, "gapd_drop_expression_id_risk_turn1", "actor/gapd/drop_expression_id_risk_turn1"
    )
    _log_non_tensor_mean_metric(
        pg_metrics, data, "gapd_drop_no_action_span_turn1", "actor/gapd/drop_no_action_span_turn1"
    )
    _log_non_tensor_mean_metric(
        pg_metrics, data, "gapd_initial_supervised_turns", "actor/gapd/initial_supervised_turns"
    )
    _log_non_tensor_mean_metric(
        pg_metrics, data, "gapd_final_supervised_turns", "actor/gapd/final_supervised_turns"
    )
    if (
        drop_answer is not None
        or drop_same_next is not None
        or drop_expression is not None
        or drop_no_action_span is not None
    ):
        pg_metrics["actor/gapd/drop_total_turns"] = Metric(
            value=torch.tensor(
                float(drop_answer or 0.0)
                + float(drop_same_next or 0.0)
                + float(drop_expression or 0.0)
                + float(drop_no_action_span or 0.0)
            ),
            aggregation=AggregationType.MEAN,
        )

    if gapd_jsd_loss is None:
        # No KL loss computed -- return weighted PG loss only
        total_loss = pg_coef * pg_loss_val
        zero_metric = pg_loss_val.new_tensor(0.0)
        pg_metrics["actor/gapd_jsd_loss"] = Metric(value=zero_metric, aggregation=AggregationType.MEAN)
        pg_metrics["actor/gapd_teacher_available"] = Metric(value=zero_metric, aggregation=AggregationType.MEAN)
        pg_metrics["actor/gapd_distill_loss"] = Metric(value=zero_metric, aggregation=AggregationType.MEAN)
        pg_metrics["actor/gapd_pg_loss"] = Metric(value=pg_loss_val, aggregation=AggregationType.MEAN)
        pg_metrics["actor/gapd_total_loss"] = Metric(value=total_loss, aggregation=AggregationType.MEAN)
        pg_metrics["actor/gapd_use_pg"] = Metric(value=torch.tensor(float(use_pg)), aggregation=AggregationType.MEAN)
        pg_metrics["actor/gapd_pg_coef"] = Metric(value=torch.tensor(float(pg_coef)), aggregation=AggregationType.MEAN)
        pg_metrics["actor/gapd_jsd_coef"] = Metric(value=torch.tensor(float(jsd_coef)), aggregation=AggregationType.MEAN)
        _ensure_gapd_metric_schema(pg_metrics, zero_metric)
        return total_loss, pg_metrics

    # Convert nested gapd_jsd_loss to padded (bsz, max_response_len) format
    gapd_jsd_loss_padded = no_padding_2_padding(gapd_jsd_loss, data)

    # Get distillation mask (gapd_turn_mask or response_mask)
    response_mask_bool = _get_gapd_mask(data)

    # For forward_kl: clamp negative values (top-k approximation can produce
    # negative KL). Matches verl's compute_forward_kl_topk which applies clamp_min(0).
    # For k1: skip clamp_min(0) since k1 CAN be negative by design (bidirectional signal).
    if kl_mode == "forward_kl":
        gapd_jsd_loss_padded = gapd_jsd_loss_padded.clamp_min(0.0)

    # Log raw KL stats BEFORE token_clip (for monitoring the true KL distribution)
    _log_kl_stats(pg_metrics, gapd_jsd_loss_padded, response_mask_bool, prefix="actor/gapd/raw")

    # Save raw KL before token clipping for potential offline analysis
    gapd_jsd_loss_raw = gapd_jsd_loss_padded

    # Apply token_clip (matching verl's loss_max_clamp pattern).
    # For forward_kl: values are already >= 0 after clamp_min(0), so
    # [-token_clip, token_clip] is effectively [0, token_clip].
    # For k1: bidirectional clamp [-token_clip, token_clip], matching verl's
    # distillation which uses [-10, 10] for k1/k3 losses.
    # Applied to BOTH supervised and PG modes for consistency with verl.
    if token_clip > 0:
        kl_before_clip = gapd_jsd_loss_padded
        if kl_mode == "k1":
            gapd_jsd_loss_padded = gapd_jsd_loss_padded.clamp(min=-token_clip, max=token_clip)
        else:
            gapd_jsd_loss_padded = gapd_jsd_loss_padded.clamp(max=token_clip)
        # Log clip impact metrics
        if response_mask_bool is not None:
            valid_kl_before = kl_before_clip[response_mask_bool]
            valid_kl_after = gapd_jsd_loss_padded[response_mask_bool]
        else:
            valid_kl_before = kl_before_clip.flatten()
            valid_kl_after = gapd_jsd_loss_padded.flatten()
        if valid_kl_before.numel() == 0:
            clip_ratio = gapd_jsd_loss_padded.new_tensor(0.0)
            kl_after_clip_mean = gapd_jsd_loss_padded.new_tensor(0.0)
        else:
            clip_ratio = (valid_kl_before.abs() > token_clip).float().mean()
            kl_after_clip_mean = valid_kl_after.mean()
        pg_metrics["actor/gapd/clip_ratio"] = Metric(value=clip_ratio, aggregation=AggregationType.MEAN)
        pg_metrics["actor/gapd/kl_after_clip_mean"] = Metric(value=kl_after_clip_mean, aggregation=AggregationType.MEAN)
        pg_metrics["actor/gapd/token_clip"] = Metric(value=torch.tensor(float(token_clip)), aggregation=AggregationType.MEAN)

    # Compute kl_mean from clamped values (used for supervised loss and monitoring)
    # For k1 mode, kl_mean can be negative (bidirectional signal).
    if response_mask_bool is not None:
        valid_count = response_mask_bool.sum().clamp(min=1)
        kl_mean = (gapd_jsd_loss_padded * response_mask_bool.float()).sum() / valid_count
        kl_nonzero_ratio = ((gapd_jsd_loss_padded != 0) & response_mask_bool).float().sum() / valid_count
    else:
        kl_mean = gapd_jsd_loss_padded.mean()
        kl_nonzero_ratio = (gapd_jsd_loss_padded != 0).float().mean()

    # Log KL mode metric
    pg_metrics["actor/gapd/kl_mode"] = Metric(value=torch.tensor(float(kl_mode == "k1")), aggregation=AggregationType.MEAN)
    pg_metrics["actor/gapd/kl_nonzero_ratio"] = Metric(value=kl_nonzero_ratio, aggregation=AggregationType.MEAN)

    if not use_pg:
        # === Supervised mode: directly backpropagate mean KL ===
        distill_loss_val = kl_mean
    else:
        # === Policy gradient mode: use -KL_per_token as advantages ===
        # Matches verl's distillation_loss() use_policy_gradient=True path:
        #   advantages = -distillation_losses.detach()
        #   distill_loss = policy_loss_fn(old_log_prob, log_prob, advantages, mask, ...)
        # For forward_kl: advantages = -KL >= 0 (always suppress, entropy explosion risk)
        # For k1: advantages = -k1 = log p_T - log p_S, bidirectional signal:
        #   positive when teacher more confident (encourage), negative when student overconfident (suppress)
        # Get log_probs and old_log_probs in padded format
        log_prob = no_padding_2_padding(model_output["log_probs"], data)

        # old_log_probs needs to be extracted from data and padded
        old_log_prob = data["old_log_probs"]
        if old_log_prob.is_nested:
            old_log_prob = old_log_prob.to_padded_tensor(0.0)

        # Per-token advantages: negative KL = reward for being close to teacher
        # detach() ensures gradients flow only through log_prob in the PG loss,
        # not through the KL computation (which was already inline during forward)
        # For forward_kl: advantages in [-token_clip, 0] (always suppress)
        # For k1: advantages in [-token_clip, token_clip] (bidirectional signal)
        distill_advantages = -gapd_jsd_loss_padded.detach()

        # --- Distill advantage stats ---
        if response_mask_bool is not None:
            valid_adv = distill_advantages[response_mask_bool]
        else:
            valid_adv = distill_advantages.flatten()
        no_valid_distill_tokens = valid_adv.numel() == 0
        zero_metric = gapd_jsd_loss_padded.new_tensor(0.0)
        if no_valid_distill_tokens:
            # Conservative rollout gates can legitimately skip every GAPD turn
            # in a mini-batch. In that case the batch should reduce to the
            # standard GRPO/PPO update, with a zero distillation loss.
            pg_metrics["actor/gapd/distill_adv_mean"] = Metric(value=zero_metric, aggregation=AggregationType.MEAN)
            pg_metrics["actor/gapd/distill_adv_std"] = Metric(value=zero_metric, aggregation=AggregationType.MEAN)
            pg_metrics["actor/gapd/distill_adv_min"] = Metric(value=zero_metric, aggregation=AggregationType.MEAN)
            pg_metrics["actor/gapd/distill_adv_max"] = Metric(value=zero_metric, aggregation=AggregationType.MEAN)
        else:
            pg_metrics["actor/gapd/distill_adv_mean"] = Metric(value=valid_adv.mean(), aggregation=AggregationType.MEAN)
            pg_metrics["actor/gapd/distill_adv_std"] = Metric(value=valid_adv.std(unbiased=False), aggregation=AggregationType.MEAN)
            pg_metrics["actor/gapd/distill_adv_min"] = Metric(value=valid_adv.min(), aggregation=AggregationType.MEAN)
            pg_metrics["actor/gapd/distill_adv_max"] = Metric(value=valid_adv.max(), aggregation=AggregationType.MEAN)

        # Use gapd_turn_mask as response_mask for the distillation PG loss,
        # so only teacher-supervised turns contribute to the distillation signal
        if response_mask_bool is None:
            # Fallback: create a mask matching the KL loss shape
            response_mask_bool = torch.ones_like(gapd_jsd_loss_padded, dtype=torch.bool)

        # Ensure shapes match (log_prob may differ from KL loss by 1 due to left-shift)
        min_len = min(log_prob.size(1), old_log_prob.size(1), distill_advantages.size(1), response_mask_bool.size(1))
        log_prob = log_prob[:, :min_len]
        old_log_prob = old_log_prob[:, :min_len]
        distill_advantages = distill_advantages[:, :min_len]
        distill_mask = response_mask_bool[:, :min_len]
        effective_distill_advantages = distill_advantages.masked_fill(~distill_mask, 0.0)

        # --- Per-sample KL stats (batch-level) ---
        if distill_mask.any():
            per_sample_kl = (gapd_jsd_loss_padded[:, :min_len] * distill_mask.float()).sum(dim=1) / distill_mask.float().sum(dim=1).clamp(min=1)
            pg_metrics["actor/gapd/per_sample_kl_mean"] = Metric(value=per_sample_kl.mean(), aggregation=AggregationType.MEAN)
            pg_metrics["actor/gapd/per_sample_kl_max"] = Metric(value=per_sample_kl.max(), aggregation=AggregationType.MEAN)
            pg_metrics["actor/gapd/per_sample_kl_min"] = Metric(value=per_sample_kl.min(), aggregation=AggregationType.MEAN)
        else:
            no_valid_distill_tokens = True
            pg_metrics["actor/gapd/per_sample_kl_mean"] = Metric(value=zero_metric, aggregation=AggregationType.MEAN)
            pg_metrics["actor/gapd/per_sample_kl_max"] = Metric(value=zero_metric, aggregation=AggregationType.MEAN)
            pg_metrics["actor/gapd/per_sample_kl_min"] = Metric(value=zero_metric, aggregation=AggregationType.MEAN)

        # rollout importance sampling weights (same as verl's distillation path)
        rollout_is_weights = data.get("rollout_is_weights", None)

        if no_valid_distill_tokens:
            distill_loss_val = pg_loss_val.new_tensor(0.0)
            pg_metrics["actor/gapd/valid_distill_tokens"] = Metric(value=zero_metric, aggregation=AggregationType.MEAN)
            pg_metrics["actor/gapd_distill_actor/pg_clipfrac"] = Metric(value=zero_metric, aggregation=AggregationType.MEAN)
            pg_metrics["actor/gapd_distill_actor/ppo_kl"] = Metric(value=zero_metric, aggregation=AggregationType.MEAN)
            pg_metrics["actor/gapd_distill_actor/pg_clipfrac_lower"] = Metric(value=zero_metric, aggregation=AggregationType.MEAN)
        else:
            pg_metrics["actor/gapd/valid_distill_tokens"] = Metric(
                value=distill_mask.float().sum(), aggregation=AggregationType.MEAN
            )
            # Compute distillation PG loss with PPO clipping (vanilla mode)
            distill_loss_val, distill_pg_metrics = compute_policy_loss_vanilla(
                old_log_prob=old_log_prob,
                log_prob=log_prob,
                advantages=distill_advantages,
                response_mask=distill_mask,
                loss_agg_mode=config.loss_agg_mode,
                config=config,
                rollout_is_weights=rollout_is_weights,
            )

            # Log distillation PG metrics
            for k, v in distill_pg_metrics.items():
                pg_metrics[f"actor/gapd_distill_{k}"] = Metric(
                    value=v if isinstance(v, torch.Tensor) else torch.tensor(v),
                    aggregation=AggregationType.MEAN,
                )

            # --- Optional: dump token-level data for offline analysis ---
            _maybe_dump_token_level_data(
                config=config,
                data=data,
                gapd_jsd_loss_raw=gapd_jsd_loss_raw,
                gapd_jsd_loss_clipped=gapd_jsd_loss_padded,
                distill_advantages=effective_distill_advantages,
                log_prob=log_prob,
                old_log_prob=old_log_prob,
                distill_mask=response_mask_bool,
            )

    # Step 3: Combine losses
    total_loss = pg_coef * pg_loss_val + jsd_coef * distill_loss_val

    # --- Core metrics (always logged) ---
    pg_metrics["actor/gapd_jsd_loss"] = Metric(value=kl_mean, aggregation=AggregationType.MEAN)
    pg_metrics["actor/gapd_distill_loss"] = Metric(value=distill_loss_val.detach(), aggregation=AggregationType.MEAN)
    pg_metrics["actor/gapd_pg_loss"] = Metric(value=pg_loss_val, aggregation=AggregationType.MEAN)
    pg_metrics["actor/gapd_total_loss"] = Metric(value=total_loss, aggregation=AggregationType.MEAN)
    pg_metrics["actor/gapd_teacher_available"] = Metric(value=torch.tensor(1.0), aggregation=AggregationType.MEAN)
    pg_metrics["actor/gapd_use_pg"] = Metric(value=torch.tensor(float(use_pg)), aggregation=AggregationType.MEAN)
    pg_metrics["actor/gapd_pg_coef"] = Metric(value=torch.tensor(float(pg_coef)), aggregation=AggregationType.MEAN)
    pg_metrics["actor/gapd_jsd_coef"] = Metric(value=torch.tensor(float(jsd_coef)), aggregation=AggregationType.MEAN)
    _ensure_gapd_metric_schema(pg_metrics, pg_loss_val.new_tensor(0.0))

    return total_loss, pg_metrics
