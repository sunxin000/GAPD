"""GAPD (Gold-Action Policy Distillation) FSDP Engine.

Subclasses FSDPEngineWithLMHead to compute distillation loss inline
during the forward pass (as a logits_processor_func), following verl's
distillation pattern. This avoids the CUDA driver error that occurs when logits
are retained and the JSD loss is computed in a separate loss function call
under FSDP + param_offload=True.

Supports two KL modes:
- forward_kl: KL(T||S) = sum p_T * (log p_T - log p_S), always >= 0
- k1: reverse KL estimator = log p_S - log p_T, can be positive or negative
  When used as advantages (-k1), provides bidirectional signal:
  positive when teacher is more confident, negative when student is overconfident.
"""

import logging

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

logger = logging.getLogger(__name__)


def _compute_gapd_kl_inline(student_logits: torch.Tensor, data: TensorDict) -> dict[str, torch.Tensor]:
    """Compute GAPD distillation loss inline during the forward pass.

    This runs as a logits_processor_func inside prepare_model_outputs,
    so the backward pass happens while FSDP still holds gathered params.

    Supports two KL modes controlled by gapd_kl_mode in batch meta
    - "forward_kl": KL(T||S) = sum p_T * (log p_T - log p_S), always >= 0
    - "k1": reverse KL estimator = log p_S(x) - log p_T(x), signed

    Args:
        student_logits: (1, total_nnz, vocab) student logits from forward.
         micro_batch TensorDict with teacher_logprobs, teacher_ids.

    Returns:
        Dict with 'gapd_jsd_loss' tensor of shape (1, total_nnz).
    """
    teacher_logprobs = data.get("teacher_logprobs", None)
    teacher_ids = data.get("teacher_ids", None)
    kl_mode = tu.get_non_tensor_data(data, "gapd_kl_mode", default="forward_kl")

    total_nnz = student_logits.size(1)
    device = student_logits.device

    if teacher_logprobs is None or teacher_ids is None:
        return {"gapd_jsd_loss": torch.zeros(1, total_nnz, device=device, dtype=student_logits.dtype)}

    # Ensure teacher data is on the same device as student logits
    teacher_logprobs = teacher_logprobs.to(device)
    teacher_ids = teacher_ids.to(device)

    input_ids = data["input_ids"]
    if not input_ids.is_nested:
        raise ValueError("GAPD inline KL expects no-padding nested input_ids")

    cu_seqlens = input_ids.offsets().to(device)
    seq_lens = cu_seqlens.diff()

    prompts = data["prompts"]
    responses = data["responses"]
    attention_mask = data.get("attention_mask", None)
    if prompts.is_nested:
        prompt_lens = prompts.offsets().diff().to(device)
        response_lens = responses.offsets().diff().to(device)
        response_ids_padded = responses.to_padded_tensor(0).to(device)
    else:
        if attention_mask is None:
            raise ValueError("GAPD inline KL needs attention_mask to align response-only teacher data")
        attention_mask = attention_mask.to(device)
        prompt_width = prompts.shape[1]
        prompt_lens = attention_mask[:, :prompt_width].sum(dim=1).long()
        response_lens = attention_mask[:, prompt_width:].sum(dim=1).long()
        response_ids_padded = responses.to(device)

    # Teacher data is response-only: (bsz, max_response_len, top_k).  Student
    # logits are full-sequence, left shifted: logit at prompt_len - 1 predicts
    # the first response token.  Build a full-sequence KL vector so verl's
    # no_padding_2_padding() slices the response segment correctly later.
    full_kl = torch.zeros(1, total_nnz, device=device, dtype=student_logits.dtype)
    teacher_lp_segments = []
    teacher_id_segments = []
    student_logit_segments = []
    sampled_id_segments = []
    sample_idx_segments = []
    response_pos_segments = []
    full_positions = []

    for i, (seq_len, prompt_len, response_len) in enumerate(zip(seq_lens, prompt_lens, response_lens, strict=True)):
        response_len_int = int(response_len.item())
        if response_len_int <= 0:
            continue
        seq_start = int(cu_seqlens[i].item())
        seq_len_int = int(seq_len.item())
        prompt_len_int = int(prompt_len.item())

        # The last prompt-token logit predicts response token 0.
        resp_logit_start = seq_start + max(prompt_len_int - 1, 0)
        resp_logit_end = min(resp_logit_start + response_len_int, seq_start + seq_len_int)
        actual_len = resp_logit_end - resp_logit_start
        if actual_len <= 0:
            continue

        teacher_lp_segments.append(teacher_logprobs[i, :actual_len])
        teacher_id_segments.append(teacher_ids[i, :actual_len])
        student_logit_segments.append(student_logits[:, resp_logit_start:resp_logit_end])
        sampled_id_segments.append(response_ids_padded[i, :actual_len])
        sample_idx_segments.append(torch.full((actual_len,), i, device=device, dtype=torch.long))
        response_pos_segments.append(torch.arange(actual_len, device=device, dtype=torch.long))
        full_positions.append(torch.arange(resp_logit_start, resp_logit_end, device=device))

    if not teacher_lp_segments:
        return {"gapd_jsd_loss": full_kl}

    teacher_topk_logprobs_aligned = torch.cat(teacher_lp_segments, dim=0).unsqueeze(0).to(device)
    teacher_topk_ids_aligned = torch.cat(teacher_id_segments, dim=0).unsqueeze(0).to(device).long()
    student_logits_aligned = torch.cat(student_logit_segments, dim=1)
    sampled_ids_aligned = torch.cat(sampled_id_segments, dim=0).unsqueeze(0).to(device).long()
    sample_indices = torch.cat(sample_idx_segments, dim=0)
    response_positions = torch.cat(response_pos_segments, dim=0)
    full_positions = torch.cat(full_positions, dim=0)

    min_len = min(
        student_logits_aligned.size(1),
        teacher_topk_logprobs_aligned.size(1),
        sampled_ids_aligned.size(1),
        full_positions.numel(),
    )
    student_logits_aligned = student_logits_aligned[:, :min_len]
    teacher_topk_logprobs_aligned = teacher_topk_logprobs_aligned[:, :min_len]
    teacher_topk_ids_aligned = teacher_topk_ids_aligned[:, :min_len]
    sampled_ids_aligned = sampled_ids_aligned[:, :min_len]
    sample_indices = sample_indices[:min_len]
    response_positions = response_positions[:min_len]
    full_positions = full_positions[:min_len]
    teacher_valid = teacher_topk_logprobs_aligned[..., 0] > -99.0

    # Detach student logits — KL values are only used as detached advantages.
    student_logits_detached = student_logits_aligned.detach()

    student_top_k = int(tu.get_non_tensor_data(data, "gapd_student_top_k", default=0) or 0)
    student_log_probs_for_dump = None

    def _maybe_store_student_topk(student_log_probs_full: torch.Tensor) -> None:
        """Store compact student top-k logprobs for offline token analysis."""
        nonlocal student_log_probs_for_dump
        if student_top_k <= 0 or student_log_probs_for_dump is not None:
            return
        k = min(student_top_k, student_log_probs_full.size(-1))
        if k <= 0:
            return

        topk_logprobs, topk_ids = torch.topk(student_log_probs_full.squeeze(0), k=k, dim=-1)
        bsz = int(seq_lens.numel())
        max_response_len = int(response_ids_padded.size(1))
        student_topk_logprobs = torch.full(
            (bsz, max_response_len, k),
            -100.0,
            device=device,
            dtype=topk_logprobs.dtype,
        )
        student_topk_ids = torch.zeros(
            (bsz, max_response_len, k),
            device=device,
            dtype=torch.int32,
        )
        valid_pos = response_positions < max_response_len
        if valid_pos.any():
            student_topk_logprobs[sample_indices[valid_pos], response_positions[valid_pos]] = topk_logprobs[valid_pos]
            student_topk_ids[sample_indices[valid_pos], response_positions[valid_pos]] = topk_ids[valid_pos].to(torch.int32)
        data.set("gapd_student_topk_logprobs", student_topk_logprobs.detach())
        data.set("gapd_student_topk_ids", student_topk_ids.detach())
        student_log_probs_for_dump = student_log_probs_full

    # Read log_prob_min_clamp from batch metadata.
    log_prob_min_clamp = tu.get_non_tensor_data(data, "gapd_log_prob_min_clamp", default=None)

    if kl_mode == "k1":
        # === K1 (reverse KL estimator): log p_S(x) - log p_T(x) ===
        # k1 > 0: student more confident than teacher → should be discouraged
        # k1 < 0: teacher more confident → should be encouraged
        # advantage = -k1 provides bidirectional signal.

        # Student log prob at sampled token
        student_log_probs_full = F.log_softmax(student_logits_detached.float(), dim=-1)
        _maybe_store_student_topk(student_log_probs_full)
        student_logprob_at_sampled = student_log_probs_full.gather(
            -1, sampled_ids_aligned.unsqueeze(-1)
        ).squeeze(-1)  # (1, min_len)

        # Teacher log prob at sampled token — look up from top-k data.
        # If sampled token is in top-k, use its logprob; else use log_prob_min_clamp.
        match_mask = (teacher_topk_ids_aligned == sampled_ids_aligned.unsqueeze(-1))  # (1, min_len, top_k)
        default_lp = log_prob_min_clamp if log_prob_min_clamp is not None else -10.0
        teacher_logprob_at_sampled = torch.full(
            (1, min_len), default_lp, device=device, dtype=teacher_topk_logprobs_aligned.dtype
        )
        # Where sampled token is in top-k, extract the matching logprob
        has_match = match_mask.any(dim=-1)  # (1, min_len)
        matched_logprobs = (teacher_topk_logprobs_aligned * match_mask.float()).sum(dim=-1)  # (1, min_len)
        teacher_logprob_at_sampled = torch.where(has_match, matched_logprobs, teacher_logprob_at_sampled)

        # Apply log_prob_min_clamp
        if log_prob_min_clamp is not None:
            student_logprob_at_sampled = student_logprob_at_sampled.clamp_min(log_prob_min_clamp)
            teacher_logprob_at_sampled = teacher_logprob_at_sampled.clamp_min(log_prob_min_clamp)

        # k1 = log p_S(x) - log p_T(x)
        kl = student_logprob_at_sampled - teacher_logprob_at_sampled  # (1, min_len) — can be positive or negative

        # Probability mass metrics
        with torch.no_grad():
            teacher_mass_by_token = teacher_topk_logprobs_aligned.float().exp().sum(dim=-1)
            student_topk_log_probs = torch.gather(student_log_probs_full, dim=-1, index=teacher_topk_ids_aligned)
            student_mass_by_token = student_topk_log_probs.exp().sum(dim=-1)
            teacher_mass = (
                teacher_mass_by_token[teacher_valid].mean().item()
                if teacher_valid.any()
                else 0.0
            )
            student_mass = (
                student_mass_by_token[teacher_valid].mean().item()
                if teacher_valid.any()
                else 0.0
            )

    else:
        # === Forward KL: KL(T||S) = sum p_T * (log p_T - log p_S) ===
        student_log_probs = F.log_softmax(student_logits_detached.float(), dim=-1)
        _maybe_store_student_topk(student_log_probs)
        student_topk_log_probs = torch.gather(
            student_log_probs, dim=-1, index=teacher_topk_ids_aligned
        )

        teacher_log_probs = teacher_topk_logprobs_aligned.float()
        if log_prob_min_clamp is not None:
            teacher_log_probs = teacher_log_probs.clamp_min(log_prob_min_clamp)
            student_topk_log_probs = student_topk_log_probs.clamp_min(log_prob_min_clamp)
        kl = (teacher_log_probs.exp() * (teacher_log_probs - student_topk_log_probs)).sum(dim=-1)

        with torch.no_grad():
            teacher_mass_by_token = teacher_log_probs.exp().sum(dim=-1)
            student_mass_by_token = student_topk_log_probs.exp().sum(dim=-1)
            teacher_mass = (
                teacher_mass_by_token[teacher_valid].mean().item()
                if teacher_valid.any()
                else 0.0
            )
            student_mass = (
                student_mass_by_token[teacher_valid].mean().item()
                if teacher_valid.any()
                else 0.0
            )

    kl = kl.masked_fill(~teacher_valid, 0.0)

    # Store mass metrics as non_tensor_data
    tu.assign_non_tensor_data(data, "gapd_teacher_mass", teacher_mass)
    tu.assign_non_tensor_data(data, "gapd_student_mass", student_mass)

    full_kl[:, full_positions] = kl[:, : full_positions.numel()].to(full_kl.dtype)
    return {"gapd_jsd_loss": full_kl}


@EngineRegistry.register(model_type="gapd_language_model", backend=["fsdp", "fsdp2"], device=["cuda"])
class GAPDFSDPEngineWithLMHead(FSDPEngineWithLMHead):
    """FSDP Engine for GAPD — computes distillation loss inline during forward.

    Instead of retaining raw logits and computing KL in a separate loss
    function, we compute the KL loss inside prepare_model_outputs as a
    logits_processor_func. This follows verl's distillation pattern and
    ensures the backward pass works correctly with FSDP + param_offload.
    """

    def prepare_model_outputs(self, output, output_args, micro_batch: TensorDict, logits_processor_func):
        """Override to compute KL loss inline when GAPD is active."""
        gapd_enabled = tu.get_non_tensor_data(micro_batch, "gapd_enable", default=False)

        if gapd_enabled:
            tu.assign_non_tensor_data(micro_batch, "distillation_use_topk", True)

            log_prob_min_clamp = tu.get_non_tensor_data(micro_batch, "gapd_log_prob_min_clamp", default=-10.0)
            tu.assign_non_tensor_data(micro_batch, "gapd_log_prob_min_clamp", log_prob_min_clamp)

            tu.assign_non_tensor_data(micro_batch, "calculate_entropy", True)

            model_output = super().prepare_model_outputs(
                output, output_args, micro_batch, logits_processor_func=_compute_gapd_kl_inline
            )

            tu.assign_non_tensor_data(micro_batch, "distillation_use_topk", False)
            tu.assign_non_tensor_data(micro_batch, "calculate_entropy", False)
            tu.assign_non_tensor_data(micro_batch, "gapd_log_prob_min_clamp", None)
            return model_output

        return super().prepare_model_outputs(output, output_args, micro_batch, logits_processor_func)
