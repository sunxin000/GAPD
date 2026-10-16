"""Generalized JSD loss using top-k teacher logprobs.

Same semantics as jsd_loss.generalized_jsd_loss but operates on top-k
teacher logprobs + ids instead of full vocab teacher logits. Student
logits are still full-vocab; we gather student logprobs at the teacher's
top-k positions before computing JSD.
"""

import torch
import torch.nn.functional as F


def generalized_jsd_loss_topk(
    student_logits: torch.Tensor,
    teacher_topk_logprobs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    response_mask: torch.Tensor = None,
    beta: float = 0.0,
    temperature: float = 1.0,
    token_clip: float = None,
    reduction: str = "batchmean",
) -> torch.Tensor:
    """Compute Generalized JSD loss using top-k teacher logprobs.

    Args:
        student_logits: (batch, resp_len, vocab) or (total_nnz, vocab).
        teacher_topk_logprobs: (batch, resp_len, top_k) teacher top-k log probs.
        teacher_topk_ids: (batch, resp_len, top_k) teacher top-k token ids.
        response_mask: (batch, resp_len) or (total_nnz,) bool mask.
        beta: 0=forward KL, 1=reverse KL, 0<beta<1=generalized JSD.
        temperature: Softmax temperature.
        token_clip: If set, clip per-token divergence.
        reduction: 'batchmean', 'sum', 'mean', or 'none'.

    Returns:
        Scalar JSD loss (or per-token if reduction='none').
    """
    student_logits = student_logits / temperature
    teacher_topk_logprobs = teacher_topk_logprobs / temperature

    # Gather student logprobs at teacher's top-k positions
    student_log_probs_full = F.log_softmax(student_logits, dim=-1)

    # Ensure shapes match for gather
    if student_log_probs_full.dim() == 2 and teacher_topk_ids.dim() == 3:
        # student is (total_nnz, vocab), teacher is (batch, resp_len, top_k)
        # Flatten teacher ids to match
        top_k = teacher_topk_ids.size(-1)
        teacher_topk_ids_flat = teacher_topk_ids.reshape(-1, top_k)  # (batch*resp_len, top_k)
        student_topk_log_probs = torch.gather(
            student_log_probs_full, dim=-1, index=teacher_topk_ids_flat
        )  # (batch*resp_len, top_k)
        student_topk_log_probs = student_topk_log_probs.reshape(
            teacher_topk_ids.shape
        )
        teacher_topk_logprobs_flat = teacher_topk_logprobs.reshape(-1, top_k)
    else:
        student_topk_log_probs = torch.gather(
            student_log_probs_full, dim=-1, index=teacher_topk_ids
        )
        teacher_topk_logprobs_flat = teacher_topk_logprobs.reshape(-1, teacher_topk_logprobs.size(-1))

    # Log-softmax teacher top-k logprobs (they are raw logprobs, need re-normalization)
    teacher_topk_log_probs = F.log_softmax(teacher_topk_logprobs, dim=-1)

    # Compute JSD on top-k subset
    if beta == 0:
        # Forward KL: KL(teacher || student)
        jsd = F.kl_div(
            student_topk_log_probs, teacher_topk_log_probs,
            reduction="none", log_target=True,
        )
    elif beta == 1:
        # Reverse KL: KL(student || teacher)
        jsd = F.kl_div(
            teacher_topk_log_probs, student_topk_log_probs,
            reduction="none", log_target=True,
        )
    else:
        # Generalized JSD
        beta_t = torch.tensor(beta, dtype=student_topk_log_probs.dtype, device=student_topk_log_probs.device)
        mixture_log_probs = torch.logsumexp(
            torch.stack([
                student_topk_log_probs + torch.log1p(-beta_t),
                teacher_topk_log_probs + torch.log(beta_t),
            ]),
            dim=0,
        )
        kl_teacher = F.kl_div(
            mixture_log_probs, teacher_topk_log_probs,
            reduction="none", log_target=True,
        )
        kl_student = F.kl_div(
            mixture_log_probs, student_topk_log_probs,
            reduction="none", log_target=True,
        )
        jsd = beta_t * kl_teacher + (1 - beta_t) * kl_student

    if token_clip is not None and token_clip > 0:
        jsd = jsd.clamp(max=token_clip)

    jsd = jsd.sum(dim=-1)  # sum over top_k

    if response_mask is not None:
        if response_mask.dim() < jsd.dim():
            response_mask = response_mask.squeeze(-1)
        jsd = jsd * response_mask.float()
        valid_count = response_mask.sum().clamp(min=1)
    else:
        valid_count = max(jsd.numel(), 1)

    if reduction == "batchmean":
        return jsd.sum() / valid_count
    elif reduction == "sum":
        return jsd.sum()
    elif reduction == "mean":
        return jsd.mean()
    else:
        return jsd