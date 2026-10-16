"""Generalized Jensen-Shannon Divergence loss for GAPD policy distillation.

Pure torch implementation with no framework dependency beyond F.kl_div.
"""

import torch
import torch.nn.functional as F


def generalized_jsd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    response_mask: torch.Tensor = None,
    beta: float = 0.0,
    temperature: float = 1.0,
    top_k: int = 0,
    token_clip: float = None,
    reduction: str = "batchmean",
) -> torch.Tensor:
    """Compute Generalized Jensen-Shannon Divergence loss.

    Args:
        student_logits: (num_tokens, vocab_size) — response tokens only.
        teacher_logits: Same shape as student_logits.
        response_mask: (num_tokens,) bool mask. If provided, only compute on True positions.
        beta: 0=forward KL, 1=reverse KL, 0<β<1=generalized JSD.
        temperature: Softmax temperature.
        top_k: If >0, restrict loss to teacher's top-k tokens.
        token_clip: If set, clip per-token divergence to this max value.
        reduction: 'batchmean', 'sum', 'mean', or 'none'.

    Returns:
        Scalar JSD loss (or per-token if reduction='none').
    """
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature

    if top_k is not None and top_k > 0:
        _, top_k_indices = torch.topk(teacher_logits, k=min(top_k, teacher_logits.size(-1)), dim=-1)
        student_logits = torch.gather(student_logits, dim=-1, index=top_k_indices)
        teacher_logits = torch.gather(teacher_logits, dim=-1, index=top_k_indices)

    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    if beta == 0:
        jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
    elif beta == 1:
        jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
    else:
        beta_t = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
        mixture_log_probs = torch.logsumexp(
            torch.stack([student_log_probs + torch.log1p(-beta_t), teacher_log_probs + torch.log(beta_t)]),
            dim=0,
        )
        kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
        kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)
        jsd = beta_t * kl_teacher + (1 - beta_t) * kl_student

    if token_clip is not None:
        jsd = jsd.clamp(max=token_clip)

    jsd = jsd.sum(dim=-1)

    if response_mask is not None:
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
