import torch
import torch.nn.functional as F


def compute_loss(logits, input_ids, theta=None, lambda_div=0.0):
    """Compute LM loss + optional load-balance auxiliary loss.

    Args:
        logits: Model output logits, shape (B, L, V).
        input_ids: Token ids, shape (B, L). Shifted internally for next-token prediction.
        theta: Selection weights from the embedding, shape (B, L, K), or None for standard.
        lambda_div: Weight for the load-balance loss (0 disables it).

    Returns:
        (total_loss, logs_dict) where logs_dict values are detached scalar tensors.
    """
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    lm_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )

    device = logits.device
    logs = {"lm_loss": lm_loss.detach()}

    if theta is not None:
        with torch.no_grad():
            active = (theta > 0).float()
            usage = active.mean(dim=(0, 1))  # (K,) per-anchor usage fraction
            logs["avg_nnz"] = active.sum(-1).mean()
            logs["dead_rate"] = (usage == 0).float().mean()
            logs["entropy"] = _entropy(theta)
    else:
        zero = torch.tensor(0.0, device=device)
        logs["avg_nnz"] = zero
        logs["dead_rate"] = zero
        logs["entropy"] = zero

    if theta is not None and lambda_div > 0:
        div_loss = load_balance(theta)
        logs["div_loss"] = div_loss.detach()
        return lm_loss + lambda_div * div_loss, logs

    return lm_loss, logs


def load_balance(theta):
    """Switch-Transformer-style load-balance loss."""
    K = theta.size(-1)
    usage = (theta > 0).float().mean(dim=(0, 1))  # (K,) hard count
    weight = theta.mean(dim=(0, 1))  # (K,) differentiable
    return K * (usage * weight).sum()


def _entropy(theta):
    p = theta.clamp_min(1e-9)
    return -(p * p.log()).sum(-1).mean()
