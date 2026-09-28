"""Anticipatory K/V pilot and loss-weight ablation. Imports launch nothing."""

ARMS = ("A", "B", "C", "D", "E")


def kv_loss_weight(arm):
    """E is exactly D with a lower weight on the mean K/V alignment loss."""
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    return 0.3 if arm == "E" else 1.0
