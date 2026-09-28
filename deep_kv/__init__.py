"""Anticipatory K/V pilot and auxiliary-loss variants. Imports launch nothing."""

ARMS = ("A", "B", "C", "D", "E", "F", "G")


def kv_loss_weight(arm):
    """Raw alignment weight for A-E; individual functional-term weight for F/G."""
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    return 0.3 if arm in ("E", "F", "G") else 1.0
