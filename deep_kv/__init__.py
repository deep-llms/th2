"""Anticipatory K/V pilot and auxiliary-loss variants. Imports launch nothing."""

BOTTLENECK_ARMS = ("Task-Aware-NoAlign", "Task-Aware-Align",
                   "Consumer-Aware-NoAlign", "Consumer-Aware-Align")
ARMS = ("A", "B", "C", "D", "E", "F", "G") + BOTTLENECK_ARMS


def code_loss_weight(arm):
    if arm not in BOTTLENECK_ARMS:
        raise ValueError(f"Not a bottleneck arm: {arm}")
    return 0.0 if arm.endswith("NoAlign") else 0.3


def kv_loss_weight(arm):
    """Raw alignment weight for A-E; individual functional-term weight for F/G."""
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    return 0.3 if arm in ("E", "F", "G") else 1.0
