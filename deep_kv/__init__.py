"""Anticipatory K/V pilot and auxiliary-loss variants. Imports launch nothing."""

BOTTLENECK_ARMS = ("Task-Aware-NoAlign", "Task-Aware-Align",
                   "Consumer-Aware-NoAlign", "Consumer-Aware-Align")
ARMS = ("A", "B", "C", "D", "E", "F", "G") + BOTTLENECK_ARMS
PROXY_ARMS = ("V1", "V3", "P1-lambda0", "P1-block", "P1-flow", "P3-lambda0", "P3-block", "P3-flow")
ANTICIPATORY_ARMS = ("P4-iso", "P6", "P5", "P4", "P6-iso", "P4-4h", "P4-iso-4h")
MEMORY_ARMS = ("P7", "P7-kq", "P7-ems", "P7-mlp", "P7-simple")
ALL_PROXY_ARMS = PROXY_ARMS + ANTICIPATORY_ARMS + MEMORY_ARMS
SCREEN_ARMS = ("A", "V1", "V3", "P1-lambda0", "P1-block", "P1-flow", "P3-lambda0", "P3-block")
ALL_ARMS = ARMS + ALL_PROXY_ARMS


def code_loss_weight(arm):
    if arm not in BOTTLENECK_ARMS:
        raise ValueError(f"Not a bottleneck arm: {arm}")
    return 0.0 if arm.endswith("NoAlign") else 0.3


def kv_loss_weight(arm):
    """Raw alignment weight for A-E; individual functional-term weight for F/G."""
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    return 0.3 if arm in ("E", "F", "G") else 1.0
