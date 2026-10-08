"""Anticipatory K/V pilot and auxiliary-loss variants. Imports launch nothing."""

BOTTLENECK_ARMS = ("Task-Aware-NoAlign", "Task-Aware-Align",
                   "Consumer-Aware-NoAlign", "Consumer-Aware-Align")
ARMS = ("A", "B", "C", "D", "E", "F", "G") + BOTTLENECK_ARMS
PROXY_ARMS = ("V1", "V3", "P1-lambda0", "P1-block", "P1-flow", "P3-lambda0", "P3-block", "P3-flow")
P6_TARGET_VARIANTS = ("P6-iso-weighted", "P6-iso-layernorm")
P6_VARIANTS = ("P6-iso-sparse", "P6-iso-short") + P6_TARGET_VARIANTS
P6_TARGET_WEIGHTS = (1.6, 1.2, 0.8, 0.4)
ANTICIPATORY_ARMS = ("P4-iso", "P6", "P5", "P4", "P6-iso", "P4-4h", "P4-iso-4h") + P6_VARIANTS
P7_P6_VARIANTS = ("P7-simple-sparse", "P7-simple-short")
SIMPLE_MEMORY_ARMS = ("P7-simple",) + P7_P6_VARIANTS
MEMORY_ARMS = ("P7", "P7-kq", "P7-ems", "P7-mlp") + SIMPLE_MEMORY_ARMS
ALL_PROXY_ARMS = PROXY_ARMS + ANTICIPATORY_ARMS + MEMORY_ARMS
SCREEN_ARMS = ("A", "V1", "V3", "P1-lambda0", "P1-block", "P1-flow", "P3-lambda0", "P3-block")
ALL_ARMS = ARMS + ALL_PROXY_ARMS


def anticipatory_layout(arm, num_hidden_layers):
    """Fixed 1-based injection blocks and MLP target length for P4–P6 arms."""
    if arm not in ANTICIPATORY_ARMS:
        raise ValueError(f"Not an anticipatory arm: {arm}")
    # Short targets retain the parent's locations; do not add late injections.
    stride = 4 if arm == "P6-iso-sparse" else 2
    return tuple(range(2, num_hidden_layers - 2, stride)), 2 if arm == "P6-iso-short" else 4


def simple_memory_layout(arm, num_hidden_layers):
    """P7-simple uses the same locations and targets as its P6-iso counterpart."""
    parents = {"P7-simple": "P6-iso", "P7-simple-sparse": "P6-iso-sparse",
               "P7-simple-short": "P6-iso-short"}
    return anticipatory_layout(parents[arm], num_hidden_layers)


def anticipatory_target_metadata(arm):
    """Raw target transformation, before the shared running standardization."""
    if arm not in ANTICIPATORY_ARMS:
        raise ValueError(f"Not an anticipatory arm: {arm}")
    if arm == "P6-iso-weighted":
        return dict(quantity="weighted_mlp_window_sum", layer_weights=list(P6_TARGET_WEIGHTS))
    if arm == "P6-iso-layernorm":
        return dict(quantity="layernorm_mlp_window_sum", per_layer_normalization=dict(
            type="layer_norm", axis="hidden", epsilon=1e-6, affine=False, dtype="float32"))
    return dict(quantity="mlp_window_sum")


def code_loss_weight(arm):
    if arm not in BOTTLENECK_ARMS:
        raise ValueError(f"Not a bottleneck arm: {arm}")
    return 0.0 if arm.endswith("NoAlign") else 0.3


def kv_loss_weight(arm):
    """Raw alignment weight for A-E; individual functional-term weight for F/G."""
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    return 0.3 if arm in ("E", "F", "G") else 1.0
