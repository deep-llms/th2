"""Anticipatory K/V pilot and auxiliary-loss variants. Imports launch nothing."""

BOTTLENECK_ARMS = ("Task-Aware-NoAlign", "Task-Aware-Align",
                   "Consumer-Aware-NoAlign", "Consumer-Aware-Align")
ARMS = ("A", "B", "C", "D", "E", "F", "G") + BOTTLENECK_ARMS
PROXY_ARMS = ("V1", "V3", "P1-lambda0", "P1-block", "P1-flow", "P3-lambda0", "P3-block", "P3-flow")
P6_TARGET_VARIANTS = ("P6-iso-weighted", "P6-iso-layernorm")
P6_VARIANTS = ("P6-iso-sparse", "P6-iso-short") + P6_TARGET_VARIANTS
P6_TARGET_WEIGHTS = (1.6, 1.2, 0.8, 0.4)
P6_ROUTING_ARMS = ("P6-iso-targetgrad", "P6-iso-shallowgrad", "P6-iso-bothgrad")
P6_ADAPTER_ARMS = ("P6-iso-adapter", "P6-iso-adapter-shallow")
P6_DEPTH_ARMS = ("P6-iso-depth", "P6-iso-depth-current")
P6_DEEP_ARMS = ("P6-iso-deep", "P6-iso-deep-shallow")
P6_EXTENSION_ARMS = P6_ADAPTER_ARMS + P6_DEPTH_ARMS + P6_DEEP_ARMS
P6_SPARSE_DEEP = ("P6-iso-sparse-deep",)
P6_CONSUMER_ARMS = ("P6-iso-consumer", "P6-iso-consumer-only")
DELAYED_ARMS = ("P6-iso-delayed", "P6-iso-sparse-delayed", "P7-simple-delayed")
PROXY_DELAY_STEPS = 500
P7_DEEP_ARMS = ("P7-simple-deep", "P7-simple-deep-shallow")
P7_ROUTING_ARMS = ("P7-simple-targetgrad", "P7-simple-lm")
# Keep the completed 24-arm screen's membership stable when adding new arms.
P7_COMBINED_ARMS = ("P7-simple-targetgrad-lm",)
P7_CONSUMER_ARMS = ("P7-simple-consumer-only", "P7-simple-consumer", "P7-simple-consumer-targetgrad")
P7_LM_PREDICTOR_ARMS = ("P7-simple-lm",) + P7_COMBINED_ARMS
FOLLOWUP_ARMS = ("P6-iso-sparse-targetgrad",) + P6_SPARSE_DEEP + P7_ROUTING_ARMS + P6_CONSUMER_ARMS + DELAYED_ARMS
P6_ADAPTER_WIDTH = 32
ANTICIPATORY_ARMS = ("P4-iso", "P6", "P5", "P4", "P6-iso", "P4-4h", "P4-iso-4h", "P6-lowaux", "P6-iso-sparse-targetgrad") + P6_VARIANTS + P6_ROUTING_ARMS + P6_EXTENSION_ARMS + P6_SPARSE_DEEP + P6_CONSUMER_ARMS + DELAYED_ARMS[:2]
P7_P6_VARIANTS = ("P7-simple-sparse", "P7-simple-short")
SIMPLE_MEMORY_ARMS = ("P7-simple",) + P7_P6_VARIANTS + P7_ROUTING_ARMS + P7_DEEP_ARMS + DELAYED_ARMS[2:] + P7_COMBINED_ARMS + P7_CONSUMER_ARMS
MEMORY_ARMS = ("P7", "P7-kq", "P7-ems", "P7-mlp") + SIMPLE_MEMORY_ARMS
ALL_PROXY_ARMS = PROXY_ARMS + ANTICIPATORY_ARMS + MEMORY_ARMS
SCREEN_ARMS = ("A", "V1", "V3", "P1-lambda0", "P1-block", "P1-flow", "P3-lambda0", "P3-block")
ALL_ARMS = ARMS + ALL_PROXY_ARMS


def auxiliary_routes(arm):
    """Explicit auxiliary routes; LM-to-estimator routing is independent."""
    return dict(shallow=arm in ("P6-iso-shallowgrad", "P6-iso-bothgrad"),
                target=arm in ("P6-iso-targetgrad", "P6-iso-bothgrad", "P6-iso-sparse-targetgrad", "P7-simple-targetgrad", "P7-simple-consumer-targetgrad") + P7_COMBINED_ARMS)


def followup_metadata(arm):
    """Additional fixed scientific contracts, shared by recording and validation."""
    result = {}
    if arm == 'P6-iso-sparse-targetgrad':result['auxiliary_routes'] = auxiliary_routes(arm)
    if arm in P7_ROUTING_ARMS + P7_COMBINED_ARMS:
        result['auxiliary_routes'] = auxiliary_routes(arm)
        result['lm_to_predictor'] = arm in P7_LM_PREDICTOR_ARMS
    if arm in P7_DEEP_ARMS:
        result['extension'] = dict(version=1, kind='deep_residual',
            source='shallow_norm' if arm.endswith('-shallow') else 'prediction',
            width=P6_ADAPTER_WIDTH, initialization='zero_added_output',
            lm_to_predictor=arm=='P7-simple-deep', deep_injection='after_target_window')
    if arm in DELAYED_ARMS:
        result['introduction'] = dict(version=1, step=PROXY_DELAY_STEPS,
            freeze_proxy_before=True, auxiliary_warmup='relative_to_introduction')
    if arm in P6_CONSUMER_ARMS:
        result['consumer_loss'] = dict(version=1, consumer='frozen_injection_decoder_block',
            loss='relative_output_mse', epsilon=1e-6, effect_weight=1.,
            cosine_weight=0. if arm.endswith('-only') else 1.,
            source_detached=True, teacher_detached=True, parameters_detached=True)
    if arm in P7_CONSUMER_ARMS:
        result.update(auxiliary_routes=auxiliary_routes(arm), lm_to_predictor=False,
            consumer_loss=dict(version=1, consumer='frozen_selected_joint_attention_heads',
                loss='relative_output_mse', epsilon=1e-6, effect_weight=1.,
                cosine_weight=0. if arm.endswith('-only') else 1.,
                source_detached=True, teacher_detached=True, parameters_detached=True,
                query_heads=4, output='before_o_proj', teacher='standardized_mlp_window_sum'))
    return result


def anticipatory_layout(arm, num_hidden_layers):
    """Fixed 1-based injection blocks and MLP target length for P4–P6 arms."""
    if arm not in ANTICIPATORY_ARMS:
        raise ValueError(f"Not an anticipatory arm: {arm}")
    # Short targets retain the parent's locations; do not add late injections.
    stride = 4 if arm.startswith("P6-iso-sparse") else 2
    return tuple(range(2, num_hidden_layers - 2, stride)), 2 if arm == "P6-iso-short" else 4


def simple_memory_layout(arm, num_hidden_layers):
    """P7-simple uses the same locations and targets as its P6-iso counterpart."""
    parents = {"P7-simple": "P6-iso", "P7-simple-sparse": "P6-iso-sparse",
               "P7-simple-short": "P6-iso-short"}
    if arm in P7_ROUTING_ARMS + P7_DEEP_ARMS + DELAYED_ARMS[2:] + P7_COMBINED_ARMS + P7_CONSUMER_ARMS:return anticipatory_layout('P6-iso',num_hidden_layers)
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
    if arm in P6_EXTENSION_ARMS + P6_SPARSE_DEEP:
        # Versioned architecture contract also checked on reports and resume.
        kind = ('shallow_adapter' if arm in P6_ADAPTER_ARMS else
                'depth_input' if arm in P6_DEPTH_ARMS else 'deep_residual')
        source = ('current_hidden' if arm.endswith('-current') else
                  'shallow_norm' if arm.endswith('-shallow') else
                  'depth_difference' if arm in P6_DEPTH_ARMS else 'prediction')
        return dict(quantity="mlp_window_sum", extension=dict(version=1, kind=kind, source=source,
            width="predictor_width" if arm in P6_DEPTH_ARMS else P6_ADAPTER_WIDTH,
            initialization="zero_added_output", lm_to_predictor=arm in ('P6-iso-deep',)+P6_SPARSE_DEEP,
            deep_injection="after_target_window" if arm in P6_DEEP_ARMS+P6_SPARSE_DEEP else None))
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
