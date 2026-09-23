"""Locked first-probe constants from research contract section 11."""
MODEL_ID = "Qwen/Qwen3-0.6B-Base"
REVISION = "ddc928429ed09d9ad603fd762053d0434c15e865"
TRANSFORMERS_VERSION = "4.57.1"
PAIRS = ((4, 16), (4, 20), (8, 20), (8, 24))
HOOKS = (4, 8, 16, 20, 24)
CONTEXT = 2048
TOKENS_PER_UPDATE = 32768
SCREEN_UPDATES = 128
SCREEN_DEV_TOKENS = 2_000_000
DATA_SEED = 20260922
SCREEN_SEED = 1701
FULL_SEED = 2901
FULL_UPDATES = 610
FULL_WARMUP = 31
PROBE_EVAL_TOKENS = 10_000_000
CALIBRATION_TOKENS = 1_000_000
CONFIG = dict(num_hidden_layers=28, hidden_size=1024, num_attention_heads=16,
              num_key_value_heads=8, head_dim=128, rms_norm_eps=1e-6,
              rope_theta=1_000_000)


def validate_config(config):
    if config.model_type != "qwen3":
        raise ValueError("The locked probe requires Qwen3")
    for key, expected in CONFIG.items():
        if getattr(config, key, None) != expected:
            raise ValueError(f"Model config {key}: expected {expected}, got {getattr(config, key, None)}")
    if getattr(config, "use_sliding_window", False) or any(
        kind != "full_attention" for kind in config.layer_types
    ):
        raise ValueError("This probe supports the pinned full-attention backbone only")
