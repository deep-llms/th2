"""Offline context-compiled memory pilot (no remote execution or downloads)."""

import os

# Every project entry point is local-only, including third-party loaders.
for _name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
    os.environ[_name] = "1"
os.environ["WANDB_MODE"] = "offline"
