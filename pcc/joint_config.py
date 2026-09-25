"""Fixed, separately versioned joint-training plan. No ML imports."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class JointSettings:
    version: str = "joint-v1"
    updates: int = 1536
    context: int = 2048
    tokens_per_update: int = 32768
    warmup: int = 77
    eval_every: int = 128
    monitor_tokens: int = 262144
    dev_tokens: int = 2000000
    s: int = 4
    d: int = 20

    def validate(self):
        for name in ("updates", "context", "tokens_per_update", "warmup", "eval_every", "monitor_tokens", "dev_tokens", "s", "d"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"Invalid joint setting: {name}")
        if not (self.context >= 2 and self.tokens_per_update % self.context == 0
                and self.warmup < self.updates and self.monitor_tokens <= self.dev_tokens
                and self.monitor_tokens % self.context == 0 and self.s < self.d):
            raise ValueError("Invalid joint schedule/context settings")
        return self


SEEDS = ((2901, 20260922), (3901, 20260923))
ARMS = ("Base", "Shallow", "Deep")


def settings_for(config):
    if config.get("experiment", "joint-v1") in ("joint-v2-ddp", "joint-local-v3"):
        return JointSettings(version=config["experiment"], updates=6144, warmup=307, eval_every=256).validate()
    return JointSettings().validate()


def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    required = {"model_path", "train_data", "val_data"}
    if not isinstance(config, dict) or not required <= set(config) or set(config) - (required | {"microbatch", "train_documents", "experiment"}):
        raise ValueError("Joint config requires model_path/train_data/val_data and optional microbatch/train_documents")
    for key in required:
        if not isinstance(config[key], str) or not config[key]:
            raise ValueError(f"Invalid path: {key}")
        config[key] = str((path.parent / config[key]).resolve())
    micro = config.setdefault("microbatch", 1)
    if type(micro) is not int or micro < 1 or 16 % micro:
        raise ValueError("microbatch must divide 16 contexts/update")
    if "train_documents" in config and (type(config["train_documents"]) is not int or config["train_documents"] <= 0):
        raise ValueError("train_documents must be a positive integer")
    if config.get("experiment", "joint-v1") not in ("joint-v1", "joint-v2-ddp", "joint-local-v3"):
        raise ValueError("Unknown fixed joint experiment")
    return config


def plan(config):
    settings = settings_for(config)
    return {"settings": asdict(settings), "config": config,
            "seeds": SEEDS, "arms": ARMS,
            "input_tokens_per_run": settings.updates * settings.tokens_per_update,
            "total_runs": 2 if config.get("experiment") == "joint-local-v3" else 6,
            "backbone_trainable": True, "remote_launch": False}
