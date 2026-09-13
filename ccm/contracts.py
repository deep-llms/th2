"""Versioned pilot contracts and small artifact helpers."""
from dataclasses import dataclass, asdict
import hashlib
import json
import math
from pathlib import Path

VERSION = "ccm-offline-v1"
ROLES = ("compile", "adapt", "common_other", "continue", "dev", "val")
ARMS = ("base", "grad", "shallow", "isolated", "contextual", "delta", "shuffled")
STAGE2_ARMS = ("base", "grad", "shuffled", "isolated", "contextual")
HOOKS = {"read": "block2_post_residual_pre_memory", "write": "last_block_pre_final_norm"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    with path.open("x", encoding="utf8") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def fresh_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


@dataclass(frozen=True)
class Budget:
    batch_tokens: int = 262144
    common_steps: int = 15259
    adapt_steps: int = 977
    continue_steps: int = 3815
    compile_tokens: int = 1000000000
    dev_tokens: int = 20000000
    val_tokens: int = 20000000
    segment_length: int = 2048
    slots: int = 262144

    def quotas(self):
        require(all(type(v) is int and v > 0 for v in asdict(self).values()),
                "Budget fields must be positive integers")
        q = dict(compile=self.compile_tokens, adapt=self.adapt_steps*self.batch_tokens,
                 common_other=self.common_steps*self.batch_tokens-self.compile_tokens-
                 self.adapt_steps*self.batch_tokens,
                 **{"continue": self.continue_steps*self.batch_tokens},
                 dev=self.dev_tokens, val=self.val_tokens)
        require(all(v > 0 for v in q.values()), "All role quotas must be positive")
        require(self.segment_length >= 3 and self.slots > 0, "Invalid segment length/capacity")
        return q

    def to_dict(self):
        return asdict(self)


PILOT = Budget()


def load_budget(value, engineering=False):
    budget = Budget(**value)
    budget.quotas()
    require(engineering or budget == PILOT, "Non-pilot budgets require --engineering")
    return budget


def seed_bundle(backbone):
    require(backbone in (17, 29, 43), "Pilot backbone seed must be 17, 29, or 43")
    return dict(backbone=backbone, data=1000+backbone, reader=100000+backbone,
                permutation=200000+backbone, grad_table=300000+backbone)


def schedule(s, total, peak, warmup_fraction, final_fraction=0.1):
    """LR applied BEFORE the 1-indexed optimizer update s."""
    require(1 <= s <= total, "Scheduler update outside its budget")
    warmup = math.ceil(total*warmup_fraction)
    if s <= warmup:
        return peak*s/warmup
    p = (s-warmup)/(total-warmup)
    return peak*(final_fraction+(1-final_fraction)*(1+math.cos(math.pi*p))/2)
