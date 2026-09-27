"""Full English pretraining recipe; cutoffs never shorten the LR schedule."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

ARMS = ("A", "B", "C", "D")
NAMES = dict(zip(ARMS, ("Base", "ExtraAttn-NoAlign", "ShallowKV-Align", "DeepKV-Align")))


@dataclass(frozen=True)
class Recipe:
    # Leave packing margin below the sampler's approximate 30B raw-token target.
    updates: int = 28600
    context: int = 2048
    tokens_per_update: int = 1048576
    warmup: int = 1430
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    eval_every: int = 512
    monitor_rows: int = 128
    eval_rows: int = 4882
    checkpoint_every: int = 512
    seed: int = 2901
    data_seed: int = 20260922
    consumer: int = 5
    deep_target: int = 21
    lm_chunk: int = 128

    def validate(self):
        for name in ("updates", "context", "tokens_per_update", "warmup", "eval_every",
                     "monitor_rows", "eval_rows", "checkpoint_every", "consumer", "deep_target", "lm_chunk"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"Invalid recipe field {name}")
        if not (self.context >= 2 and self.tokens_per_update % self.context == 0
                and self.warmup < self.updates and self.monitor_rows <= self.eval_rows
                and self.consumer < self.deep_target and self.learning_rate > 0 and self.weight_decay >= 0):
            raise ValueError("Invalid training recipe")
        return self

    @property
    def train_rows(self):
        return self.updates * self.tokens_per_update // self.context

    def end_update(self, stop_after=None):
        end = self.updates if stop_after is None else stop_after
        if type(end) is not int or not 1 <= end <= self.updates:
            raise ValueError("Invalid fixed-schedule cutoff")
        return end


def load_config(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    required = {"model_config", "tokenizer", "train_data", "eval_data"}
    if not isinstance(value, dict) or not required <= value.keys() or value.keys() - (required | {"microbatch"}):
        raise ValueError("Config requires model_config/tokenizer/train_data/eval_data and optional microbatch")
    for key in required:
        value[key] = str((path.parent / value[key]).resolve())
    value.setdefault("microbatch", 1)
    rows_per_rank = Recipe().tokens_per_update // Recipe().context // 8
    if (type(value["microbatch"]) is not int or value["microbatch"] <= 0
            or rows_per_rank % value["microbatch"]):
        raise ValueError(f"Eight-GPU microbatch must divide {rows_per_rank} contexts per rank")
    if value["train_data"] == value["eval_data"]:
        raise ValueError("Training and evaluation splits must differ")
    return value


def plan(config, stop_after=None):
    recipe = Recipe().validate()
    end = recipe.end_update(stop_after)
    return {"format": "deep-kv-v2", "recipe": asdict(recipe), "config": config,
            "arms": NAMES, "tokens_per_arm": recipe.updates * recipe.tokens_per_update,
            "total_input_tokens": len(ARMS) * recipe.updates * recipe.tokens_per_update,
            "stop_after": stop_after, "run_updates_per_arm": end,
            "run_tokens_per_arm": end * recipe.tokens_per_update,
            "run_total_input_tokens": len(ARMS) * end * recipe.tokens_per_update,
            "initialization": "random Qwen3; shared backbone seed; byte-identical B/C/D branch",
            "alignment_weight": 1.0, "gpus_per_arm": 8, "training_launched": False}
