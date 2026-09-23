"""Standard-library-only pipeline configuration and read-only execution plans."""
import json
from pathlib import Path

from .protocol import CONTEXT, TOKENS_PER_UPDATE


PATH_FIELDS = ("model_path", "train_data", "val_data", "test_data")


def load_pipeline_config(path):
    path = Path(path).absolute()
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict) or set(config) - (set(PATH_FIELDS) | {"microbatch"}):
        raise ValueError("Pipeline config permits only model/data paths and microbatch")
    result = {}
    for key in PATH_FIELDS:
        value = config.get(key)
        if key == "test_data" and value is None:
            result[key] = None
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Pipeline config requires a nonempty {key}")
        candidate = Path(value)
        # Lexical absolutization only: do not resolve symlinks/stat/open test data.
        result[key] = str(candidate if candidate.is_absolute() else path.parent / candidate)
    microbatch = config.get("microbatch", 1)
    if type(microbatch) is not int or microbatch < 1 or (TOKENS_PER_UPDATE // CONTEXT) % microbatch:
        raise ValueError("microbatch must be a positive integer divisor of 16")
    result["microbatch"] = microbatch
    return result


def pipeline_plan(config, *, check_only=False):
    result = {"mode": "pcc_pipeline", "config": config,
        "before_training": "validate full train/validation budgets, vocabulary, split policies, and screen prefixes",
        "check_only": check_only, "ordered_stages": [
        {"name": "screen", "when": "always", "steps": ["correctness preflight", "four-pair screen", "pair selection"]},
        {"name": "probe", "when": "screen selects an eligible pair", "steps": [
            "fresh teacher training", "correction calibration", "matched student/control training",
            "development evaluation", "conditional target-permuted control",
            "freeze development decisions", "one locked-test evaluation only if supplied and all dev gates pass"]}],
        "without_test_data": "finish at validation; no confirmatory-test claim",
        "negative_screen": "complete successfully and skip full probe",
        "software_failure": "stop immediately; no completion marker or automatic retry",
        "pretraining_authorized": False}
    if check_only:
        result["ordered_stages"] = [{"name": "input_checks", "when": "always", "steps": [
            "synthetic model correctness preflight", "full train/validation input checks", "stop before experiment training"]}]
    return result
