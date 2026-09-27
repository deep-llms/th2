"""Validate matched completed arms and report fixed-evaluation LM contrasts."""
import json
from dataclasses import asdict
from pathlib import Path
import torch
from .config import ARMS, Recipe
from .training import write_json


def report(directory, *, recipe=None):
    root = Path(directory)
    identities, results = {}, {}
    recipe = recipe or Recipe()
    for arm in ARMS:
        identity = json.loads((root / arm / "run.json").read_text())
        result = json.loads((root / arm / "complete.json").read_text())
        state = torch.load(root / arm / "checkpoint.pt", map_location="cpu", weights_only=True, mmap=True)
        if (identity["recipe"] != asdict(recipe) or state.get("format") != "deep-kv-checkpoint-v2"
                or identity["arm"] != arm or result["status"] != "complete" or result["arm"] != arm or result["update"] != recipe.updates
                or result["input_tokens"] != recipe.updates * recipe.tokens_per_update
                or state["update"] != recipe.updates or state["identity"] != identity
                or result["final_evaluation"]["rows"] != recipe.eval_rows
                or result["final_evaluation"] != state["history"]["evaluation"][-1]):
            raise ValueError(f"Invalid/incomplete final result: {arm}")
        identities[arm], results[arm] = identity, result
        del state
    common = {key: value for key, value in identities["A"].items() if key not in ("arm", "initial_aux_sha256")}
    for arm in "BCD":
        if {key: value for key, value in identities[arm].items() if key not in ("arm", "initial_aux_sha256")} != common:
            raise ValueError("Arms differ in initialization/data/training configuration")
    if len({identities[a]["initial_aux_sha256"] for a in "BCD"}) != 1:
        raise ValueError("Auxiliary initialization differs across B/C/D")
    nll = {a: r["final_evaluation"]["lm_loss"] for a, r in results.items()}
    contrasts = {f"{a}-{b}": nll[a] - nll[b] for a, b in (("B", "A"), ("C", "B"), ("D", "B"), ("D", "C"), ("D", "A"))}
    summary = {"status": "complete", "arms": results, "lm_loss": nll, "nll_differences": contrasts,
               "pilot_result": recipe == Recipe(),
               "negative_difference_favors_first_arm": True,
               "deep_gain_pattern": all(nll["D"] < nll[a] for a in "ABC"),
               "interpretation": "Single-seed pilot; direction of differences is not proof of statistical reliability"}
    write_json(root / "comparison.json", summary)
    return summary
