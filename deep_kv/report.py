"""Validate matched completed arms and report fixed-evaluation LM contrasts."""
import json
from dataclasses import asdict
from pathlib import Path
from .config import ARMS, Recipe
from .training import write_json, verify_checkpoint


def report(directory, *, recipe=None, stop_after=None):
    root = Path(directory)
    identities, results = {}, {}
    recipe = (recipe or Recipe()).validate()
    end = recipe.end_update(stop_after)
    status = "complete" if end == recipe.updates else "stopped"
    for arm in ARMS:
        identity = json.loads((root / arm / "run.json").read_text())
        result = json.loads((root / arm / f"{status}.json").read_text())
        if result.get("checkpoint") != f"checkpoint-{end}":
            raise ValueError("Result references the wrong HF checkpoint")
        state = verify_checkpoint(root / arm / result["checkpoint"], identity)
        if (identity["recipe"] != asdict(recipe)
                or identity["arm"] != arm or result["status"] != status or result["arm"] != arm or result["update"] != end
                or result["input_tokens"] != end * recipe.tokens_per_update
                or state["update"] != end or state["identity"] != identity
                or result["final_evaluation"]["update"] != end
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
               "training_complete": end == recipe.updates, "compared_update": end,
               "schedule_updates": recipe.updates, "input_tokens_per_arm": end * recipe.tokens_per_update,
               "pilot_result": recipe == Recipe(),
               "negative_difference_favors_first_arm": True,
               "deep_gain_pattern": all(nll["D"] < nll[a] for a in "ABC"),
               "interpretation": "Single-seed pilot; direction of differences is not proof of statistical reliability"}
    write_json(root / "comparison.json", summary)
    return summary
