"""Compare selected matched train.py results, using held-out LM loss."""
import json
from pathlib import Path
from . import ARMS, kv_loss_weight


def report(directory, arms="ABCD"):
    arms = tuple(arms)
    if not arms or len(set(arms)) != len(arms) or any(arm not in ARMS for arm in arms):
        raise ValueError("Select nonempty, unique arms from A/B/C/D/E")
    root = Path(directory)
    results, common = {}, None
    for arm in arms:
        path = root / arm
        config = json.loads((path / "train_config.json").read_text())
        result = json.loads((path / "result.json").read_text())
        state = json.loads((path / "trainer_state.json").read_text())
        evaluation = json.loads((path / "eval_results.json").read_text())
        if (config["pilot"].pop("arm") != arm or result["arm"] != arm
                or state["global_step"] != result["global_step"]
                or state["max_steps"] != result["schedule_steps"]
                or result["schedule_steps"] != config["training"]["max_steps"]
                or not 0 < result["global_step"] <= result["schedule_steps"]
                or result["input_tokens"] != result["global_step"] * config["tokens_per_update"]
                or result["evaluation"] != evaluation
                or evaluation["eval_rows"] != config["data"]["eval_rows"]
                or not (path / "model.safetensors").is_file()):
            raise ValueError(f"Incomplete or inconsistent result: {arm}")
        expected = "complete" if result["global_step"] == result["schedule_steps"] else "stopped"
        if result["status"] != expected:
            raise ValueError(f"Incorrect completion status: {arm}")
        matched = (config, result["global_step"])
        if common is not None and matched != common:
            raise ValueError("Arms differ in configuration, data, or stopping step")
        common, results[arm] = matched, result
    nll = {arm: value["evaluation"]["eval_lm_loss"] for arm, value in results.items()}
    summary = {"status": "complete", "compared_update": results[arms[0]]["global_step"], "lm_loss": nll,
               "nll_differences": {f"{a}-{b}": nll[a] - nll[b] for a, b in
                                   (("B", "A"), ("C", "B"), ("D", "B"), ("D", "C"), ("D", "A"),
                                    ("E", "D"), ("E", "B"), ("E", "C"), ("E", "A")) if a in nll and b in nll},
               "deep_gain_pattern": all(nll["D"] < nll[a] for a in "ABC") if set("ABCD") <= nll.keys() else None,
               "interpretation": "Negative difference favors first arm; single-seed exploratory comparison."}
    if "E" in arms:
        summary["kv_loss_weights"] = {arm: kv_loss_weight(arm) for arm in arms}
    (root / "comparison.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary
