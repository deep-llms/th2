"""Compare selected matched train.py results, using held-out LM loss."""
import json
from pathlib import Path
from . import ALL_ARMS, BOTTLENECK_ARMS, code_loss_weight, kv_loss_weight


def report(directory, arms="ABCD"):
    arms = tuple(arms)
    if not arms or len(set(arms)) != len(arms) or any(arm not in ALL_ARMS for arm in arms):
        raise ValueError("Select nonempty, unique arms from " + "/".join(ALL_ARMS))
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
                                    ("E", "D"), ("E", "B"), ("E", "C"), ("E", "A"),
                                    ("F", "B"), ("G", "F"), ("G", "B"),
                                    ("F", "A"), ("G", "A"), ("F", "D"), ("G", "D"),
                                    ("F", "E"), ("G", "E"),
                                    ("Task-Aware-Align", "Task-Aware-NoAlign"),
                                    ("Consumer-Aware-Align", "Consumer-Aware-NoAlign"),
                                    ("Consumer-Aware-Align", "Task-Aware-Align")) if a in nll and b in nll},
               "deep_gain_pattern": all(nll["D"] < nll[a] for a in "ABC") if set("ABCD") <= nll.keys() else None,
               "interpretation": "Negative difference favors first arm; single-seed exploratory comparison."}
    if "E" in arms:
        summary["kv_loss_weights"] = {arm: kv_loss_weight(arm) for arm in arms if arm in "ABCDE"}
    if set(arms) & set(BOTTLENECK_ARMS):
        summary["bottleneck_loss_weights"] = {arm: {"extractor": 1.0, "alignment": code_loss_weight(arm)}
                                              for arm in arms if arm in BOTTLENECK_ARMS}
        summary["training_cost"] = {arm: results[arm].get("training_cost") for arm in arms}
    if set(arms) & {"F", "G"}:
        summary["functional_loss_weights"] = {arm: {"route": .3, "message": .3 if arm == "G" else 0.}
                                               for arm in arms if arm in ("F", "G")}
    if common[0]['pilot'].get('proxy_screen'):
        for family in ('P1','P3'):
            for route in ('block','flow'):
                arm = family+'-'+route
                for control in (family+'-lambda0','V'+family[1],'A'):
                    if arm in nll and control in nll:
                        summary['nll_differences'][arm+'-'+control] = nll[arm]-nll[control]
        summary['training_cost'] = {arm:results[arm]['training_cost'] for arm in arms}
        summary['reliance_loss_increase'] = {arm:results[arm]['evaluation'].get('eval_reliance_loss_increase') for arm in arms}
    (root / "comparison.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def report_seeds(directory, arms, seeds):
    import statistics
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError('Seeds must be distinct nonnegative integers')
    root = Path(directory)
    rows = {seed:report(root/f'seed-{seed}',arms) for seed in seeds}
    recipes = []
    for seed in seeds:
        cfg = json.loads((root/f'seed-{seed}'/arms[0]/'train_config.json').read_text())
        if cfg['training']['seed'] != seed or cfg['training']['data_seed'] != seed:
            raise ValueError('Incorrect screening seed')
        for key in ('seed','data_seed'):
            cfg['training'].pop(key)
        # HF shuffle fingerprints intentionally differ across seeds.
        cfg.pop('train_fingerprint')
        recipes.append(cfg)
    if any(cfg != recipes[0] for cfg in recipes) or len({r['compared_update'] for r in rows.values()}) != 1:
        raise ValueError('Seed runs differ in recipe or cutoff')
    losses = {arm:[rows[seed]['lm_loss'][arm] for seed in seeds] for arm in arms}
    result = dict(status='complete',seeds=list(seeds),lm_loss={arm:dict(values=values,mean=statistics.mean(values),
                  stdev=statistics.stdev(values) if len(values)>1 else None,range=max(values)-min(values)) for arm,values in losses.items()},
                  per_seed_differences={str(seed):r['nll_differences'] for seed,r in rows.items()},
                  interpretation='Inspect paired differences and seed spread; no significance claim from three seeds alone.')
    (root/'comparison-seeds.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result
