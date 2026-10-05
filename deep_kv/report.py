"""Compare selected matched train.py results, using held-out LM loss."""
import json
import copy
from pathlib import Path
from . import ALL_ARMS, BOTTLENECK_ARMS, code_loss_weight, kv_loss_weight


def comparable_config(config):
    """Compare the shared recipe; proxy target identity is checked separately."""
    config = copy.deepcopy(config)
    config['pilot'].pop('arm')
    if config['pilot'].get('proxy_screen'):
        config['pilot'].pop('proxy_channel_mask', None)
        config.pop('proxy_mask', None)
        config.pop('proxy_target', None)
        for key in ('target_version','variance_floor','target_clip','momentum','loss_form'):
            config['pilot'].pop('proxy_'+key,None)
    return config


def result_path(root, arm, baseline_dir=None):
    if arm == 'A' and baseline_dir is not None:
        source = Path(baseline_dir)
        if (root/arm).exists() and (root/arm).resolve() != source.resolve():
            raise ValueError('Ambiguous local and reused baseline')
        return source
    return root/arm


def report(directory, arms="ABCD", baseline_dir=None, expected_step=None, expected_seed=None):
    arms = tuple(arms)
    if not arms or len(set(arms)) != len(arms) or any(arm not in ALL_ARMS for arm in arms):
        raise ValueError("Select nonempty, unique arms from " + "/".join(ALL_ARMS))
    root = Path(directory)
    if baseline_dir is not None and 'A' not in arms:
        raise ValueError('Baseline reuse requires arm A')
    results, common = {}, None
    targets = {}
    for arm in arms:
        path = result_path(root, arm, baseline_dir)
        config = json.loads((path / "train_config.json").read_text())
        result = json.loads((path / "result.json").read_text())
        state = json.loads((path / "trainer_state.json").read_text())
        evaluation = json.loads((path / "eval_results.json").read_text())
        if (config["pilot"]["arm"] != arm or result["arm"] != arm
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
        if expected_step is not None and result['global_step'] != expected_step:
            raise ValueError('Incorrect reused baseline stopping step')
        if expected_seed is not None and any(config['training'][key] != expected_seed for key in ('seed','data_seed')):
            raise ValueError('Incorrect screening seed')
        if config['pilot'].get('proxy_screen') and arm.startswith(('P1-', 'P3-')):
            target = config.get('proxy_target')
            if not target or target.get('target_version') != 'r7' or config['pilot'].get('proxy_channel_mask'):
                raise ValueError('Proxy arms require the r7 target definition without a channel mask')
            targets[arm] = target
            for other, definition in targets.items():
                # Families differ only in the raw target quantity and target index set.
                ignored = set() if other[:2] == arm[:2] else {'quantity','bands'}
                if ({k:v for k,v in target.items() if k not in ignored} !=
                        {k:v for k,v in definition.items() if k not in ignored}):
                    raise ValueError('Proxy arms differ in target normalization/loss definition')
            for key, value in [('target_version',target['target_version']),('variance_floor',target['variance_floor']),
                               ('target_clip',target['clip']),('momentum',target['momentum']),('loss_form',target['loss_form'])]:
                if config['pilot'].get('proxy_'+key) != value:
                    raise ValueError('Proxy target metadata differs from pilot configuration')
        matched = (comparable_config(config), result["global_step"])
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
        summary['proxy_targets'] = targets
        for family in ('P1','P3'):
            for route in ('block','flow'):
                arm = family+'-'+route
                for control in (family+'-lambda0','V'+family[1],'A'):
                    if arm in nll and control in nll:
                        summary['nll_differences'][arm+'-'+control] = nll[arm]-nll[control]
        summary['training_cost'] = {arm:results[arm]['training_cost'] for arm in arms}
        summary['reliance_loss_increase'] = {arm:results[arm]['evaluation'].get('eval_reliance_loss_increase') for arm in arms}
    if baseline_dir is not None:
        summary['reused_baseline'] = str(Path(baseline_dir).resolve())
    root.mkdir(parents=True, exist_ok=True)
    (root / "comparison.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def report_seeds(directory, arms, seeds, reuse_baselines=None):
    import statistics
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError('Seeds must be distinct nonnegative integers')
    root = Path(directory)
    reuse_baselines = reuse_baselines or {}
    if set(reuse_baselines)-set(seeds) or (reuse_baselines and 'A' not in arms):
        raise ValueError('Reused baselines must belong to selected seeds and arm A')
    rows = {seed:report(root/f'seed-{seed}',arms,baseline_dir=reuse_baselines.get(seed),expected_seed=seed) for seed in seeds}
    recipes = []
    for seed in seeds:
        path = result_path(root/f'seed-{seed}',arms[0],reuse_baselines.get(seed))
        cfg = comparable_config(json.loads((path/'train_config.json').read_text()))
        if cfg['training']['seed'] != seed or cfg['training']['data_seed'] != seed:
            raise ValueError('Incorrect screening seed')
        for key in ('seed','data_seed'):
            cfg['training'].pop(key)
        # HF shuffle fingerprints intentionally differ across seeds.
        cfg.pop('train_fingerprint')
        recipes.append(cfg)
    if any(cfg != recipes[0] for cfg in recipes) or len({r['compared_update'] for r in rows.values()}) != 1:
        raise ValueError('Seed runs differ in recipe or cutoff')
    if any(row.get('proxy_targets') != rows[seeds[0]].get('proxy_targets') for row in rows.values()):
        raise ValueError('Seed runs differ in proxy target definition')
    losses = {arm:[rows[seed]['lm_loss'][arm] for seed in seeds] for arm in arms}
    result = dict(status='complete',seeds=list(seeds),lm_loss={arm:dict(values=values,mean=statistics.mean(values),
                  stdev=statistics.stdev(values) if len(values)>1 else None,range=max(values)-min(values)) for arm,values in losses.items()},
                  per_seed_differences={str(seed):r['nll_differences'] for seed,r in rows.items()},
                  interpretation='Inspect paired differences and seed spread; no significance claim from three seeds alone.')
    (root/'comparison-seeds.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result
