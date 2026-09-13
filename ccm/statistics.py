"""Paired cluster bootstrap. Policies are explicit, not silently scientific defaults."""
import json
from pathlib import Path
import numpy as np
from .contracts import require, read_json, write_json, file_hash


def records(path, population, cluster):
    result = {}
    meta = read_json(Path(path)/"metrics.json")
    require(file_hash(Path(path)/"segments.jsonl") == meta["segments_sha256"], "Evaluation records checksum mismatch")
    with (Path(path)/"segments.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            key = r[cluster]
            result.setdefault(key, np.zeros(2, dtype=np.float64))
            result[key] += r[population]
    total = np.sum(list(result.values()), axis=0)
    expected = meta["metrics"][population]
    require(total[1] == expected["count"] and np.isclose(total[0], expected["loss_sum"], rtol=1e-10, atol=1e-7),
            "Evaluation summary disagrees with its segment records")
    return result


def paired_arrays(left, right, population="overall", cluster="content_hash", allow_final=False):
    lm, rm = read_json(Path(left)/"metrics.json"), read_json(Path(right)/"metrics.json")
    for key in ("corpus_hash", "vocabulary_hash", "role", "phase", "seed", "step", "total_steps",
                "source_checkpoint_hash", "engineering"):
        require(lm[key] == rm[key], f"Unpaired evaluation metadata: {key}")
    require(lm["role"] == "dev" or (allow_final and lm["role"] == "val" and
            lm.get("final_evaluation") and rm.get("final_evaluation")),
            "Development decision statistics cannot consume locked validation; use explicit final reporting")
    if lm["role"] == "val":
        require(lm.get("final_lock_hash") is not None and lm["final_lock_hash"] == rm.get("final_lock_hash"),
                "Final reports must share the same decision lock")
    if lm["arm"] != "base" and rm["arm"] != "base":
        require(lm["paired_initial_reader_hash"] == rm["paired_initial_reader_hash"], "Reader initializations are not paired")
    l, r = records(left, population, cluster), records(right, population, cluster)
    require(set(l) == set(r), "Different evaluation clusters")
    ids = sorted(l)
    a, b = np.array([l[x] for x in ids]), np.array([r[x] for x in ids])
    require(np.array_equal(a[:, 1], b[:, 1]), "Mismatched paired target counts")
    require(a[:, 1].sum() > 0, "No targets in selected population")
    return ids, a, b


def bootstrap(pairs, replicates=10000, seed=20260913, coupling="shared"):
    require(coupling in ("shared", "independent"), "Choose cross-seed document coupling explicitly")
    require(replicates > 0 and pairs, "Empty bootstrap")
    for ids, a, b in pairs:
        require(ids == pairs[0][0], "Backbones must evaluate identical clusters")
    rng = np.random.default_rng(seed)
    n = len(pairs[0][0])
    effects = [float(a[:, 0].sum()/a[:, 1].sum()-b[:, 0].sum()/b[:, 1].sum()) for _, a, b in pairs]
    draws = []
    # Zero-hit/miss clusters can yield an undefined resample on tiny toy data.
    # Reject that replicate instead of turning 0/0 into fabricated zero effect.
    attempts = 0
    while len(draws) < replicates:
        attempts += 1
        require(attempts <= replicates*100, "Too few populated clusters for bootstrap")
        shared = rng.integers(0, n, size=n)
        values = []
        for _, a, b in pairs:
            ix = shared if coupling == "shared" else rng.integers(0, n, size=n)
            denominator = a[ix, 1].sum()
            if denominator == 0:
                break
            values.append(float((a[ix, 0].sum()-b[ix, 0].sum())/denominator))
        if len(values) == len(pairs):
            draws.append(float(np.mean(values)))
    lo, hi = map(float, np.quantile(draws, [.025, .975]))
    return dict(mean_difference=float(np.mean(effects)), per_seed_effects=effects, lower95=lo, upper95=hi,
                favorable_all_seeds=all(x < 0 for x in effects), replicates=replicates, bootstrap_seed=seed,
                cross_seed_coupling=coupling, interpretation="Evaluation-cluster uncertainty conditional on fixed trained backbones")


def compare(args):
    require(len(args.left) == len(args.right), "Need paired arm paths for each seed")
    lm_all = [read_json(Path(p)/"metrics.json") for p in args.left]
    rm_all = [read_json(Path(p)/"metrics.json") for p in args.right]
    require(lm_all, "Need at least one paired evaluation")
    require(all(m["engineering"] for m in lm_all + rm_all) or args.replicates == 10000,
            "Scientific comparisons require 10000 bootstrap replicates")
    if args.final_report:
        policy = dict(cluster=args.cluster, cross_seed_coupling=args.cross_seed_coupling)
        for meta in lm_all + rm_all:
            require(meta.get("statistical_policy") == policy,
                    "Final report statistical policy differs from the frozen decision lock")
    seeds = [m["seed"] for m in lm_all]
    require(len(set(seeds)) == len(seeds), "Duplicate backbone seed in combined comparison")
    for key in ("corpus_hash", "vocabulary_hash", "phase", "step", "total_steps", "arm", "role", "engineering"):
        require(all(m[key] == lm_all[0][key] for m in lm_all) and all(m[key] == rm_all[0][key] for m in rm_all),
                f"Cross-seed comparison mixes contracts: {key}")
    pairs = [paired_arrays(l, r, args.population, args.cluster, args.final_report) for l, r in zip(args.left, args.right)]
    result = bootstrap(pairs, args.replicates, coupling=args.cross_seed_coupling)
    result.update(left=args.left, right=args.right, population=args.population, cluster=args.cluster,
                  sign="negative means left is better", backbone_seeds=seeds,
                  final_report=args.final_report, three_seed_panel=set(seeds) == {17, 29, 43})
    # Per-seed intervals must remain visible; never report only a pooled CI.
    result["per_seed"] = [bootstrap([p], args.replicates) for p in pairs]
    result["binned_differences"] = {}
    for family in ("frequency", "variance"):
        by_seed = []
        for l, r in zip(args.left, args.right):
            lm, rm = read_json(Path(l)/"metrics.json"), read_json(Path(r)/"metrics.json")
            if family == "variance":
                if lm.get("diagnostic_table_hash") != rm.get("diagnostic_table_hash"):
                    by_seed.append({"unavailable": "Different bin-defining tables; overall comparison remains valid"})
                    continue
            differences = []
            for (ls, ln), (rs, rn) in zip(lm[family], rm[family]):
                require(ln == rn, "Binned target counts differ across paired arms")
                differences.append((ls-rs)/ln if ln else None)
            by_seed.append(differences)
        result["binned_differences"][family] = by_seed
    write_json(args.output, result)
    return result


def delta_decision(args):
    paths = dict(delta=args.delta, contextual=args.contextual, shuffled=args.shuffled)
    metas = {k: read_json(Path(p)/"metrics.json") for k, p in paths.items()}
    for arm, meta in metas.items():
        require(meta["arm"] == arm and meta["phase"] == "stage1" and meta["seed"] == 17,
                "Delta inclusion is an explicit seed-17 Stage-1 screen decision")
        require(meta["step"] == meta["total_steps"], "Delta decision requires completed Stage-1 adaptation")
    results = {}
    for control in ("contextual", "shuffled"):
        results[f"hit_vs_{control}"] = bootstrap([paired_arrays(args.delta, paths[control], "hit", args.cluster)])
    results["miss_vs_contextual"] = bootstrap([paired_arrays(args.delta, args.contextual, "miss", args.cluster)])
    include = (results["hit_vs_contextual"]["upper95"] < 0 and results["hit_vs_shuffled"]["upper95"] < 0
               and results["miss_vs_contextual"]["upper95"] <= .002
               and metas["delta"]["metrics"]["overall"]["nll"] <= metas["contextual"]["metrics"]["overall"]["nll"])
    require(args.replication_policy == "seed17_then_all", "Explicit replication policy required")
    report = dict(include_delta=include, results=results, cluster=args.cluster,
                  replication_policy=args.replication_policy, decision_seed=17,
                  corpus_hash=metas["delta"]["corpus_hash"], vocabulary_hash=metas["delta"]["vocabulary_hash"],
                  source_evaluations=paths)
    write_json(args.output, report)
    return report
