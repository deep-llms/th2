#!/usr/bin/env python3
"""Paired per-example analysis of lm-eval `--log_samples` runs.

Reference implementation for downstream_paired_logprob_analysis_spec.md (Revision 2).
Differences are always B - A (positive values favour B). Subcommands:
  admit    --a10 DIR --a20 DIR [--tasks ...] --out admission.json     task admission from A only (Section 7)
  pair     DIR_A DIR_B [--tasks ...] [--admission admission.json]       paired analysis (Sections 6, 8, 9)
  paloma   DIR_A DIR_B [--tasks ...]                                    bits-per-byte by source (Section 8.4)
  position LOSS_A.npy LOSS_B.npy                                        loss by token position (Section 8.5)
A run directory holds one subdirectory per task: RUN_DIR/<task>/**/results_*.json and samples_*.jsonl
(a group task such as mmlu_continuation has one samples file per subtask; all subtasks are pooled).
"""
import argparse, glob, json, math, os, re
import numpy as np

# task -> (kind, scoring rule, reported accuracy key). kind: "mc" multiple choice, "ll" one continuation.
TASKS = {
    "arc_easy": ("mc", "char", "acc_norm"), "arc_challenge": ("mc", "char", "acc_norm"),
    "hellaswag": ("mc", "char", "acc_norm"), "piqa": ("mc", "char", "acc_norm"),
    "sciq": ("mc", "char", "acc_norm"), "openbookqa": ("mc", "char", "acc_norm"),
    "social_iqa_norm": ("mc", "char", "acc_norm"), "commonsense_qa_cf": ("mc", "char", "acc_norm"),
    "mmlu_continuation": ("mc", "char", "acc_norm"),
    "winogrande": ("mc", "raw", "acc"), "xnli_en": ("mc", "raw", "acc"), "xstorycloze_en": ("mc", "raw", "acc"),
    "paws_en": ("mc", "raw", "acc"), "blimp": ("mc", "raw", "acc"),
    "babi_cf_qa1": ("mc", "raw", "acc"), "babi_cf_qa2": ("mc", "raw", "acc"), "babi_cf_qa3": ("mc", "raw", "acc"),
    "lambada_openai": ("ll", None, "acc"),
}
SUITE_Z = ["arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande", "xnli_en", "xstorycloze_en",
           "lambada_openai", "sciq", "openbookqa", "social_iqa_norm", "commonsense_qa_cf",
           "mmlu_continuation", "blimp"]
PROBES = ["babi_cf_qa1", "babi_cf_qa2", "babi_cf_qa3"]
PALOMA = ["paloma_c4_en", "paloma_falcon-refinedweb", "paloma_wikitext_103", "paloma_m2d2_s2orc_unsplit",
          "paloma_dolma_100_programing_languages", "paloma_dolma_100_subreddits"]
# Subtasks whose choices sit in the context and share one continuation; gold index as in the task's doc_to_text.
MULTI_INPUT_GOLD = {"winogrande": lambda doc: {"1": 0, "2": 1}[doc["answer"]]}
METRICS = ("M1", "M2", "M3", "ACC")
SAMPLES_RE = re.compile(r"samples_(.+)_(\d{4}-\d{2}-\d{2}T[\d\-.]+)\.jsonl$")


def one_file(pattern):
    files = sorted(glob.glob(pattern, recursive=True))
    if len(files) != 1:
        raise RuntimeError(f"expected exactly one file for {pattern}, found {len(files)}")
    return files[0]


def _args(s):
    args = s["arguments"]
    if isinstance(args, dict):  # lm-eval >= 0.4.3: {"gen_args_N": {"arg_0": ..., "arg_1": ...}}
        args = [args[k] for k in sorted(args, key=lambda k: int(k.rsplit("_", 1)[1]))]
        args = [[a[f"arg_{j}"] for j in range(len(a))] for a in args]
    return [tuple(a) for a in args]


def _parse(s, kind, sub, delim, rule):
    args = _args(s)
    r = {"args": args, "doc_hash": s["doc_hash"], "prompt_hash": s["prompt_hash"],
         "logged": {k: s[k] for k in ("acc", "acc_norm", "bits_per_byte") if k in s}}
    resp = s["filtered_resps"]
    if kind == "rolling":                                   # whole document: ll and UTF-8 bytes
        r["ll"] = float(resp[0] if isinstance(resp, list) else resp)
        r["bytes"] = len(s["target"].encode("utf-8"))
        return r
    if kind == "ll":                                        # one continuation (LAMBADA)
        cont = args[0][1]
        r.update(ll=float(resp[0][0]), greedy=str(resp[0][1]) == "True", n_ans=float(len(cont.strip())))
        return r
    cont = [a[1] for a in args]
    if not all(c.startswith(delim) for c in cont):
        raise RuntimeError(f"{sub} doc {s['doc_id']}: continuation lacks the target delimiter")
    r["ll"] = np.array([float(x[0]) for x in resp], dtype=np.float64)
    r["n"] = np.array([len(c) - len(delim) for c in cont], dtype=np.float64)  # = len(choice), as acc_norm
    r["multi"] = sub in MULTI_INPUT_GOLD
    if r["multi"]:
        if rule == "char" or len(set(cont)) != 1 or cont[0] != delim + s["target"]:
            raise RuntimeError(f"{sub} doc {s['doc_id']}: not a shared-continuation task scored raw")
        r["gold"], r["n_ans"] = MULTI_INPUT_GOLD[sub](s["doc"]), r["n"][0]
    else:
        r["gold"] = int(s["target"])
        r["n_ans"] = r["n"][r["gold"]]
    return r


def load_task(run_dir, task, kind=None):
    """Parse one run's output for one task. Returns ({(subtask, doc_id): record}, results JSON)."""
    kind = kind or TASKS[task][0]
    rule = TASKS[task][1] if task in TASKS else None
    tdir = os.path.join(run_dir, task)
    with open(one_file(os.path.join(tdir, "**", "results_*.json")), encoding="utf-8") as f:
        results = json.load(f)
    files = sorted(glob.glob(os.path.join(tdir, "**", "samples_*.jsonl"), recursive=True))
    if not files:
        raise RuntimeError(f"{task}: no samples files under {tdir}")
    recs = {}
    for fn in files:
        sub = SAMPLES_RE.search(os.path.basename(fn)).group(1)
        delim = results["configs"][sub].get("target_delimiter")
        delim = " " if delim is None else delim
        with open(fn, encoding="utf-8") as f:
            for line in f:
                s = json.loads(line)
                key = (sub, s["doc_id"])
                if key in recs:
                    raise RuntimeError(f"{task}: duplicate example {key}")
                recs[key] = _parse(s, kind, sub, delim, rule)
    return recs, results


def example_metrics(r, kind, rule):
    if kind == "ll":
        return {"M1": r["ll"] / r["n_ans"], "M2": np.nan, "M3": np.nan, "ACC": float(r["greedy"])}
    ll, g = r["ll"], r["gold"]
    s = ll / r["n"] if rule == "char" else ll               # scores of the reported accuracy
    p = np.exp(s - s.max())
    p /= p.sum()
    return {"M1": ll[g] / r["n_ans"],                       # per-character log-likelihood of the gold answer
            "M2": p[g],                                     # normalized probability of the gold answer
            "M3": s[g] - np.delete(s, g).max(),             # margin over the best wrong choice
            "ACC": float(np.argmax(s) == g)}                # reported accuracy (acc_norm or acc)


# ---------------------------------------------------------------- integration tests T1, T2, T5
def _close(a, b, rel=1e-9):
    return abs(a - b) <= rel * max(1.0, abs(b))


def check_t1(recs, results, task, kind=None):
    """Recomputed per-example metrics equal the logged ones; subtask means equal the results JSON."""
    kind = kind or TASKS[task][0]
    key = TASKS[task][2] if task in TASKS else None
    bad, by_sub = 0, {}
    for (sub, _), r in recs.items():
        by_sub.setdefault(sub, []).append(r)
    for sub, rs in by_sub.items():
        agg = results["results"][sub]
        if kind == "rolling":
            ll, nb = sum(r["ll"] for r in rs), sum(r["bytes"] for r in rs)
            bad += sum(r["logged"]["bits_per_byte"][1] != r["bytes"] for r in rs)
            bad += not _close(-ll / nb / math.log(2), agg["bits_per_byte,none"])
            continue
        if any(key not in r["logged"] for r in rs):
            return False, -1
        if kind == "ll":
            acc = [float(r["greedy"]) for r in rs]
            bad += sum(float(r["logged"]["acc"]) != a for r, a in zip(rs, acc))
            bad += not _close(math.exp(-np.mean([r["ll"] for r in rs])), agg["perplexity,none"])
        else:
            acc = [float(np.argmax(r["ll"]) == r["gold"]) for r in rs]
            bad += sum("acc" in r["logged"] and r["logged"]["acc"] != a for r, a in zip(rs, acc))
            if "acc_norm" in rs[0]["logged"] and not rs[0]["multi"]:
                accn = [float(np.argmax(r["ll"] / r["n"]) == r["gold"]) for r in rs]
                bad += sum(r["logged"]["acc_norm"] != a for r, a in zip(rs, accn))
                bad += abs(np.mean(accn) - agg["acc_norm,none"]) > 1e-12
        bad += abs(np.mean(acc) - agg["acc,none"]) > 1e-12
    return bool(bad == 0), int(bad)


def check_t2(ra, rb):
    """Same examples, documents, prompts and continuations in both runs."""
    if set(ra) != set(rb):
        return False
    return all(ra[i]["doc_hash"] == rb[i]["doc_hash"] and ra[i]["prompt_hash"] == rb[i]["prompt_hash"]
               and ra[i]["args"] == rb[i]["args"] for i in ra)


def check_t5(recs, results):
    counts = {}
    for sub, _ in recs:
        counts[sub] = counts.get(sub, 0) + 1
    ns = results.get("n-samples", {})
    return all(sub not in ns or n == ns[sub]["effective"] == ns[sub]["original"] for sub, n in counts.items())


def load_pair(dir_a, dir_b, task, kind=None):
    ra, res_a = load_task(dir_a, task, kind)
    rb, res_b = load_task(dir_b, task, kind)
    (t1a, bad_a), (t1b, bad_b) = check_t1(ra, res_a, task, kind), check_t1(rb, res_b, task, kind)
    chk = {"T1": t1a and t1b, "T1_mismatches": bad_a + bad_b, "T2": bool(check_t2(ra, rb)),
           "T5": bool(check_t5(ra, res_a) and check_t5(rb, res_b))}
    if not (chk["T1"] and chk["T2"] and chk["T5"]):
        raise RuntimeError(f"integration test failed for {task}: {chk}")
    return ra, rb, chk


def metric_arrays(recs, task):
    kind, rule, _ = TASKS[task]
    ids = sorted(recs)
    ms = [example_metrics(recs[i], kind, rule) for i in ids]
    return {m: np.array([x[m] for x in ms]) for m in METRICS}, ids


# ---------------------------------------------------------------- statistics (unchanged from Revision 1)
def _boot(d, B, rng, chunk):
    """Bootstrap replicates of the mean and of the standardized mean (mean / sd) of d."""
    N, m, z = len(d), np.empty(B), np.empty(B)
    for s in range(0, B, chunk):
        b = min(chunk, B - s)
        x = d[rng.integers(0, N, size=(b, N))]
        mu, sd = x.mean(axis=1), x.std(axis=1, ddof=1)
        m[s:s + b], z[s:s + b] = mu, np.divide(mu, sd, out=np.zeros(b), where=sd > 0)
    return m, z


def _flip(d, B, rng, chunk):
    """Sign-flip (paired permutation) replicates of the mean and standardized mean under H0."""
    N, ss, m, z = len(d), float(d @ d), np.empty(B), np.empty(B)
    for s in range(0, B, chunk):
        b = min(chunk, B - s)
        signs = rng.integers(0, 2, size=(b, N), dtype=np.int8) * 2 - 1
        mu = (signs @ d) / N
        sd = np.sqrt(np.maximum(ss - N * mu ** 2, 0.0) / (N - 1))
        m[s:s + b], z[s:s + b] = mu, np.divide(mu, sd, out=np.zeros(b), where=sd > 0)
    return m, z


def _std(d):
    sd = d.std(ddof=1)
    return d.mean() / sd if sd > 0 else 0.0


def _p(null, obs):
    return (1 + np.sum(np.abs(null) >= abs(obs) * (1 - 1e-9))) / (len(null) + 1)


def task_stats(d, B=10_000, seed=0, chunk=250):
    rng = np.random.default_rng(seed)
    bm, bz = _boot(d, B, rng, chunk)
    fm, fz = _flip(d, B, rng, chunk)
    return {"N": len(d), "mean": d.mean(), "ci": np.percentile(bm, [2.5, 97.5]).tolist(), "p": _p(fm, d.mean()),
            "std_effect": _std(d), "std_ci": np.percentile(bz, [2.5, 97.5]).tolist(), "p_std": _p(fz, _std(d)),
            "frac_pos": float(np.mean(d > 0)), "frac_neg": float(np.mean(d < 0))}


def aggregate_stats(d_by_task, B=10_000, seed=0, chunk=250):
    """Equal task weights; stratified bootstrap and per-task sign flips, combined replicate by replicate."""
    rng, T = np.random.default_rng(seed), len(d_by_task)
    bm, bz, fm, fz = (np.zeros(B) for _ in range(4))
    for d in d_by_task:
        m, z = _boot(d, B, rng, chunk); bm += m / T; bz += z / T
        m, z = _flip(d, B, rng, chunk); fm += m / T; fz += z / T
    mean = float(np.mean([d.mean() for d in d_by_task]))
    std = float(np.mean([_std(d) for d in d_by_task]))
    return {"mean": mean, "ci": np.percentile(bm, [2.5, 97.5]).tolist(), "p": _p(fm, mean),
            "std_effect": std, "std_ci": np.percentile(bz, [2.5, 97.5]).tolist(), "p_std": _p(fz, std)}


# ---------------------------------------------------------------- additional statistics (Revision 2)
def contrast_stats(d1, d2, B=10_000, seed=0, chunk=250):
    """Difference of standardized effects, delta(d2) - delta(d1), for two independent tasks."""
    rng = np.random.default_rng(seed)
    _, z1 = _boot(d1, B, rng, chunk)
    _, z2 = _boot(d2, B, rng, chunk)
    return {"contrast": _std(d2) - _std(d1), "ci": np.percentile(z2 - z1, [2.5, 97.5]).tolist()}


def ratio_stats(s, c, B=10_000, seed=0, chunk=250):
    """Ratio sum(s)/sum(c) over clusters (documents) with cluster bootstrap CI and sign-flip p-value."""
    rng, N = np.random.default_rng(seed), len(s)
    est, bt, fl = s.sum() / c.sum(), np.empty(B), np.empty(B)
    for k in range(0, B, chunk):
        b = min(chunk, B - k)
        idx = rng.integers(0, N, size=(b, N))
        bt[k:k + b] = s[idx].sum(axis=1) / c[idx].sum(axis=1)
        signs = rng.integers(0, 2, size=(b, N), dtype=np.int8) * 2 - 1
        fl[k:k + b] = (signs @ s) / c.sum()
    return {"estimate": float(est), "ci": np.percentile(bt, [2.5, 97.5]).tolist(), "p": _p(fl, est), "N": N}


def ratio_contrast(s1, c1, s2, c2, B=10_000, seed=0, chunk=250):
    """sum(s2)/sum(c2) - sum(s1)/sum(c1) on the same clusters, cluster bootstrap CI."""
    rng, N = np.random.default_rng(seed), len(s1)
    bt = np.empty(B)
    for k in range(0, B, chunk):
        b = min(chunk, B - k)
        idx = rng.integers(0, N, size=(b, N))
        bt[k:k + b] = s2[idx].sum(1) / c2[idx].sum(1) - s1[idx].sum(1) / c1[idx].sum(1)
    return {"contrast": float(s2.sum() / c2.sum() - s1.sum() / c1.sum()),
            "ci": np.percentile(bt, [2.5, 97.5]).tolist()}


# ---------------------------------------------------------------- drivers
def admit(dir_a10, dir_a20, tasks=SUITE_Z):
    """Admission (Section 7): uses A only. ACC under each task's scoring rule, all subtasks pooled."""
    out = {"tasks": {}, "admitted": []}
    for t in tasks:
        r10, r20, chk = load_pair(dir_a10, dir_a20, t)
        a10, ids = metric_arrays(r10, t)
        a20, _ = metric_arrays(r20, t)
        N, acc10, acc20 = len(ids), a10["ACC"].mean(), a20["ACC"].mean()
        chance = 0.0 if TASKS[t][0] == "ll" else float(np.mean([1.0 / len(r20[i]["ll"]) for i in ids]))
        se = math.sqrt(acc20 * (1 - acc20) / N)              # binomial SE of the 20k accuracy
        d = a20["ACC"] - a10["ACC"]                            # same examples in both runs (T2)
        sd = d.std(ddof=1)
        above = (acc20 - chance) / se if se > 0 else 0.0
        gain_se = d.mean() / (sd / math.sqrt(N)) if sd > 0 else 0.0   # paired SE of the 10k->20k gain
        nonrandom, sensitive = bool(above >= 3), bool(gain_se >= 2)
        out["tasks"][t] = {"N": N, "chance": chance, "acc_10k": acc10, "acc_20k": acc20, "se_20k": se,
                           "above_chance_se": above, "gain": acc20 - acc10, "gain_se": gain_se,
                           "nonrandom": nonrandom, "sensitive": sensitive, "checks": chk}
        if nonrandom and sensitive:
            out["admitted"].append(t)
    return out


def analyze(dir_a, dir_b, tasks, B=10_000, seed=20261012, confirmatory=None):
    out = {"tasks": {}, "aggregate": {}, "checks": {}, "contrasts": {}}
    diffs = {}
    for k, t in enumerate(tasks):
        ra, rb, out["checks"][t] = load_pair(dir_a, dir_b, t)
        ma, _ = metric_arrays(ra, t)
        mb, _ = metric_arrays(rb, t)
        out["tasks"][t] = {}
        for m in METRICS:
            if np.isnan(ma[m]).any():
                continue
            d = mb[m] - ma[m]
            diffs[(t, m)] = d
            out["tasks"][t][m] = {"mean_A": ma[m].mean(), "mean_B": mb[m].mean(), **task_stats(d, B, seed + k)}
    sets = {"all": list(tasks)}
    if confirmatory is not None:
        sets["confirmatory"] = [t for t in tasks if t in confirmatory]
    for name, ts in sets.items():
        out["aggregate"][name] = {}
        for j, m in enumerate(METRICS):
            ds = [diffs[(t, m)] for t in ts if (t, m) in diffs]
            if ds:
                out["aggregate"][name][m] = {"tasks": [t for t in ts if (t, m) in diffs],
                                             **aggregate_stats(ds, B, seed + 1000 + j)}
    if all(t in tasks for t in PROBES):                     # hop contrasts (Section 8.3)
        for j, m in enumerate(("M1", "M2")):
            for t2 in PROBES[1:]:
                out["contrasts"][f"{m}:{t2}-{PROBES[0]}"] = contrast_stats(
                    diffs[(PROBES[0], m)], diffs[(t2, m)], B, seed + 2000 + 10 * j + PROBES.index(t2))
    if confirmatory is not None:                            # fixed-sequence test (Section 9)
        agg, c = out["aggregate"]["confirmatory"], sets["confirmatory"]
        s1 = len(c) >= 3 and "M1" in agg and agg["M1"]["p_std"] < 0.05
        h2 = s1 and "M2" in agg and len(agg["M2"]["tasks"]) >= 3 and agg["M2"]["p_std"] < 0.05
        out["confirmatory"] = {"set": c, "tested": len(c) >= 3, "H1_M1_rejected": bool(s1),
                               "H2_M2_rejected": bool(h2)}
    return out


def paloma(dir_a, dir_b, tasks=PALOMA, B=10_000, seed=20261012):
    """Bits-per-byte reduction BPB_A - BPB_B per source (byte-weighted), paired over documents."""
    out = {}
    for k, t in enumerate(tasks):
        ra, rb, chk = load_pair(dir_a, dir_b, t, kind="rolling")
        ids = sorted(ra)
        s = np.array([(rb[i]["ll"] - ra[i]["ll"]) / math.log(2) for i in ids])
        c = np.array([float(ra[i]["bytes"]) for i in ids])
        bpb = {arm: -sum(r[i]["ll"] for i in ids) / c.sum() / math.log(2) for arm, r in (("A", ra), ("B", rb))}
        out[t] = {"bpb_A": bpb["A"], "bpb_B": bpb["B"], "reduction": ratio_stats(s, c, B, seed + k), "checks": chk}
    return out


BUCKETS = [(1, 64), (64, 128), (128, 256), (256, 512), (512, 1024), (1024, 2048)]


def position(loss_a, loss_b, buckets=BUCKETS, B=10_000, seed=20261012):
    """Loss reduction (A - B, nats/token) by token position; arrays [documents, positions], NaN = no token."""
    if loss_a.shape != loss_b.shape or not np.array_equal(np.isnan(loss_a), np.isnan(loss_b)):
        raise RuntimeError("loss arrays must cover the same documents and positions")
    diff = loss_a - loss_b
    out, sc = {}, {}
    for k, (lo, hi) in enumerate(buckets):
        seg = diff[:, lo:hi]
        s, c = np.nansum(seg, axis=1), np.sum(~np.isnan(seg), axis=1).astype(float)
        keep = c > 0
        sc[(lo, hi)] = (s, c)
        out[f"{lo}-{hi - 1}"] = ratio_stats(s[keep], c[keep], B, seed + k)
    (s1, c1), (s2, c2) = sc[buckets[0]], sc[buckets[-1]]
    keep = (c1 > 0) & (c2 > 0)
    out["last_minus_first"] = ratio_contrast(s1[keep], c1[keep], s2[keep], c2[keep], B, seed + 100)
    out["overall"] = ratio_stats(np.nansum(diff, 1), np.sum(~np.isnan(diff), 1).astype(float), B, seed + 200)
    return out


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("admit"); a.add_argument("--a10", required=True); a.add_argument("--a20", required=True)
    a.add_argument("--tasks", nargs="+", default=SUITE_Z); a.add_argument("--out", default="admission.json")
    p = sp.add_parser("pair"); p.add_argument("run_a"); p.add_argument("run_b")
    p.add_argument("--tasks", nargs="+", default=SUITE_Z); p.add_argument("--admission")
    p.add_argument("--B", type=int, default=10_000); p.add_argument("--seed", type=int, default=20261012)
    p.add_argument("--out", default="paired_result.json")
    q = sp.add_parser("paloma"); q.add_argument("run_a"); q.add_argument("run_b")
    q.add_argument("--tasks", nargs="+", default=PALOMA); q.add_argument("--out", default="paloma_result.json")
    r = sp.add_parser("position"); r.add_argument("loss_a"); r.add_argument("loss_b")
    r.add_argument("--out", default="position_result.json")
    args = ap.parse_args()
    if args.cmd == "admit":
        res = admit(args.a10, args.a20, args.tasks)
        for t, x in res["tasks"].items():
            print(f"{t:18s} N {x['N']:6d} acc10 {x['acc_10k']:.4f} acc20 {x['acc_20k']:.4f} chance {x['chance']:.3f} "
                  f"above {x['above_chance_se']:+6.1f} SE gain {x['gain_se']:+5.1f} paired SE -> "
                  f"{'ADMIT' if t in res['admitted'] else 'descriptive'}")
    elif args.cmd == "pair":
        conf = None
        if args.admission:
            with open(args.admission) as f:
                conf = json.load(f)["admitted"]
        res = analyze(args.run_a, args.run_b, args.tasks, args.B, args.seed, conf)
        for name, agg in res["aggregate"].items():
            for m, g in agg.items():
                print(f"{name:12s} {m:3s} D {g['mean']:+.6f} [{g['ci'][0]:+.6f}, {g['ci'][1]:+.6f}] p {g['p']:.4f} | "
                      f"delta {g['std_effect']:+.4f} [{g['std_ci'][0]:+.4f}, {g['std_ci'][1]:+.4f}] p {g['p_std']:.4f}")
        for t, ms in res["tasks"].items():
            for m, x in ms.items():
                print(f"{t:18s} {m:3s} A {x['mean_A']:+.5f} B {x['mean_B']:+.5f} d {x['mean']:+.6f} "
                      f"[{x['ci'][0]:+.6f}, {x['ci'][1]:+.6f}] p {x['p']:.4f} >0 {x['frac_pos']:.3f}")
        print(res.get("contrasts"), res.get("confirmatory"))
    elif args.cmd == "paloma":
        res = paloma(args.run_a, args.run_b, args.tasks)
        for t, x in res.items():
            g = x["reduction"]
            print(f"{t:40s} BPB A {x['bpb_A']:.5f} B {x['bpb_B']:.5f} reduction {g['estimate']:+.6f} "
                  f"[{g['ci'][0]:+.6f}, {g['ci'][1]:+.6f}] p {g['p']:.4f}")
    else:
        res = position(np.load(args.loss_a), np.load(args.loss_b))
        for k, x in res.items():
            v = x.get("estimate", x.get("contrast"))
            print(f"{k:16s} {v:+.6f} [{x['ci'][0]:+.6f}, {x['ci'][1]:+.6f}]" + (f" p {x['p']:.4f}" if "p" in x else ""))
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1, default=float)


if __name__ == "__main__":
    main()
