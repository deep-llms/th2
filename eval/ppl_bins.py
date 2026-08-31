"""Offline frequency-binned perplexity from ``eval/ppl_bytoken.py`` outputs.

Runs on CPU with numpy only. Two binnings are produced:

* **level sets** — the frequency-ranked token blocks that received distinct
  ranks in the tiered arms (default ``2048,6144,24576,119168`` = the T4
  profile: rank 1024 / 512 / 192 / 64).
* **mass bins** — cumulative-occurrence bins (default head = the smallest
  prefix covering 90% of mass, torso = the next prefix through 99%, tail =
  the rest), matching the round-2 protocol.

Token order is ``(-count, token_id)`` with a stable sort, matching
``compositional.compression_init.frequency_rank_order``. Repeating a run name
merges disjoint language shards, which permits two one-GPU jobs per checkpoint:

  python eval/ppl_bins.py \
      --counts resources/token_freq_sample10.npz \
      --run dense=out/dense_a/eval_ppl_bytoken.npz \
      --run dense=out/dense_b/eval_ppl_bytoken.npz \
      --run control=out/control_a/eval_ppl_bytoken.npz \
      --run control=out/control_b/eval_ppl_bytoken.npz \
      --reference dense --output bins.json

For every bin and arm the script reports PPL, realized type/training/eval mass,
and, against ``--reference``, the exact contribution to the mean log-PPL gap.
"""

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np


RUN_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


def frequency_rank_order(counts):
    """Stable descending order with token id as the deterministic tie-break."""
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 1 or not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("counts must be a finite nonnegative 1-D vector")
    return np.argsort(-counts, kind="stable")


def _validate_order(order, vocab_size):
    order = np.asarray(order)
    if order.shape != (vocab_size,) or not np.issubdtype(order.dtype, np.integer):
        raise ValueError("order must be a one-dimensional integer vocabulary permutation")
    if not np.array_equal(np.sort(order), np.arange(vocab_size)):
        raise ValueError("order must contain every token id exactly once")
    return order.astype(np.int64, copy=False)


def level_set_bins(order, populations):
    """Return bin id per token for contiguous frequency-rank blocks."""
    order = _validate_order(order, np.asarray(order).size)
    populations = [int(population) for population in populations]
    if not populations or any(population <= 0 for population in populations):
        raise ValueError("populations must contain positive integers")
    if sum(populations) != order.size:
        raise ValueError("populations must sum to the vocabulary size")

    bin_of_token = np.empty(order.size, dtype=np.int64)
    labels = []
    start = 0
    for bin_id, size in enumerate(populations):
        end = start + size
        bin_of_token[order[start:end]] = bin_id
        labels.append(f"rank[{start}:{end})")
        start = end
    return bin_of_token, labels


def mass_bins(order, counts, thresholds):
    """Return bins whose prefixes minimally cover each cumulative threshold."""
    counts = np.asarray(counts, dtype=np.float64)
    order = _validate_order(order, counts.size)
    if counts.ndim != 1 or not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("counts must be a finite nonnegative 1-D vector")
    total = float(counts.sum())
    if total <= 0:
        raise ValueError("mass bins require a positive total occurrence count")

    edges = sorted(float(threshold) for threshold in thresholds)
    if not edges or any(not math.isfinite(edge) for edge in edges):
        raise ValueError("mass thresholds must be finite and non-empty")
    if any(not 0 < edge < 1 for edge in edges):
        raise ValueError("mass thresholds must lie strictly between 0 and 1")
    if any(left >= right for left, right in zip(edges, edges[1:])):
        raise ValueError("mass thresholds must be unique and strictly increasing")

    cumulative = np.cumsum(counts[order], dtype=np.float64)
    # Include the crossing token in the earlier bin: each prefix is the
    # smallest prefix whose mass is >= its threshold.
    ends = [
        int(np.searchsorted(cumulative, edge * total, side="left") + 1)
        for edge in edges
    ]
    bin_of_token = np.empty(order.size, dtype=np.int64)
    start = 0
    for bin_id, end in enumerate(ends):
        bin_of_token[order[start:end]] = bin_id
        start = end
    bin_of_token[order[start:]] = len(edges)

    labels = [f"mass<={edges[0]:g}"]
    labels.extend(
        f"mass({left:g},{right:g}]"
        for left, right in zip(edges, edges[1:])
    )
    labels.append(f"mass>{edges[-1]:g}")
    return bin_of_token, labels


def load_run(path, langs=None):
    """Return ``{lang: (nll_sum[V], count[V])}`` from one by-token shard."""
    selected = set(langs) if langs else None
    found = {}
    with np.load(path, allow_pickle=False) as data:
        nll_keys = sorted(key for key in data.files if key.endswith("_nll"))
        count_keys = {key for key in data.files if key.endswith("_cnt")}
        for key in nll_keys:
            lang = key[:-4]
            count_key = f"{lang}_cnt"
            if selected is not None and lang not in selected:
                continue
            if count_key not in data.files:
                raise ValueError(f"{path}: missing {count_key}")
            raw_nll = data[key]
            raw_count = data[count_key]
            if raw_nll.ndim != 1 or raw_count.shape != raw_nll.shape:
                raise ValueError(f"{path}: {lang} arrays must be matching vectors")
            if not np.issubdtype(raw_nll.dtype, np.number):
                raise ValueError(f"{path}: {key} must be numeric")
            if not np.issubdtype(raw_count.dtype, np.integer):
                raise ValueError(f"{path}: {count_key} must have integer dtype")
            nll = raw_nll.astype(np.float64, copy=True)
            count = raw_count.astype(np.int64, copy=True)
            if not np.isfinite(nll).all() or (nll < 0).any():
                raise ValueError(f"{path}: {key} must be finite and nonnegative")
            if (count < 0).any():
                raise ValueError(f"{path}: {count_key} must be nonnegative")
            found[lang] = (nll, count)

        selected_count_keys = {f"{lang}_cnt" for lang in found}
        unexpected_counts = {
            key
            for key in count_keys
            if (selected is None or key[:-4] in selected)
            and key not in selected_count_keys
        }
        if unexpected_counts:
            raise ValueError(
                f"{path}: count arrays without matching NLL arrays: "
                f"{sorted(unexpected_counts)}"
            )
    if not found:
        raise ValueError(f"no selected <lang>_nll arrays in {path}")
    return found


def merge_run_shards(name, shards):
    """Merge shards for one run, rejecting overlapping languages."""
    merged = {}
    for path, shard in shards:
        overlap = sorted(set(merged).intersection(shard))
        if overlap:
            raise ValueError(
                f"run {name!r} has duplicate languages {overlap} in shard {path}"
            )
        merged.update(shard)
    if not merged:
        raise ValueError(f"run {name!r} has no languages")
    return merged


def _validate_runs(runs, vocab_size):
    """Require identical languages and exact per-token evaluation counts."""
    if not runs:
        raise ValueError("at least one run is required")
    reference_name = next(iter(runs))
    reference = runs[reference_name]
    reference_languages = set(reference)
    if not reference_languages:
        raise ValueError(f"run {reference_name!r} has no languages")

    for name, run in runs.items():
        if set(run) != reference_languages:
            raise ValueError(
                f"run {name!r} languages {sorted(run)} differ from "
                f"{reference_name!r} languages {sorted(reference)}"
            )
        for lang in sorted(reference_languages):
            nll, count = run[lang]
            if nll.shape != (vocab_size,) or count.shape != (vocab_size,):
                raise ValueError(
                    f"run {name!r} language {lang}: expected vectors of "
                    f"length {vocab_size}, got {nll.shape} and {count.shape}"
                )
            if not np.array_equal(count, reference[lang][1]):
                raise ValueError(
                    f"run {name!r} language {lang}: per-token counts differ "
                    f"from run {reference_name!r}"
                )


def bin_table(run, bin_of_token, num_bins, vocab_size):
    """Per-bin NLL sums/counts pooled over languages, plus per language."""
    pooled_nll = np.zeros(num_bins, dtype=np.float64)
    pooled_count = np.zeros(num_bins, dtype=np.int64)
    per_language = {}
    for lang, (nll, count) in sorted(run.items()):
        if nll.shape != (vocab_size,) or count.shape != (vocab_size,):
            raise ValueError(f"{lang}: arrays do not match vocabulary size {vocab_size}")
        bin_nll = np.bincount(
            bin_of_token, weights=nll, minlength=num_bins
        ).astype(np.float64, copy=False)
        bin_count = np.zeros(num_bins, dtype=np.int64)
        np.add.at(bin_count, bin_of_token, count)
        per_language[lang] = (bin_nll, bin_count)
        pooled_nll += bin_nll
        pooled_count += bin_count
    return pooled_nll, pooled_count, per_language


def perplexity(nll, count):
    """Return finite PPL, or ``None`` for a bin with no eval targets."""
    if count <= 0:
        return None
    value = float(math.exp(float(nll) / int(count)))
    if not math.isfinite(value):
        raise ValueError(f"non-finite perplexity from nll={nll}, count={count}")
    return value


def analyze(runs, bin_of_token, labels, reference, frequency_counts):
    vocab_size = bin_of_token.size
    num_bins = len(labels)
    _validate_runs(runs, vocab_size)
    tables = {
        name: bin_table(run, bin_of_token, num_bins, vocab_size)
        for name, run in runs.items()
    }

    reference_counts = tables[next(iter(tables))][1]
    total_tokens = int(reference_counts.sum())
    if total_tokens <= 0:
        raise ValueError("by-token runs contain no counted evaluation targets")
    type_count = np.bincount(bin_of_token, minlength=num_bins).astype(np.int64)
    training_mass = np.bincount(
        bin_of_token, weights=frequency_counts, minlength=num_bins
    )
    total_training_mass = float(training_mass.sum())
    if total_training_mass <= 0:
        raise ValueError("training frequency counts have zero total mass")

    result = {
        "labels": labels,
        "type_count": type_count.tolist(),
        "training_mass_share": (training_mass / total_training_mass).tolist(),
        "eval_token_count": reference_counts.tolist(),
        "eval_token_share": (reference_counts / total_tokens).tolist(),
        "ppl": {},
        "per_language_ppl": {},
        "gap_vs_reference": {},
    }
    for name, (nll, count, per_language) in tables.items():
        result["ppl"][name] = [
            perplexity(nll[index], count[index]) for index in range(num_bins)
        ]
        result["ppl"][name].append(perplexity(nll.sum(), count.sum()))
        result["per_language_ppl"][name] = {
            lang: [
                perplexity(bin_nll[index], bin_count[index])
                for index in range(num_bins)
            ]
            for lang, (bin_nll, bin_count) in per_language.items()
        }

    if reference:
        reference_nll = tables[reference][0]
        for name, (nll, _, _) in tables.items():
            if name == reference:
                continue
            contributions = (nll - reference_nll) / total_tokens
            total = float(contributions.sum())
            result["gap_vs_reference"][name] = {
                "total_mean_log_ppl_gap": total,
                "per_bin_contribution": contributions.tolist(),
                "per_bin_share_of_gap": (
                    (contributions / total).tolist() if total != 0 else None
                ),
            }
    return result


def _format_ppl(value):
    return "n/a" if value is None else f"{value:.2f}"


def render(title, result, reference):
    lines = [f"### {title}", ""]
    header = ["bin", "types", "train%", "eval%"] + list(result["ppl"])
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for index, label in enumerate(result["labels"]):
        row = [
            label,
            str(result["type_count"][index]),
            f"{100 * result['training_mass_share'][index]:.4f}",
            f"{100 * result['eval_token_share'][index]:.4f}",
        ]
        row.extend(
            _format_ppl(result["ppl"][name][index]) for name in result["ppl"]
        )
        lines.append("| " + " | ".join(row) + " |")
    pooled = [
        "pooled",
        str(sum(result["type_count"])),
        "100.0000",
        "100.0000",
    ]
    pooled.extend(_format_ppl(result["ppl"][name][-1]) for name in result["ppl"])
    lines.append("| " + " | ".join(pooled) + " |")

    if reference and result["gap_vs_reference"]:
        lines.extend(["", f"Share of mean log-PPL gap vs `{reference}` by bin:", ""])
        for name, gap in result["gap_vs_reference"].items():
            shares = gap["per_bin_share_of_gap"]
            shown = (
                ", ".join(
                    f"{label}: {100 * share:.1f}%"
                    for label, share in zip(result["labels"], shares)
                )
                if shares is not None
                else "n/a"
            )
            lines.append(
                f"- {name}: total gap {gap['total_mean_log_ppl_gap']:+.4f} "
                f"nats -> {shown}"
            )
    return "\n".join(lines)


def write_merged_runs(output_dir, runs):
    """Write one combined by-token NPZ per logical run."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, run in runs.items():
        if not RUN_NAME_PATTERN.fullmatch(name):
            raise ValueError(
                f"run name {name!r} is unsafe for --write-merged-dir; use "
                "letters, digits, '.', '_' or '-'"
            )
        arrays = {}
        for lang, (nll, count) in sorted(run.items()):
            arrays[f"{lang}_nll"] = nll
            arrays[f"{lang}_cnt"] = count
        path = output_dir / f"{name}_eval_ppl_bytoken.npz"
        np.savez_compressed(path, **arrays)
        written[name] = str(path)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--counts", required=True, help="token frequency .npz")
    parser.add_argument("--counts-key", default="counts")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="name=path/to/eval_ppl_bytoken.npz; repeat a name for disjoint shards",
    )
    parser.add_argument("--reference", default=None, help="run name for gap decomposition")
    parser.add_argument(
        "--populations",
        default="2048,6144,24576,119168",
        help="level-set block sizes in frequency order",
    )
    parser.add_argument("--mass-thresholds", default="0.9,0.99")
    parser.add_argument("--langs", nargs="+", default=None)
    parser.add_argument("--output", default=None, help="write full JSON here")
    parser.add_argument(
        "--write-merged-dir",
        default=None,
        help="also write one merged by-token NPZ for each logical run",
    )
    args = parser.parse_args()

    with np.load(args.counts, allow_pickle=False) as archive:
        if args.counts_key not in archive.files:
            raise KeyError(f"{args.counts} has no array {args.counts_key!r}")
        counts = archive[args.counts_key].astype(np.float64, copy=True)
    order = frequency_rank_order(counts)

    shard_specs = {}
    for spec in args.run:
        name, separator, path = spec.partition("=")
        if not separator or not name or not path:
            raise ValueError(f"--run expects name=path, got {spec!r}")
        shard_specs.setdefault(name, []).append(path)
    runs = {
        name: merge_run_shards(
            name,
            [(path, load_run(path, args.langs)) for path in paths],
        )
        for name, paths in shard_specs.items()
    }
    if args.reference and args.reference not in runs:
        raise ValueError(f"--reference {args.reference!r} is not a loaded run")

    populations = [value for value in args.populations.split(",") if value]
    thresholds = [value for value in args.mass_thresholds.split(",") if value]
    level_bin_ids, level_labels = level_set_bins(order, populations)
    mass_bin_ids, mass_labels = mass_bins(order, counts, thresholds)

    report = {
        "metadata": {
            "counts": args.counts,
            "counts_key": args.counts_key,
            "run_shards": shard_specs,
            "reference": args.reference,
            "languages": sorted(next(iter(runs.values()))),
            "populations": [int(value) for value in populations],
            "mass_thresholds": [float(value) for value in thresholds],
        },
        "level_sets": analyze(
            runs, level_bin_ids, level_labels, args.reference, counts
        ),
        "mass_bins": analyze(
            runs, mass_bin_ids, mass_labels, args.reference, counts
        ),
    }
    print(
        render(
            "Level-set bins (tiered-rank blocks)",
            report["level_sets"],
            args.reference,
        )
    )
    print()
    print(
        render(
            "Cumulative-mass bins (head/torso/tail)",
            report["mass_bins"],
            args.reference,
        )
    )

    if args.write_merged_dir:
        report["metadata"]["merged_outputs"] = write_merged_runs(
            args.write_merged_dir, runs
        )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
        print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
