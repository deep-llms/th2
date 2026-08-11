"""Driver for the cross-lingual transfer test battery.

Loads each checkpoint ONCE and runs all requested tests on it, then writes
per-model JSONs plus a merged markdown comparison (deltas vs the designated
baseline model).

Tests (all architecture-independent — plain forward passes only):
  t6       BLI translation retrieval via CSLS (word-level, embedding layer)
  t8       MEXA parallel-sentence alignment on FLORES-200 (all layers)
  probe_b  translation-vs-random cosine gap (word-level, embedding layer)
  t5       per-layer in-context similarity sweep
  t7       generative code-switch log-prob margins
  t1       XNLI transfer probe (English-trained classifier, 6-language test)

Usage (labels derived from the checkpoint's parent dir name):
  python crosslingual/run_crosslingual.py \
      --checkpoints /path/baseline/checkpoint-10000 /path/ant_ours/checkpoint-10000 \
      --baseline-label baseline \
      --tests t6 t8 probe_b \
      --eval-dir /opt/dlami/nvme/sparse_emb_data/Qwen_Qwen3-0.6B/eval \
      --output-dir /opt/dlami/nvme/sparse_emb_outputs/crosslingual
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from crosslingual.utils import (DISTANT_PAIRS, NON_EN_LANGS, RELATED_PAIRS,
                                filter_loanwords_per_pair, get_tokenizer,
                                load_model, load_translations)

ALL_TESTS = ["t6", "t8", "probe_b", "t5", "t7", "t1"]


def label_for(ckpt_path):
    """original_ant/checkpoint-10000 -> original_ant"""
    return os.path.basename(os.path.dirname(os.path.normpath(ckpt_path)))


def run_tests_for_model(ckpt, tests, tokenizer, translations, args, device):
    model = load_model(ckpt, device, dtype=torch.bfloat16)
    results = {"checkpoint": ckpt}

    if "t6" in tests:
        from crosslingual.t6_bli import run_t6
        print("  t6 (BLI/CSLS)...")
        results["t6"] = run_t6(model, tokenizer, translations, device,
                               layer_idx=None, csls_k=args.csls_k)

    if "probe_b" in tests:
        from crosslingual.probe_b import run_probe_b
        print("  probe_b (cosine gap)...")
        results["probe_b"] = run_probe_b(model, tokenizer, translations,
                                         device, layer_idx=None)

    if "t8" in tests:
        from crosslingual.t8_mexa import load_flores_all, run_t8
        print("  t8 (MEXA)...")
        flores = load_flores_all(args.flores_dir, args.mexa_sentences)
        if "en" not in flores:
            results["t8"] = {"error": f"FLORES data not found "
                                      f"(dir={args.flores_dir})"}
        else:
            results["t8"] = run_t8(model, tokenizer, flores, device)

    if "t5" in tests:
        from crosslingual.t5_layer_sweep import run_t5
        print("  t5 (layer sweep)...")
        layers = [None] + args.t5_layers
        results["t5"] = run_t5(model, tokenizer, translations, args.eval_dir,
                               layers, device)

    if "t7" in tests:
        from crosslingual.t7_codeswitch import run_t7
        print("  t7 (code-switch)...")
        results["t7"] = run_t7(model, tokenizer, translations, device)

    if "t1" in tests:
        from crosslingual.t1_xnli import run_t1
        print("  t1 (XNLI probe)...")
        results["t1"] = run_t1(model, tokenizer, device,
                               layers=args.t1_layers,
                               max_train=args.t1_max_train)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return results


def write_report(all_results, baseline_label, tests, path):
    lines = ["# Cross-lingual transfer test results\n"]
    labels = list(all_results.keys())
    base = all_results.get(baseline_label, {})

    if "t6" in tests:
        lines.append("\n## T6 — BLI retrieval (P@1, CSLS, embedding layer)\n")
        header = "| pair | " + " | ".join(labels) + " |"
        lines += [header, "|" + "---|" * (len(labels) + 1)]
        for pair in [f"en-{l}" for l in NON_EN_LANGS]:
            row = [pair]
            for lb in labels:
                r = all_results[lb].get("t6", {}).get("per_lang", {}).get(pair)
                row.append(f"{r['p_at_1']:.4f} (n={r['n_pairs']})" if r else "-")
            lines.append("| " + " | ".join(row) + " |")
        for key in ("mean", "related_mean", "distant_mean"):
            row = [key]
            for lb in labels:
                s = all_results[lb].get("t6", {}).get("summary", {})
                v = s.get(key)
                d = ""
                if v is not None and lb != baseline_label and base:
                    bv = base.get("t6", {}).get("summary", {}).get(key)
                    if bv is not None:
                        d = f" ({v - bv:+.4f})"
                row.append(f"{v:.4f}{d}" if v is not None else "-")
            lines.append("| " + " | ".join(row) + " |")
        lines.append(f"\nrelated={RELATED_PAIRS}, distant={DISTANT_PAIRS}; "
                     "en-vi is small-n — do not over-read.")

    if "t8" in tests:
        lines.append("\n## T8 — MEXA (FLORES-200)\n")
        header = "| pair | " + " | ".join(
            f"{lb} mean / best(L)" for lb in labels) + " |"
        lines += [header, "|" + "---|" * (len(labels) + 1)]
        for pair in [f"en-{l}" for l in NON_EN_LANGS]:
            row = [pair]
            for lb in labels:
                r = all_results[lb].get("t8", {}).get(pair)
                row.append(f"{r['mean_over_layers']:.4f} / "
                           f"{r['best_score']:.4f} (L{r['best_layer']})"
                           if r else "-")
            lines.append("| " + " | ".join(row) + " |")
        row = ["avg"]
        for lb in labels:
            s = all_results[lb].get("t8", {}).get("summary", {})
            v = s.get("avg_mean_over_layers")
            d = ""
            if v is not None and lb != baseline_label and base:
                bv = base.get("t8", {}).get("summary", {}).get(
                    "avg_mean_over_layers")
                if bv is not None:
                    d = f" ({v - bv:+.4f})"
            row.append(f"{v:.4f}{d}" if v is not None else "-")
        lines.append("| " + " | ".join(row) + " |")

    if "probe_b" in tests:
        lines.append("\n## Probe B — cosine gap (embedding layer)\n")
        lines.append("| model | cos_trans | cos_rand | gap | p | n |")
        lines.append("|---|---|---|---|---|---|")
        for lb in labels:
            r = all_results[lb].get("probe_b")
            if r:
                p = f"{r['pvalue']:.2e}" if r["pvalue"] is not None else "-"
                lines.append(
                    f"| {lb} | {r['cos_translation']:.4f} | "
                    f"{r['cos_random']:.4f} | {r['cos_gap']:.4f} | {p} | "
                    f"{r['n_translation']} |")

    if "t5" in tests:
        lines.append("\n## T5 — layer sweep (gap by layer)\n")
        for lb in labels:
            r = all_results[lb].get("t5")
            if r:
                lines.append(f"- **{lb}**: peak {r['peak_layer']} "
                             f"gap={r['peak_gap']:.4f} "
                             f"(words found: {r['n_words_found']})")

    if "t7" in tests:
        lines.append("\n## T7 — code-switch margins\n")
        header = "| pair | " + " | ".join(labels) + " |"
        lines += [header, "|" + "---|" * (len(labels) + 1)]
        for pair in [f"en-{l}" for l in NON_EN_LANGS]:
            row = [pair]
            for lb in labels:
                r = all_results[lb].get("t7", {}).get(pair)
                row.append(f"{r['mean_margin']:.3f}" if r else "-")
            lines.append("| " + " | ".join(row) + " |")

    if "t1" in tests:
        lines.append("\n## T1 — XNLI transfer probe (acc, mean over seeds)\n")
        for lb in labels:
            r = all_results[lb].get("t1")
            if not r:
                continue
            for layer_name, ld in sorted(r.items()):
                accs = " ".join(f"{l}={a:.3f}"
                                for l, a in ld["mean"]["per_lang"].items())
                lines.append(f"- **{lb}** {layer_name}: {accs}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nReport written to {path}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Cross-lingual test battery")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--baseline-label", default="baseline",
                        help="Label whose numbers are the delta reference")
    parser.add_argument("--tests", nargs="+", default=["t6", "t8", "probe_b"],
                        choices=ALL_TESTS)
    parser.add_argument("--tokenizer-name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--translations", default=None)
    parser.add_argument("--eval-dir", default=None,
                        help="Per-language eval corpus (required for t5)")
    parser.add_argument("--flores-dir", default=None,
                        help="FLORES-200 text files (required for t8)")
    parser.add_argument("--csls-k", type=int, default=10)
    parser.add_argument("--mexa-sentences", type=int, default=100)
    parser.add_argument("--t5-layers", nargs="+", type=int,
                        default=[0, 4, 9, 14, 19, 27])
    parser.add_argument("--t1-layers", nargs="+", type=int, default=[14, 27])
    parser.add_argument("--t1-max-train", type=int, default=20000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default="temp/crosslingual")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if "t5" in args.tests and not args.eval_dir:
        parser.error("t5 requires --eval-dir")

    tokenizer = get_tokenizer(args.tokenizer_name)
    translations = filter_loanwords_per_pair(
        load_translations(args.translations))
    print(f"Translation tuples (loanword-filtered): {len(translations)}")

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}
    for ckpt in args.checkpoints:
        lb = label_for(ckpt)
        print(f"\n=== {lb}: {ckpt}")
        results = run_tests_for_model(ckpt, args.tests, tokenizer,
                                      translations, args, device)
        all_results[lb] = results
        with open(os.path.join(args.output_dir, f"{lb}.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)

    report = write_report(all_results, args.baseline_label, args.tests,
                          os.path.join(args.output_dir, "report.md"))
    print(report)


if __name__ == "__main__":
    main()
