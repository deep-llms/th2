"""Anchor-usage distribution analysis for compositional checkpoints.

Answers: are anchors used equally, or do a few carry most of the load?

For static-routing arms (ant, original_ant) usage is EXACT over the full
vocabulary: theta depends only on the token id, so one pass over all
vocab ids gives every type's anchor selection. Loads are reported two ways:
  - type-weighted: every vocab type counts once
  - freq-weighted: types weighted by training-corpus counts (token_freq.npz),
    i.e. the load anchors actually carry under the real token distribution.

For context-routing arms (v2, isolation_control) usage is measured
empirically over eval text (each position's theta), which is inherently
occurrence-weighted.

Loads use |theta| as the weight (theta is non-negative for these arms
anyway). Reported metrics per weighting:
  dead        anchors with zero total load
  gini        inequality of the load distribution (0=equal, 1=one anchor)
  eff_1/HHI   effective #anchors = 1/sum(p^2)
  eff_exp(H)  effective #anchors = exp(entropy)
  topX%_share fraction of total load carried by the top X% anchors

Saves <output-prefix>.npz (per-anchor load arrays) and <output-prefix>.json
(metrics), and prints the metrics.

Usage (static arm):
  python scripts/analyze_anchor_usage.py --checkpoint <ckpt-dir> \
      --freq-npz <token_freq.npz> --output-prefix <path>
Usage (context arm):
  python scripts/analyze_anchor_usage.py --checkpoint <ckpt-dir> \
      --eval-dir <eval-dir> --tokenizer-name Qwen/Qwen3-0.6B \
      --max-tokens-per-lang 500000 --output-prefix <path>
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch


def build_embed(checkpoint_dir, device):
    """Build just the embedding module (not the full LM) from a checkpoint."""
    from compositional.loading import (_build_arm_from_config,
                                       _find_config_path,
                                       _infer_comp_config_from_state)
    from transformers import AutoConfig

    state = torch.load(os.path.join(checkpoint_dir, "embedding.pt"),
                       map_location="cpu", weights_only=True)
    config_path = _find_config_path(checkpoint_dir)
    if config_path is not None:
        with open(config_path) as f:
            comp_config = json.load(f)["compositional"]
    else:
        comp_config = _infer_comp_config_from_state(state)
        print(f"No train_config.json — inferred: {comp_config}")

    hf_config = AutoConfig.from_pretrained(checkpoint_dir)
    embed = _build_arm_from_config(comp_config, hf_config.vocab_size,
                                   hf_config.hidden_size)
    embed.load_state_dict(state)
    embed.to(device).eval()
    return embed, comp_config, hf_config.vocab_size


def metrics_from_load(load):
    """Distribution metrics for a non-negative per-anchor load vector."""
    load = np.asarray(load, dtype=np.float64)
    K = load.size
    total = load.sum()
    out = {"K": int(K), "dead": int((load == 0).sum())}
    if total <= 0:
        return out
    p = load / total
    srt = np.sort(load)
    cum = np.cumsum(srt)
    # Gini via the standard sorted formula
    out["gini"] = float((K + 1 - 2 * (cum / cum[-1]).sum()) / K)
    out["eff_inv_hhi"] = float(1.0 / (p ** 2).sum())
    nz = p[p > 0]
    out["eff_exp_entropy"] = float(np.exp(-(nz * np.log(nz)).sum()))
    desc = np.sort(load)[::-1]
    for frac in (0.01, 0.05, 0.10, 0.50):
        k = max(1, int(round(K * frac)))
        out[f"top{int(frac*100)}%_share"] = float(desc[:k].sum() / total)
    return out


@torch.no_grad()
def run_static(embed, vocab_size, device, freq, batch=8192):
    K = None
    arrs = {}
    for start in range(0, vocab_size, batch):
        ids_t = torch.arange(start, min(start + batch, vocab_size),
                             device=device).unsqueeze(0)
        _, theta = embed(ids_t)
        theta = theta.squeeze(0).float().cpu().numpy().astype(np.float64)  # [B,K]
        active = theta > 0
        if K is None:
            K = theta.shape[-1]
            for k in ("sel_type", "w_type", "sel_freq", "w_freq"):
                arrs[k] = np.zeros(K, dtype=np.float64)
        arrs["sel_type"] += active.sum(0)
        arrs["w_type"] += theta.sum(0)
        if freq is not None:
            f = freq[start:start + theta.shape[0]][:, None]
            arrs["sel_freq"] += (active * f).sum(0)
            arrs["w_freq"] += (theta * f).sum(0)
    return arrs


@torch.no_grad()
def run_context(embed, eval_dir, tokenizer_name, langs, max_tokens, device,
                block=2048):
    from datasets import load_from_disk
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    K = None
    arrs = {}
    n_positions = 0
    for lang in langs:
        ds = load_from_disk(os.path.join(eval_dir, lang))
        ids = tok("\n\n".join(ds["text"]), return_tensors="pt",
                  add_special_tokens=False).input_ids[0][:max_tokens]
        for start in range(0, ids.numel() - block + 1, block):
            chunk = ids[start:start + block].unsqueeze(0).to(device)
            _, theta = embed(chunk)
            theta = theta.squeeze(0).float().cpu().numpy().astype(np.float64)
            if K is None:
                K = theta.shape[-1]
                arrs["sel_occ"] = np.zeros(K, dtype=np.float64)
                arrs["w_occ"] = np.zeros(K, dtype=np.float64)
            arrs["sel_occ"] += (theta > 0).sum(0)
            arrs["w_occ"] += theta.sum(0)
            n_positions += theta.shape[0]
    arrs["n_positions"] = np.array([n_positions], dtype=np.int64)
    print(f"context usage measured over {n_positions:,} positions")
    return arrs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--freq-npz", default=None,
                        help="token_freq.npz for freq-weighted loads (static arms)")
    parser.add_argument("--eval-dir", default=None,
                        help="eval data dir (context arms)")
    parser.add_argument("--tokenizer-name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--langs", nargs="+",
                        default=["en", "vi", "zh", "ru", "de", "ar"])
    parser.add_argument("--max-tokens-per-lang", type=int, default=500000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    embed, comp_config, vocab_size = build_embed(args.checkpoint, args.device)
    arm = comp_config["arm"]
    print(f"arm={arm}  vocab={vocab_size}")

    static = arm in ("ant", "original_ant", "v0", "v1")
    if static:
        freq = None
        if args.freq_npz:
            freq = np.load(args.freq_npz)["counts"].astype(np.float64)
            assert freq.size == vocab_size, (freq.size, vocab_size)
        arrs = run_static(embed, vocab_size, args.device, freq)
    else:
        if args.eval_dir is None:
            raise SystemExit("context arm requires --eval-dir")
        arrs = run_context(embed, args.eval_dir, args.tokenizer_name,
                           args.langs, args.max_tokens_per_lang, args.device)

    report = {"arm": arm, "checkpoint": args.checkpoint}
    for key, arr in arrs.items():
        if key == "n_positions":
            report["n_positions"] = int(arr[0])
            continue
        if arr.sum() == 0 and key.endswith("_freq"):
            continue  # freq weighting not requested
        report[key] = metrics_from_load(arr)

    np.savez_compressed(args.output_prefix + ".npz", **arrs)
    with open(args.output_prefix + ".json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
