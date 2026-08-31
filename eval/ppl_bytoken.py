"""Per-token-id NLL accumulation for frequency-binned perplexity.

Uses the EXACT same sliding-window scheme as eval/ppl.py (same windows, same
stride, same -100 masking of context positions), but instead of keeping only
the scalar mean NLL it accumulates, per target token id:

    nll_sum[id] = sum of -log p(target) over all counted positions with that id
    count[id]   = number of counted positions with that id

Bins are formed OFFLINE from these arrays: for any set of token ids B,

    PPL_B = exp( sum(nll_sum[B]) / sum(count[B]) )

With B = the full vocab this reproduces the standard PPL exactly, which is
verified in-run: every chunk compares the per-position mean against the scalar
loss the model itself computes from the same forward pass (labels path), and
the summary reports the max gap seen.

Outputs (in --output-dir, default = checkpoint dir):
    eval_ppl_bytoken.npz           per lang: <lang>_nll (f64), <lang>_cnt (i64)
    eval_ppl_bytoken_summary.json  overall loss/ppl/tokens per lang + max
                                   verification gap (must be ~1e-3 or smaller)

Usage:
  python eval/ppl_bytoken.py --checkpoint path/to/ckpt \
      --eval-dir data/Qwen_Qwen3-0.6B/eval --tokenizer-name Qwen/Qwen3-0.6B --bf16
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer


def validate_tokenizer_ids(tokenizer, model_vocab_size):
    """Require every tokenizer id to fit the model's possibly padded vocab."""
    if model_vocab_size <= 0:
        raise ValueError("model vocabulary size must be positive")
    vocabulary = tokenizer.get_vocab()
    if not vocabulary:
        raise ValueError("tokenizer vocabulary is empty")
    token_ids = list(vocabulary.values())
    if any(not isinstance(token_id, int) for token_id in token_ids):
        raise ValueError("tokenizer vocabulary contains a non-integer token id")
    minimum = min(token_ids)
    maximum = max(token_ids)
    if minimum < 0 or maximum >= model_vocab_size:
        raise ValueError(
            "tokenizer id range does not fit model vocabulary: "
            f"[{minimum}, {maximum}] vs [0, {model_vocab_size})"
        )
    return {
        "tokenizer_size": len(tokenizer),
        "tokenizer_id_min": minimum,
        "tokenizer_id_max": maximum,
        "model_vocab_size": model_vocab_size,
        "unused_model_rows": model_vocab_size - len(set(token_ids)),
    }


@torch.no_grad()
def accumulate_bytoken(model, input_ids, max_length, stride, device, vocab_size):
    """Sliding-window pass mirroring eval/ppl.py's compute_perplexity, but
    accumulating per-token-id NLL sums and counts.

    Returns (nll_sum[vocab], count[vocab], stats dict).
    """
    seq_len = input_ids.size(1)

    nll_sum = torch.zeros(vocab_size, dtype=torch.float64)
    count = torch.zeros(vocab_size, dtype=torch.int64)
    max_gap = 0.0
    prev_end_loc = 0

    for begin_loc in tqdm(range(0, seq_len, stride), desc="  PPL-bytoken"):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc

        chunk_input_ids = input_ids[:, begin_loc:end_loc].to(device)
        target_ids = chunk_input_ids.clone()
        target_ids[:, :-trg_len] = -100

        # Labels path: the model computes the scalar mean loss itself — this is
        # byte-for-byte the quantity eval/ppl.py accumulates. We reuse the same
        # forward's logits for the per-token version and verify the two agree.
        outputs = model(chunk_input_ids, labels=target_ids)
        ref_loss = outputs.loss.item()

        # Same shift + float32 cast the HF loss path applies internally.
        shift_logits = outputs.logits.float()[:, :-1, :]
        shift_labels = target_ids[:, 1:]
        mask = shift_labels != -100

        flat_labels = shift_labels[mask]                      # [M]
        flat_logits = shift_logits[mask]                      # [M, V]
        nll = F.cross_entropy(flat_logits, flat_labels, reduction="none")  # [M] f32

        gap = abs(nll.mean().item() - ref_loss)
        max_gap = max(max_gap, gap)

        labels_cpu = flat_labels.cpu()
        nll_sum += torch.bincount(labels_cpu, weights=nll.double().cpu(),
                                  minlength=vocab_size)
        count += torch.bincount(labels_cpu, minlength=vocab_size)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    n_tokens = int(count.sum().item())
    avg_nll = float(nll_sum.sum().item()) / n_tokens
    stats = {
        "loss": avg_nll,
        "perplexity": math.exp(avg_nll),
        "num_tokens": n_tokens,
        "max_chunk_gap": max_gap,
    }
    return nll_sum.numpy(), count.numpy(), stats


def main():
    parser = argparse.ArgumentParser(description="Per-token-id PPL accumulation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--langs", nargs="+", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--max-verification-gap",
        type=float,
        default=1e-3,
        help="fail if any chunk's manual-vs-model loss gap exceeds this value",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.checkpoint
    stride = args.stride or args.block_size // 2
    if args.block_size <= 1:
        raise ValueError("--block-size must be greater than 1")
    if not 0 < stride <= args.block_size:
        raise ValueError("--stride must be positive and no larger than --block-size")
    if args.max_verification_gap < 0 or not math.isfinite(args.max_verification_gap):
        raise ValueError("--max-verification-gap must be finite and nonnegative")
    if args.langs and len(args.langs) != len(set(args.langs)):
        raise ValueError("--langs must not contain duplicates")

    dtype = torch.bfloat16 if args.bf16 else None
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name or args.checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    from eval.eval_checkpoint import load_model
    print("=" * 60)
    print(f"  Checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, args.device, dtype=dtype)
    vocab_size = model.config.vocab_size
    vocab_info = validate_tokenizer_ids(tokenizer, vocab_size)
    print(
        f"  Vocab: tokenizer={vocab_info['tokenizer_size']} "
        f"id_range=[{vocab_info['tokenizer_id_min']}, "
        f"{vocab_info['tokenizer_id_max']}] model={vocab_size} "
        f"unused_model_rows={vocab_info['unused_model_rows']}"
    )
    print("=" * 60)

    langs = args.langs or sorted(
        d for d in os.listdir(args.eval_dir)
        if os.path.isdir(os.path.join(args.eval_dir, d)))

    arrays = {}
    summary = {}
    for lang in langs:
        ds = load_from_disk(os.path.join(args.eval_dir, lang))
        encodings = tokenizer("\n\n".join(ds["text"]), return_tensors="pt",
                              add_special_tokens=False)
        input_ids = encodings.input_ids
        if input_ids.size(1) < args.block_size:
            raise ValueError(
                f"{lang}: only {input_ids.size(1)} tokens, fewer than "
                f"block size {args.block_size}"
            )
        print(f"  [{lang}] {input_ids.size(1):,} tokens")
        nll_sum, count, stats = accumulate_bytoken(
            model, input_ids, args.block_size, stride, args.device, vocab_size)
        arrays[f"{lang}_nll"] = nll_sum
        arrays[f"{lang}_cnt"] = count
        summary[lang] = stats
        print(f"  [{lang}] loss={stats['loss']:.4f}  ppl={stats['perplexity']:.2f}  "
              f"max_chunk_gap={stats['max_chunk_gap']:.2e}")

    os.makedirs(args.output_dir, exist_ok=True)
    npz_path = os.path.join(args.output_dir, "eval_ppl_bytoken.npz")
    np.savez_compressed(npz_path, **arrays)
    with open(os.path.join(args.output_dir, "eval_ppl_bytoken_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, allow_nan=False)

    worst = max((s["max_chunk_gap"] for s in summary.values()), default=0.0)
    if not summary:
        raise RuntimeError("no languages were evaluated")
    print(f"\n  Saved {npz_path}")
    print(f"  Max verification gap across all chunks: {worst:.2e} "
          f"({'OK' if worst <= args.max_verification_gap else 'INVALID'})")
    if worst > args.max_verification_gap:
        raise RuntimeError(
            f"max verification gap {worst:.6g} exceeds "
            f"limit {args.max_verification_gap:.6g}"
        )


if __name__ == "__main__":
    main()
