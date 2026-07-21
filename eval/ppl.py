"""Core perplexity evaluation using sliding-window strategy.

Following HuggingFace's recommended approach:
https://huggingface.co/docs/transformers/en/perplexity
"""

import math
import os

import torch
from datasets import load_from_disk
from tqdm import tqdm


@torch.no_grad()
def compute_perplexity(model, input_ids, max_length, stride, device):
    seq_len = input_ids.size(1)

    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0

    for begin_loc in tqdm(range(0, seq_len, stride), desc="  PPL"):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc

        chunk_input_ids = input_ids[:, begin_loc:end_loc].to(device)
        target_ids = chunk_input_ids.clone()
        target_ids[:, :-trg_len] = -100

        outputs = model(chunk_input_ids, labels=target_ids)

        num_loss_tokens = (target_ids[:, 1:] != -100).sum().item()
        nll_sum += outputs.loss.item() * num_loss_tokens
        n_tokens += num_loss_tokens

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    avg_nll = nll_sum / n_tokens
    perplexity = math.exp(avg_nll)
    return {"perplexity": perplexity, "loss": avg_nll, "num_tokens": n_tokens}


def eval_ppl(model, tokenizer, eval_dir, block_size=2048, stride=None, device="cuda", langs=None):
    if stride is None:
        stride = block_size // 2

    if langs is None:
        langs = sorted([d for d in os.listdir(eval_dir) if os.path.isdir(os.path.join(eval_dir, d))])

    results = {}
    for lang in langs:
        lang_dir = os.path.join(eval_dir, lang)
        if not os.path.isdir(lang_dir):
            print(f"  [{lang}] Not found, skipping.")
            continue

        ds = load_from_disk(lang_dir)
        texts = ds["text"]
        encodings = tokenizer("\n\n".join(texts), return_tensors="pt", add_special_tokens=False)
        input_ids = encodings.input_ids
        seq_len = input_ids.size(1)

        if seq_len < block_size:
            print(f"  [{lang}] Only {seq_len} tokens (< block_size {block_size}), skipping.")
            continue

        print(f"  [{lang}] {seq_len:,} tokens")
        r = compute_perplexity(model, input_ids, block_size, stride, device)
        results[lang] = r
        print(f"  [{lang}] loss={r['loss']:.4f}  ppl={r['perplexity']:.2f}")

    return results


def print_ppl_results(results):
    if not results:
        return
    print("\n  " + "=" * 52)
    print(f"  {'Language':<10} {'Loss':>10} {'PPL':>12} {'Tokens':>14}")
    print("  " + "-" * 52)
    for lang, r in sorted(results.items()):
        print(f"  {lang:<10} {r['loss']:>10.4f} {r['perplexity']:>12.2f} {r['num_tokens']:>14,}")

    total_tokens = sum(r["num_tokens"] for r in results.values())
    avg_loss = sum(r["loss"] * r["num_tokens"] for r in results.values()) / total_tokens
    avg_ppl = math.exp(avg_loss)
    print("  " + "-" * 52)
    print(f"  {'Overall':<10} {avg_loss:>10.4f} {avg_ppl:>12.2f} {total_tokens:>14,}")
    print("  " + "=" * 52)
