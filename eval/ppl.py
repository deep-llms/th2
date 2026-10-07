"""Core perplexity evaluation using sliding-window strategy.

Following HuggingFace's recommended approach:
https://huggingface.co/docs/transformers/en/perplexity
"""

import math

import torch
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from deep_kv.packing import document_end_id


@torch.no_grad()
def compute_perplexity(model, input_ids, max_length, stride, device):
    if not 1 <= stride < max_length:
        raise ValueError('Use 1 <= stride < max_length so window-boundary targets are scored')
    seq_len = input_ids.size(1)

    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0

    for begin_loc in range(0, seq_len, stride):
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

    if not n_tokens:
        raise ValueError('No next-token targets to evaluate')
    avg_nll = nll_sum / n_tokens
    perplexity = math.exp(avg_nll)
    return {"perplexity": perplexity, "loss": avg_nll, "num_tokens": n_tokens}


def eval_ppl(model, tokenizer, eval_dir, block_size=2048, stride=None, device="cuda", langs=None):
    if stride is None:
        stride = block_size // 2

    if not 1 <= stride < block_size or block_size > model.config.max_position_embeddings:
        raise ValueError('Invalid perplexity stride/window or window exceeds model context limit')
    root = Path(eval_dir)
    direct = (root / 'state.json').is_file() or any(root.glob('shard_*')) or any(root.glob('*.parquet'))
    if direct:
        if langs is not None:
            raise ValueError('--langs requires a parent directory containing language folders')
        paths = {'dataset': root}
    else:
        langs = langs or sorted(p.name for p in root.iterdir() if p.is_dir())
        paths = {lang: root / lang for lang in langs}
    end_id = document_end_id(tokenizer)

    results = {}
    for lang, lang_dir in paths.items():
        parquet = sorted(str(p) for p in lang_dir.glob('*.parquet'))
        if parquet:
            ds = load_dataset('parquet', data_files=parquet, split='train')
        else:
            from train import load_text
            ds = load_text(str(lang_dir))
        nll_sum, tokens, documents = 0., 0, 0
        for row in tqdm(ds, desc=f'  {lang} documents'):
            ids = tokenizer.encode(row['text'], add_special_tokens=False) + [end_id]
            if len(ids) < 2:
                continue
            r = compute_perplexity(model, torch.tensor([ids]), block_size, stride, device)
            nll_sum += r['loss'] * r['num_tokens']
            tokens += r['num_tokens']
            documents += 1
        if not tokens:
            raise ValueError(f'No scored documents in {lang_dir}')
        loss = nll_sum / tokens
        r = dict(perplexity=math.exp(loss), loss=loss, num_tokens=tokens, documents=documents,
                 policy='independent_documents_with_eos_sliding_window_v1', block_size=block_size, stride=stride)
        results[lang] = r
        print(f"  [{lang}] loss={r['loss']:.4f}  ppl={r['perplexity']:.2f}")

    if not results:
        raise ValueError('No evaluation data selected')
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
