"""Per-language exact PPL, without packing across language boundaries."""
import math
import torch
from torch.utils.data import DataLoader
from capacity_allocation.data import load_text_data, preprocess_text
from eval.runtime import languages


def evaluate(model, tokenizer, data_dir, selected_languages='en', *, device='cpu',
             precision='fp32', batch_size=1, block_size=2048, workers=160,
             map_batch_size=1000, cache_dir=None):
    results = {}
    if batch_size <= 0:
        raise ValueError('Positive PPL batch size required')
    model.eval()
    for lang in languages(selected_languages):
        data = preprocess_text(load_text_data(data_dir, (lang,)), tokenizer,
            block_size=block_size, num_proc=workers, batch_size=map_batch_size, cache_dir=cache_dir)
        total_nll, count = 0., 0
        with torch.no_grad():
            for batch in DataLoader(data, batch_size=batch_size, shuffle=False):
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.autocast(device, dtype=torch.bfloat16, enabled=precision == 'bf16'):
                    loss = model(**batch, use_cache=False).loss.item()
                targets = batch['labels'][:, 1:].ne(-100).sum().item()
                if not math.isfinite(loss) or not targets:
                    raise RuntimeError('Nonfinite PPL loss or empty targets')
                total_nll += loss * targets
                count += targets
        if count != len(data) * (block_size - 1):
            raise RuntimeError('PPL coverage mismatch')
        nll = total_nll/count
        results[lang] = dict(nll=nll, ppl=math.exp(nll) if nll < 700 else None,
            scored_targets=count, processed_tokens=len(data)*block_size,
            data_fingerprint=data._fingerprint)
    total = sum(row['scored_targets'] for row in results.values())
    nll = sum(row['nll']*row['scored_targets'] for row in results.values()) / total
    return dict(by_language=results, token_weighted=dict(nll=nll,
        ppl=math.exp(nll) if nll < 700 else None, scored_targets=total))
