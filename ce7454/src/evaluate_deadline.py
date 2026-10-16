"""Fixed endpoint evaluation using only the original train-derived holdout.

Reuses the nine existing processing functions and original chunk RNG streams.
No development/test images, estimated labels, ensembles or TTA.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from dataset import load_meta, stratified_split
from make_protocols import PROTOCOLS, apply
from models import build_model, count_trainable
from train import normalize, set_norm


@torch.no_grad()
def score(model, images, truth, batch=32):
    groups = defaultdict(list)
    for i, im in enumerate(images):
        groups[im.shape].append(i)
    logits = np.empty((len(images), 3), dtype=np.float32)
    for indices in groups.values():
        for start in range(0, len(indices), batch):
            ix = indices[start:start + batch]
            x = torch.from_numpy(np.stack([images[i] for i in ix])).permute(0, 3, 1, 2)
            x = normalize(x.cuda()).contiguous(memory_format=torch.channels_last)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits[ix] = model(x).float().cpu().numpy()
    if not np.isfinite(logits).all():
        raise ValueError('Nonfinite evaluation logits')
    pred = logits.argmax(1)
    loss = torch.nn.functional.cross_entropy(torch.from_numpy(logits), torch.from_numpy(truth)).item()
    conf = np.bincount(truth * 3 + pred, minlength=9).reshape(3, 3)
    return logits, {'n': len(truth), 'accuracy': float((pred == truth).mean()),
                    'loss': loss, 'confusion': conf.tolist()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if any(ck['args'].get(k) != v for k, v in {'train_split': 'train', 'val_frac': .1, 'split_seed': 42}.items()):
        raise ValueError('Wrong evaluation partition')
    model = build_model(ck['args']['model']).cuda().eval().to(memory_format=torch.channels_last)
    model.load_state_dict(ck['model'], strict=True)
    if count_trainable(model) > 12_300_000:
        raise ValueError('Parameter cap exceeded')
    set_norm(ck['args']['norm'])
    labels = np.asarray(load_meta(args.cache, 'train')['labels'], dtype=np.int64)
    _, val = stratified_split(labels, .1, 42)
    arr = np.load(args.cache / 'train_images.npy', mmap_mode='r')
    metrics = {}
    for pi, name in enumerate(['clean', *PROTOCOLS]):
        images = []
        for chunk, indices in enumerate(np.array_split(val, 8)):
            # pi=1 for first processing protocol, matching make_protocols.py.
            rng = np.random.RandomState(1000 * pi + chunk)
            images.extend(np.array(arr[i]) if name == 'clean' else apply(name, np.array(arr[i]), rng)
                          for i in indices)
        logits, metrics[name] = score(model, images, labels[val])
        np.save(args.output / f'{name}_logits.npy', logits)
        print(name, metrics[name], flush=True)
    np.save(args.output / 'val_indices.npy', val)
    with args.checkpoint.open('rb') as f:
        sha = hashlib.file_digest(f, 'sha256').hexdigest()
    result = {'status': 'passed', 'checkpoint_sha256': sha, 'model': ck['args']['model'],
              'epoch': ck['epoch'], 'metrics': metrics,
              'processed_mean_accuracy': float(np.mean([metrics[n]['accuracy'] for n in PROTOCOLS])),
              'dev_test_used': False, 'tta': False}
    (args.output / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
