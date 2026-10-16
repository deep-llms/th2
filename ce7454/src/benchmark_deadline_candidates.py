"""Bounded CUDA readiness checks; generated checkpoints are not scientific runs."""
import argparse
import json
from itertools import islice
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from dataset import CachedImageDataset, load_meta, stratified_split
from models import build_model, count_trainable
from train import normalize, seed_all, set_norm


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--cache', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--crop', type=int, default=256)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    seed_all(0)
    set_norm('imagenet')
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    labels = np.asarray(load_meta(args.cache, 'train')['labels'])
    fit, val = stratified_split(labels, 0.1, 42)
    # Every image used for fitting is in the original training partition.
    ds = CachedImageDataset(args.cache, 'train', fit, labels[fit], train=True,
        crop=args.crop, rrc_size=args.crop, crop_upscale=args.crop == 512,
        flips='rot90_hflip', views=2, social_p=1.0,
        social_region=True, social_cf_interp=True, rrc_p=0.2,
        grip_blur_p=0.1, grip_blur_smax=3.0, sharpen_p=0.5)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True,
        generator=torch.Generator().manual_seed(42))
    model = build_model(args.model).cuda().to(memory_format=torch.channels_last)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    times, end_to_end_times, losses = [], [], []
    torch.cuda.reset_peak_memory_stats()
    last = time.perf_counter()
    for step, (x, y) in enumerate(islice(loader, args.steps)):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        x = normalize(x.cuda(non_blocking=True)).flatten(0, 1).contiguous(memory_format=torch.channels_last)
        y = y.cuda(non_blocking=True).repeat_interleave(2)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = F.cross_entropy(model(x).float(), y)
        loss.backward()
        grads = [p.grad for p in model.parameters()]
        if any(g is None for g in grads):
            raise ValueError('Unused parameters')
        finite = torch.stack([torch.isfinite(loss), *[torch.isfinite(g).all() for g in grads]]).all()
        if not finite.item():
            raise ValueError('Nonfinite loss/gradient or unused parameters')
        optimizer.step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        end_to_end_times.append(time.perf_counter() - last)
        last = time.perf_counter()
        losses.append(loss.item())
    peak = torch.cuda.max_memory_allocated()
    # Independent FP32 reference at identical weights with frozen BN/dropout.
    model.eval()
    if len(times) != args.steps or x.ndim != 4 or y.ndim != 1:
        raise ValueError('Incomplete benchmark or invalid last processed batch')
    def numerical_check(sample, target):
        model.zero_grad(set_to_none=True)
        ref = F.cross_entropy(model(sample).float(), target)
        ref.backward()
        ref_grad = torch.cat([p.grad.detach().float().reshape(-1) for p in model.parameters()])
        model.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            amp = F.cross_entropy(model(sample).float(), target)
        amp.backward()
        amp_grad = torch.cat([p.grad.detach().float().reshape(-1) for p in model.parameters()])
        rel = ((amp_grad - ref_grad).norm() / ref_grad.norm().clamp_min(1e-12)).item()
        cosine = F.cosine_similarity(ref_grad, amp_grad, dim=0).item()
        return {'shape': list(sample.shape), 'fp32_loss': ref.item(), 'bf16_loss': amp.item(),
                'gradient_relative_l2': rel, 'gradient_cosine': cosine,
                'passed': abs(ref.item()-amp.item()) <= .05 and rel <= .15 and cosine >= .98}
    # Retain the failed tiny-crop diagnostic instead of hiding it. Gate on TWO
    # actual-resolution groups, with unchanged tolerances. A 64px crop is not
    # either training resolution and has very different pooled feature support.
    numerics = {'tiny64_diagnostic': numerical_check(x[:2, :, :64, :64].detach(), y[:2]),
                'native_first8': numerical_check(x[:8].detach(), y[:8]),
                'native_last8': numerical_check(x[-8:].detach(), y[-8:])}
    (args.output / 'numerics.json').write_text(json.dumps(numerics, indent=2) + '\n')
    print('NUMERICS', json.dumps(numerics), flush=True)
    if not all(numerics[key]['passed'] for key in ('native_first8', 'native_last8')):
        raise ValueError('Native-resolution FP32/BF16 numerical gate failed; see numerics.json')
    ref = numerics['native_first8']
    # Use a fit image, not dev/test; check full-size inference and exact reload.
    full_ds = CachedImageDataset(args.cache, 'train', fit[:1], labels[fit[:1]])
    full = normalize(full_ds[0][0].unsqueeze(0).cuda()).contiguous(memory_format=torch.channels_last)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        before = model(full).float()
    checkpoint = args.output / 'smoke.pt'
    torch.save({'model': model.state_dict(), 'args': {'model': args.model}, 'smoke_only': True}, checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location='cuda', weights_only=True)['model'])
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        after = model(full).float()
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    if not torch.isfinite(after).all():
        raise ValueError('Nonfinite full-resolution output')
    # Full-validation latency, at the actual evaluation batch size (32).
    val_input = full.expand(32, -1, -1, -1).contiguous(memory_format=torch.channels_last)
    eval_times = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for _ in range(4):
            torch.cuda.synchronize(); start = time.perf_counter()
            model(val_input)
            torch.cuda.synchronize(); eval_times.append(time.perf_counter() - start)
    result = {'model': args.model, 'parameters': count_trainable(model),
        'steps': len(times), 'source_batch_size': args.batch_size, 'views': 2,
        'crop': args.crop,
        'median_step_seconds_after_warmup': float(np.median(times[5:])),
        'mean_end_to_end_step_seconds_after_warmup': float(np.mean(end_to_end_times[5:])),
        'full512_eval_batch32_seconds': float(np.median(eval_times[1:])),
        'gpu_allocated_peak_gib': peak / 2**30, 'losses': losses,
        'fp32_loss': ref['fp32_loss'], 'bf16_loss': ref['bf16_loss'],
        'gradient_relative_l2': ref['gradient_relative_l2'],
        'gradient_cosine': ref['gradient_cosine'], 'numerics': numerics, 'full512_reload_exact': True,
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'note': 'Compute timing includes finite-gradient checks; excludes loader wait. Not an accuracy result.'}
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
