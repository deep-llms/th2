"""Validation monitoring that cannot change training modes, gradients or RNG state."""
import random
import time
import numpy as np
import torch
from .model import ORDERS


@torch.no_grad()
def probe(model, rows, ids, depth, device):
    if depth not in (1, 2, 3, 4) or len(ids) == 0:
        raise ValueError('Invalid monitoring request')
    modes = [(m, m.training) for m in model.modules()]
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("FP32 validation requires FP32 parameters")
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else None
    np_rng, py_rng = np.random.get_state(), random.getstate()
    started = time.monotonic()
    try:
        model.eval()
        full, active = [], []
        for offset in range(0, len(ids), 8):
            tokens = rows.batch(ids[offset:offset + 8], device)
            costs = torch.stack([model.nll(tokens, path) for path in ORDERS], 1)
            short = costs if depth == 4 else torch.stack([model.nll(tokens, path[:depth]) for path in ORDERS], 1)
            full.append(costs.float().cpu().numpy())
            active.append(short.float().cpu().numpy())
        full, active = np.concatenate(full), np.concatenate(active)
        if not np.isfinite(full).all() or not np.isfinite(active).all():
            raise ValueError('Nonfinite validation probe')
        return dict(depth=depth, sequences=len(ids), precision='fp32',
                    full_order_means=full.mean(0, dtype=np.float64).tolist(),
                    active_order_means=active.mean(0, dtype=np.float64).tolist(),
                    block_tokens=(48 + (8 * (2 + depth) if depth < 4 else 0)) * len(ids) * model.c.length,
                    seconds=time.monotonic() - started)
    finally:
        for module, mode in modes:
            module.training = mode
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        np.random.set_state(np_rng)
        random.setstate(py_rng)
