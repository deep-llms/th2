"""Explicit device scope; ownership/admission is enforced by the node driver."""
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

import torch

_DEVICE = ContextVar('embedding_anchor_device', default='cpu')
OLD_GUARD = """        if x.device.type != 'cpu' or self.embed.weight.device.type != 'cpu':
            raise RuntimeError('Prototype is CPU-only; production admission absent')
"""
NEW_GUARD = """        from .embedding_anchor_runtime import anchor_device
        if x.device.type != anchor_device() or self.embed.weight.device.type != anchor_device():
            raise RuntimeError('Explicit matching anchor device scope required')
"""


def anchor_device(): return _DEVICE.get()


@contextmanager
def anchor_runtime(device):
    if device not in ('cpu', 'cuda'): raise ValueError('Explicit CPU/CUDA required')
    if device == 'cuda' and torch.cuda.device_count() != 1:
        raise ValueError('Exactly one visible owned GPU required')
    token = _DEVICE.set(device)
    try: yield
    finally: _DEVICE.reset(token)


def verify_forward_bridge(old_path):
    """Only the device guard may differ from authenticated CPU forward source."""
    old = Path(old_path).read_text()
    if old.count(OLD_GUARD) != 1:
        raise ValueError('Unexpected CPU reference source guard')
    current = Path(__file__).with_name('embedding_anchor.py').read_text()
    if old.replace(OLD_GUARD, NEW_GUARD) != current:
        raise ValueError('CPU/GPU source bridge changed computation')
