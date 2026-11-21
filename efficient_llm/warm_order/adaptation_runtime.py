"""Explicit measurement arithmetic; ownership/admission belongs to the caller.

Evaluation uses FP32 on CPU or CUDA with math SDPA and TF32 disabled. BF16 is
the continuation training policy, not the component-intervention policy.
"""
from contextlib import contextmanager
from contextvars import ContextVar

import torch

_DEVICE = ContextVar('adaptation_measurement_device', default='cpu')


def measurement_device():
    return _DEVICE.get()


def measurement_policy():
    cuda = measurement_device() == 'cuda'
    return dict(device=measurement_device(), precision='fp32',
                attention_backend='math' if cuda else 'cpu_native',
                tf32=False if cuda else None,
                component_atol=1e-4 if cuda else 2e-6,
                component_rtol=1e-4 if cuda else 1e-5)


@contextmanager
def measurement_runtime(device='cpu'):
    if device not in ('cpu', 'cuda'): raise ValueError('Explicit CPU or CUDA measurement required')
    if device == 'cuda' and torch.cuda.device_count() != 1:
        raise ValueError('Exactly one visible GPU for measurement')
    token = _DEVICE.set(device)
    try:
        if device == 'cpu':
            yield measurement_policy()
        else:
            from torch.nn.attention import sdpa_kernel, SDPBackend
            previous = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
            try:
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                with torch.autocast('cuda', enabled=False), sdpa_kernel(SDPBackend.MATH):
                    yield measurement_policy()
            finally:
                torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous
    finally:
        _DEVICE.reset(token)


def training_policy(device):
    return dict(device=device, precision='bf16' if device=='cuda' else 'fp32',
                attention_backend='math' if device=='cuda' else 'cpu_native',
                deterministic=device=='cuda', tf32=False if device=='cuda' else None,
                cublas_workspace=':4096:8' if device=='cuda' else None)


@contextmanager
def training_runtime(device):
    """Fixed arithmetic for exact paired continuation/resume, restored on exit."""
    if device == 'cpu':
        yield training_policy(device)
        return
    if device != 'cuda':raise ValueError('Explicit cpu/cuda training device required')
    import os
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise ValueError('Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialization')
    from torch.nn.attention import sdpa_kernel, SDPBackend
    previous=(torch.are_deterministic_algorithms_enabled(),
              torch.is_deterministic_algorithms_warn_only_enabled(),
              torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        with sdpa_kernel(SDPBackend.MATH):
            yield training_policy(device)
    finally:
        torch.use_deterministic_algorithms(previous[0],warn_only=previous[1])
        torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=previous[2:]
