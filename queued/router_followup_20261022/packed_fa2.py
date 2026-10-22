"""Pinned FA2 adapter using the same document layout and native Qwen operations."""
from functools import lru_cache
from importlib.metadata import version

from transformers import AttentionInterface
from packed_fa4 import varlen_attention

BACKEND = 'template_fa2'
FA2_VERSION = '2.8.3.post2'


@lru_cache(maxsize=1)
def load_kernel():
    installed = version('flash-attn')
    if installed.split('+')[0] != FA2_VERSION:
        raise RuntimeError(f'FA2 requires flash-attn=={FA2_VERSION}, found {installed}')
    from flash_attn import flash_attn_varlen_func

    def kernel(*args, **kwargs):
        return flash_attn_varlen_func(*args, dropout_p=0., **kwargs), None
    return kernel


def register_backend():
    load_kernel()
    AttentionInterface.register(BACKEND, attention_forward)
    return BACKEND


def attention_forward(module, query, key, value, attention_mask, **kwargs):
    return varlen_attention(load_kernel, module, query, key, value, attention_mask, **kwargs)
