"""FA4 variable-length attention for the document-isolated vanilla baseline."""
from importlib.metadata import version
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path

import torch


def load_kernel():
    # Same pinned API used by the full-training document-isolation benchmark.
    try:
        installed = version('flash-attn-4')
        from flash_attn.cute.interface import flash_attn_varlen_func
    except ImportError as error:
        raise RuntimeError('FA4 baseline requires envs/attention_bench.txt; no SDPA fallback') from error
    if installed != '4.0.0b33':
        raise RuntimeError(f'FA4 baseline requires flash-attn-4==4.0.0b33, found {installed}')
    return flash_attn_varlen_func, dict(package='flash-attn-4', version=installed,
                                      entry_point='flash_attn.cute.interface.flash_attn_varlen_func')


def document_layout(context):
    """Flatten packed rows into independent contiguous document fragments."""
    segments = context.segments
    if segments is None or not bool(context.valid.all()):
        raise ValueError('FA4 isolation requires fully packed document metadata')
    starts = torch.ones_like(segments, dtype=torch.bool)
    starts[:, 1:] = segments[:, 1:] != segments[:, :-1]
    boundaries = starts.flatten().nonzero().flatten()
    cu = torch.cat((boundaries, boundaries.new_tensor([segments.numel()]))).to(torch.int32)
    maximum = int((cu[1:]-cu[:-1]).max().item())
    return cu, maximum


def attention(query, key, value, layout, scaling, kernel):
    """Preserve native GQA, QK-norm/RoPE results and differentiable token order."""
    b, heads, length, dim = query.shape
    def flatten(x):
        return x.to(value.dtype).transpose(1, 2).reshape(b*length, x.shape[1], dim).contiguous()
    q, k, v = map(flatten, (query, key, value))
    cu, maximum = layout
    output, _ = kernel(q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu,
                       max_seqlen_q=maximum, max_seqlen_k=maximum,
                       causal=True, softmax_scale=scaling)
    return output.reshape(b, length, heads, dim)


@contextmanager
def trainer_audit(trainer, phase):
    """Record actual FA4 wrapper calls and gradients on the first GPU microbatch."""
    model = trainer.model
    receipt = dict(phase=phase, backend=model.attention_runtime, global_step=trainer.state.global_step,
                   rank=trainer.args.process_index, world_size=trainer.args.world_size,
                   calls=0, query_gradient_calls=0, causal=True, document_isolation=True,
                   positions='reset_per_document', dense_mask=False,
                   note='Wrapper calls and query gradients; not CUDA kernel profiling or numerical validation')
    def backward(gradient):
        receipt['query_gradient_calls'] += 1
        return gradient
    def observe(q,k,v,layout):
        receipt['calls'] += 1
        if receipt['calls'] == 1:
            receipt.update(query_shape=list(q.shape),key_shape=list(k.shape),
                           kernel_dtype=str(v.dtype),fragments=layout[0].numel()-1,max_seqlen=layout[1])
        if q.requires_grad:
            q.register_hook(backward)
    previous = model._fa4_observer
    model._fa4_observer = observe
    try:
        yield
    finally:
        model._fa4_observer = previous
    layers = len(model.backbone.model.layers)
    if receipt['calls'] < layers or (phase == 'train' and receipt['query_gradient_calls'] < layers):
        raise RuntimeError('Incomplete FA4 forward/backward receipt')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path = Path(trainer.args.output_dir)/f'fa4-{phase}-{stamp}.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:
        json.dump(receipt,stream,indent=2,allow_nan=False)
    if not hasattr(trainer,'_fa4_receipts'):
        trainer._fa4_receipts = []
    trainer._fa4_receipts.append(path.name)
