"""Experimental kernels only; does not change DeepKV or its packing policy."""
import argparse
import json
import time
from itertools import accumulate

import torch
import torch.nn.functional as F


class Layout:
    def __init__(self, rows, device):
        self.batch, self.length = len(rows), sum(rows[0])
        assert all(sum(row) == self.length and all(n > 0 for n in row) for row in rows)
        self.rows = rows
        segments, q_indices, k_indices, lengths = [], [], [], []
        for batch, row in enumerate(rows):
            segments.append([i for i, n in enumerate(row) for _ in range(n)])
            offset = batch * self.length
            for n in row:
                if n > 1:
                    q_indices.extend(range(offset + 1, offset + n))
                    k_indices.extend(range(offset, offset + n - 1))
                    lengths.append(n - 1)
                offset += n
        self.segments = torch.tensor(segments, device=device)
        self.q_indices = torch.tensor(q_indices, device=device, dtype=torch.long)
        self.k_indices = torch.tensor(k_indices, device=device, dtype=torch.long)
        self.lengths = [n for row in rows for n in row]
        self.cu = torch.tensor([0, *accumulate(self.lengths)], device=device, dtype=torch.int32)
        self.strict_cu = torch.tensor([0, *accumulate(lengths)], device=device, dtype=torch.int32)
        self.maximum = max(self.lengths)
        self.strict_maximum = max(lengths, default=0)

    def allowed(self, strict):
        n = self.length
        causal = torch.ones(n, n, dtype=torch.bool, device=self.segments.device).tril(-int(strict))
        return causal[None, None] & (self.segments[:, None, :, None] == self.segments[:, None, None, :])


def repeat_heads(x, heads):
    return x.repeat_interleave(heads // x.shape[1], dim=1)


def dense(q, k, v, allowed):
    nonempty = allowed.any(-1, keepdim=True)
    eye = torch.eye(q.shape[2], dtype=torch.bool, device=q.device)[None, None]
    safe = allowed | (~nonempty & eye)
    mask = torch.zeros_like(safe, dtype=q.dtype).masked_fill(~safe, torch.finfo(q.dtype).min)
    value = F.scaled_dot_product_attention(q, repeat_heads(k, q.shape[1]),
                                          repeat_heads(v, q.shape[1]), attn_mask=mask)
    return value.masked_fill(~nonempty, 0)


def packed_varlen(q, k, v, layout, strict, kernel):
    b, h, n, d = q.shape
    def flatten(x):
        return x.transpose(1, 2).reshape(b * n, x.shape[1], d).contiguous()
    # Native varlen supports GQA, though available backends differ by device.
    fq, fk, fv = map(flatten, (q, k, v))
    if strict:
        if layout.strict_maximum == 0:
            return q * 0 + (k.sum() + v.sum()) * 0
        fq = fq.index_select(0, layout.q_indices)
        fk, fv = (x.index_select(0, layout.k_indices) for x in (fk, fv))
        cu, maximum = layout.strict_cu, layout.strict_maximum
    else:
        cu, maximum = layout.cu, layout.maximum
    out = kernel(fq, fk, fv, cu, cu, maximum, maximum,
                 window_size=(-1, 0), enable_gqa=q.shape[1] != k.shape[1])
    if strict:
        out = out.new_zeros(b * n, h, d).index_copy(0, layout.q_indices, out)
    return out.reshape(b, n, h, d).transpose(1, 2)


def factory(name, layout, strict):
    started = time.perf_counter()
    if name == 'dense':
        allowed = layout.allowed(strict)
        fn = lambda q, k, v: dense(q, k, v, allowed)
    elif name == 'varlen':
        from torch.nn.attention.varlen import varlen_attn
        fn = lambda q, k, v: packed_varlen(q, k, v, layout, strict, varlen_attn)
    elif name == 'fa4':
        from flash_attn.cute.interface import flash_attn_varlen_func
        def kernel(q, k, v, cq, ck, mq, mk, **kwargs):
            return flash_attn_varlen_func(q, k, v, cu_seqlens_q=cq, cu_seqlens_k=ck,
                                          max_seqlen_q=mq, max_seqlen_k=mk, causal=True)
        fn = lambda q, k, v: packed_varlen(q, k, v, layout, strict, kernel)
    else:
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention
        segments = layout.segments
        def mask_mod(b, h, q, k):
            same = segments[b, q] == segments[b, k]
            # Dummy self edge for a document's first query, zeroed below.
            first = (q == 0) | (segments[b, (q - 1).clamp_min(0)] != segments[b, q])
            return same & ((q > k) | (first & (q == k))) if strict else same & (q >= k)
        mask = create_block_mask(mask_mod, layout.batch, None, layout.length, layout.length,
                                 device=str(segments.device))
        options = {'BACKEND': 'FLASH'} if name == 'flex_fa4' else {}
        flex = torch.compile(flex_attention, dynamic=False)
        nonempty = layout.allowed(strict).any(-1, keepdim=True)
        def fn(q, k, v):
            return flex(q, k, v, block_mask=mask, enable_gqa=True,
                        kernel_options=options).masked_fill(~nonempty, 0)
    if layout.segments.is_cuda:
        torch.cuda.synchronize()
    return fn, (time.perf_counter() - started) * 1000


def validate(fn, inputs, allowed):
    # FP32 dense SDPA is the independent numerical reference.
    reference_inputs = [x.detach().float().requires_grad_() for x in inputs]
    reference = dense(*reference_inputs, allowed)
    probe = torch.randn_like(reference)
    expected_grads = torch.autograd.grad((reference * probe).sum(), reference_inputs)
    actual = fn(*inputs)
    grads = torch.autograd.grad((actual.float() * probe).sum(), inputs)
    errors = []
    for label, a, e in zip(('output', 'dq', 'dk', 'dv'), (actual, *grads), (reference, *expected_grads)):
        delta = a.float() - e
        norm = delta.norm().item() / max(e.norm().item(), 1e-12)
        maximum = delta.abs().max().item() / max(e.abs().max().item(), 1e-12)
        assert torch.isfinite(a).all() and norm < .02 and maximum < .03, (label, norm, maximum)
        errors.append(dict(tensor=label, relative_l2=norm, relative_max=maximum))
    # Changing the first document must leave every later document unchanged.
    if allowed.shape[-1] > 1:
        changed = [x.detach().clone() for x in inputs]
        # Position 0 belongs to the first document in each row.
        changed[1][:, :, 0] += 1
        changed[2][:, :, 0] -= 1
        after = fn(*changed)
        invisible = ~allowed[:, 0, :, 0]
        # Exclude empty rows: they are explicitly zero in both paths anyway.
        delta = (after - actual.detach()).masked_select(invisible[:, None, :, None].expand_as(after))
        assert delta.count_nonzero().item() == 0, 'Cross-document or causal leakage'
    return errors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backends', nargs='+', default=['dense', 'varlen', 'flex', 'fa4', 'flex_fa4'])
    p.add_argument('--iterations', type=int, default=20)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    assert args.iterations > 0
    torch.set_num_threads(1); torch.manual_seed(42)
    print('KERNEL_ENV', json.dumps(dict(torch=torch.__version__, cuda=torch.version.cuda,
                                      gpu=torch.cuda.get_device_name())), flush=True)
    report = []
    layouts = {'equal': [[512] * 4] * 2,
               'ragged': [[1, 127, 384, 1536], [63, 1, 448, 512, 1024]]}
    for case, rows in layouts.items():
        start = time.perf_counter(); layout = Layout(rows, 'cuda')
        torch.cuda.synchronize(); layout_ms = (time.perf_counter() - start) * 1000
        for strict in (False, True):
            allowed = layout.allowed(strict)
            inputs = [torch.randn(2, h, 2048, 128, device='cuda', dtype=torch.bfloat16,
                                  requires_grad=True) for h in (16, 8, 8)]
            for name in args.backends:
                record = dict(layout=case, strict_past=strict, backend=name, layout_ms=layout_ms)
                try:
                    fn, setup_ms = factory(name, layout, strict)
                    validation_start = time.perf_counter()
                    errors = validate(fn, inputs, allowed)
                    torch.cuda.synchronize()
                    validation_seconds = time.perf_counter() - validation_start
                    def step():
                        for x in inputs: x.grad = None
                        fn(*inputs).float().square().mean().backward()
                    for _ in range(3): step()
                    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    for _ in range(args.iterations): step()
                    torch.cuda.synchronize()
                    ms = (time.perf_counter() - start) * 1000 / args.iterations
                    record.update(status='passed', forward_backward_ms=ms, setup_ms=setup_ms,
                                  compile_and_validation_seconds=validation_seconds,
                                  peak_mib=torch.cuda.max_memory_allocated()/1024**2, errors=errors)
                except Exception as error:
                    record.update(status='failed_or_unavailable', error=repr(error))
                report.append(record); print('KERNEL_RESULT', json.dumps(record), flush=True)
                from pathlib import Path
                Path(args.output).write_text(json.dumps(report, indent=2))
    assert all(r['status'] == 'passed' for r in report if r['backend'] == 'dense')
    print('PACKED_KERNEL_BENCHMARK_FINISHED', flush=True)


if __name__ == '__main__':
    main()
