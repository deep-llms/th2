"""Small SDPA document-mask benchmark, not a full-model throughput estimate.

Run only on an explicitly allocated, free GPU. No process management or downloads.
"""
import argparse
import json
import time

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations', type=int, default=20)
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error('iterations must be positive')
    torch.set_num_threads(1)
    torch.manual_seed(42)
    b, h, n, d = 2, 16, 2048, 128
    q, k, v = [torch.randn(b, h, n, d, device='cuda', dtype=torch.bfloat16,
                          requires_grad=True) for _ in range(3)]
    causal = torch.ones(n, n, device='cuda', dtype=torch.bool).tril()[None, None].expand(b, 1, n, n)
    segments = torch.arange(n, device='cuda') // 512
    isolated = causal & (segments[:, None] == segments[None, :])[None, None]
    print(json.dumps(dict(torch=torch.__version__, cuda=torch.version.cuda,
                          gpu=torch.cuda.get_device_name(), shape=[b, h, n, d],
                          dtype='bfloat16', iterations=args.iterations)), flush=True)
    for name, allowed, implicit in [('implicit_causal', None, True),
                                    ('explicit_causal', causal, False),
                                    ('document_isolated', isolated, False)]:
        mask = (None if allowed is None else torch.zeros_like(allowed, dtype=torch.bfloat16)
                .masked_fill(~allowed, torch.finfo(torch.bfloat16).min))

        def step():
            for x in (q, k, v):
                x.grad = None
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=implicit)
            y.float().square().mean().backward()

        for _ in range(3):
            step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(args.iterations):
            step()
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000 / args.iterations
        peak = torch.cuda.max_memory_allocated() / 1024**2
        with profile(activities=[ProfilerActivity.CPU]) as trace:
            step()
        operators = [e.key for e in trace.key_averages() if 'scaled_dot_product' in e.key]
        assert all(torch.isfinite(x.grad).all() for x in (q, k, v))
        print(json.dumps(dict(mode=name, forward_backward_ms=elapsed,
                              peak_allocated_mib=peak, operators=operators)), flush=True)
    print('DOCUMENT_ATTENTION_BENCHMARK_PASSED', flush=True)


if __name__ == '__main__':
    main()
