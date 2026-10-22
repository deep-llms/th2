"""Real varlen kernel forward/backward vs per-document FP32 SDPA math."""
import argparse
import json
import sys
from pathlib import Path
from importlib import import_module
from importlib.metadata import version
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

def relative(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))


def kernel_probe(kernel):
    from torch.nn.attention import sdpa_kernel, SDPBackend
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checks = []
    for lengths in ([1, 17, 37, 73], [128] * 16, [2048]):
        total = sum(lengths)
        cu = torch.tensor([0] + list(__import__('itertools').accumulate(lengths)),
                          device='cuda', dtype=torch.int32)
        xs = [torch.randn(total, h, 128, device='cuda', dtype=torch.bfloat16,
                          requires_grad=True) for h in (8, 2, 2)]
        refs = [x.detach().float().requires_grad_() for x in xs]
        out, _ = kernel(*xs, cu_seqlens_q=cu, cu_seqlens_k=cu,
                            max_seqlen_q=max(lengths), max_seqlen_k=max(lengths),
                            causal=True, softmax_scale=128 ** -.5)
        outputs = []
        start = 0
        with sdpa_kernel(SDPBackend.MATH):
            for length in lengths:
                q, k, v = [x[start:start+length].transpose(0, 1).unsqueeze(0) for x in refs]
                y = torch.nn.functional.scaled_dot_product_attention(
                    q, k.repeat_interleave(4, dim=1), v.repeat_interleave(4, dim=1),
                    is_causal=True, dropout_p=0., scale=128 ** -.5)
                outputs.append(y.squeeze(0).transpose(0, 1))
                start += length
        ref = torch.cat(outputs)
        grad = torch.randn_like(out)
        out.backward(grad)
        ref.backward(grad.float())
        errors = [relative(x.grad, r.grad) for x, r in zip(xs, refs)]
        row = dict(lengths=lengths, logits_relative_l2=relative(out, ref),
                   qkv_gradient_relative_l2=errors)
        assert row['logits_relative_l2'] < .01 and max(errors) < .02, row
        assert all(torch.isfinite(x.grad).all() for x in xs)
        checks.append(row)
    return dict(status='passed', checks=checks,
                reference='FP32 SDPA math, TF32 disabled, separate causal documents',
                tolerances=dict(output_relative_l2=.01, gradient_relative_l2=.02))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=('fa2','fa4'),required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise ValueError('Receipt already exists')
    kernel=import_module('packed_'+a.backend).load_kernel()
    result=kernel_probe(kernel)
    result.update(backend=a.backend,torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),kernel_version=version('flash-attn' if a.backend=='fa2' else 'flash-attn-4'))
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
