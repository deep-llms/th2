"""Short FA4 kernel and Accelerate/DDP runtime check; synthetic data only."""
import argparse
import json
from pathlib import Path

import torch
from accelerate import Accelerator
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from deep_kv.fa4 import attention, load_kernel


def relative_error(actual, reference):
    return float((actual.float() - reference.float()).norm() /
                 reference.float().norm().clamp_min(1e-12))


def numerical_check(device, kernel):
    torch.manual_seed(42)
    boundaries = [0, 1, 129, 777, 1024, 2048]
    cu = torch.tensor(boundaries, dtype=torch.int32, device=device)
    layout = (cu, max(b - a for a, b in zip(boundaries, boundaries[1:])))
    source = [torch.randn(1, h, 2048, 128, device=device, dtype=torch.bfloat16)
              for h in (16, 8, 8)]
    q, k, v = [x.detach().requires_grad_() for x in source]
    actual = attention(q, k, v, layout, 128 ** -.5, kernel)
    upstream = torch.randn_like(actual)
    grads = torch.autograd.grad(actual, (q, k, v), upstream)
    rq, rk, rv = [x.float().detach().requires_grad_() for x in source]
    parts = []
    with sdpa_kernel(SDPBackend.MATH):
        for a, b in zip(boundaries, boundaries[1:]):
            parts.append(F.scaled_dot_product_attention(
                rq[:, :, a:b], rk[:, :, a:b], rv[:, :, a:b],
                is_causal=True, enable_gqa=True).transpose(1, 2))
    reference = torch.cat(parts, dim=1)
    ref_grads = torch.autograd.grad(reference, (rq, rk, rv), upstream.float())
    errors = dict(output=relative_error(actual, reference),
                  **{name: relative_error(a, b) for name, a, b in
                     zip(('query', 'key', 'value'), grads, ref_grads)})
    assert errors['output'] < .02 and all(errors[x] < .03 for x in ('query', 'key', 'value')), errors
    assert all(torch.isfinite(g).all() for g in grads)
    # Change only earlier documents; later outputs and their gradients must be isolated.
    changed = [x.clone() for x in source]
    for tensor in changed:
        tensor[:, :, :777] = 2
    with torch.no_grad():
        altered = attention(*changed, layout, 128 ** -.5, kernel)
    assert torch.equal(actual[:, 777:], altered[:, 777:])
    isolated = attention(q, k, v, layout, 128 ** -.5, kernel)
    leakage = torch.autograd.grad(isolated[:, 777:].float().square().sum(), (q, k, v))
    assert all(torch.count_nonzero(g[:, :, :777]) == 0 for g in leakage)
    return errors


class AttentionProbe(nn.Module):
    def __init__(self, kernel, device):
        super().__init__()
        self.projection = nn.Linear(128, 4096, bias=False)
        self.kernel = kernel
        self.register_buffer('cu', torch.tensor([0, 129, 1024, 2048], dtype=torch.int32, device=device))

    def forward(self, inputs):
        projected = self.projection(inputs).to(torch.bfloat16)
        q, k, v = projected.split((2048, 1024, 1024), dim=-1)
        def reshape(x, heads):
            return x.reshape(1, 2048, heads, 128).transpose(1, 2)
        return attention(reshape(q, 16), reshape(k, 8), reshape(v, 8),
                         (self.cu, 1024), 128 ** -.5, self.kernel).float().square().mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=2)
    assert accelerator.num_processes == 8 and accelerator.device.type == 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert not args.output.exists(), args.output
    kernel, runtime = load_kernel()
    errors = numerical_check(accelerator.device, kernel)
    torch.manual_seed(1042)
    model = AttentionProbe(kernel, accelerator.device).to(accelerator.device)
    initial = model.projection.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model, optimizer = accelerator.prepare(model, optimizer)
    torch.manual_seed(1042 + accelerator.process_index)
    losses = []
    for _ in range(3):
        for _ in range(2):
            inputs = torch.randn(1, 2048, 128, device=accelerator.device)
            with accelerator.accumulate(model), accelerator.autocast():
                loss = model(inputs)
                assert torch.isfinite(loss)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    for parameter in model.parameters():
                        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                        gathered = accelerator.gather(parameter.grad.detach().flatten())
                        gathered = gathered.reshape(8, -1)
                        assert torch.equal(gathered, gathered[0].expand_as(gathered)), 'DDP gradients differ'
                optimizer.step()
                optimizer.zero_grad()
            losses.append(float(loss.detach()))
    weight = accelerator.unwrap_model(model).projection.weight.detach()
    assert not torch.equal(initial, weight), 'Optimizer did not update weights'
    weights = accelerator.gather(weight.flatten()).reshape(8, -1)
    assert torch.equal(weights, weights[0].expand_as(weights)), 'Weights differ across ranks'
    ranks = accelerator.gather(torch.tensor([accelerator.process_index + 1], device=accelerator.device))
    assert ranks.sum().item() == 36
    all_errors = accelerator.gather(torch.tensor(list(errors.values()), device=accelerator.device)).reshape(8, -1)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result = dict(status='passed', world_size=8, optimizer_steps=3, accumulation=2,
                      sequence_length=2048, runtime=runtime, losses_rank0=losses,
                      errors_by_rank=all_errors.tolist(), error_columns=list(errors),
                      document_isolation=True, synchronized_gradients=True,
                      synchronized_updated_weights=True, synthetic_runtime_test=True)
        with args.output.open('x') as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
        print('FA4_EIGHT_GPU_SMOKE_PASSED', json.dumps(result), flush=True)
    accelerator.wait_for_everyone()


if __name__ == '__main__':
    main()
