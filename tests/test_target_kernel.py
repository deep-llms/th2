"""CPU interpreter regression for custom target arithmetic and first-order VJP.

The compiled CUDA implementation is additionally gated on B200 before timing.
"""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import unittest


class TargetKernelTests(unittest.TestCase):
    def run_interpreter(self, code):
        result = subprocess.run([sys.executable, '-c', code],
            cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, 'TRITON_INTERPRET': '1', 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1'},
            capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless(importlib.util.find_spec('triton'), 'Triton not installed')
    def test_interpreted_primitive_and_overlapping_gradients(self):
        # Isolated process: interpreter selection must precede Triton imports.
        # Bypass only the device guard; all tensors/allocations stay on CPU.
        code = """
import torch
from unittest.mock import patch
from scripts.p7_kernel_trial import primitive_gate
with patch.object(torch.Tensor, 'is_cuda', property(lambda self: True)):
    results = primitive_gate(torch.device('cpu'))
assert len(results) == 13 and all(r['status'] == 'passed' for r in results)
"""
        self.run_interpreter(code)

    @unittest.skipUnless(importlib.util.find_spec('triton'), 'Triton not installed')
    def test_interpreted_model_routes_optimizer_and_statistics(self):
        # Exercise the actual kernel in the production model, not the torch
        # reference used by the other CPU integration tests. This complements
        # (and does not replace) the compiled CUDA gates recorded on B200.
        self.run_interpreter("""
import torch
from unittest.mock import patch
from deep_kv.target_kernel import target_window
from tests.test_proxy_memory import model, batch, objective

torch.set_num_threads(1)
calls = []
def interpreted_target(*args):
    calls.append(1)
    # Restrict the device-guard bypass to kernel preparation. The model,
    # autograd, optimizer and all tensor allocations remain ordinary CPU code.
    with patch.object(torch.Tensor, 'is_cuda', property(lambda self: True)):
        return target_window(*args)

def compare(a, b, label):
    assert a.keys() == b.keys(), label
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=5e-4, atol=5e-6,
                                   msg=label+'/'+key)

with patch('deep_kv.target_kernel.target_window', interpreted_target):
    for checkpointed in (False, True):
        eager = model('P7-simple-targetgrad', checkpoint=checkpointed)
        fused = model('P7-simple-targetgrad', checkpoint=checkpointed, fused_targets=True)
        eager.mu.fill_(.1)
        eager.sigma2.fill_(.02)
        fused.load_state_dict(eager.state_dict(), strict=True)
        optimizers = [torch.optim.AdamW(m.parameters(), lr=.0003) for m in (eager, fused)]
        for step in range(2):
            captures = []
            for m, opt in zip((eager, fused), optimizers):
                opt.zero_grad(set_to_none=True)
                out = m(batch(), collect_target_statistics=True)
                named = dict(m.named_parameters())
                routes = {}
                for label, loss in (('lm', out['lm_sum']/out['lm_count']),
                                    ('aux', out['aux_sum']/out['aux_count'])):
                    grads = torch.autograd.grad(loss, list(named.values()),
                                                allow_unused=True, retain_graph=True)
                    routes.update({label+'/'+n: g.clone() for n, g in zip(named, grads) if g is not None})
                assert any(n.startswith('aux/backbone.') and g.abs().sum()>0 for n, g in routes.items())
                assert not any(n.startswith('lm/heads.') and '.w1.' in n for n in routes)
                objective(out).backward()
                grads = {n: p.grad.clone() for n, p in named.items() if p.grad is not None}
                opt.step()
                m.update_statistics(out['center_sums'], out['center_squares'], out['center_counts'])
                captures.append((out, routes, grads, m.state_dict()))
            for label, a, b in zip(('output', 'routes', 'gradients', 'state'), *captures):
                compare(a, b, f'{checkpointed}/{step}/{label}')
        # Auxiliary gradients disabled and no-grad evaluation must stay valid
        # even though the same fused target implementation remains installed.
        outs = [m(batch(), auxiliary_grad=False) for m in (eager, fused)]
        compare(*outs, 'auxiliary_grad=False')
        assert not outs[1]['aux_sum'].requires_grad
        with torch.no_grad():
            compare(eager(batch()), fused(batch()), 'no_grad')
assert calls, 'The actual kernel was never exercised'
""")
