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
        result = subprocess.run([sys.executable, '-c', code],
            cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, 'TRITON_INTERPRET': '1', 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1'},
            capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
