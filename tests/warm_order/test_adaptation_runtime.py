import unittest
from unittest.mock import patch
import numpy as np
import torch
from efficient_llm.warm_order.adaptation_runtime import measurement_runtime, measurement_policy, measurement_device, training_runtime
from efficient_llm.warm_order.adaptation_gpu_probe import exact
from efficient_llm.warm_order.adaptation_components import component_crossovers
from efficient_llm.warm_order.model import Model,Config

class AdaptationRuntime(unittest.TestCase):
    def test_explicit_cpu_preserves_component_results_and_context_after_failure(self):
        torch.set_num_threads(1);torch.manual_seed(143)
        m=Model(Config(d=8,heads=1,kv_heads=1,head_dim=8,ff=16,vocab=19,prefix=2,length=8)).eval().requires_grad_(False)
        x=torch.randint(0,19,(2,8));a=component_crossovers(m,x,2,'attention')
        with measurement_runtime('cpu'):
            b=component_crossovers(m,x,2,'attention')
            for key in a['cells']:torch.testing.assert_close(a['cells'][key]['hidden'],b['cells'][key]['hidden'],atol=0,rtol=0)
        with self.assertRaises(RuntimeError):
            with measurement_runtime('cpu'):raise RuntimeError('injected')
        self.assertEqual(measurement_device(),'cpu')
        with patch('torch.cuda.device_count',return_value=0):
            with self.assertRaisesRegex(ValueError,'one visible'):
                with measurement_runtime('cuda'):pass
        self.assertEqual(measurement_policy()['device'],'cpu')
        with self.assertRaises(ValueError):
            with measurement_runtime('cuda:0'):pass

    def test_training_determinism_policy_and_failure_restore(self):
        import os
        before=(torch.are_deterministic_algorithms_enabled(),torch.is_deterministic_algorithms_warn_only_enabled(),
                torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
        with patch.dict(os.environ,{'CUBLAS_WORKSPACE_CONFIG':''}):
            with self.assertRaisesRegex(ValueError,'CUBLAS_WORKSPACE_CONFIG'):
                with training_runtime('cuda'):pass
        with patch.dict(os.environ,{'CUBLAS_WORKSPACE_CONFIG':':4096:8'}):
            with self.assertRaisesRegex(RuntimeError,'injected'):
                with training_runtime('cuda') as policy:
                    self.assertTrue(torch.are_deterministic_algorithms_enabled())
                    self.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())
                    self.assertFalse(torch.backends.cuda.flash_sdp_enabled())
                    self.assertTrue(torch.backends.cuda.math_sdp_enabled())
                    self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                    self.assertEqual(policy['attention_backend'],'math')
                    raise RuntimeError('injected')
        self.assertEqual(before,(torch.are_deterministic_algorithms_enabled(),torch.is_deterministic_algorithms_warn_only_enabled(),
                         torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32))

    def test_semantic_fullstate_comparison_checks_rng_dtype_and_nonfinite(self):
        a=dict(model=torch.tensor([1.]),rng=np.array([1,2],dtype=np.uint32),cuda_rng=[torch.tensor([2],dtype=torch.uint8)],step=40)
        exact(a,a)
        for b in (dict(a,step=39),dict(a,model=torch.tensor([1.],dtype=torch.float64)),
                  dict(a,rng=np.array([1,3],dtype=np.uint32))):
            with self.assertRaises(ValueError):exact(a,b)
        bad=dict(a,model=torch.tensor([float('inf')]))
        with self.assertRaisesRegex(ValueError,'Nonfinite'):exact(bad,bad)

if __name__=='__main__':unittest.main()
