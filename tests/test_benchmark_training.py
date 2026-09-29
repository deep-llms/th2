"""Benchmark substitutions must preserve the original objective and gradients."""
import unittest
import torch
from scripts.benchmark_training import configure_model, validate_variant
from tests.test_deep_kv import config, context


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_causal_loss_and_gradients_for_all_measured_arms(self):
        for arm in 'ABFG':
            with self.subTest(arm=arm):
                result=validate_variant('causal',False,arm)
                self.assertLess(result['gradient_relative_l2'],2e-5)

    def test_native_qwen_matches_wrapped_objective_and_gradients(self):
        self.assertLess(validate_variant('original',True,'A')['gradient_relative_l2'],2e-5)

    def test_chunk_size_and_checkpoint_recomputation_preserve_gradients(self):
        for enabled in (True,False):
            value=validate_variant('causal',False,'G',checkpoint_layers=enabled,lm_chunk=64)
            self.assertLess(value['gradient_relative_l2'],2e-5)

    def test_fast_path_refuses_padding_and_segment_masks(self):
        model=configure_model('causal').from_scratch(config(),'B',consumer=2,deep_target=4)
        with self.assertRaisesRegex(ValueError,'fully packed'):
            model(context())


if __name__=='__main__':
    unittest.main()
