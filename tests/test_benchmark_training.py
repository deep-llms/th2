"""Benchmark substitutions must preserve the original objective and gradients."""
import unittest
from unittest.mock import patch
import types
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

    def test_fa4_tuple_api_preserves_attention_output_and_gradients(self):
        # Exercise the real Qwen adapter with FA4's pinned (output, lse) API.
        # CPU SDPA substitutes only for the unavailable CUDA kernel here.
        calls=[]
        def flash(q,k,v,*,causal,softmax_scale):
            calls.append((q.shape,k.shape,v.shape,causal))
            out=torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),
                is_causal=causal,scale=softmax_scale,enable_gqa=True)
            return out.transpose(1,2),None
        package=types.ModuleType('flash_attn')
        cute=types.ModuleType('flash_attn.cute');cute.flash_attn_func=flash
        with patch.dict('sys.modules',{'flash_attn':package,'flash_attn.cute':cute}):
            result=validate_variant('fa4',False,'B')
        self.assertTrue(calls)
        self.assertTrue(all(causal for *_,causal in calls))
        self.assertLess(result['gradient_relative_l2'],2e-5)

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
