import unittest
import tempfile
import torch

from deep_kv.model import Context
from eval.recurrent_p6 import recurrent_hidden, injected_residual, project_kv
from scripts.evaluate_proxy_gates import state_hash
from tests.test_proxy_heads import model, batch


def context(ids, docs):
    starts = torch.ones_like(docs, dtype=torch.bool)
    starts[:, 1:] = docs[:, 1:] != docs[:, :-1]
    offsets = torch.arange(ids.shape[1]).expand_as(ids)
    pos = offsets-offsets.masked_fill(~starts, 0).cummax(-1).values
    return Context(ids, torch.ones_like(ids, dtype=torch.bool), pos, docs, ids)


class RecurrentP6Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_hf_runtime_does_not_reset_accelerator_after_creation(self):
        from scripts.evaluate_recurrent_p6 import evaluation_runtime
        with tempfile.TemporaryDirectory() as output:
            args, accelerator = evaluation_runtime(dict(use_cpu=True, bf16=False), output)
            with args.main_process_first(desc='CPU cache setup regression'):
                self.assertEqual(accelerator.process_index, 0)
                self.assertEqual(accelerator.num_processes, 1)
            accelerator.wait_for_everyone()

    def test_native_matches_parallel_and_disabled_without_mutation(self):
        m = model('P6-iso').eval(); ctx = batch()
        before = state_hash(m)
        with torch.no_grad():
            actual = recurrent_hidden(m, ctx, 'native_proxy')
            reference = m.hidden_states(ctx)[0]
            torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-6)
            with m.without_proxy():
                reference = m.hidden_states(ctx)[0]
            torch.testing.assert_close(recurrent_hidden(m, ctx, 'disabled'), reference, rtol=2e-5, atol=2e-6)
        self.assertEqual(before, state_hash(m))

    def test_real_uses_no_predictor_and_publishes_exact_target_kv(self):
        m = model('P6-iso').eval(); ctx = batch(); seen = []
        before = state_hash(m)
        def observe(t, number, source, prediction, k, v, mlps):
            raw = sum(mlps[number-1:number+3])
            expected = m.normalize_target(raw, number)
            torch.testing.assert_close(prediction, expected, rtol=0, atol=0)
            layer = m.backbone.model.layers[number-1]
            residual = injected_residual(m, number, source, expected)
            rotary = m.backbone.model.rotary_emb(source, ctx.position_ids[:, t:t+1])
            rk, rv = project_kv(layer, layer.input_layernorm(residual), rotary)
            torch.testing.assert_close(k, rk, rtol=0, atol=0)
            torch.testing.assert_close(v, rv, rtol=0, atol=0)
            seen.append((t, number))
        # Any invocation of a learned predictor in the real-only pass is a bug.
        handles = [h.w1.register_forward_pre_hook(lambda *_: self.fail('Predictor called')) for h in m.heads.values()]
        try:
            real = recurrent_hidden(m, ctx, 'past_real', observer=observe)
        finally:
            for handle in handles:handle.remove()
        self.assertEqual(len(seen), ctx.input_ids.numel()*len(m.layers))
        disabled = recurrent_hidden(m, ctx, 'disabled')
        # A document's first token has no prior memory and must be native.
        torch.testing.assert_close(real[:, [0, 3]], disabled[:, [0, 3]], rtol=0, atol=0)
        self.assertGreater(float((real[:, 1:3]-disabled[:, 1:3]).abs().max()), 1e-6)
        self.assertEqual(before, state_hash(m))

    def test_causality_document_reset_and_manual_attention_oracle(self):
        m = model('P6-iso').eval(); ctx = batch()
        def oracle(q, k, v, *, attn_mask, dropout_p, is_causal, scale):
            self.assertFalse(is_causal); self.assertEqual(dropout_p, 0)
            logits = (q@k.transpose(-1, -2))*scale
            return logits.masked_fill(~attn_mask, -torch.inf).softmax(-1)@v
        for mode in ('native_proxy', 'past_proxy', 'past_real', 'disabled'):
            real = recurrent_hidden(m, ctx, mode)
            reference = recurrent_hidden(m, ctx, mode, attention_fn=oracle)
            torch.testing.assert_close(real, reference, rtol=2e-5, atol=2e-6)
            changed = ctx.input_ids.clone(); changed[:, -2:] = 6
            other = recurrent_hidden(m, context(changed, ctx.segments), mode)
            torch.testing.assert_close(real[:, :-2], other[:, :-2], rtol=0, atol=0)
            changed = ctx.input_ids.clone(); changed[:, :3] = 6
            other = recurrent_hidden(m, context(changed, ctx.segments), mode)
            torch.testing.assert_close(real[:, 3:], other[:, 3:], rtol=0, atol=0)
            standalone = context(ctx.input_ids[:, 3:], torch.zeros_like(ctx.segments[:, 3:]))
            torch.testing.assert_close(real[:, 3:], recurrent_hidden(m, standalone, mode), rtol=2e-5, atol=2e-6)

    def test_zero_gates_all_modes_agree_and_invalid_inputs_fail(self):
        m = model('P6-iso').eval(); ctx = batch()
        with torch.no_grad():
            for head in m.heads.values():head.alpha.zero_()
        reference = recurrent_hidden(m, ctx, 'disabled')
        for mode in ('native_proxy', 'past_proxy', 'past_real'):
            torch.testing.assert_close(recurrent_hidden(m, ctx, mode), reference, rtol=0, atol=0)
        with self.assertRaises(ValueError):recurrent_hidden(m.train(), ctx, 'past_real')
        m.eval()
        ctx.position_ids[:, -1] += 1
        with self.assertRaises(ValueError):recurrent_hidden(m, ctx, 'past_real')

    def test_batched_full_depth_and_bfloat16_controls(self):
        from tests.test_proxy_heads import cfg, settings
        from deep_kv.proxy import ProxyModel
        config = cfg(); config.num_hidden_layers = 28; config.layer_types = ['full_attention']*28
        m = ProxyModel.from_scratch(config, 'P6-iso', proxy_settings=settings(),
                                    checkpoint_layers=False,checkpoint_lm=False,checkpoint_aux=False).eval()
        ids = torch.tensor([[3,4,5,6,7,8],[9,8,7,6,5,4]])
        docs = torch.tensor([[0,0,1,1,1,2],[5,5,5,6,6,6]])
        ctx = context(ids, docs)
        for bf16 in (False, True):
            with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16, enabled=bf16):
                reference = m.hidden_states(ctx)[0]
                actual = recurrent_hidden(m, ctx, 'native_proxy')
                relative = float((actual-reference).norm()/reference.norm())
                self.assertLess(relative, .02 if bf16 else 2e-5)
                real = recurrent_hidden(m, ctx, 'past_real')
                for i in range(2):
                    single = context(ids[i:i+1],docs[i:i+1])
                    other = recurrent_hidden(m, single, 'past_real')
                    torch.testing.assert_close(real[i:i+1],other,rtol=.02 if bf16 else 2e-5,
                                               atol=.02 if bf16 else 2e-6)
