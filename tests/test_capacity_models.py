import math
import tempfile
import unittest

import torch
from transformers import AutoModelForCausalLM, Qwen3ForCausalLM, set_seed
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

from capacity_allocation.modeling import (
    ARMS, EXPECTED_COUNTS, SHARED_ARMS, AllocationConfig,
    DirectWideningDecoderLayer,
    FanInLinear, FixedResidualDecoderLayer, ScaledFanInLinear,
    ScaledVocabularyHead,
    activation_report, build_model, experiment_config, parameter_report,
)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_production_parameter_counts(self):
        for arm in ARMS:
            with self.subTest(arm=arm), torch.device("meta"):
                model = build_model(experiment_config(arm))
                self.assertEqual(parameter_report(model)["total"], EXPECTED_COUNTS[arm])
        for arm, count in [("B0", 344354816), ("A128", 345403392)]:
            with torch.device("meta"):
                self.assertEqual(parameter_report(build_model(experiment_config(arm, depth=12)))["total"], count)

    def test_stock_baseline_equivalence(self):
        config = experiment_config("B0", tiny=True)
        set_seed(7)
        actual = build_model(config)
        set_seed(7)
        expected = Qwen3ForCausalLM(config)
        x = torch.arange(8).unsqueeze(0)
        torch.testing.assert_close(actual(x).logits, expected(x).logits, rtol=0, atol=0)

    def test_qwen_reference_configuration_and_stage_heads(self):
        config = experiment_config('B0')
        for key, value in dict(vocab_size=151936, hidden_size=1024, intermediate_size=3072,
                               num_hidden_layers=6, num_attention_heads=16, num_key_value_heads=8,
                               head_dim=128, max_position_embeddings=40960,
                               eos_token_id=151645, pad_token_id=151643,
                               tie_word_embeddings=True).items():
            self.assertEqual(getattr(config, key), value, key)
        self.assertEqual(config.layer_types, ['full_attention']*6)
        self.assertEqual(config.rope_parameters['rope_theta'], 1000000.)
        for arm in ('C', 'D'):
            with torch.device('meta'):
                model = build_model(experiment_config(arm))
            for layer, width in zip(model.model.layers, model.config.widths):
                self.assertEqual(layer.self_attn.q_proj.out_features, 2*width)
                self.assertEqual(layer.self_attn.k_proj.out_features, width)
                self.assertEqual(layer.self_attn.v_proj.out_features, width)
                self.assertEqual(layer.self_attn.num_key_value_groups, 2)
                self.assertEqual(layer.mlp.intermediate_size, 3*width)
                self.assertEqual(layer.self_attn.q_norm.weight.numel(), 128)

    def test_new_config_rejects_ambiguous_or_unsupported_shapes(self):
        base = dict(vocab_size=97, head_dim=4)
        with self.assertRaisesRegex(ValueError, 'widening only'):
            AllocationConfig(widths=[32, 24], input_rank=8, output_rank=8,
                             body_type='direct', **base)
        with self.assertRaisesRegex(ValueError, 'Skipped vocabulary adapters'):
            AllocationConfig(widths=[32], input_rank=8, output_rank=8,
                             skip_input_adapter=True, **base)
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            AllocationConfig(widths=[32], interface_type='shared', shared_rank=8,
                             input_private_rank=4, input_rank=8, **base)
        with self.assertRaisesRegex(ValueError, 'full-width shared table'):
            AllocationConfig(widths=[32], interface_type='shared', shared_rank=16,
                             output_private_rank=16, skip_output_adapter=True, **base)

    def test_forward_backward_causality_cache_and_roundtrip(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                set_seed(3)
                model = build_model(experiment_config(arm, tiny=True))
                parameter_report(model)
                x = torch.randint(0, 96, (2, 9))
                x[:, 4] = 96  # EOS remains a scored target.
                out = model(x, labels=x)
                explicit = torch.nn.functional.cross_entropy(
                    out.logits[:, :-1].reshape(-1, 97), x[:, 1:].reshape(-1))
                torch.testing.assert_close(out.loss, explicit)
                out.loss.backward()
                for name, p in model.named_parameters():
                    self.assertIsNotNone(p.grad, name)
                    self.assertTrue(torch.isfinite(p.grad).all(), name)
                    self.assertGreater(p.grad.abs().sum().item(), 0, name)
                shared = model.get_input_embeddings().weight is model.get_output_embeddings().weight
                self.assertEqual(shared, arm == "B0" or arm in SHARED_ARMS)
                model.eval()
                with torch.no_grad():
                    full = model(x).logits
                    changed = x.clone()
                    changed[:, 5:] = (changed[:, 5:] + 1) % 97
                    torch.testing.assert_close(full[:, :5], model(changed).logits[:, :5])
                    cache, parts = None, []
                    for i in range(x.shape[1]):
                        step = model(x[:, i:i+1], past_key_values=cache, use_cache=True)
                        cache = step.past_key_values
                        parts.append(step.logits)
                    torch.testing.assert_close(full, torch.cat(parts, dim=1), atol=2e-6, rtol=2e-5)
                    widths = (model.config.compute_widths
                              if getattr(model.config, 'body_type', None) == 'fixed_residual'
                              else getattr(model.config, "widths", [model.config.hidden_size]*6))
                    for i, width in enumerate(widths):
                        self.assertEqual(tuple(cache.layers[i].keys.shape),
                                         (2, width//model.config.head_dim, 9, model.config.head_dim))
                    generated = model.generate(x, attention_mask=torch.ones_like(x),
                                               max_new_tokens=2, min_new_tokens=2,
                                               do_sample=False, use_cache=True,
                                               pad_token_id=96)
                    self.assertEqual(tuple(generated.shape), (2, 11))
                scales = activation_report(model, x)
                self.assertTrue(all(math.isfinite(v) and v > 0 for v in scales.values()))
                with tempfile.TemporaryDirectory() as path:
                    model.save_pretrained(path)
                    loaded = AutoModelForCausalLM.from_pretrained(path, local_files_only=True).eval()
                    with torch.no_grad():
                        torch.testing.assert_close(full, loaded(x).logits, rtol=0, atol=0)
                    self.assertEqual(loaded.get_input_embeddings().weight is
                                     loaded.get_output_embeddings().weight,
                                     arm == "B0" or arm in SHARED_ARMS)

    def test_new_variant_topology_and_partial_logits(self):
        tied = build_model(experiment_config('T768', tiny=True))
        self.assertIs(tied.model.embed_tokens.shared.weight, tied.lm_head.shared_head.weight)
        self.assertEqual(tied.model.embed_tokens.projection.in_features, 24)
        self.assertEqual(tied.lm_head.shared_projection.out_features, 24)

        partial = build_model(experiment_config('P512-128-384', tiny=True)).eval()
        self.assertIs(partial.model.embed_tokens.shared.weight,
                      partial.lm_head.shared_head.weight)
        self.assertIsNotNone(partial.model.embed_tokens.input_private)
        self.assertIsNotNone(partial.lm_head.output_private_head)
        hidden = torch.randn(2, 3, 32)
        expected = (torch.nn.functional.linear(
                        partial.lm_head.shared_projection(hidden),
                        partial.model.embed_tokens.shared.weight) +
                    torch.nn.functional.linear(
                        partial.lm_head.output_private_projection(hidden),
                        partial.lm_head.output_private_head.weight))
        torch.testing.assert_close(partial.lm_head(hidden), expected)

        direct = build_model(experiment_config('A768-Direct', tiny=True))
        self.assertIsInstance(direct.model.embed_tokens.projection, torch.nn.Identity)
        self.assertIsInstance(direct.model.layers[0], DirectWideningDecoderLayer)
        self.assertEqual(direct.model.layers[0].hidden_size, 24)
        self.assertEqual(direct.model.layers[0].mlp.down_proj.out_features, 32)
        self.assertEqual(len(direct.model.layers), 6)

        fixed = build_model(experiment_config('FixedResidual', tiny=True))
        self.assertIsInstance(fixed.model.embed_tokens.projection, torch.nn.Identity)
        self.assertIsInstance(fixed.lm_head.shared_projection, torch.nn.Identity)
        self.assertEqual(fixed.config.widths, [32] * 6)
        self.assertEqual(fixed.config.compute_widths, [16, 16, 24, 24, 32, 32])
        self.assertIsInstance(fixed.model.layers[0], FixedResidualDecoderLayer)
        self.assertEqual(fixed.model.layers[0].down.in_features, 32)
        self.assertEqual(fixed.model.layers[0].down.out_features, 16)

    def test_direct_and_fixed_residual_equations(self):
        direct = build_model(experiment_config('A768-Direct', tiny=True)).eval()
        layer = direct.model.layers[0]
        x = torch.randn(2, 5, 24)
        position_ids = torch.arange(5).unsqueeze(0)
        positions = direct.model.rotary_emb(x, position_ids)
        actual = layer(x, position_embeddings=positions)
        attention, _ = layer.self_attn(
            hidden_states=layer.input_layernorm(x), attention_mask=None,
            position_ids=position_ids, past_key_values=None, use_cache=False,
            position_embeddings=positions)
        residual = x + attention
        expected = torch.nn.functional.pad(residual, (0, 8)) + layer.mlp(
            layer.post_attention_layernorm(residual))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        fixed = build_model(experiment_config('FixedResidual', tiny=True)).eval()
        layer = fixed.model.layers[0]
        x = torch.randn(2, 5, 32)
        positions = fixed.model.rotary_emb(x, position_ids)
        actual = layer(x, position_embeddings=positions)
        narrow = layer.down(x)
        updated = Qwen3DecoderLayer.forward(layer, narrow,
                                            position_ids=position_ids,
                                            position_embeddings=positions)
        torch.testing.assert_close(actual, x + layer.up(updated - narrow), rtol=0, atol=0)
        full = fixed.model.layers[-1]
        expected = Qwen3DecoderLayer.forward(full, x, position_ids=position_ids,
                                             position_embeddings=positions)
        torch.testing.assert_close(full(x, position_ids=position_ids,
                                        position_embeddings=positions), expected,
                                   rtol=0, atol=0)

    def test_gradient_checkpointing(self):
        for arm in ("B0", "A128", "C", "D", "T768", "P512-128-384",
                    "A768-Direct", "FixedResidual", "WNW", "O1280", "C-Direct"):
            with self.subTest(arm=arm):
                model = build_model(experiment_config(arm, tiny=True))
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                model.train()
                x = torch.randint(0, 97, (2, 8))
                model(x, labels=x).loss.backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                    for p in model.parameters()))

    def test_initialization(self):
        set_seed(41)
        model = build_model(experiment_config("D", tiny=True))
        for module in model.modules():
            if isinstance(module, FanInLinear):
                # Tiny input maps can have only 64 samples. Allow sampling
                # variance, while strongly rejecting accidental std=0.02.
                self.assertAlmostEqual(module.weight.std().item() * math.sqrt(module.in_features), 1, delta=.3)
            if isinstance(module, ScaledVocabularyHead):
                target = .02 * math.sqrt(module.reference_width/module.in_features)
                self.assertAlmostEqual(module.weight.std().item()/target, 1, delta=.1)
        tied = build_model(experiment_config('T768', tiny=True))
        projection = tied.lm_head.shared_projection
        self.assertIsInstance(projection, ScaledFanInLinear)
        target = math.sqrt(32/24) / math.sqrt(32)
        self.assertAlmostEqual(projection.weight.std().item()/target, 1, delta=.1)


if __name__ == "__main__":
    unittest.main()
