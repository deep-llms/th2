import math
import tempfile
import unittest

import torch
from transformers import AutoModelForCausalLM, Qwen3ForCausalLM, set_seed

from capacity_allocation.modeling import (
    ARMS, EXPECTED_COUNTS, FanInLinear, ScaledVocabularyHead, activation_report,
    build_model, experiment_config, parameter_report,
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
                tied = model.get_input_embeddings().weight is model.get_output_embeddings().weight
                self.assertEqual(tied, arm == "B0")
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
                    widths = getattr(model.config, "widths", [model.config.hidden_size]*6)
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
                                     loaded.get_output_embeddings().weight, arm == "B0")

    def test_gradient_checkpointing(self):
        for arm in ("B0", "A128", "C", "D"):
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


if __name__ == "__main__":
    unittest.main()
