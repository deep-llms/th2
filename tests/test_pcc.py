"""CPU-only PCC checks using an actual tiny Qwen3, no downloads or GPU queries."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
    import transformers
    HAS_ML = np is not None
except ImportError:
    HAS_ML = False

from pcc.protocol import PAIRS, REVISION, validate_config
if np is not None:
    from pcc.statistics import paired_bootstrap, select_pair

if HAS_ML:
    from pcc.diagnostics import tiny_backbone, run_preflight, paired_adapters
    from pcc.model import Context, CorrectionAdapter, load_local
    from pcc.objectives import correction_scale, alignment_loss, learning_rate, optimizer
    from pcc.data import PreparedContexts, save_contexts, validate_matched_data
    from pcc.protocol import MODEL_ID, DATA_SEED


@unittest.skipIf(np is None, "PCC statistics require numpy in the project environment")
class StatisticsTests(unittest.TestCase):
    def test_token_weighted_and_paired(self):
        counts = np.array([1, 99])
        result = paired_bootstrap({"a": [2., 99.], "b": [3., 198.]}, counts, resamples=50)
        self.assertEqual(result["nll"]["a"], 1.01)
        np.testing.assert_allclose(result["samples"]["b"] - result["samples"]["a"], 1.)
        again = paired_bootstrap({"a": [2., 99.], "b": [3., 198.]}, counts, resamples=50)
        np.testing.assert_array_equal(result["samples"]["a"], again["samples"]["a"])

    def test_negative_screen_and_tie_break(self):
        counts = np.array([4, 8, 3])
        losses = {"Base": counts * 3.}
        for s, d in PAIRS:
            losses[f"deep-{s}-{d}"] = counts * 3.
            losses[f"shallow-{s}-{d}"] = counts * 3.
        self.assertIsNone(select_pair(losses, counts)["selected_pair"])
        for s, d in PAIRS:
            losses[f"deep-{s}-{d}"] = counts * 2.
        result = select_pair(losses, counts)
        self.assertEqual(result["selected_pair"], [4, 16])
        self.assertFalse(result["test_unlocked"])
        losses["deep-8-24"] = counts * 1.5
        self.assertEqual(select_pair(losses, counts)["selected_pair"], [8, 24])

    def test_bad_records_fail(self):
        for values in ([1., np.nan], [1.], [1., np.inf]):
            with self.assertRaises(ValueError):
                paired_bootstrap({"a": values}, [1, 2])
        with self.assertRaises(ValueError):
            paired_bootstrap({"a": [1, 2]}, [1, 0])
        with self.assertRaises(ValueError):
            select_pair({"Base": [1, 2]}, [1, 2])
        for bad_counts in ([1.5, 2.], [True, True]):
            with self.assertRaises(ValueError):
                paired_bootstrap({"a": [1, 2]}, bad_counts)

    def test_both_screen_gates_and_uncertainty_are_required(self):
        counts = np.ones(6, dtype=np.int64)
        for base, shallow, deep in ((3., 1., np.full(6, 2.)),
                                    (1., 3., np.full(6, 2.)),
                                    (3., 3., np.array([1., 1., 1., 1., 5., 5.]))):
            losses = {"Base": counts * base}
            for s, d in PAIRS:
                losses[f"deep-{s}-{d}"] = deep
                losses[f"shallow-{s}-{d}"] = counts * shallow
            self.assertIsNone(select_pair(losses, counts)["selected_pair"])


@unittest.skipUnless(HAS_ML, "PCC tests require the project ML environment")
class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.backbone = tiny_backbone()

    def context(self, length=12):
        ids = torch.arange(length).unsqueeze(0) % 67
        return Context(ids, torch.ones_like(ids, dtype=torch.bool), torch.arange(length).unsqueeze(0))

    def test_complete_preflight(self):
        report = run_preflight(self.backbone)
        self.assertEqual(report["status"], "ok")
        self.assertFalse(report["scientific_gain_established"])

    def test_bf16_preflight(self):
        model = tiny_backbone()
        model.model.to(torch.bfloat16)
        self.assertEqual(run_preflight(model)["status"], "ok")

    def test_preflight_rejects_detached_lm_path(self):
        model = tiny_backbone()
        real_tail = model.tail
        with patch.object(model, "tail", side_effect=lambda *a, **k: real_tail(*a, **k).detach()):
            with self.assertRaisesRegex(RuntimeError, "LM loss is detached"):
                run_preflight(model)

    def test_preflight_rejects_small_future_leak(self):
        real_forward = CorrectionAdapter.forward
        def leaking(adapter, shallow, source, allowed, rotary, **kwargs):
            result = real_forward(adapter, shallow, source, allowed, rotary, **kwargs)
            if torch.count_nonzero(adapter.out.weight) == 0:
                return result
            mask = allowed & torch.ones(shallow.shape[1], shallow.shape[1], dtype=torch.bool).tril(-1)
            leak = shallow.roll(-1, dims=1) * 1e-4 * mask[:, 0].any(-1)[..., None]
            return (result[0] + leak, result[1]) if isinstance(result, tuple) else result + leak
        with patch.object(CorrectionAdapter, "forward", leaking):
            with self.assertRaises(AssertionError):
                run_preflight(tiny_backbone())

    def test_preflight_rejects_cross_segment_visibility(self):
        original = Context.allowed
        def ignores_segments(context):
            return original(Context(context.input_ids, context.valid, context.position_ids))
        with patch.object(Context, "allowed", ignores_segments):
            with self.assertRaises(AssertionError):
                run_preflight(tiny_backbone())

    def test_nonzero_tail_matches_full_forward_intervention_and_gradients(self):
        context = self.context()
        context.segments = torch.tensor([[0] * 6 + [1] * 6])
        context.position_ids[:, 6:] -= 6
        clean = self.backbone.clean(context, (4,))
        delta = torch.randn_like(clean.states[4]) * 0.02
        direct_delta = delta.clone().requires_grad_()
        recompute_delta = delta.clone().requires_grad_()
        full_delta = delta.clone().requires_grad_()
        direct = self.backbone.tail(clean, 4, direct_delta)
        recompute = self.backbone.tail(clean, 4, recompute_delta, checkpoint_layers=True)
        handle = self.backbone.model.model.layers[3].register_forward_hook(lambda m, a, out: out + full_delta)
        try:
            full = self.backbone.model.model(input_ids=context.input_ids,
                attention_mask=context.additive_mask(torch.float32), position_ids=context.position_ids,
                use_cache=False).last_hidden_state
        finally:
            handle.remove()
        for hidden in (direct, recompute, full):
            sums, counts = self.backbone.losses(hidden, context, chunk_size=5)
            (sums.sum() / counts.sum()).backward()
        torch.testing.assert_close(direct, full, atol=0, rtol=0)
        torch.testing.assert_close(recompute, full, atol=0, rtol=0)
        torch.testing.assert_close(direct_delta.grad, full_delta.grad)
        torch.testing.assert_close(recompute_delta.grad, full_delta.grad)

    def test_rotary_uses_explicit_positions(self):
        context = self.context()
        context.position_ids = context.position_ids + 17
        clean = self.backbone.clean(context, (4, 20))
        # Independent analytic default-Qwen RoPE reference (non-interleaved halves).
        inv_freq = 1 / (1_000_000 ** (torch.arange(0, 128, 2).float() / 128))
        angles = context.position_ids.float()[..., None] * inv_freq
        angles = torch.cat([angles, angles], dim=-1)
        torch.testing.assert_close(clean.rotary[0], angles.cos())
        torch.testing.assert_close(clean.rotary[1], angles.sin())

    def test_bf16_attention_matches_fp32_score_reference(self):
        model = tiny_backbone()
        model.model.to(torch.bfloat16)
        context = self.context()
        context.position_ids += 13
        clean = model.clean(context, (4, 20))
        adapter, _ = paired_adapters(model)
        captured = {}
        handles = [projection.register_forward_hook(
            lambda module, args, output, key=key: captured.update({key: output.detach()}))
            for key, projection in (("q", adapter.q), ("k", adapter.k))]
        try:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                _, diag = adapter(clean.states[4], clean.states[20], context.allowed(), clean.rotary,
                                  diagnostics=True)
        finally:
            for handle in handles:
                handle.remove()
        cos, sin = (x[:, None] for x in clean.rotary)
        def rotate(value):
            value = value.reshape(1, 12, 2, 128).transpose(1, 2)
            half = torch.cat([-value[..., 64:], value[..., :64]], dim=-1)
            return value * cos + half * sin
        q, k = rotate(captured["q"]), rotate(captured["k"])
        scores = q.float() @ k.float().transpose(-1, -2) / (128 ** 0.5)
        mask = context.allowed() & torch.ones(12, 12, dtype=torch.bool).tril(-1)
        scores.masked_fill_(~mask, -torch.inf)
        scores[:, :, 0] = 0
        expected = scores.softmax(-1).masked_fill(~mask, 0).to(torch.bfloat16)
        torch.testing.assert_close(diag["attention"], expected, atol=0, rtol=0)

    def test_clean_sources_and_tail_only(self):
        context = self.context()
        clean = self.backbone.clean(context, (4, 20))
        saved = clean.states[20].clone()
        calls = []
        handles = [layer.register_forward_hook(lambda m, a, o, i=i: calls.append(i))
                   for i, layer in enumerate(self.backbone.model.model.layers)]
        try:
            self.backbone.tail(clean, 4, torch.ones_like(clean.states[4]) * 0.1)
        finally:
            for h in handles:
                h.remove()
        self.assertEqual(calls, list(range(4, 28)))
        torch.testing.assert_close(saved, clean.states[20], atol=0, rtol=0)

    def test_chunked_loss_matches_vanilla_and_gradient(self):
        context = self.context()
        clean = self.backbone.clean(context, (4,))
        a = clean.final_hidden.clone().requires_grad_()
        b = a.detach().clone().requires_grad_()
        sums, counts = self.backbone.losses(a, context, chunk_size=5)
        got = sums.sum() / counts.sum()
        logits = self.backbone.model.lm_head(b)
        expected = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 67), context.input_ids[:, 1:].reshape(-1))
        got.backward()
        expected.backward()
        torch.testing.assert_close(got, expected)
        torch.testing.assert_close(a.grad, b.grad)

    def test_scale_and_teacher_detach(self):
        teacher = torch.randn(2, 8, 32, requires_grad=True)
        student = torch.randn(2, 8, 32, requires_grad=True)
        eligible = torch.ones(2, 7, dtype=torch.bool)
        eligible[1, 4:] = False
        scale = correction_scale(teacher, eligible)
        reference = teacher[:, :-1][eligible].detach().double().square().sum()
        reference = (reference / (eligible.sum() * 32)).sqrt().item()
        self.assertAlmostEqual(scale, reference, places=14)
        alignment_loss(student, teacher, eligible, scale).backward()
        self.assertIsNone(teacher.grad)
        self.assertGreater(student.grad.abs().sum(), 0)
        for bad in (0, 1e-9, float("nan")):
            with self.assertRaises(ValueError):
                alignment_loss(student, teacher, eligible, bad)

    def test_initialization_and_checkpoint_roundtrip(self):
        teacher, student = paired_adapters(self.backbone)
        for key, value in teacher.state_dict().items():
            torch.testing.assert_close(value, student.state_dict()[key], atol=0, rtol=0)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "adapter.pt"
            torch.save(student.state_dict(), path)
            restored = copy.deepcopy(student)
            restored.load_state_dict(torch.load(path, weights_only=True))
            for key, value in restored.state_dict().items():
                torch.testing.assert_close(value, student.state_dict()[key], atol=0, rtol=0)

    def test_pinned_config_and_missing_snapshot(self):
        with self.assertRaises(ValueError):
            validate_config(self.backbone.model.config)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                load_local(folder, "cpu")
            with self.assertRaises(FileNotFoundError):
                load_local(Path(folder) / REVISION, "cpu")

    def test_incomplete_pretrained_load_is_rejected(self):
        from pcc.protocol import CONFIG
        from types import SimpleNamespace
        config = SimpleNamespace(**CONFIG, model_type="qwen3", layer_types=["full_attention"] * 28)
        validate_config(config)
        with tempfile.TemporaryDirectory() as folder:
            snapshot = Path(folder) / REVISION
            snapshot.mkdir()
            for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
                with patch("pcc.model.AutoConfig.from_pretrained", return_value=config), \
                     patch("pcc.model.AutoTokenizer.from_pretrained"), \
                     patch("pcc.model.AutoModelForCausalLM.from_pretrained", return_value=(None, {key: ["bad weight"]})):
                    with self.assertRaisesRegex(ValueError, "did not load exactly"):
                        load_local(snapshot, "cpu")

    def test_optimizer_groups_and_scale_exclusions(self):
        adapter, _ = paired_adapters(self.backbone)
        opt = optimizer(adapter)
        decay = {id(p): group["weight_decay"] for group in opt.param_groups for p in group["params"]}
        self.assertEqual(decay[id(adapter.q.weight)], 0.01)
        self.assertEqual(decay[id(adapter.gate.weight)], 0.)
        self.assertEqual(decay[id(adapter.gate.bias)], 0.)
        self.assertEqual(decay[id(adapter.q_norm.weight)], 0.)
        self.assertFalse(set(decay) & {id(p) for p in self.backbone.model.parameters()})
        values = torch.ones(1, 5, 32)
        eligible = torch.tensor([[True, False, True, False]])
        values[:, 1::2] = 1e6
        values[:, -1] = 1e6
        self.assertEqual(correction_scale(values, eligible), 1.)

    def test_scheduler(self):
        self.assertEqual(learning_rate(7), 3e-4)
        self.assertEqual(learning_rate(128), 3e-5)
        self.assertEqual(learning_rate(31, total=610, warmup=31), 3e-4)
        self.assertEqual(learning_rate(610, total=610, warmup=31), 3e-5)

    def write_contexts(self, path, split, rows, width):
        ids = np.arange(rows * width).reshape(rows, width) % 67
        metadata = dict(split=split, language="en", model_id=MODEL_ID,
                        tokenizer_revision=REVISION, data_order_seed=DATA_SEED,
                        source=f"selected-culturax/{split}", preprocessing="fixture-policy",
                        packing="full_causal")
        save_contexts(path, ids, np.ones_like(ids, dtype=bool),
                      np.tile(np.arange(width), (rows, 1)), metadata)

    def test_prepared_data_budget_policy_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            a, b = Path(folder) / "train.npz", Path(folder) / "dev.npz"
            self.write_contexts(a, "train", 2, 2048)
            self.write_contexts(b, "dev", 2, 2048)
            train, dev = PreparedContexts(a, "train", 4096), PreparedContexts(b, "dev", 4096)
            validate_matched_data(train, dev)
            torch.testing.assert_close(train.batch(0, 1, "cpu").input_ids,
                                       train.batch(0, 2, "cpu").input_ids[:1], atol=0, rtol=0)
            modified = train.batch(0, 1, "cpu")
            modified.input_ids.fill_(0)
            self.assertGreater(int(train.batch(0, 1, "cpu").input_ids.sum()), 0)
            with self.assertRaises(ValueError):
                PreparedContexts(a, "train", 4095)
            with self.assertRaises(ValueError):
                PreparedContexts(a, "dev", 4096)
            with self.assertRaises(FileExistsError):
                self.write_contexts(a, "train", 2, 2048)
            dev.metadata["preprocessing"] = "changed"
            with self.assertRaises(ValueError):
                validate_matched_data(train, dev)

    def test_segment_validation_and_explicit_target_counts(self):
        with tempfile.TemporaryDirectory() as folder, patch("pcc.data.CONTEXT", 6):
            root = Path(folder)
            ids = np.arange(6).reshape(1, 6)
            metadata = dict(split="dev", language="en", model_id=MODEL_ID,
                            tokenizer_revision=REVISION, data_order_seed=DATA_SEED,
                            source="fixed-dev", preprocessing="no-boundary-loss", packing="block_isolated")
            for name, segments in (("reuse", [0, 0, 1, 1, 0, 0]), ("empty", [0, 1, 2, 3, 4, 5]),
                                   ("ok", [0, 0, 0, 1, 1, 1])):
                path = root / (name + ".npz")
                save_contexts(path, ids, np.ones_like(ids, dtype=bool), ids, metadata,
                              np.array([segments]))
                if name != "ok":
                    with self.assertRaises(ValueError):
                        PreparedContexts(path, "dev", 6)
                else:
                    context = PreparedContexts(path, "dev", 6).batch(0, 1, "cpu")
                    torch.testing.assert_close(context.targets(), torch.tensor([[True, True, False, True, True]]))
                    clean = self.backbone.clean(context, (4,))
                    sums, counts = self.backbone.losses(clean.final_hidden, context)
                    logits = self.backbone.model.lm_head(clean.final_hidden)
                    expected = torch.nn.functional.cross_entropy(logits[:, [0, 1, 3, 4]].reshape(-1, 67),
                                                                  context.input_ids[:, [1, 2, 4, 5]].reshape(-1), reduction="sum")
                    self.assertEqual(int(counts.sum()), 4)
                    torch.testing.assert_close(sums.sum(), expected)

    def test_screen_end_to_end_and_failure_artifacts(self):
        import pcc.screen as screen_module
        import json
        with tempfile.TemporaryDirectory() as folder, \
             patch("pcc.data.CONTEXT", 12), \
             patch.multiple(screen_module, CONTEXT=12, TOKENS_PER_UPDATE=24,
                            SCREEN_UPDATES=2, SCREEN_DEV_TOKENS=24):
            root = Path(folder)
            train, dev = root / "train.npz", root / "dev.npz"
            self.write_contexts(train, "train", 4, 12)
            self.write_contexts(dev, "dev", 2, 12)
            backbone = tiny_backbone()
            backbone.model.to(torch.bfloat16)
            decision = screen_module.screen(backbone, train, dev, root / "run")
            self.assertEqual(len(decision["pairs"]), 4)
            self.assertFalse(decision["test_unlocked"])
            self.assertTrue((root / "run" / "complete.json").is_file())
            self.assertEqual(len(list((root / "run").glob("*.pt"))), 8)
            self.assertTrue((root / "run" / "provenance.json").is_file())
            saved = torch.load(root / "run" / "deep-4-16.pt", weights_only=True)
            self.assertEqual(saved["updates"], 2)
            self.assertEqual(saved["input_tokens"], 48)
            self.assertGreater(int(torch.count_nonzero(saved["state_dict"]["out.weight"])), 0)
            restored, _ = paired_adapters(backbone)
            restored.load_state_dict(saved["state_dict"])
            restored.eval()
            dev_contexts = PreparedContexts(dev, "dev", 24)
            reproduced = []
            with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
                for i in range(len(dev_contexts)):
                    context = dev_contexts.batch(i, i + 1, "cpu")
                    clean = backbone.clean(context, (4, 16))
                    delta = restored(clean.states[4], clean.states[16], context.allowed(), clean.rotary)
                    sums, _ = backbone.losses(backbone.tail(clean, 4, delta), context)
                    reproduced.extend(sums.double().tolist())
            with np.load(root / "run" / "eval-deep-4-16.npz") as evaluated:
                np.testing.assert_array_equal(reproduced, evaluated["loss_sums"])
            with self.assertRaises(FileExistsError):
                screen_module.screen(backbone, train, dev, root / "run")
            with patch.object(screen_module, "run_preflight", side_effect=RuntimeError("bad hook")):
                with self.assertRaisesRegex(RuntimeError, "bad hook"):
                    screen_module.screen(backbone, train, dev, root / "failed")
            self.assertFalse((root / "failed" / "complete.json").exists())
            self.assertEqual(json.loads((root / "failed" / "failure.json").read_text())["error"], "bad hook")
            real_write = screen_module.write_json
            def fail_provenance(path, value):
                if Path(path).name == "provenance.json":
                    raise OSError("provenance unavailable")
                real_write(path, value)
            with patch.object(screen_module, "write_json", side_effect=fail_provenance):
                with self.assertRaisesRegex(OSError, "provenance unavailable"):
                    screen_module.screen(backbone, train, dev, root / "no-provenance")
            self.assertFalse((root / "no-provenance" / "complete.json").exists())
            self.assertTrue((root / "no-provenance" / "failure.json").is_file())

    def test_fp32_global_update_normalization_and_paired_inputs(self):
        from pcc.screen import train_update
        with tempfile.TemporaryDirectory() as folder, patch("pcc.data.CONTEXT", 12):
            path = Path(folder) / "train.npz"
            ids = np.arange(24).reshape(2, 12) % 67
            segments = np.array([[0] * 12, [0] * 3 + [1] * 9])
            metadata = dict(split="train", language="en", model_id=MODEL_ID,
                            tokenizer_revision=REVISION, data_order_seed=DATA_SEED,
                            source="fixed-train", preprocessing="no-boundary-loss", packing="block_isolated")
            save_contexts(path, ids, np.ones_like(ids, dtype=bool), np.tile(np.arange(12), (2, 1)), metadata, segments)
            data = PreparedContexts(path, "train", 24)
            # The examples have different target counts (11 and 10), so averaging
            # microbatch means incorrectly would fail against the full-batch result.
            states = []
            for microbatch in (1, 2):
                deep, shallow = paired_adapters(self.backbone)
                adapters = {"deep": (deep, 20), "shallow": (shallow, 4)}
                opts = {name: optimizer(adapter) for name, (adapter, _) in adapters.items()}
                seen = {name: [] for name in adapters}
                def capture(name):
                    def hook(module, args):
                        seen[name].append(args[0].detach().clone())
                    return hook
                handles = [adapter.register_forward_pre_hook(capture(name)) for name, (adapter, _) in adapters.items()]
                try:
                    for update in (1, 2):
                        records = train_update(self.backbone, data, adapters, opts, 4, 20, 0, 2, microbatch, update)
                        self.assertTrue(all(r["target_tokens"] == 21 for r in records))
                finally:
                    for h in handles:
                        h.remove()
                torch.testing.assert_close(torch.cat(seen["deep"]), torch.cat(seen["shallow"]), atol=0, rtol=0)
                states.append({name: copy.deepcopy(adapter.state_dict()) for name, (adapter, _) in adapters.items()})
            for name in states[0]:
                for key in states[0][name]:
                    torch.testing.assert_close(states[0][name][key], states[1][name][key], atol=2e-7, rtol=1e-5)

    def test_json_publication_rejects_partial_or_overwritten_results(self):
        from pcc.screen import write_json
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "complete.json"
            with self.assertRaises(ValueError):
                write_json(path, {"bad": float("nan")})
            self.assertFalse(path.exists())
            write_json(path, {"ok": True})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                write_json(path, {"ok": False})
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(folder).glob("*.part")), [])


if __name__ == "__main__":
    unittest.main()
