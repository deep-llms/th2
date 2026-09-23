"""CPU-only full-probe acceptance tests; synthetic models/data are not research evidence."""
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_pcc import HAS_ML

if HAS_ML:
    import numpy as np
    import torch
    from torch.nn import functional as F
    from pcc.data import PreparedContexts, save_contexts
    from pcc.diagnostics import tiny_backbone, paired_adapters, fingerprint
    from pcc.model import Context
    from pcc.objectives import optimizer
    from pcc.permutation import permute_targets
    from pcc.probe import probe, freeze, screen_evidence, verify_slice
    from pcc.protocol import MODEL_ID, REVISION, DATA_SEED, FULL_SEED, PAIRS
    from pcc.screen import write_json
    from pcc.statistics import probe_decision, select_pair
    from pcc.training import calibrate_scale, prefix_batches, student_update
    from pcc.evaluation import evaluate


class UntouchableTestPath:
    def __fspath__(self):
        raise AssertionError("Test data accessed before gates passed")


@unittest.skipUnless(HAS_ML, "PCC tests require the project ML environment")
class ProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def data(self, root, split="train", rows=4, name=None, segments=None):
        ids = np.tile(np.arange(12), (rows, 1)).astype(np.int64)
        metadata = {"split": split, "language": "en", "model_id": MODEL_ID,
            "tokenizer_revision": REVISION, "data_order_seed": DATA_SEED,
            "source": f"synthetic-{split}", "preprocessing": "synthetic-only",
            "packing": "block_isolated" if segments is not None else "full_causal"}
        path = root / (name or f"{split}.npz")
        save_contexts(path, ids, np.ones_like(ids, dtype=bool), ids, metadata, segments=segments)
        with patch("pcc.data.CONTEXT", 12):
            return PreparedContexts(path, split, rows * 12)

    def teacher(self, backbone):
        teacher, _ = paired_adapters(backbone, seed=FULL_SEED)
        with torch.no_grad():
            teacher.out.weight.copy_(torch.linspace(-.03, .02, teacher.out.weight.numel()).reshape_as(teacher.out.weight))
        return freeze(teacher)

    def test_dev_gates_conditional_control_and_confirmatory_budget(self):
        counts = np.array([3, 7, 11, 8])
        levels = {"Base": 3., "Privileged-Deep": 2., "Shallow-ExtraAttn": 2.9, "Student-PCC": 2.5}
        losses = {name: counts * value for name, value in levels.items()}
        initial = probe_decision(losses, counts)
        self.assertTrue(initial["conditional_control_required"])
        self.assertFalse(initial["all_dev_gates_pass"])
        self.assertEqual(initial["recovery_ratio"], .5)
        self.assertEqual(initial["bootstrap_resamples"], 2000)
        with self.assertRaisesRegex(ValueError, "five frozen arms"):
            probe_decision(losses, counts, split="test")
        losses["Target-Permuted"] = counts * 2.8
        self.assertTrue(probe_decision(losses, counts)["all_dev_gates_pass"])
        confirm = probe_decision(losses, counts, split="test")
        self.assertEqual(confirm["bootstrap_resamples"], 5000)
        self.assertTrue(confirm["directionally_consistent"])
        losses["Target-Permuted"] = counts * 2.4
        self.assertFalse(probe_decision(losses, counts)["all_dev_gates_pass"])
        self.assertFalse(probe_decision(losses, counts, split="test")["directionally_consistent"])

    def test_other_failed_gate_does_not_skip_conditional_control(self):
        counts = np.array([2, 5, 7])
        # PCC/distillation pass despite failed teacher, source, and recovery gates.
        losses = {name: counts * value for name, value in {
            "Base": 3., "Privileged-Deep": 3.1, "Shallow-ExtraAttn": 2.9, "Student-PCC": 2.5}.items()}
        result = probe_decision(losses, counts)
        self.assertTrue(result["conditional_control_required"])
        self.assertFalse(result["all_dev_gates_pass"])
        self.assertIsNone(result["recovery_ratio"])
        losses["Privileged-Deep"] = counts * 2.9999
        result = probe_decision(losses, counts)
        self.assertFalse(result["gates"]["privileged"])
        losses["Privileged-Deep"] = counts * 1.
        losses["Student-PCC"] = counts * 2.8
        self.assertFalse(probe_decision(losses, counts)["gates"]["recovery"])

    def test_permutation_deranges_preserves_buckets_and_is_reproducible(self):
        ids = torch.arange(12).repeat(8, 1)
        valid = torch.ones_like(ids, dtype=torch.bool)
        valid[:, -2:] = False
        context = Context(ids, valid, ids)
        targets = torch.zeros(8, 12, 8, requires_grad=True)
        with torch.no_grad():
            for row in range(8):
                targets[row, :, row] = torch.arange(1, 13)
        permuted, audit = permute_targets(targets, context, 1)
        again, duplicate = permute_targets(targets, context, 1)
        self.assertFalse(permuted.requires_grad)
        self.assertTrue(torch.equal(permuted, again))
        mapping, buckets = audit["mapping"], audit["buckets"]
        self.assertTrue((mapping != np.arange(len(mapping))).all())
        np.testing.assert_array_equal(np.sort(mapping), np.arange(len(mapping)))
        np.testing.assert_array_equal(buckets, buckets[mapping])
        np.testing.assert_array_equal(mapping, duplicate["mapping"])
        self.assertFalse(np.array_equal(mapping, permute_targets(targets, context, 2)[1]["mapping"]))
        eligible = context.targets()
        self.assertTrue(torch.equal(permuted[:, :-1][eligible], targets[:, :-1][eligible][mapping]))
        self.assertTrue(torch.equal(permuted[:, :-1][~eligible], targets[:, :-1][~eligible]))
        self.assertTrue(torch.equal(permuted[:, -1], targets[:, -1]))
        self.assertFalse(torch.equal(permuted, targets))

    def test_singleton_bucket_fails_instead_of_preserving_target(self):
        ids = torch.arange(12).unsqueeze(0)
        context = Context(ids, torch.ones_like(ids, dtype=torch.bool), ids)
        with self.assertRaisesRegex(ValueError, "singleton"):
            permute_targets(torch.arange(12.).view(1, 12, 1), context, 1)

    def test_exact_calibration_prefix_and_fp64_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            data = self.data(Path(temp))
            backbone = tiny_backbone()
            teacher = self.teacher(backbone)
            before = fingerprint(teacher)
            contexts = list(prefix_batches(data, 25, 2, "cpu"))
            self.assertEqual([int(c.valid.sum()) for c in contexts], [24, 1])
            calibrated = calibrate_scale(backbone, teacher, data, 4, 16, 25, 2)
            vectors = []
            with torch.no_grad():
                for context in contexts:
                    clean = backbone.clean(context, (4, 16))
                    correction = teacher(clean.states[4], clean.states[16], context.allowed(), clean.rotary)
                    vectors.append(correction[:, :-1][context.targets()].double())
            reference = torch.cat(vectors).square().mean().sqrt().item()
            self.assertAlmostEqual(calibrated["sigma_delta"], reference, places=14)
            self.assertEqual(calibrated["input_tokens"], 25)
            self.assertEqual(calibrated["target_tokens"], 22)
            self.assertEqual(fingerprint(teacher), before)
            self.assertTrue(all(p.grad is None for p in teacher.parameters()))
            zero, _ = paired_adapters(backbone)
            with self.assertRaisesRegex(ValueError, "usable"):
                calibrate_scale(backbone, freeze(zero), data, 4, 16, 25, 2)
            teacher.train()
            with self.assertRaisesRegex(ValueError, "eval mode"):
                calibrate_scale(backbone, teacher, data, 4, 16, 25, 2)

    def test_one_pass_matches_tail_and_visits_each_block_once(self):
        for dtype in (torch.float32, torch.bfloat16):
            backbone = tiny_backbone()
            backbone.model.to(dtype)
            ids = torch.arange(12).repeat(2, 1)
            segments = torch.zeros_like(ids)
            segments[1, 6:] = 1
            context = Context(ids, torch.ones_like(ids, dtype=torch.bool), ids, segments)
            for nonzero in (False, True):
                adapter, _ = paired_adapters(backbone)
                if nonzero:
                    with torch.no_grad():
                        adapter.out.weight.fill_(.02)
                copy_adapter = copy.deepcopy(adapter)
                calls = [0] * 28
                def counter(i):
                    def hook(*args):
                        calls[i] += 1
                    return hook
                handles = [block.register_forward_hook(counter(i)) for i, block in enumerate(backbone.model.model.layers)]
                with torch.autocast("cpu", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
                    actual, _, _ = backbone.student(context, 4, adapter)
                    for handle in handles:
                        handle.remove()
                    self.assertEqual(calls, [1] * 28)
                    clean = backbone.clean(context, (4,))
                    delta = copy_adapter(clean.states[4], clean.states[4], context.allowed(), clean.rotary)
                    expected = backbone.tail(clean, 4, delta, checkpoint_layers=True)
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    backbone.losses(actual, context)[0].sum().backward()
                    backbone.losses(expected, context)[0].sum().backward()
                for a, b in zip(adapter.parameters(), copy_adapter.parameters()):
                    torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0)
                self.assertTrue(all(p.grad is None for p in backbone.model.parameters()))

    def test_student_update_global_normalization_and_gradient_routes(self):
        with tempfile.TemporaryDirectory() as temp:
            segments = np.zeros((2, 12), dtype=np.int64)
            segments[1, 6:] = 1
            data = self.data(Path(temp), rows=2, segments=segments)
            backbone = tiny_backbone()
            teacher = self.teacher(backbone)
            frozen_hashes = fingerprint(backbone.model), fingerprint(teacher)
            initial, _ = paired_adapters(backbone, seed=FULL_SEED)
            results = []
            for microbatch in (1, 2):
                arms = {name: copy.deepcopy(initial) for name in ("Shallow-ExtraAttn", "Student-PCC")}
                opts = {name: optimizer(arm) for name, arm in arms.items()}
                records, audit = student_update(backbone, teacher, data, arms, opts, 4, 16, 0, 2,
                    microbatch, 1, .01, total_updates=2, warmup=1)
                self.assertIsNone(audit)
                self.assertEqual({r["target_tokens"] for r in records}, {21})
                self.assertEqual({r["sigma_delta"] for r in records}, {.01})
                self.assertGreater(records[1]["correction_loss"], 0)
                self.assertEqual(records[0]["correction_loss"], 0)
                results.append(arms)
            for name in results[0]:
                for a, b in zip(results[0][name].parameters(), results[1][name].parameters()):
                    torch.testing.assert_close(a, b, atol=2e-7, rtol=1e-5)
            # Independent full-batch expression for the documented two-term objective.
            context = data.batch(0, 2, "cpu")
            clean = backbone.clean(context, (4, 16))
            target = teacher(clean.states[4], clean.states[16], context.allowed(), clean.rotary)
            for name in results[1]:
                reference = copy.deepcopy(initial)
                opt = optimizer(reference)
                delta = reference(clean.states[4], clean.states[4], context.allowed(), clean.rotary)
                loss = backbone.losses(backbone.tail(clean, 4, delta), context)[0].sum() / 21
                if name == "Student-PCC":
                    eligible = context.targets()
                    loss = loss + F.smooth_l1_loss(delta[:, :-1][eligible] / .01,
                                                  target[:, :-1][eligible] / .01, beta=1.)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.)
                opt.step()
                for a, b in zip(reference.parameters(), results[1][name].parameters()):
                    torch.testing.assert_close(a, b, atol=2e-7, rtol=1e-5)
            self.assertEqual(frozen_hashes, (fingerprint(backbone.model), fingerprint(teacher)))
            self.assertTrue(all(p.grad is None for p in teacher.parameters()))
            self.assertNotEqual(fingerprint(results[0]["Student-PCC"]), fingerprint(results[0]["Shallow-ExtraAttn"]))

    def test_screen_evidence_recomputes_selection_without_loading_checkpoints(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            backbone = tiny_backbone()
            counts = np.array([11, 11])
            losses = {"Base": counts * 3.}
            for s, d in PAIRS:
                losses[f"deep-{s}-{d}"] = counts * 2.
                losses[f"shallow-{s}-{d}"] = counts * 2.5
            for name, values in losses.items():
                np.savez(root / f"eval-{name}.npz", loss_sums=values, target_counts=counts, sequence_indices=np.arange(2))
            write_json(root / "provenance.json", {"model_id": MODEL_ID, "model_revision": REVISION})
            write_json(root / "preflight.json", {"status": "ok", "checks": {"synthetic": {"passed": True}},
                "backbone_sha256_before_and_after": fingerprint(backbone.model)})
            write_json(root / "complete.json", {"status": "ok"})
            write_json(root / "data.json", {})
            decision = select_pair(losses, counts)
            write_json(root / "decision.json", decision)
            # No .pt exists. Selection needs loss evidence only.
            with patch("torch.load", side_effect=AssertionError("Must not load a screened checkpoint")):
                self.assertEqual(screen_evidence(root, backbone)[0]["selected_pair"], [4, 16])
            decision["selected_pair"] = [8, 24]
            (root / "decision.json").write_text(json.dumps(decision))
            with self.assertRaisesRegex(ValueError, "disagrees"):
                screen_evidence(root, backbone)

    def setup_probe(self, root, stack):
        backbone = tiny_backbone()
        backbone.model.to(torch.bfloat16)
        train = self.data(root)
        screen_train = self.data(root, rows=2, name="screen-train.npz")
        dev = self.data(root, "dev", rows=2)
        data_record = {"train_path": str(screen_train.path), "dev_path": str(dev.path),
                       "train": screen_train.metadata, "dev": dev.metadata}
        selected = {"selected_pair": [4, 16], "decision": "proceed_to_full_probe"}
        stack.enter_context(patch("pcc.probe.screen_evidence", return_value=(selected, data_record)))
        for target, value in {"pcc.data.CONTEXT": 12, "pcc.probe.CONTEXT": 12,
            "pcc.probe.TOKENS_PER_UPDATE": 24, "pcc.probe.FULL_UPDATES": 2,
            "pcc.probe.FULL_WARMUP": 1, "pcc.probe.SCREEN_UPDATES": 1,
            "pcc.probe.SCREEN_DEV_TOKENS": 24, "pcc.probe.PROBE_EVAL_TOKENS": 24,
            "pcc.probe.CALIBRATION_TOKENS": 25}.items():
            stack.enter_context(patch(target, value))
        return backbone, train, dev

    def test_actual_tiny_full_training_evaluation_and_freshness(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp)
            backbone, train, dev = self.setup_probe(root, stack)
            backbone_before = fingerprint(backbone.model)
            steps = []
            def tracked_optimizer(adapter):
                opt = optimizer(adapter)
                steps.append(stack.enter_context(patch.object(opt, "step", wraps=opt.step)))
                return opt
            stack.enter_context(patch("pcc.probe.optimizer", side_effect=tracked_optimizer))
            # A failing primary gate is injected solely to guarantee this test never reads test data.
            from pcc.statistics import probe_decision as real_decision
            def negative_decision(*args, **kwargs):
                decision = real_decision(*args, **kwargs)
                decision.update(conditional_control_required=False, all_dev_gates_pass=False)
                return decision
            stack.enter_context(patch("pcc.probe.probe_decision", side_effect=negative_decision))
            out = root / "run"
            with patch("torch.load", side_effect=AssertionError("Full run must start fresh")):
                result = probe(backbone, root / "screen", train.path, dev.path, UntouchableTestPath(), out)
            self.assertEqual(result["decision"], "stop_dev_gates")
            self.assertFalse((out / "test-started.json").exists())
            self.assertTrue((out / "complete.json").exists())
            # Actual optimizer invocations, not just budget labels in artifacts:
            # teacher first, followed by the matched shallow and PCC optimizers.
            self.assertEqual([step.call_count for step in steps], [2, 2, 2])
            self.assertEqual(fingerprint(backbone.model), backbone_before)
            initial = torch.load(out / "initial.pt", weights_only=True)
            expected, _ = paired_adapters(backbone, FULL_SEED)
            screen_initial, _ = paired_adapters(backbone)
            self.assertNotEqual(fingerprint(expected), fingerprint(screen_initial))
            for key, value in expected.state_dict().items():
                self.assertTrue(torch.equal(value, initial["state_dict"][key]))
            teacher, _ = paired_adapters(backbone)
            teacher.load_state_dict(torch.load(out / "Privileged-Deep.pt", weights_only=True)["state_dict"])
            self.assertNotEqual(fingerprint(teacher), fingerprint(expected))
            freeze(teacher)
            students = {}
            for name in ("Shallow-ExtraAttn", "Student-PCC"):
                adapter, _ = paired_adapters(backbone)
                payload = torch.load(out / f"{name}.pt", weights_only=True)
                self.assertEqual(payload["init_seed"], FULL_SEED)
                self.assertEqual(payload["input_tokens"], 48)
                self.assertEqual(payload["updates"], 2)
                adapter.load_state_dict(payload["state_dict"])
                self.assertNotEqual(fingerprint(adapter), fingerprint(expected))
                students[name] = freeze(adapter)
            for filename, names in (("train-teacher.jsonl", ("Privileged-Deep",)),
                                    ("train-students.jsonl", ("Shallow-ExtraAttn", "Student-PCC"))):
                records = [json.loads(line) for line in (out / filename).read_text().splitlines()]
                for name in names:
                    matched = [record for record in records if record["arm"] == name]
                    self.assertEqual([record["update"] for record in matched], [1, 2])
                    self.assertEqual([record["input_tokens"] for record in matched], [24, 48])
            repeated, counts = evaluate(backbone, teacher, students, dev, 4, 16, root / "repeat")
            for name, values in repeated.items():
                with np.load(out / "dev" / f"{name}.npz") as saved:
                    np.testing.assert_array_equal(values, saved["loss_sums"])
                    np.testing.assert_array_equal(counts, saved["target_counts"])
            with self.assertRaises(FileExistsError):
                probe(backbone, "unused", train.path, dev.path, UntouchableTestPath(), out)

    def test_conditional_control_and_test_lock_state_machine(self):
        for semantics_pass, test_supplied in ((False, True), (True, True), (True, False)):
            with self.subTest(semantics_pass=semantics_pass, test_supplied=test_supplied), tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
                root = Path(temp)
                backbone, train, dev = self.setup_probe(root, stack)
                test = self.data(root, "test", rows=2)
                out = root / "run"
                calls = []
                # Mock scientific metrics, but exercise real adapter/control training and permutation.
                def synthetic_evaluation(backbone, teacher, students, data, s, d, output, **kwargs):
                    calls.append(data.metadata["split"])
                    if data.metadata["split"] == "test":
                        self.assertTrue((out / "frozen-decisions.json").exists())
                        self.assertTrue((out / "test-started.json").exists())
                        self.assertTrue(json.loads((out / "dev-decision.json").read_text())["all_dev_gates_pass"])
                    levels = {"Base": 3., "Privileged-Deep": 2., "Shallow-ExtraAttn": 2.9,
                              "Student-PCC": 2.5, "Target-Permuted": 2.8 if semantics_pass else 2.4}
                    names = (["Base", "Privileged-Deep"] if kwargs.get("include_reference", True) else []) + list(students)
                    return {name: [levels[name] * 11] * 2 for name in names}, [11, 11]
                stack.enter_context(patch("pcc.probe.evaluate", side_effect=synthetic_evaluation))
                original_loader = PreparedContexts
                test_opens = []
                def load(path, split, budget):
                    if split == "test":
                        self.assertTrue((out / "frozen-decisions.json").exists())
                        test_opens.append(path)
                    return original_loader(path, split, budget)
                stack.enter_context(patch("pcc.probe.PreparedContexts", side_effect=load))
                result = probe(backbone, "unused", train.path, dev.path,
                    (test.path if test_supplied else None) if semantics_pass else UntouchableTestPath(), out)
                test_expected = semantics_pass and test_supplied
                self.assertEqual(result["test_unlocked"], test_expected)
                if semantics_pass and not test_supplied:
                    self.assertEqual(result["decision"], "validation_complete_test_not_supplied")
                    self.assertFalse(result["pilot_specification_recommended"])
                    self.assertFalse((out / "test-started.json").exists())
                    self.assertTrue((out / "frozen-decisions.json").exists())
                self.assertFalse(result["pretraining_authorized"])
                self.assertEqual(calls, ["dev", "dev", "test"] if test_expected else ["dev", "dev"])
                self.assertEqual(len(test_opens), int(test_expected))
                audits = list((out / "permutations").glob("*.npz"))
                self.assertEqual(len(audits), 2)
                for audit_path in audits:
                    with np.load(audit_path) as audit:
                        self.assertTrue((audit["mapping"] != np.arange(len(audit["mapping"]))).all())
                shared_scale = json.loads((out / "calibration.json").read_text())["sigma_delta"]
                for name in ("Shallow-ExtraAttn", "Student-PCC", "Target-Permuted"):
                    self.assertEqual(torch.load(out / f"{name}.pt", weights_only=True)["sigma_delta"], shared_scale)

    def test_failure_leaves_no_complete_and_negative_screen_reads_no_data(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp)
            backbone, train, dev = self.setup_probe(root, stack)
            with patch("pcc.probe.student_update", side_effect=RuntimeError("injected training failure")):
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    probe(backbone, "unused", train.path, dev.path, UntouchableTestPath(), root / "failed")
            self.assertTrue((root / "failed" / "failure.json").exists())
            self.assertFalse((root / "failed" / "complete.json").exists())
            with patch("pcc.probe.screen_evidence", return_value=({"selected_pair": None}, {})):
                result = probe(backbone, "unused", UntouchableTestPath(), UntouchableTestPath(),
                               UntouchableTestPath(), root / "negative")
            self.assertEqual(result["decision"], "stop_negative_screen")

    def test_screen_full_prefix_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            full = self.data(root)
            subset = self.data(root, rows=2, name="prefix.npz")
            verify_slice(subset, full)
            subset.ids[0, 4] += 1
            with self.assertRaisesRegex(ValueError, "token-identical"):
                verify_slice(subset, full)

    def test_one_pass_causality_isolation_padding_and_hook_cleanup(self):
        backbone = tiny_backbone()
        backbone.model.to(torch.bfloat16)
        ids = torch.arange(12).repeat(2, 1)
        valid = torch.ones_like(ids, dtype=torch.bool)
        valid[:, 10:] = False
        segments = torch.zeros_like(ids)
        segments[:, 5:] = 1
        # Position resets are supplied by the fixture, not inferred by the model.
        positions = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6]]).repeat(2, 1)
        context = Context(ids, valid, positions, segments)
        adapter = self.teacher(backbone)
        before = fingerprint(backbone.model)
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            for s in (4, 8):
                reference, corrections, _ = backbone.student(context, s, adapter)
                for changed_positions, compared_positions in ((slice(8, 12), slice(0, 8)),
                        (slice(0, 5), slice(5, 10)), (slice(10, 12), slice(0, 10))):
                    changed = ids.clone()
                    changed[:, changed_positions] += 20
                    perturbed = Context(changed, valid, positions, segments)
                    actual, delta, _ = backbone.student(perturbed, s, adapter)
                    torch.testing.assert_close(actual[:, compared_positions], reference[:, compared_positions], atol=0, rtol=0)
                    torch.testing.assert_close(delta[:, compared_positions], corrections[:, compared_positions], atol=0, rtol=0)
                self.assertTrue(torch.equal(corrections[:, [0, 5, 10, 11]], torch.zeros_like(corrections[:, [0, 5, 10, 11]])))
                with patch.object(adapter, "forward", side_effect=RuntimeError("injected hook failure")):
                    with self.assertRaisesRegex(RuntimeError, "injected hook"):
                        backbone.student(context, s, adapter)
                self.assertFalse(backbone.model.model.layers[s - 1]._forward_hooks)
                # A failed student forward must not poison subsequent vanilla calls.
                clean = backbone.clean(context, (s,))
                self.assertTrue(torch.isfinite(clean.final_hidden).all())
        self.assertEqual(before, fingerprint(backbone.model))

    def test_permuted_update_uses_detached_permuted_supervision(self):
        with tempfile.TemporaryDirectory() as temp:
            data = self.data(Path(temp), rows=2)
            backbone = tiny_backbone()
            teacher = self.teacher(backbone)
            initial, _ = paired_adapters(backbone, seed=FULL_SEED)
            actual = copy.deepcopy(initial)
            real_permute = permute_targets
            captured = {}
            def permutation_spy(targets, context, update):
                # A controlled, distinct detached target tests the loss wiring;
                # the standalone permutation test verifies the actual derangement.
                result, audit = real_permute(targets, context, update)
                self.assertFalse(targets.requires_grad)
                result = result + .05
                captured["target"] = result
                return result, audit
            with patch("pcc.training.permute_targets", side_effect=permutation_spy) as permute:
                records, audit = student_update(backbone, teacher, data, {"Target-Permuted": actual},
                    {"Target-Permuted": optimizer(actual)}, 4, 16, 0, 2, 1, 1, .01,
                    total_updates=2, warmup=1, permuted=True)
            self.assertEqual(permute.call_count, 1)
            self.assertEqual(len(audit["mapping"]), 22)
            context = data.batch(0, 2, "cpu")
            clean = backbone.clean(context, (4,))
            reference = copy.deepcopy(initial)
            opt = optimizer(reference)
            delta = reference(clean.states[4], clean.states[4], context.allowed(), clean.rotary)
            eligible = context.targets()
            alignment = F.smooth_l1_loss(delta[:, :-1][eligible] / .01,
                                        captured["target"][:, :-1][eligible] / .01, beta=1.)
            loss = backbone.losses(backbone.tail(clean, 4, delta), context)[0].sum() / 22 + alignment
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.)
            opt.step()
            self.assertAlmostEqual(records[0]["correction_loss"], float(alignment.detach()), places=5)
            for a, b in zip(actual.parameters(), reference.parameters()):
                torch.testing.assert_close(a, b, atol=2e-7, rtol=1e-5)

    def test_failed_test_attempt_is_recorded_and_cannot_resume(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp)
            backbone, train, dev = self.setup_probe(root, stack)
            # Bypass metrics only to reach the test-open failure with actual training.
            stack.enter_context(patch("pcc.probe.probe_decision", return_value={
                "conditional_control_required": False, "all_dev_gates_pass": True, "gates": {}}))
            out = root / "run"
            with self.assertRaises(FileNotFoundError):
                probe(backbone, "unused", train.path, dev.path, root / "missing-test.npz", out)
            self.assertTrue((out / "frozen-decisions.json").exists())
            self.assertTrue((out / "test-started.json").exists())
            self.assertTrue((out / "failure.json").exists())
            self.assertFalse((out / "complete.json").exists())
            with self.assertRaises(FileExistsError):
                probe(backbone, "unused", train.path, dev.path, root / "missing-test.npz", out)


if __name__ == "__main__":
    unittest.main()
