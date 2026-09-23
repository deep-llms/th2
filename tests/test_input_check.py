"""Full-budget checks must succeed before any scientific adapter training."""
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_sampling_boundary import HAS_SAMPLING

if HAS_SAMPLING:
    import numpy as np
    import torch
    from pcc.data import SampledDataLoader
    from pcc.diagnostics import fingerprint, tiny_backbone
    from pcc.input_check import check_inputs, context_fingerprint
    from pcc.pipeline import pipeline
    from test_sampled_data import sampled_fixture
    from test_probe import UntouchableTestPath


@unittest.skipUnless(HAS_SAMPLING, "Requires the project ML/data environment")
class InputCheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        for target, value in {"pcc.data.CONTEXT": 12, "pcc.protocol.CONTEXT": 12,
                              "pcc.protocol.TOKENS_PER_UPDATE": 24,
                              "pcc.protocol.FULL_UPDATES": 5,
                              "pcc.protocol.SCREEN_UPDATES": 2,
                              "pcc.protocol.PROBE_EVAL_TOKENS": 50,
                              "pcc.protocol.SCREEN_DEV_TOKENS": 26}.items():
            self.stack.enter_context(patch(target, value))
        self.train, tokenizer, _ = sampled_fixture(self.root / "train")
        self.val, _, _ = sampled_fixture(self.root / "val", sharded=False, documents=10)
        self.loader = SampledDataLoader(tokenizer, self.root / "cache")
        self.backbone = tiny_backbone()
        self.backbone.model.to(torch.bfloat16)
        self.config = {"train_data": str(self.train), "val_data": str(self.val),
                       "test_data": None, "microbatch": 1}

    def test_full_budgets_counts_prefixes_and_stable_fingerprints(self):
        report = check_inputs(self.backbone, self.train, self.val, load_data=self.loader)
        self.assertEqual(report["streams"]["train"]["input_tokens"], 120)
        self.assertEqual(report["streams"]["train"]["target_tokens"], 110)
        self.assertEqual(report["streams"]["dev"]["target_tokens"], 45)
        self.assertEqual(report["streams"]["screen_dev"]["target_tokens"], 23)
        self.assertFalse(report["test_data_inspected"])
        self.assertEqual(report, check_inputs(self.backbone, self.train, self.val, load_data=self.loader))
        self.assertEqual(len(self.loader.packed), 2)

    def test_fingerprint_covers_order_masks_positions_and_segments(self):
        data = self.loader(self.train, "train", 120)
        expected = context_fingerprint(data)
        # Storage integer widths do not change the model inputs.
        narrower = copy.copy(data)
        narrower.ids = data.ids.astype(np.int32)
        self.assertEqual(context_fingerprint(narrower), expected)
        for field in ("ids", "valid", "positions", "segments"):
            changed = copy.copy(data)
            values = getattr(data, field)
            values = np.zeros_like(data.ids) if values is None else values.copy()
            values[0, 0] = not values[0, 0] if field == "valid" else values[0, 0] + 1
            setattr(changed, field, values)
            self.assertNotEqual(context_fingerprint(changed), expected)
        changed = copy.copy(data)
        changed.ids = data.ids[::-1]
        self.assertNotEqual(context_fingerprint(changed), expected)

    def test_short_full_validation_fails_before_screen_even_when_screen_fits(self):
        # The screen needs 26 tokens, which fit; a 120-token full eval does not.
        with patch("pcc.protocol.PROBE_EVAL_TOKENS", 120), \
                patch("pcc.pipeline.screen", side_effect=AssertionError("Must fail before screen")) as screen:
            out = self.root / "failed"
            with self.assertRaisesRegex(ValueError, "no resampling"):
                pipeline(self.backbone, self.config, out, load_data=self.loader)
        screen.assert_not_called()
        self.assertTrue((out / "failure.json").exists())
        self.assertFalse((out / "screen-started.json").exists())
        self.assertFalse((out / "complete.json").exists())

    def test_check_only_runs_model_preflight_without_screen_probe_or_test_reads(self):
        out = self.root / "checked"
        before = fingerprint(self.backbone.model)
        # A string path is recorded lexically, but the loader must never receive it.
        config = {**self.config, "test_data": str(self.root / "never-open-test")}
        def load(path, split, budget):
            if split == "test":
                raise AssertionError("Check-only cannot open test data")
            return self.loader(path, split, budget)
        with patch("pcc.pipeline.screen", side_effect=AssertionError("No screen")), \
                patch("pcc.pipeline.probe", side_effect=AssertionError("No full probe")):
            result = pipeline(self.backbone, config, out, check_only=True, load_data=load)
        self.assertEqual(result["decision"], "inputs_validated")
        self.assertFalse(result["scientific_training_performed"])
        self.assertEqual(json.loads((out / "complete.json").read_text()), result)
        self.assertEqual(json.loads((out / "preflight.json").read_text())["status"], "ok")
        self.assertTrue((out / "input-check.json").exists())
        self.assertFalse((out / "screen").exists())
        self.assertEqual(before, fingerprint(self.backbone.model))

    def test_changed_screen_prefix_and_out_of_vocabulary_ids_fail(self):
        for fault in ("prefix", "vocabulary"):
            def load(path, split, budget):
                data = self.loader(path, split, budget)
                if fault == "prefix" and budget == 48:
                    data.ids[0, 0] += 1
                if fault == "vocabulary" and split == "dev":
                    data.ids[0, 0] = self.backbone.model.config.vocab_size
                return data
            with self.subTest(fault=fault), self.assertRaisesRegex(ValueError, "token-identical|vocabulary"):
                check_inputs(self.backbone, self.train, self.val, load_data=load)

    def test_unfrozen_model_rejected_before_reading_inputs(self):
        self.backbone.model.train()
        with self.assertRaisesRegex(ValueError, "frozen backbone"):
            check_inputs(self.backbone, UntouchableTestPath(), UntouchableTestPath(), load_data=self.loader)


if __name__ == "__main__":
    unittest.main()
