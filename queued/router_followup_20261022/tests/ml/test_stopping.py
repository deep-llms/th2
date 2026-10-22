"""Full-schedule cutoff and native HF optimizer/scheduler/RNG/data resume."""
import json
from pathlib import Path
import tempfile
import unittest

from test_document_isolation import model, row, torch
from packed_training import PackedCollator, PackedTrainer
from train import (BaselineTrainingArguments, StopAfterCallback, check_resume_steps,
                   SaveConfigCallback, ModelArguments, DataArguments)
from resume_recipe import build_training_recipe, require_resume_training
from transformers import TrainerCallback, TrainerControl, TrainerState


class RecordingTrainer(PackedTrainer):
    def compute_loss(self, model, inputs, *args, **kwargs):
        self.seen.append(inputs["input_ids"].detach().cpu().tolist())
        return super().compute_loss(model, inputs, *args, **kwargs)


class LrTrace(TrainerCallback):
    def __init__(self):
        self.values = []

    def on_step_begin(self, args, state, control, optimizer, **kwargs):
        self.values.append([g["lr"] for g in optimizer.param_groups])


class StoppingTests(unittest.TestCase):
    def trainer(self, output, stop, max_steps=8):
        m = model()
        # Nonzero SDPA dropout makes RNG restoration consequential.
        for layer in m.model.layers:
            layer.self_attn.attention_dropout = .1
        args = BaselineTrainingArguments(output_dir=str(output), use_cpu=True, report_to="none",
            remove_unused_columns=False, stop_after=stop, max_steps=max_steps,
            per_device_train_batch_size=1, gradient_accumulation_steps=2,
            learning_rate=.001, lr_scheduler_type="cosine", warmup_ratio=.5,
            save_strategy="no", logging_strategy="no", dataloader_num_workers=0,
            dataloader_pin_memory=False, disable_tqdm=True, seed=42, data_seed=123,
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
        rows = [row([i + 3, i + 4, 31, i + 5, i + 6, i + 7, 31],
                    [0, 0, 0, 1, 1, 1, 1]) for i in range(10)]
        trace = LrTrace()
        t = RecordingTrainer(model=m, args=args, train_dataset=rows,
            data_collator=PackedCollator(), callbacks=[StopAfterCallback(), trace])
        t.training_recipe = build_training_recipe(m, args, 'sdpa', t.accelerator)
        t.add_callback(SaveConfigCallback(ModelArguments(), DataArguments(),
                                         {'test_fixture': True}, t.training_recipe))
        t.seen = []
        return t, trace

    def assert_nested_equal(self, a, b):
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                self.assert_nested_equal(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_nested_equal(x, y)
        else:
            self.assertEqual(a, b)

    def test_cutoff_resume_matches_uninterrupted_across_epoch_and_warmup(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            full, full_lr = self.trainer(root / "full", 8)
            full.train()
            first, first_lr = self.trainer(root / "split", 3)
            first.train()
            checkpoint = root / "split/checkpoint-3"
            self.assertEqual(first.state.global_step, 3)
            state = json.loads((checkpoint / "trainer_state.json").read_text())
            self.assertEqual(state["max_steps"], 8)
            self.assertEqual(first.args.get_warmup_steps(8), 4)
            for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
                self.assertTrue((checkpoint / name).is_file(), name)
            resumed, resumed_lr = self.trainer(root / "split", 8)
            check_resume_steps(checkpoint, resumed.args)
            require_resume_training(checkpoint, resumed.training_recipe, resumed.args)
            resumed.train(resume_from_checkpoint=str(checkpoint))
            self.assertEqual(first.seen + resumed.seen, full.seen)
            self.assertEqual(first_lr.values + resumed_lr.values, full_lr.values)
            self.assertEqual(len(full_lr.values), 8)
            self.assertEqual(resumed.state.global_step, 8)
            self.assert_nested_equal(full.model.state_dict(), resumed.model.state_dict())
            self.assert_nested_equal(full.optimizer.state_dict(), resumed.optimizer.state_dict())
            self.assert_nested_equal(full.lr_scheduler.state_dict(), resumed.lr_scheduler.state_dict())
            # Final RNG states also match, not just the deterministic weights.
            a = torch.load(root / "full/checkpoint-8/rng_state.pth", weights_only=False, map_location="cpu")
            b = torch.load(root / "split/checkpoint-8/rng_state.pth", weights_only=False, map_location="cpu")
            self.assert_nested_equal(a["cpu"], b["cpu"])

    def test_argument_validation_and_existing_cutoff_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            for extra in ({"stop_after": 0}, {"stop_after": 9}, {"stop_after": 1.5},
                          {"stop_after": 3, "save_only_model": True},
                          {"stop_after": 3, "ignore_data_skip": True},
                          {"stop_after": 3, "max_steps": -1}):
                with self.assertRaises(ValueError):
                    BaselineTrainingArguments(**dict(output_dir=folder, use_cpu=True,
                        report_to="none", max_steps=8) | extra)
            args = BaselineTrainingArguments(output_dir=folder, use_cpu=True, report_to="none", max_steps=8, stop_after=3)
            with self.assertRaisesRegex(ValueError, "already reached"):
                StopAfterCallback().on_train_begin(args, TrainerState(global_step=3), TrainerControl())
            with self.assertRaisesRegex(RuntimeError, "ended at"):
                StopAfterCallback().on_train_end(args, TrainerState(global_step=2), TrainerControl())
            p = Path(folder)
            (p / "trainer_state.json").write_text(json.dumps({"global_step": 3, "max_steps": 8}))
            with self.assertRaisesRegex(ValueError, "already reached"):
                check_resume_steps(p, args)
            args.max_steps, args.stop_after = 10, 6
            with self.assertRaisesRegex(ValueError, "full max_steps"):
                check_resume_steps(p, args)
            (p / "train_config.json").write_text(json.dumps({"training": {"max_steps": 8}}))
            args.max_steps, args.stop_after = -1, None
            with self.assertRaisesRegex(ValueError, "full max_steps"):
                check_resume_steps(p, args)


if __name__ == "__main__":
    unittest.main()
