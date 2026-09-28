"""Offline CPU checks of the actual train.py entry point, packing, and HF resume."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", WANDB_MODE="offline")
import numpy as np
import torch
from datasets import Dataset
from safetensors.torch import load_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3Config, TrainingArguments
import train
from deep_kv.__main__ import jobs
from deep_kv.report import report
from deep_kv.training import DeepKVTrainer, compute_metrics
from deep_kv.model import Context, DeepKV


def fixture(root, world=1, bf16=False, microbatch=2, accumulation=2):
    cfg = Qwen3Config(vocab_size=32, hidden_size=32, intermediate_size=48, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                      max_position_embeddings=64, tie_word_embeddings=True)
    cfg.save_pretrained(root / "model")
    backend = Tokenizer(models.WordLevel({**{str(i): i for i in range(31)}, "<|endoftext|>": 31}, unk_token="0"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="0", eos_token="<|endoftext|>", model_max_length=64)
    tokenizer.save_pretrained(root / "model")
    rng = np.random.default_rng(42)
    for split, rows in (("train", 4 * microbatch * accumulation * world + 5), ("eval", 8)):
        Dataset.from_dict({"text": [" ".join(map(str, row)) for row in rng.integers(0, 30, (rows, 7))]}).save_to_disk(root / split / "en")
    return dict(config_name=str(root / "model"), tokenizer_name=str(root / "model"),
                data_dir=str(root / "train"), eval_data_dir=str(root / "eval"),
                output_dir=str(root / "run"), arm="A", consumer=2, deep_target=4, lm_chunk=4,
                block_size=8, preprocessing_num_workers=1, eval_rows=5, monitor_rows=2,
                max_steps=3, warmup_steps=1, learning_rate=3e-4, lr_scheduler_type="cosine_with_min_lr",
                lr_scheduler_kwargs={"min_lr_rate": .1}, seed=42, data_seed=42,
                per_device_train_batch_size=microbatch, per_device_eval_batch_size=microbatch, gradient_accumulation_steps=accumulation,
                optim="adamw_torch", use_cpu=True, bf16=bf16, report_to="none", save_steps=1,
                save_total_limit=2, eval_strategy="steps", eval_steps=1, logging_steps=1,
                dataloader_num_workers=0, disable_tqdm=True)


def invoke(root, config):
    path = root / "invocation.json"
    path.write_text(json.dumps(config))
    with patch("sys.argv", ["train.py", str(path)]):
        train.main()


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_real_entry_point_all_arms_cutoff_report_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = fixture(root)
            for arm in "ABCDE":
                args = {**config, "arm": arm, "output_dir": str(root / arm), "stop_after": 2}
                invoke(root, args)
                state = json.loads((root / arm / "checkpoint-2/trainer_state.json").read_text())
                self.assertEqual(state["max_steps"], 3)
                self.assertEqual(state["global_step"], 2)
                self.assertEqual(json.loads((root / arm / "result.json").read_text())["evaluation"]["eval_rows"], 5)
            self.assertEqual(report(root)["compared_update"], 2)
            comparison = report(root, "ABCDE")
            self.assertIn("E-D", comparison["nll_differences"])
            self.assertEqual(comparison["kv_loss_weights"]["E"], 0.3)
            self.assertEqual(set(report(root, "E")["lm_loss"]), {"E"})
            e_eval = json.loads((root / "E/eval_results.json").read_text())
            self.assertAlmostEqual(e_eval["eval_loss"], e_eval["eval_lm_loss"] +
                                   0.3 * (e_eval["eval_loss_k"] + e_eval["eval_loss_v"]) / 2)
            # The same weights are not permission to resume D as E.
            with self.assertRaisesRegex(ValueError, "Resume configuration"):
                invoke(root, {**config, "arm": "E", "output_dir": str(root / "D"), "stop_after": 2})
            # C and D have identical tensor names: HF alone cannot reject this
            # scientifically invalid cross-arm restore.
            result_before = (root / "D/result.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "must belong to this output"):
                invoke(root, {**config, "arm": "D", "output_dir": str(root / "D"),
                              "stop_after": 2, "resume_from_checkpoint": str(root / "C/checkpoint-2")})
            self.assertEqual((root / "D/result.json").read_bytes(), result_before)
            # Same-cutoff resume must take no extra optimizer step.
            before = load_file(root / "A/model.safetensors")
            invoke(root, {**config, "output_dir": str(root / "A"), "stop_after": 2,
                          "resume_from_checkpoint": str(root / "A/checkpoint-2")})
            for name, value in load_file(root / "A/model.safetensors").items():
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)
            # Compare real interrupted/resumed training against an uninterrupted run.
            for arm in "ABCDE":
                invoke(root, {**config, "arm": arm, "output_dir": str(root / arm)})
                invoke(root, {**config, "arm": arm, "output_dir": str(root / (arm + "-full"))})
                actual, expected = (load_file(root / p / "model.safetensors") for p in (arm, arm + "-full"))
                for name in actual:
                    torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
            self.assertEqual(report(root)["compared_update"], 3)
            self.assertEqual(report(root, "DE")["compared_update"], 3)
            with self.assertRaisesRegex(ValueError, "Resume configuration"):
                invoke(root, {**config, "output_dir": str(root / "A"), "seed": 43})
            result_path = root / "D/result.json"
            broken = json.loads(result_path.read_text()); broken["input_tokens"] += 1
            result_path.write_text(json.dumps(broken))
            with self.assertRaises(ValueError):
                report(root)

    def test_invalid_cutoff_and_chunk_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); config = fixture(root)
            for field, values in (("stop_after", (True, 1.5, 0, 4)), ("lm_chunk", (-1, 0, 1.5))):
                for value in values:
                    with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, field):
                        invoke(root, {**config, field: value})
            self.assertFalse((root / "run").exists())

    def test_eos_packing_cache_and_order_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); cfg = fixture(root)
            tokenizer = PreTrainedTokenizerFast.from_pretrained(root / "model", local_files_only=True)
            # Two workers: each has 20 x 4 tokens, producing 13 six-token rows.
            Dataset.from_dict({"text": ["1 2 3"] * 20 + ["7 8 9"] * 20}).save_to_disk(root / "packing")
            raw = train.load_text(root / "packing")
            args = TrainingArguments(output_dir=str(root / "args"), use_cpu=True, report_to="none")
            first = train.preprocess_dataset(raw, tokenizer, 6, args, num_proc=2).shuffle(seed=42)
            expected = []
            for doc in ([1, 2, 3, 31], [7, 8, 9, 31]):
                flat = doc * 20
                expected.extend(flat[i:i + 6] for i in range(0, 78, 6))
            baseline = Dataset.from_dict({"input_ids": expected, "labels": expected, "attention_mask": [[1]*6]*26}).shuffle(seed=42)
            self.assertEqual(first.to_dict(), baseline.to_dict())
            caches = {p: p.stat().st_mtime_ns for p in root.rglob("cache-*.arrow")}
            cached = train.preprocess_dataset(raw, tokenizer, 6, args, num_proc=2).shuffle(seed=42)
            self.assertEqual(caches, {p: p.stat().st_mtime_ns for p in root.rglob("cache-*.arrow")})
            rebuilt = train.preprocess_dataset(raw, tokenizer, 6, args, num_proc=2, overwrite_cache=True).shuffle(seed=42)
            self.assertEqual(first.to_dict(), cached.to_dict()); self.assertEqual(first.to_dict(), rebuilt.to_dict())
            orders = []
            for arm, data in zip("ABCDE", (first, cached, rebuilt, first, cached)):
                model_cfg = Qwen3Config.from_pretrained(root / "model", local_files_only=True)
                model_cfg._attn_implementation = "sdpa"
                model = DeepKV.from_scratch(model_cfg, arm, consumer=2, deep_target=4)
                args.remove_unused_columns = False
                trainer = DeepKVTrainer(model=model, args=args, train_dataset=data)
                orders.append([b["input_ids"].tolist() for b in trainer.get_train_dataloader()])
            self.assertTrue(all(x == orders[0] for x in orders))

    def test_native_gradient_accumulation_scaling(self):
        from tests.test_deep_kv import model
        ids = torch.arange(24).reshape(4, 6) % 30
        with tempfile.TemporaryDirectory() as tmp:
            for arm in "ABCDE":
                accumulated, whole = model(arm), model(arm)
                if accumulated.aux is not None:
                    torch.nn.init.normal_(accumulated.aux.out.weight, std=.03)
                    whole.load_state_dict(accumulated.state_dict())
                args = TrainingArguments(output_dir=tmp, use_cpu=True, report_to="none", gradient_accumulation_steps=4)
                trainer = DeepKVTrainer(model=accumulated, args=args)
                trainer.current_gradient_accumulation_steps = 4
                for row in ids:
                    trainer.training_step(accumulated, {"input_ids": row[None]})
                result = whole(Context(ids, torch.ones_like(ids, dtype=torch.bool), torch.arange(6).expand_as(ids)))
                (result["lm_sum"] / 20 + whole.kv_loss_weight * (result["k_sum"] + result["v_sum"]) / 48).backward()
                for (name, p), (_, q) in zip(accumulated.named_parameters(), whole.named_parameters()):
                    torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-5, msg=name)

    def test_direct_entry_point_restores_baseline_environment(self):
        # A fresh process catches queue launches which bypass the shell exports.
        env = dict(os.environ, WANDB_MODE="online")
        env.pop("NCCL_NVLS_ENABLE", None)
        subprocess.run([sys.executable, "-c", """
import importlib, os
import train
assert os.environ['NCCL_NVLS_ENABLE'] == '0'
assert os.environ['WANDB_MODE'] == 'offline'
assert all(os.environ[key] == '1' for key in (
    'HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'))
# Preserve an operator's explicit NCCL override, as documented.
os.environ['NCCL_NVLS_ENABLE'] = '1'
importlib.reload(train)
assert os.environ['NCCL_NVLS_ENABLE'] == '1'
"""], env=env, check=True, timeout=120)

    def test_e_changes_only_alignment_weight_and_gradient(self):
        from types import SimpleNamespace
        from tests.test_deep_kv import model, parameter_hash
        models = {arm: model(arm) for arm in "BDE"}
        self.assertEqual(models["D"].kv_loss_weight, 1.0)
        self.assertEqual(models["E"].kv_loss_weight, 0.3)
        self.assertEqual(parameter_hash(models["D"]), parameter_hash(models["E"]))
        torch.nn.init.normal_(models["B"].aux.out.weight, std=.03)
        for arm in "DE":
            models[arm].load_state_dict(models["B"].state_dict())
        ids = torch.arange(24).reshape(4, 6) % 30
        # Compare raw outputs before applying either loss coefficient, with a
        # nonzero branch so this also exercises its learned attention path.
        ctx = Context(ids, torch.ones_like(ids, dtype=torch.bool), torch.arange(6).expand_as(ids))
        with torch.no_grad():
            d_outputs, e_outputs = models["D"](ctx), models["E"](ctx)
        for key in d_outputs:
            torch.testing.assert_close(d_outputs[key], e_outputs[key], rtol=0, atol=0)
        losses, gradients, metrics = {}, {}, {}
        with tempfile.TemporaryDirectory() as tmp:
            for arm, current in models.items():
                trainer = DeepKVTrainer(model=current, compute_metrics=compute_metrics,
                    args=TrainingArguments(output_dir=tmp, use_cpu=True, report_to="none"))
                loss, outputs = trainer.compute_loss(current, {"input_ids": ids}, return_outputs=True)
                losses[arm] = loss.detach()
                metrics[arm] = trainer.compute_metrics(SimpleNamespace(predictions=outputs["statistics"].numpy()))
                self.assertAlmostEqual(metrics[arm]["loss"], loss.item(), places=5)
                loss.backward()
                gradients[arm] = {name: p.grad.clone() for name, p in current.named_parameters()}
        self.assertGreater((losses["D"] - losses["B"]).item(), 0)
        torch.testing.assert_close(losses["E"] - losses["B"],
                                   .3 * (losses["D"] - losses["B"]), atol=1e-6, rtol=1e-5)
        for name in gradients["E"]:
            expected = gradients["B"][name] + .3 * (gradients["D"][name] - gradients["B"][name])
            torch.testing.assert_close(gradients["E"][name], expected, atol=2e-6, rtol=2e-5, msg=name)
        for key in ("lm_loss", "loss_k", "loss_v"):
            self.assertAlmostEqual(metrics["E"][key], metrics["D"][key], places=6)

    def test_queue_uses_train_py_and_unchanged_schedule(self):
        from run_experiments import load_jobs
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "jobs.json"
            path.write_text(json.dumps(jobs("deep_kv.b200.json", 2500)))
            queue = load_jobs(path)
            self.assertEqual(len(queue), 5)
            for job in queue[:4]:
                self.assertEqual(job["gpus"], list(range(8)))
                argv = job["argv"]
                self.assertTrue(any(a.endswith("/train.py") for a in argv))
                self.assertEqual(argv[argv.index("--max_steps") + 1], "28600")
                self.assertEqual(argv[argv.index("--stop_after") + 1], "2500")
            # Parse the exact generated CLI with HF's parser to catch shell/JSON drift.
            parser = train.HfArgumentParser((train.ModelArguments, train.DataArguments, train.PilotArguments, TrainingArguments))
            argv = queue[0]["argv"]; start = next(i for i,a in enumerate(argv) if a.endswith("/train.py")) + 1
            values = parser.parse_args_into_dataclasses(argv[start:] + ["--use_cpu", "true", "--bf16", "false", "--report_to", "none"])
            self.assertEqual(values[-1].gradient_accumulation_steps, 4)
            self.assertEqual(values[-1].lr_scheduler_kwargs, {"min_lr_rate": .1})
            selected = jobs("deep_kv.b200.json", 2500, arms="E")["jobs"]
            self.assertEqual([j["name"] for j in selected], ["arm-E", "compare"])
            self.assertEqual(selected[0]["gpus"], list(range(8)))
            argv = selected[0]["argv"]
            start = next(i for i, a in enumerate(argv) if a.endswith("/train.py")) + 1
            parsed = parser.parse_args_into_dataclasses(argv[start:] + ["--use_cpu", "true", "--bf16", "false", "--report_to", "none"])
            self.assertEqual(parsed[2].arm, "E")
            self.assertEqual(parsed[-1].max_steps, 28600)
            self.assertEqual(parsed[-1].warmup_steps, 1430)
            self.assertEqual(selected[-1]["argv"][-2:], ["--arms", "E"])
            # All executable training settings must match, apart from the arm
            # selector and the names/paths that keep their results separate.
            d_job, e_job = jobs("deep_kv.b200.json", 2500, arms="DE")["jobs"][:2]
            normalized = []
            for job in (d_job, e_job):
                argv = list(job["argv"])
                for key in ("--arm", "--output_dir", "--run_name"):
                    argv[argv.index(key) + 1] = "ARM_SPECIFIC"
                normalized.append(argv)
            self.assertEqual(normalized[0], normalized[1])
            self.assertEqual(d_job["gpus"], e_job["gpus"])
            for invalid in ("", "EE", "F"):
                with self.assertRaises(ValueError):
                    jobs("deep_kv.b200.json", arms=invalid)
            config = json.loads(Path("deep_kv.b200.json").read_text())
            config.update(use_cpu=True, bf16=False, label_names=["labels"], report_to=["none"])
            recipe = Path(tmp) / "recipe.json"
            for report_to in (["none"], [], ["wandb"]):
                config["report_to"] = report_to
                recipe.write_text(json.dumps(config))
                argv = jobs(recipe, 2500)["jobs"][0]["argv"]
                start = next(i for i, a in enumerate(argv) if a.endswith("/train.py")) + 1
                parsed = parser.parse_args_into_dataclasses(argv[start:])[-1]
                self.assertEqual(parsed.label_names, ["labels"])
                self.assertEqual(parsed.report_to, ["wandb"] if report_to == ["wandb"] else [])


if __name__ == "__main__":
    unittest.main()
