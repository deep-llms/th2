"""Offline CPU checks of the actual train.py entry point, packing, and HF resume."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import shutil
import copy
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


def fixture_target_counts(root, config):
    """Independent counts from document lengths in the tiny evaluation fixture."""
    from itertools import groupby
    tokenizer = PreTrainedTokenizerFast.from_pretrained(root / 'model', local_files_only=True)
    texts = train.load_text(root / 'eval')['text']
    lengths = [len(ids) + 1 for ids in tokenizer(list(texts), add_special_tokens=False)['input_ids']]
    documents = [i for i, length in enumerate(lengths) for _ in range(length)]
    size = config['block_size']
    fragments = [len(list(group)) for start in range(0, config['eval_rows'] * size, size)
                 for _, group in groupby(documents[start:start + size])]
    return sum(length - 1 for length in fragments), sum(max(0, length - 2) for length in fragments)


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_observed_resume_gate_and_generated_queue(self):
        from scripts.check_optimized_resume import FAST, worker, compare, make_jobs
        from scripts.stage_deep_kv_resume import stage
        from run_experiments import load_jobs
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = fixture(root)
            source = root / 'source' / 'G'
            invoke(root, {**config, 'arm': 'G', 'output_dir': str(source), 'stop_after': 2})
            smoke = root / 'smoke'
            for mode in ('control', 'optimized'):
                stage({'G': str(source)}, smoke / mode, 2)
                output = smoke / mode / 'G'
                args = {**config, 'arm': 'G', 'output_dir': str(output), 'stop_after': 3,
                        'resume_from_checkpoint': str(output / 'checkpoint-2')}
                if mode == 'optimized':
                    args.update(**FAST, allow_performance_change_on_resume=True)
                invocation = root / 'gate.json'
                invocation.write_text(json.dumps(args))
                with patch.object(train, 'DeepKVTrainer', train.DeepKVTrainer), patch('sys.argv', []):
                    worker(str(output / 'resume-check'), [str(invocation)])
            compare(smoke, 2, arms='G', world=1)
            self.assertEqual(json.loads((smoke / 'verified.json').read_text())['status'], 'ok')
            recipe = root / 'recipe.json'
            recipe.write_text(json.dumps(config))
            queue = root / 'queue'
            make_jobs(recipe, queue, root / 'source', 2, 3)
            load_jobs(queue / 'jobs.json')
            generated = json.loads((queue / 'jobs.json').read_text())['jobs']
            self.assertEqual([j['name'] for j in generated][-6:],
                             ['verify-resume', 'stage-continuation', 'arm-B', 'arm-F', 'arm-G', 'compare'])
            for job in generated:
                if 'gpus' in job:
                    self.assertEqual(job['gpus'], list(range(8)))
                    self.assertIn('--resume_from_checkpoint', job['argv'])
                    self.assertEqual(job['argv'][job['argv'].index('--max_steps') + 1], '3')
            reused = root / 'reuse'
            make_jobs(recipe, reused, root / 'source', 2, 3, smoke / 'control')
            load_jobs(reused / 'jobs.json')
            generated = json.loads((reused / 'jobs.json').read_text())['jobs']
            self.assertFalse(any('smoke-control' in job['name'] for job in generated))
            # A read-only external control gives the same numerical/state checks.
            external = root / 'external-control'
            shutil.move(smoke / 'control', external)
            (smoke / 'verified.json').unlink()
            compare(smoke, 2, arms='G', world=1, control_root=external)

    def test_real_entry_point_all_arms_cutoff_report_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = fixture(root)
            target_count, _ = fixture_target_counts(root, config)
            for arm in "ABCDEFG":
                args = {**config, "arm": arm, "output_dir": str(root / arm), "stop_after": 2}
                invoke(root, args)
                state = json.loads((root / arm / "checkpoint-2/trainer_state.json").read_text())
                self.assertEqual(state["max_steps"], 3)
                self.assertEqual(state["global_step"], 2)
                self.assertEqual(json.loads((root / arm / "result.json").read_text())["evaluation"]["eval_rows"], 5)
            self.assertEqual(report(root)["compared_update"], 2)
            comparison = report(root, "ABCDEFG")
            self.assertIn("F-B", comparison["nll_differences"])
            self.assertIn("G-F", comparison["nll_differences"])
            self.assertIn("G-B", comparison["nll_differences"])
            self.assertEqual(comparison["functional_loss_weights"]["G"], {"route": .3, "message": .3})
            self.assertEqual(set(report(root, "FG")["lm_loss"]), {"F", "G"})
            for arm in "FG":
                metrics = json.loads((root / arm / "eval_results.json").read_text())
                self.assertAlmostEqual(metrics["eval_loss"], metrics["eval_lm_loss"] +
                                       .3 * metrics["eval_loss_route"] + .3 * metrics["eval_loss_msg"])
                self.assertEqual(metrics["eval_route_queries"], target_count)
                self.assertNotIn("eval_loss_k", metrics)
            self.assertEqual(json.loads((root / "F/eval_results.json").read_text())["eval_loss_msg"], 0.)
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
            for arm in "ABCDEFG":
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

    def test_performance_resume_preserves_state_and_data_and_rejects_other_changes(self):
        class RecordingTrainer(DeepKVTrainer):
            def _load_optimizer_and_scheduler(self, checkpoint):
                super()._load_optimizer_and_scheduler(checkpoint)
                if checkpoint:
                    torch.testing.assert_close(self.optimizer.state_dict(),
                        torch.load(Path(checkpoint) / 'optimizer.pt', map_location='cpu', weights_only=True), rtol=0, atol=0)
                    self.loaded_scheduler = copy.deepcopy(self.lr_scheduler.state_dict())
                    self.seen = []
                    records.append(self)

            def compute_loss(self, model, inputs, *args, **kwargs):
                if model.training and self.is_in_train and hasattr(self, 'seen'):
                    self.seen.append(inputs['input_ids'].clone())
                return super().compute_loss(model, inputs, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); cfg = {**fixture(root), 'isolate_documents': False}
            for arm in 'BFG':
                old = root / arm
                invoke(root, {**cfg, 'arm': arm, 'output_dir': str(old), 'stop_after': 2})
                # Simulate an existing checkpoint/config from before these flags existed.
                metadata = json.loads((old / 'train_config.json').read_text())
                for key in ('checkpoint_lm', 'checkpoint_aux', 'causal_attention'):
                    metadata['pilot'].pop(key)
                (old / 'train_config.json').write_text(json.dumps(metadata))
                optimized = root / (arm + '-optimized')
                shutil.copytree(old, optimized)
                changes = dict(checkpoint_layers=False, checkpoint_lm=False, checkpoint_aux=False,
                               causal_attention=True, lm_chunk=8)
                args = {**cfg, 'arm': arm, 'output_dir': str(optimized), **changes}
                with self.assertRaisesRegex(ValueError, 'allow_performance_change'):
                    invoke(root, args)
                for extra in ({'seed': 43}, {'learning_rate': .001}, {'ignore_data_skip': True},
                              {'per_device_train_batch_size': 1, 'gradient_accumulation_steps': 4}, {'arm': 'D'}):
                    with self.assertRaisesRegex(ValueError, 'Resume configuration'):
                        invoke(root, {**args, 'allow_performance_change_on_resume': True, **extra})
                self.assertFalse(list(optimized.glob('resume-transition-*.json')))
                records = []
                with patch.object(train, 'DeepKVTrainer', RecordingTrainer):
                    invoke(root, {**cfg, 'arm': arm, 'output_dir': str(old)})
                    invoke(root, {**args, 'allow_performance_change_on_resume': True})
                a, b = records
                self.assertEqual(a.loaded_scheduler, b.loaded_scheduler)
                self.assertEqual(len(a.seen), cfg['gradient_accumulation_steps'])
                torch.testing.assert_close(a.seen, b.seen, rtol=0, atol=0)
                torch.testing.assert_close(a.model.state_dict(), b.model.state_dict(), rtol=1e-5, atol=1e-7)
                torch.testing.assert_close(a.optimizer.state_dict(), b.optimizer.state_dict(), rtol=1e-4, atol=1e-7)
                self.assertEqual(a.lr_scheduler.state_dict(), b.lr_scheduler.state_dict())
                transitions = list(optimized.glob('resume-transition-*.json'))
                self.assertEqual(len(transitions), 1)
                record = json.loads(transitions[0].read_text())
                self.assertEqual(record['global_step'], 2)
                self.assertEqual(set(record['changes']), set(changes))
                self.assertEqual(record['previous'], metadata)
                self.assertNotIn('allow_performance_change_on_resume', record['requested']['pilot'])

    def test_resume_retention_keeps_checkpoints_without_changing_training(self):
        from scripts.check_optimized_resume import state_hash
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = {**fixture(root), 'max_steps': 5}
            control, keep = root / 'control', root / 'keep'
            invoke(root, {**cfg, 'output_dir': str(control), 'stop_after': 2})
            shutil.copytree(control, keep)
            previous = json.loads((keep / 'train_config.json').read_text())
            for field, value in [('seed', 43), ('learning_rate', .001),
                                 ('ignore_data_skip', True), ('save_steps', 2)]:
                changed = copy.deepcopy(previous)
                changed['training'].update(save_total_limit=0, **{field: value})
                with self.assertRaisesRegex(ValueError, 'Resume configuration'):
                    train.resume_performance_changes(previous, changed)
            invoke(root, {**cfg, 'output_dir': str(control)})
            invoke(root, {**cfg, 'output_dir': str(keep), 'save_total_limit': 0})
            self.assertEqual({p.name for p in keep.glob('checkpoint-*')},
                             {f'checkpoint-{i}' for i in range(1, 6)})
            self.assertEqual(len(list(control.glob('checkpoint-*'))), 2)
            for name, value in load_file(keep / 'model.safetensors').items():
                torch.testing.assert_close(value, load_file(control / 'model.safetensors')[name], rtol=0, atol=0)
            for name in ('optimizer.pt', 'scheduler.pt', 'rng_state.pth'):
                states = [torch.load(p / 'checkpoint-5' / name, map_location='cpu',
                                     weights_only=name != 'rng_state.pth') for p in (control, keep)]
                self.assertEqual(state_hash(states[0]), state_hash(states[1]), name)
            records = list(keep.glob('resume-transition-*.json'))
            self.assertEqual(len(records), 1)
            self.assertEqual(json.loads(records[0].read_text())['changes'],
                             {'save_total_limit': {'before': 2, 'after': 0}})

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
            for arm, data in zip("ABCDEFG", (first, cached, rebuilt, first, cached, rebuilt, cached)):
                model_cfg = Qwen3Config.from_pretrained(root / "model", local_files_only=True)
                model_cfg._attn_implementation = "sdpa"
                model = DeepKV.from_scratch(model_cfg, arm, consumer=2, deep_target=4)
                args.remove_unused_columns = False
                trainer = DeepKVTrainer(model=model, args=args, train_dataset=data)
                orders.append([b["input_ids"].tolist() for b in trainer.get_train_dataloader()])
            self.assertTrue(all(x == orders[0] for x in orders))

    def test_isolation_packing_cache_boundaries_and_all_arm_entry_points(self):
        from deep_kv import ARMS
        from deep_kv.packing import isolated_data_collator
        from scripts.benchmark_document_training import Collator

        class RecordingTrainer(DeepKVTrainer):
            def compute_loss(self, model, inputs, *args, **kwargs):
                ctx = self.context(inputs)
                self_outer.assertIsNotNone(ctx.segments)
                # Different documents must never attend or form an LM target.
                boundary = ctx.segments[:, 1:] != ctx.segments[:, :-1]
                self_outer.assertFalse((ctx.targets() & boundary).any())
                different = ctx.segments[:, :, None] != ctx.segments[:, None, :]
                self_outer.assertFalse((ctx.allowed()[:, 0] & different).any())
                if boundary.any():
                    seen.add((self.model.arm, model.training))
                return super().compute_loss(model, inputs, *args, **kwargs)

        self_outer = self
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); cfg = fixture(root)
            cfg.update(block_size=6, stop_after=1)
            tokenizer = PreTrainedTokenizerFast.from_pretrained(root / 'model', local_files_only=True)
            raw = train.load_text(root / 'train')
            args = TrainingArguments(output_dir=str(root / 'args'), use_cpu=True, report_to='none')
            cross = train.preprocess_dataset(raw, tokenizer, 6, args, num_proc=2)
            isolated = train.preprocess_dataset(raw, tokenizer, 6, args, num_proc=2, isolate_documents=True)
            self.assertNotEqual(cross._fingerprint, isolated._fingerprint)
            self.assertEqual(cross['input_ids'], isolated['input_ids'])
            # Global indices remain distinct across multiprocessing shards.
            self.assertGreater(max(max(s) for s in isolated['segments']), len(raw) // 2)
            times = {p: p.stat().st_mtime_ns for p in root.rglob('cache-*.arrow')}
            cached = train.preprocess_dataset(raw, tokenizer, 6, args, num_proc=2, isolate_documents=True)
            self.assertEqual(times, {p: p.stat().st_mtime_ns for p in root.rglob('cache-*.arrow')})
            rebuilt = train.preprocess_dataset(raw, tokenizer, 6, args, num_proc=2,
                                               isolate_documents=True, overwrite_cache=True)
            self.assertEqual(isolated.to_dict(), cached.to_dict())
            self.assertEqual(isolated.to_dict(), rebuilt.to_dict())
            # Production collator must exactly match the B200-tested semantics.
            rows = list(isolated.select(range(4)))
            torch.testing.assert_close(isolated_data_collator(rows), Collator('sdpa_isolated')(rows), rtol=0, atol=0)
            with self.assertRaises(KeyError):
                isolated_data_collator(list(cross.select(range(2))))
            seen = set()
            with patch.object(train, 'DeepKVTrainer', RecordingTrainer):
                for arm in ARMS:
                    invoke(root, {**cfg, 'arm': arm, 'output_dir': str(root / arm)})
                    saved = json.loads((root / arm / 'train_config.json').read_text())
                    self.assertTrue(saved['data']['isolate_documents'])
            self.assertEqual(seen, {(arm, phase) for arm in ARMS for phase in (True, False)})
            # Even performance-change permission cannot change isolation on resume.
            with self.assertRaisesRegex(ValueError, 'Resume configuration'):
                invoke(root, {**cfg, 'output_dir': str(root / 'A'), 'isolate_documents': False,
                              'allow_performance_change_on_resume': True})
            with self.assertRaisesRegex(ValueError, 'document-aware attention'):
                invoke(root, {**cfg, 'causal_attention': True})

    def test_native_gradient_accumulation_scaling(self):
        from tests.test_deep_kv import model, objective
        ids = torch.arange(24).reshape(4, 6) % 30
        with tempfile.TemporaryDirectory() as tmp:
            for arm in "ABCDEFG":
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
                objective(whole, result).backward()
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

    def test_functional_trainer_weights_metrics_and_gradient(self):
        from types import SimpleNamespace
        from tests.test_deep_kv import model, parameter_hash
        models = {arm: model(arm) for arm in "BFG"}
        self.assertEqual(parameter_hash(models["B"]), parameter_hash(models["F"]))
        self.assertEqual(parameter_hash(models["F"]), parameter_hash(models["G"]))
        torch.nn.init.normal_(models["B"].aux.out.weight, std=.03)
        for arm in "FG":
            models[arm].load_state_dict(models["B"].state_dict())
        ids = torch.arange(24).reshape(4, 6) % 30
        ctx = Context(ids, torch.ones_like(ids, dtype=torch.bool), torch.arange(6).expand_as(ids))
        losses, route, message, gradients = {}, {}, {}, {}
        with tempfile.TemporaryDirectory() as tmp:
            for arm, current in models.items():
                trainer = DeepKVTrainer(model=current, compute_metrics=compute_metrics,
                    args=TrainingArguments(output_dir=tmp, use_cpu=True, report_to="none"))
                loss, outputs = trainer.compute_loss(current, {"input_ids": ids}, return_outputs=True)
                losses[arm] = loss.detach()
                metrics = trainer.compute_metrics(SimpleNamespace(predictions=outputs["statistics"].numpy()))
                self.assertAlmostEqual(metrics["loss"], loss.item(), places=5)
                if arm in "FG":
                    self.assertEqual(current.kv_loss_weight, .3)
                    self.assertEqual(outputs["route_count"].item(), 20)
                    route[arm] = outputs["route_sum"].detach() / 20
                    message[arm] = outputs["msg_sum"].detach() / 20
                    self.assertNotIn("loss_k", metrics)
                    self.assertNotIn("k_sum", outputs)
                loss.backward()
                gradients[arm] = {name: p.grad.clone() for name, p in current.named_parameters()}
        torch.testing.assert_close(route["F"], route["G"], rtol=0, atol=0)
        self.assertEqual(message["F"], 0.)
        self.assertGreater(message["G"], 0.)
        torch.testing.assert_close(losses["F"], losses["B"] + .3 * route["F"])
        torch.testing.assert_close(losses["G"], losses["F"] + .3 * message["G"])
        # G's additional gradient must be exactly .3 times the message gradient;
        # routing retains F's coefficient, and the shared query stays detached.
        current = models["G"]
        current.zero_grad(set_to_none=True)
        output = current(ctx)
        (.3 * output["msg_sum"] / output["route_count"]).backward()
        for name, p in current.named_parameters():
            extra = torch.zeros_like(p) if p.grad is None else p.grad
            torch.testing.assert_close(gradients["G"][name], gradients["F"][name] + extra,
                                       atol=2e-6, rtol=2e-5, msg=name)

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
            selected_jobs = jobs("deep_kv.b200.json", 2500, arms="DEFG")["jobs"][:-1]
            normalized = []
            for job in selected_jobs:
                argv = list(job["argv"])
                for key in ("--arm", "--output_dir", "--run_name"):
                    argv[argv.index(key) + 1] = "ARM_SPECIFIC"
                normalized.append(argv)
            self.assertTrue(all(argv == normalized[0] for argv in normalized))
            self.assertTrue(all(job["gpus"] == list(range(8)) for job in selected_jobs))
            fg = jobs("deep_kv.b200.json", arms="FG")["jobs"]
            self.assertEqual([j["name"] for j in fg], ["arm-F", "arm-G", "compare"])
            for job, arm in zip(fg, "FG"):
                argv = job["argv"]
                start = next(i for i,a in enumerate(argv) if a.endswith("/train.py")) + 1
                values = parser.parse_args_into_dataclasses(argv[start:] + ["--use_cpu", "true", "--bf16", "false", "--report_to", "none"])
                self.assertEqual(values[2].arm, arm)
            for invalid in ("", "EE", "H"):
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
