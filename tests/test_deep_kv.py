"""Mechanism acceptance, budget/resume, and fixed-stream contracts (CPU only)."""
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import transformers
from transformers import Qwen3Config
from deep_kv.config import Recipe
from deep_kv.data import load_data
from deep_kv.model import DeepKV
from deep_kv.training import parameter_hash, train, DeepKVTrainer, training_arguments, verify_checkpoint
from pcc.model import Context


def config():
    result = Qwen3Config(vocab_size=32, hidden_size=32, intermediate_size=48,
                        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                        head_dim=8, max_position_embeddings=64, attention_dropout=0.0,
                        tie_word_embeddings=True)
    result._attn_implementation = "sdpa"
    return result


def model(arm, checkpoint=False):
    return DeepKV.from_scratch(config(), arm, consumer=2, deep_target=4,
                               checkpoint_layers=checkpoint, lm_chunk=3)


def context():
    ids = torch.tensor([[3, 8, 2, 6, 9, 1], [4, 7, 8, 0, 0, 0]])
    valid = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]], dtype=torch.bool)
    positions = torch.tensor([[0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 0, 0]])
    segments = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]])
    return Context(ids, valid, positions, segments)


@unittest.skipUnless(transformers.__version__ == "5.9.0", "Deep-KV uses the B200-matched Transformers 5.9.0 environment")
class DeepKVAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_base_equivalence_and_identical_initialization(self):
        batch = context()
        base = model("A").eval()
        with torch.no_grad():
            native = base.backbone(input_ids=batch.input_ids, attention_mask=batch.additive_mask(torch.float32),
                                   position_ids=batch.position_ids, use_cache=False).logits
            base_logits = base.backbone.lm_head(base.hidden_states(batch)[0])
        torch.testing.assert_close(base_logits, native)
        branch_hash = None
        for arm in "BCD":
            m = model(arm).eval()
            self.assertEqual(parameter_hash(base.backbone), parameter_hash(m.backbone))
            if branch_hash is None:
                branch_hash = parameter_hash(m.aux)
            self.assertEqual(branch_hash, parameter_hash(m.aux))
            with torch.no_grad():
                logits = m.backbone.lm_head(m.hidden_states(batch)[0])
            torch.testing.assert_close(logits, native, atol=1e-6, rtol=1e-5)

    def test_strict_mask_empty_rows_and_reference_attention(self):
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv
        m = model("B")
        torch.nn.init.normal_(m.aux.out.weight, std=.1)
        batch = context()
        u = torch.randn(2, 6, 32, requires_grad=True)
        q = torch.randn(2, 4, 6, 8)
        rotary = m.backbone.model.rotary_emb(u, batch.position_ids)
        qr, _ = apply_rotary_pos_emb(q, q, *rotary)
        actual, k, v = m.aux(u, qr, rotary, batch.allowed())
        _, kr = apply_rotary_pos_emb(qr, k, *rotary)
        allowed = batch.allowed() & torch.ones(6, 6, dtype=torch.bool).tril(-1)
        nonempty = allowed.any(-1, keepdim=True)
        scores = qr @ repeat_kv(kr, 2).transpose(-1, -2) / 8 ** .5
        scores = scores.masked_fill(~allowed, -torch.inf)
        weights = torch.where(nonempty, scores, torch.zeros_like(scores)).softmax(-1).masked_fill(~allowed, 0)
        expected = m.aux.out((weights @ repeat_kv(v, 2)).transpose(1, 2).reshape(2, 6, -1))
        torch.testing.assert_close(actual, expected)
        self.assertEqual(weights.masked_select(~allowed.expand_as(weights)).count_nonzero().item(), 0)
        self.assertTrue(torch.equal(actual[0, 0], torch.zeros(32)))
        self.assertTrue(torch.equal(actual[0, 3], torch.zeros(32)))
        self.assertTrue(torch.equal(actual[1, 3:], torch.zeros(3, 32)))
        actual[0, 1].sum().backward()
        self.assertEqual(u.grad[0, 1:].count_nonzero().item(), 0)
        changed = u.detach().clone()
        changed[0, 2:] += 100
        after = m.aux(changed, qr, rotary, batch.allowed())[0]
        torch.testing.assert_close(actual[0, :2], after[0, :2])

    def test_native_targets_exact_and_one_projection_per_forward(self):
        for arm, target_index in (("C", 4), ("D", 20)):
            cfg = config()
            cfg.num_hidden_layers = 28
            cfg.layer_types = ["full_attention"] * 28
            m = DeepKV.from_scratch(cfg, arm, checkpoint_layers=False).eval()
            captures, handles = {}, []
            def hook(name):
                def save(module, args, output):
                    captures.setdefault(name, []).append(output)
                return save
            for i, layer in enumerate(m.backbone.model.layers):
                for key in ("q_proj", "k_norm", "v_proj"):
                    handles.append(getattr(layer.self_attn, key).register_forward_hook(hook((i, key))))
            try:
                _, predicted, target = m.hidden_states(context())
            finally:
                for h in handles:
                    h.remove()
            self.assertTrue(all(len(x) == 1 for x in captures.values()))
            native_k = captures[(target_index, "k_norm")][0].transpose(1, 2)
            native_v = captures[(target_index, "v_proj")][0].view(2, 6, 2, 8).transpose(1, 2)
            self.assertTrue(torch.equal(target[0], native_k))
            self.assertTrue(torch.equal(target[1], native_v))
            self.assertEqual(predicted[0].shape, (2, 2, 6, 8))

    def test_target_stopgrad_and_padding_exclusion(self):
        for arm, target_index in (("C", 1), ("D", 3)):
            m = model(arm)
            batch = context()
            _, predicted, target = m.hidden_states(batch)
            for t in target:
                t.retain_grad()
            k, v = m.alignment(predicted, target, batch.valid)
            expected = [(p - t.detach()).abs().permute(0, 2, 1, 3)[batch.valid].mean() for p, t in zip(predicted, target)]
            torch.testing.assert_close(k / batch.valid.sum(), expected[0])
            torch.testing.assert_close(v / batch.valid.sum(), expected[1])
            ((k + v) / (2 * batch.valid.sum())).backward()
            self.assertTrue(all(t.grad is None for t in target))
            a = m.backbone.model.layers[target_index].self_attn
            self.assertIsNone(a.k_proj.weight.grad)
            self.assertIsNone(a.v_proj.weight.grad)
            self.assertGreater(m.aux.k.weight.grad.abs().sum().item(), 0)
            self.assertGreater(m.aux.v.weight.grad.abs().sum().item(), 0)
            self.assertGreater(m.backbone.model.embed_tokens.weight.grad.abs().sum().item(), 0)

    def test_checkpoint_recomputation_and_lm_gradient_path(self):
        for arm in "ABCD":
            plain, checked = model(arm), model(arm, checkpoint=True)
            if plain.aux is not None:
                torch.nn.init.normal_(plain.aux.out.weight, std=.03)
                checked.load_state_dict(plain.state_dict())
            for m in (plain, checked):
                result = m(context())
                loss = result["lm_sum"] / result["lm_count"] + (result["k_sum"] + result["v_sum"]) / (2 * result["kv_count"])
                loss.backward()
            for (name, p), (_, q) in zip(plain.named_parameters(), checked.named_parameters()):
                self.assertIsNotNone(p.grad, name)
                torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-5, msg=name)
            if plain.aux is not None:
                plain.zero_grad(set_to_none=True)
                plain(context())["lm_sum"].backward()
                self.assertGreater(plain.aux.k.weight.grad.abs().sum().item(), 0)
                self.assertGreater(plain.backbone.model.layers[-1].self_attn.v_proj.weight.grad.abs().sum().item(), 0)

    def test_bfloat16_forward_backward(self):
        reference = None
        for arm in "ABCD":
            m = model(arm, checkpoint=True)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                result = m(context())
                loss = result["lm_sum"] / result["lm_count"] + (result["k_sum"] + result["v_sum"]) / (2 * result["kv_count"])
            if reference is None:
                reference = result["lm_sum"].detach()
            torch.testing.assert_close(result["lm_sum"], reference, atol=1e-5, rtol=1e-5)
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_query_width_can_exceed_hidden_size_as_in_qwen_06b(self):
        cfg = config()
        cfg.head_dim = 16  # four query heads = 64 features, hidden size = 32
        m = DeepKV.from_scratch(cfg, "D", consumer=2, deep_target=4)
        self.assertEqual(m.aux.out.in_features, 64)
        result = m(context())
        loss = result["lm_sum"] / result["lm_count"] + (result["k_sum"] + result["v_sum"]) / (2 * result["kv_count"])
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_modified_arms_have_identical_lm_path_after_branch_learns(self):
        # Zero-output equivalence alone would hide mistakes in the live branch.
        for bf16 in (False, True):
            models = [model(arm, checkpoint=True) for arm in "BCD"]
            torch.nn.init.normal_(models[0].aux.out.weight, std=.03)
            for other in models[1:]:
                other.load_state_dict(models[0].state_dict())
            reference = None
            for current in models:
                with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
                    result = current(context())
                if reference is None:
                    reference = result["lm_sum"].detach()
                torch.testing.assert_close(result["lm_sum"], reference, atol=1e-6, rtol=1e-6)
                result["lm_sum"].backward()
            for other in models[1:]:
                for (name, p), (_, q) in zip(models[0].named_parameters(), other.named_parameters()):
                    torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-6, msg=name)


def toy_data(recipe):
    from datasets import Dataset
    result = []
    for split, rows in (("train", recipe.train_rows), ("eval", recipe.eval_rows)):
        ids = np.random.default_rng(42 + (split == "eval")).integers(0, 31, (rows, recipe.context))
        result.append(Dataset.from_dict({"input_ids": ids.tolist(), "labels": ids.tolist(),
                                         "attention_mask": np.ones_like(ids).tolist()}).shuffle(seed=recipe.seed))
    return tuple(result)


@unittest.skipUnless(transformers.__version__ == "5.9.0", "Deep-KV uses the B200-matched Transformers 5.9.0 environment")
class DeepKVTraining(unittest.TestCase):
    def test_hf_cached_text_pipeline_and_cache_miss_order(self):
        from datasets import Dataset
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast
        recipe = Recipe(dataloader_workers=0, updates=3, context=6, tokens_per_update=12, warmup=1,
                        monitor_rows=1, eval_rows=2, consumer=2, deep_target=4)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backend = Tokenizer(models.WordLevel({**{str(i): i for i in range(31)}, "<|endoftext|>": 31}, unk_token="0"))
            backend.pre_tokenizer = pre_tokenizers.Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="0", eos_token="<|endoftext|>")
            tokenizer.save_pretrained(root / "tokenizer")
            Dataset.from_dict({"text": ["1 2 3"] * 20}).save_to_disk(root / "train" / "shard_0000")
            Dataset.from_dict({"text": ["7 8 9"] * 20}).save_to_disk(root / "train" / "shard_0001")
            Dataset.from_dict({"text": ["4 5 6"] * 20}).save_to_disk(root / "eval")
            inputs = {"tokenizer": str(root / "tokenizer"), "train_data": str(root / "train"),
                      "eval_data": str(root / "eval"), "preprocessing_num_workers": 2}
            args = training_arguments(recipe, root / "args", 1, cpu=True, mixed_precision=False)
            data, dev, meta = load_data(inputs, recipe, args)
            self.assertGreater(len(data), recipe.train_rows)  # no prefix truncation
            self.assertEqual(len(dev), recipe.eval_rows)
            self.assertEqual(dev[0]["input_ids"], [4, 5, 6, 31, 4, 5])
            # Each of the two map workers handles 20 four-token documents:
            # 80 tokens -> 13 contexts, dropping two tokens at its boundary.
            expected = []
            for document in ([1, 2, 3, 31], [7, 8, 9, 31]):
                flat = document * 20
                expected.extend(flat[i:i + 6] for i in range(0, 78, 6))
            baseline = Dataset.from_dict({"input_ids": expected, "labels": expected,
                                          "attention_mask": [[1] * 6] * 26}).shuffle(seed=42)
            self.assertEqual(data.to_dict(), baseline.to_dict())
            caches = {p: p.stat().st_mtime_ns for p in root.rglob("cache-*.arrow")}
            cached, _, cached_meta = load_data(inputs, replace(recipe, logging_every=1), args)
            self.assertEqual(meta, cached_meta)
            self.assertEqual(data.to_dict(), cached.to_dict())
            self.assertEqual(caches, {p: p.stat().st_mtime_ns for p in root.rglob("cache-*.arrow")})
            rebuilt, _, rebuilt_meta = load_data({**inputs, "overwrite_cache": True}, recipe, args)
            self.assertEqual(meta, rebuilt_meta)
            self.assertEqual(data.to_dict(), rebuilt.to_dict())
            self.assertTrue(any(p.stat().st_mtime_ns != mtime for p, mtime in caches.items()))
            # Native Trainer sampling must also produce the same example order
            # across arms, including an arm that rebuilt its own cache.
            orders = []
            for arm, dataset in zip("ABCD", (data, cached, rebuilt, data)):
                trainer = DeepKVTrainer(model=model(arm), args=args, train_dataset=dataset, eval_dataset=dev)
                orders.append([batch["input_ids"].tolist() for batch in trainer.get_train_dataloader()])
            self.assertTrue(all(order == orders[0] for order in orders))
            with self.assertRaisesRegex(ValueError, "Insufficient train data"):
                load_data(inputs, replace(recipe, updates=100), args)

    def test_offline_wandb_per_arm_and_override(self):
        import wandb
        # Stop the SDK service explicitly; unittest exits with a bool, which
        # this SDK's protobuf atexit handler rejects as an integer exit code.
        self.addCleanup(wandb.teardown, exit_code=0)
        recipe = Recipe(dataloader_workers=0, updates=3, context=6, tokens_per_update=12,
                        warmup=1, monitor_rows=1, eval_rows=2, consumer=2, deep_target=4)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"WANDB_SILENT": "true"}):
            os.environ.pop("WANDB_DIR", None)
            root = Path(tmp)
            data, dev = toy_data(recipe)
            meta = {"fingerprint": data._fingerprint}

            def with_wandb(*args, **kwargs):
                result = training_arguments(*args, **kwargs)
                result.report_to = ["wandb"]
                return result

            # Opt CPU tests into the real offline SDK/HF callback, without GPUs.
            with patch("deep_kv.training.training_arguments", side_effect=with_wandb):
                for arm in "ABC":
                    directory = root / arm
                    if arm == "C":
                        os.environ["WANDB_DIR"] = str(root / "operator-logs")
                    train(model(arm), data, dev, recipe, directory,
                          {"recipe": asdict(recipe), "arm": arm},
                          mixed_precision=False, stop_after=1)
                    log_root = root / "operator-logs" if arm == "C" else directory
                    self.assertEqual(len(list((log_root / "wandb").glob("offline-run-*/*.wandb"))), 1)
                    self.assertIsNone(wandb.run)
                self.assertFalse((root / "C/wandb").exists())
                self.assertEqual(os.environ["WANDB_DIR"], str(root / "operator-logs"))
                os.environ.pop("WANDB_DIR")
                with patch.object(DeepKVTrainer, "train", side_effect=RuntimeError("injected failure")):
                    with self.assertRaisesRegex(RuntimeError, "injected failure"):
                        train(model("D"), data, dev, recipe, root / "D",
                              {"recipe": asdict(recipe), "arm": "D"}, mixed_precision=False)
                self.assertIsNone(wandb.run)
                self.assertNotIn("WANDB_DIR", os.environ)

    def test_packing_resume_and_fixed_budget(self):
        torch.set_num_threads(1)
        # Exercise worker prefetch plus HF's skipped-batch restore, not only the
        # synchronous loader used by the smallest mechanism tests.
        recipe = Recipe(dataloader_workers=2, logging_every=1, updates=3, context=6, tokens_per_update=12, warmup=1, eval_every=1,
                        monitor_rows=1, eval_rows=2, checkpoint_every=1, consumer=2, deep_target=4)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, dev = toy_data(recipe)
            full = model("D", checkpoint=True)
            resumed = model("D", checkpoint=True)
            identity = {"recipe": asdict(recipe), "arm": "D"}
            train(full, data, dev, recipe, root / "full", identity, mixed_precision=False)
            train(resumed, data, dev, recipe, root / "resumed", identity, mixed_precision=False, stop_after=2)
            self.assertFalse((root / "resumed/complete.json").exists())
            changed = replace(recipe, logging_every=2)
            # Reusable data does not make an existing run resumable under new
            # operational settings: full run identity remains authoritative.
            with self.assertRaisesRegex(ValueError, "Run receipt differs"):
                train(model("D"), data, dev,
                      changed, root / "resumed", {"recipe": asdict(changed), "arm": "D"},
                      mixed_precision=False, resume=True)
            with self.assertRaisesRegex(ValueError, "precedes checkpoint"):
                train(resumed, data, dev, recipe, root / "resumed", identity,
                      mixed_precision=False, resume=True, stop_after=1)
            self.assertTrue((root / "resumed/stopped.json").exists())
            train(resumed, data, dev, recipe, root / "resumed", identity, mixed_precision=False, resume=True)
            for p, q in zip(full.parameters(), resumed.parameters()):
                torch.testing.assert_close(p, q, atol=0, rtol=0)
            report = json.loads((root / "resumed/complete.json").read_text())
            self.assertEqual(report["input_tokens"], 36)
            self.assertEqual(report["final_evaluation"]["rows"], 2)
            self.assertFalse((root / "resumed/stopped.json").exists())
            # Simulate a crash after the final checkpoint was safely saved but
            # before final JSON publication. Resume must restore both receipts.
            (root / "resumed/complete.json").unlink()
            (root / "resumed/metrics.json").unlink()
            recovered = model("D", checkpoint=True)
            train(recovered, data, dev, recipe, root / "resumed", identity, mixed_precision=False, resume=True)
            saved_history = json.loads((root / "resumed/metrics.json").read_text())
            self.assertEqual(saved_history["train"][-1]["update"], 3)
            for p, q in zip(full.parameters(), recovered.parameters()):
                torch.testing.assert_close(p, q, atol=0, rtol=0)
            checkpoint = root / "resumed/checkpoint-3"
            verify_checkpoint(checkpoint, identity)
            scheduler = checkpoint / "scheduler.pt"
            original = scheduler.read_bytes()
            scheduler.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify_checkpoint(checkpoint, identity)
            scheduler.write_bytes(original)

    def test_queue_and_budget(self):
        from deep_kv.__main__ import build_identity, jobs
        from deep_kv.config import load_config, plan
        from run_experiments import load_jobs
        self.assertEqual(Recipe().tokens_per_update, 1048576)
        self.assertEqual(Recipe().warmup, 500)
        self.assertEqual(Recipe().checkpoint_every, 250)
        self.assertEqual(Recipe().train_rows, 14643200)
        self.assertEqual(Recipe().updates * Recipe().tokens_per_update, 29989273600)
        self.assertEqual(Recipe().tokens_per_update // Recipe().context // 8, 64)
        identity = build_identity({}, Recipe(), model("D"), {}, mixed_precision=False)
        self.assertEqual(json.loads(json.dumps(identity)), identity)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "jobs.json"
            path.write_text(json.dumps(jobs("deep_kv.b200.json")))
            queue = load_jobs(path)
            self.assertEqual(len(queue), 5)
            for job, arm in zip(queue[:4], "ABCD"):
                self.assertEqual(job["gpus"], list(range(8)))
                self.assertIn("accelerate.commands.launch", job["argv"])
                self.assertIn("--multi_gpu", job["argv"])
                self.assertNotIn("--data-dir", job["argv"])
                self.assertNotIn("prepare", job["argv"])
                self.assertEqual(job["required_outputs"][0]["json_equals"]["arm"], arm)
                self.assertEqual(job["required_outputs"][0]["json_equals"]["input_tokens"], 29989273600)
            path.write_text(json.dumps(jobs("deep_kv.b200.json", stop_after=2000)))
            queue = load_jobs(path)
            for job, arm in zip(queue[:4], "ABCD"):
                self.assertEqual(job["argv"][-2:], ["--stop-after", "2000"])
                self.assertEqual(job["required_outputs"][0], {"path": f"{arm}/stopped.json", "json_equals": {
                    "status": "stopped", "arm": arm, "update": 2000, "input_tokens": 2097152000}})
            self.assertEqual(queue[-1]["argv"][-2:], ["--stop-after", "2000"])
            self.assertFalse(queue[-1]["required_outputs"][0]["json_equals"]["training_complete"])
            for invalid in (0, -1, 28601, True, 1.5):
                with self.assertRaises(ValueError):
                    jobs("deep_kv.b200.json", stop_after=invalid)
            planned = plan(load_config("deep_kv.b200.json"), 2000)
            self.assertEqual(planned["run_tokens_per_arm"], 2097152000)
            self.assertEqual(planned["run_total_input_tokens"], 8388608000)
            self.assertEqual(planned["recipe"]["updates"], 28600)
            for microbatch in (1, 2, 4, 8, 16, 32, 64, 3, 128):
                path.write_text(json.dumps({**json.loads(Path("deep_kv.b200.json").read_text()), "microbatch": microbatch}))
                if microbatch in (3, 128):
                    with self.assertRaises(ValueError):
                        load_config(path)
                else:
                    self.assertEqual(load_config(path)["microbatch"], microbatch)

    def test_interrupted_save_preserves_certified_checkpoint(self):
        from deep_kv.training import write_json
        recipe = Recipe(dataloader_workers=0, updates=5, context=6, tokens_per_update=12,
                        warmup=1, monitor_rows=1, eval_rows=2, checkpoint_every=1,
                        consumer=2, deep_target=4)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, dev = toy_data(recipe)
            meta = {"fingerprint": data._fingerprint}
            output = root / "run"
            identity = {"recipe": asdict(recipe), "arm": "D"}
            train(model("D"), data, dev, recipe, output, identity,
                  mixed_precision=False, stop_after=2)
            # A failed later save was ignored during restore. The user now
            # selects an earlier cutoff, still after the last certified save.
            (output / "checkpoint-4").mkdir()

            def fail_certification(path, value):
                if Path(path) == output / "checkpoint-3/deep_kv.json":
                    raise OSError("simulated interrupted certification")
                return write_json(path, value)

            with patch("deep_kv.training.write_json", side_effect=fail_certification):
                with self.assertRaisesRegex(OSError, "interrupted certification"):
                    train(model("D"), data, dev, recipe, output, identity,
                          mixed_precision=False, stop_after=3, resume=True)
            verify_checkpoint(output / "checkpoint-2", identity)
            train(model("D"), data, dev, recipe, output, identity,
                  mixed_precision=False, stop_after=3, resume=True)
            verify_checkpoint(output / "checkpoint-3", identity)
            self.assertTrue((output / "checkpoint-2").exists())
            self.assertFalse((output / "checkpoint-1").exists())

    def test_hf_accumulation_gradient_matches_whole_global_batch(self):
        torch.set_num_threads(1)
        recipe = Recipe(dataloader_workers=0, updates=3, context=6, tokens_per_update=24,
                        warmup=1, monitor_rows=1, eval_rows=2, consumer=2, deep_target=4)
        ids = torch.arange(24).reshape(4, 6) % 30
        with tempfile.TemporaryDirectory() as tmp:
            for arm in "ABCD":
                accumulated, whole = model(arm), model(arm)
                if accumulated.aux is not None:
                    torch.nn.init.normal_(accumulated.aux.out.weight, std=.03)
                    whole.load_state_dict(accumulated.state_dict())
                args = training_arguments(recipe, tmp, 1, cpu=True, mixed_precision=False)
                args.eval_strategy = "no"
                trainer = DeepKVTrainer(model=accumulated, args=args)
                trainer.current_gradient_accumulation_steps = 4
                for row in ids:
                    trainer.training_step(accumulated, {"input_ids": row[None]})
                batch = Context(ids, torch.ones_like(ids, dtype=torch.bool), torch.arange(6).expand_as(ids))
                result = whole(batch)
                loss = result["lm_sum"] / 20 + (result["k_sum"] + result["v_sum"]) / 48
                loss.backward()
                for (name, p), (_, q) in zip(accumulated.named_parameters(), whole.named_parameters()):
                    torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-5, msg=name)

    def test_cutoff_report_and_full_schedule_resume(self):
        from deep_kv.__main__ import build_identity
        from deep_kv.report import report
        torch.set_num_threads(1)
        recipe = Recipe(dataloader_workers=0, logging_every=1, updates=4, context=6, tokens_per_update=12, warmup=1, eval_every=2,
                        monitor_rows=1, eval_rows=2, checkpoint_every=2, consumer=2, deep_target=4)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, dev = toy_data(recipe)
            meta = {"fingerprint": data._fingerprint}
            identities = {}
            for arm in "ABCD":
                current = model(arm)
                identities[arm] = build_identity({}, recipe, current, meta, mixed_precision=False)
                history = train(current, data, dev, recipe, root / arm, identities[arm],
                                mixed_precision=False, stop_after=2)
                # Native HF schedule: step 1 uses LR 0; step 2 uses the peak
                # after one warmup update. The cutoff never shortens the curve.
                self.assertEqual(history["train"][-1]["lr"], recipe.learning_rate)
                self.assertEqual(history["evaluation"][-1]["rows"], 2)
                self.assertFalse((root / arm / "complete.json").exists())
                state = json.loads((root / arm / "checkpoint-2/trainer_state.json").read_text())
                for row in state["log_history"]:
                    if "eval_objective" in row:
                        self.assertEqual(row["eval_loss"], row["eval_objective"])
            result = report(root, recipe=recipe, stop_after=2)
            self.assertFalse(result["training_complete"])
            self.assertEqual(result["compared_update"], 2)
            self.assertEqual(result["input_tokens_per_arm"], 24)
            with self.assertRaises(FileNotFoundError):
                report(root, recipe=recipe)
            with self.assertRaises(ValueError):
                report(root, recipe=recipe, stop_after=1)
            # A periodic checkpoint at the chosen cutoff may only have monitor
            # metrics. Resume at that same step must add full evaluation/save.
            checkpoint = root / "A/checkpoint-2/deep_kv.json"
            saved = json.loads(checkpoint.read_text())
            saved["history"]["evaluation"][-1]["rows"] = 1
            checkpoint.write_text(json.dumps(saved))
            train(model("A"), data, dev, recipe, root / "A", identities["A"],
                  mixed_precision=False, resume=True, stop_after=2)
            self.assertFalse(report(root, recipe=recipe, stop_after=2)["training_complete"])
            for arm in "ABCD":
                train(model(arm), data, dev, recipe, root / arm, identities[arm],
                      mixed_precision=False, resume=True)
            self.assertTrue(report(root, recipe=recipe)["training_complete"])


if __name__ == "__main__":
    unittest.main()
