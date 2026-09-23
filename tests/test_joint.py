import copy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from pcc.data import PreparedContexts
from pcc.joint_config import JointSettings, load_config
from pcc.joint_model import JointQwen
from pcc.joint_training import (evaluate, make_optimizer, train, train_update,
                                load_checkpoint, schedule)
from pcc.model import Context, CorrectionAdapter


def tiny(arm="Deep", checkpoint_layers=True):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        config = Qwen3Config(vocab_size=67, hidden_size=32, intermediate_size=64,
            num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1,
            head_dim=128, rope_theta=1_000_000, attention_dropout=0., tie_word_embeddings=True)
        config._attn_implementation = "eager"
        return JointQwen(Qwen3ForCausalLM(config), arm, s=1, d=2, checkpoint_layers=checkpoint_layers)


def contexts(rows=8, length=8):
    data = PreparedContexts.__new__(PreparedContexts)
    data.ids = np.random.default_rng(21).integers(0, 67, (rows, length))
    data.valid = np.ones((rows, length), dtype=bool)
    data.positions = np.broadcast_to(np.arange(length), (rows, length))
    data.segments = None
    data.path = Path("synthetic")
    return data


def settings():
    return JointSettings(updates=4, context=8, tokens_per_update=16, warmup=1,
                         eval_every=2, monitor_tokens=16, dev_tokens=32, s=1, d=2)


class JointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_all_zero_init_arms_match_native_full_forward(self):
        context = contexts(2).batch(0, 2, "cpu")
        for arm in ("Base", "Shallow", "Deep"):
            model = tiny(arm)
            for bf16 in (False, True):
                with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
                    reference = model.model.model(input_ids=context.input_ids,
                        attention_mask=context.additive_mask(torch.float32),
                        position_ids=context.position_ids, use_cache=False).last_hidden_state
                    actual = model(context)
                torch.testing.assert_close(actual, reference, atol=0, rtol=0)

    def test_deep_source_receives_gradient_with_nonzero_branch(self):
        model = tiny(checkpoint_layers=False)
        torch.nn.init.normal_(model.adapter.out.weight, std=.02)
        outputs = []
        def capture(module, args, output):
            output.retain_grad()
            outputs.append(output)
        hook = model.model.model.layers[1].register_forward_hook(capture)
        context = contexts(2).batch(0, 2, "cpu")
        model.losses(model(context), context)[0].sum().backward()
        hook.remove()
        self.assertEqual(len(outputs), 2)
        for value in outputs:
            self.assertIsNotNone(value.grad)
            self.assertGreater(float(value.grad.abs().sum()), 0)

    def test_checkpointed_gradients_match_direct_gradients(self):
        direct = tiny(checkpoint_layers=False)
        torch.nn.init.normal_(direct.adapter.out.weight, std=.02)
        checked = copy.deepcopy(direct)
        checked.checkpoint_layers = True
        context = contexts(2).batch(0, 2, "cpu")
        for model in (direct, checked):
            model.losses(model(context), context)[0].sum().backward()
        for a, b in zip(direct.parameters(), checked.parameters()):
            self.assertIsNotNone(a.grad)
            torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-5)

    def test_backbone_and_branch_really_update_all_arms(self):
        for arm in ("Base", "Shallow", "Deep"):
            model = tiny(arm)
            original = {name: p.detach().clone() for name, p in model.named_parameters()}
            opt = make_optimizer(model)
            for step in (1, 2):
                train_update(model, opt, contexts(), step, settings(), 1, mixed_precision=False)
            for name in ("model.model.embed_tokens.weight", "model.model.layers.0.mlp.down_proj.weight",
                         "model.model.layers.2.mlp.down_proj.weight"):
                self.assertFalse(torch.equal(original[name], dict(model.named_parameters())[name]))
            if arm != "Base":
                self.assertFalse(torch.equal(original["adapter.out.weight"], model.adapter.out.weight))
                self.assertFalse(torch.equal(original["adapter.q.weight"], model.adapter.q.weight))
            self.assertIs(model.model.lm_head.weight, model.model.model.embed_tokens.weight)
            for state in opt.state.values():
                self.assertEqual(state["exp_avg"].dtype, torch.float32)
                self.assertEqual(state["exp_avg_sq"].dtype, torch.float32)

    def test_accumulation_matches_global_update(self):
        data = contexts()
        for arm in ("Base", "Shallow", "Deep"):
            a, b = tiny(arm), tiny(arm)
            # SGD makes this an accumulation/normalization test. AdamW can
            # amplify fp32 near-zero gradient differences through epsilon.
            def sgd(model):
                return torch.optim.SGD([{"params": list(model.parameters()), "peak_lr": 1e-3}], lr=1e-3)
            ra = train_update(a, sgd(a), data, 1, settings(), 1, mixed_precision=False)
            rb = train_update(b, sgd(b), data, 1, settings(), 2, mixed_precision=False)
            self.assertAlmostEqual(ra["nll"], rb["nll"], places=6)
            for x, y in zip(a.parameters(), b.parameters()):
                torch.testing.assert_close(x, y, atol=2e-7, rtol=1e-5)

    def test_future_and_other_segments_cannot_affect_predictions(self):
        for arm in ("Base", "Shallow", "Deep"):
            model = tiny(arm)
            if model.adapter:
                torch.nn.init.normal_(model.adapter.out.weight, std=.1)
            context = contexts(1).batch(0, 1, "cpu")
            changed = copy.deepcopy(context)
            changed.input_ids[:, 5:] = (changed.input_ids[:, 5:] + 9) % 67
            with torch.no_grad():
                torch.testing.assert_close(model(context)[:, :5], model(changed)[:, :5], atol=0, rtol=0)
            context.segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
            changed = copy.deepcopy(context)
            changed.input_ids[:, :4] = (changed.input_ids[:, :4] + 9) % 67
            with torch.no_grad():
                torch.testing.assert_close(model(context)[:, 4:], model(changed)[:, 4:], atol=0, rtol=0)

    def test_native_token_loss_and_partial_context_counts(self):
        model = tiny("Base")
        data = contexts(2)
        data.valid[1, -3:] = False
        context = data.batch(0, 2, "cpu")
        with torch.no_grad():
            logits = model.model(input_ids=context.input_ids,
                attention_mask=context.additive_mask(torch.float32), position_ids=context.position_ids,
                use_cache=False).logits
            expected = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 67),
                context.input_ids[:, 1:].masked_fill(~context.targets(), -100).reshape(-1), reduction="sum")
        sums, counts = evaluate(model, data, 1, mixed_precision=False)
        self.assertEqual(counts.tolist(), [7, 4])
        self.assertAlmostEqual(sums.sum(), float(expected), places=5)

    def test_resume_matches_uninterrupted_training(self):
        data, dev = contexts(), contexts(4)
        for arm in ("Base", "Shallow", "Deep"):
            identity = {"settings": asdict(settings()), "arm": arm, "test_identity": "fixed"}
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                full = tiny(arm)
                train(full, data, dev, root / "full", identity, mixed_precision=False)
                partial = tiny(arm)
                result = train(partial, data, dev, root / "partial", identity, mixed_precision=False, stop_after=2)
                self.assertEqual(result["status"], "stopped_at_step")
                self.assertFalse((root / "partial/complete.json").exists())
                resumed = tiny(arm)
                train(resumed, data, dev, root / "resumed", identity, mixed_precision=False,
                      resume=root / "partial/latest.pt")
                for a, b in zip(full.parameters(), resumed.parameters()):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
                for name in ("full", "resumed"):
                    state = torch.load(root / name / "final.pt", weights_only=True)
                    self.assertEqual(state["update"], 4)
                    self.assertEqual(state["next_context"], 8)
                    self.assertEqual([r["update"] for r in state["history"]["validation"]], [0, 2, 4])
                with np.load(root / "full/eval.npz") as a, np.load(root / "resumed/eval.npz") as b:
                    np.testing.assert_array_equal(a["loss_sums"], b["loss_sums"])

    def test_resume_rejects_changed_identity(self):
        model = tiny()
        identity = {"settings": asdict(settings()), "arm": "Deep", "inputs": "original"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train(model, contexts(), contexts(4), root / "partial", identity, stop_after=2, mixed_precision=False)
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                train(tiny(), contexts(), contexts(4), root / "bad", {**identity, "inputs": "changed"},
                      resume=root / "partial/latest.pt", mixed_precision=False)
            self.assertTrue((root / "bad/failure.json").exists())
            self.assertFalse((root / "bad/complete.json").exists())

    def test_failure_does_not_publish_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            with patch("pcc.joint_training.train_update", side_effect=RuntimeError("injected failure")):
                with self.assertRaisesRegex(RuntimeError, "injected failure"):
                    train(tiny(), contexts(), contexts(4), root,
                          {"settings": asdict(settings()), "arm": "Deep"}, mixed_precision=False)
            self.assertFalse((root / "complete.json").exists())
            self.assertEqual(json.loads((root / "failure.json").read_text())["status"], "failed")

    def test_scheduler_endpoints_and_optimizer_tied_deduplication(self):
        model = tiny()
        opt = make_optimizer(model)
        ids = [id(p) for g in opt.param_groups for p in g["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        schedule(opt, 4, settings())
        for group in opt.param_groups:
            self.assertAlmostEqual(group["lr"], .1 * group["peak_lr"])

    def test_config_and_manifest(self):
        from pcc.joint import manifest
        from run_experiments import load_jobs
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = root / "config.json"
            cfg.write_text(json.dumps({"model_path": "model", "train_data": "train", "val_data": "val", "microbatch": 1}))
            self.assertEqual(load_config(cfg)["train_data"], str(root / "train"))
            manifest(cfg, 0, root / "jobs.json")
            jobs = load_jobs(root / "jobs.json")
            self.assertEqual(len(jobs), 8)
            self.assertEqual(sum("gpus" in j for j in jobs), 6)
            cfg.write_text(json.dumps({"model_path": "model", "train_data": "train", "val_data": "val", "microbatch": True}))
            with self.assertRaises(ValueError):
                load_config(cfg)

    def test_seed_orders_share_exact_pool_and_detect_changed_inputs(self):
        from pcc.data import save_contexts
        from pcc.input_check import describe_contexts
        from pcc.joint import load_inputs
        from pcc.joint_config import plan
        from pcc.protocol import MODEL_ID, REVISION
        small = JointSettings(updates=4, tokens_per_update=4096, warmup=1,
                              monitor_tokens=2048, dev_tokens=4096)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"model_path": "model", "train_data": "train", "val_data": "val", "microbatch": 1}
            streams = {}
            for split, count in (("train", 8), ("dev", 2)):
                data = contexts(count, 2048)
                data.path = root / f"{split}.npz"
                data.metadata = {"source": split, "split": split, "language": "en", "model_id": MODEL_ID,
                    "tokenizer_revision": REVISION, "data_order_seed": 20260922,
                    "preprocessing": {"policy": "fixture"}, "packing": "full_causal"}
                save_contexts(data.path, data.ids, data.valid, data.positions, data.metadata)
                streams[split] = describe_contexts(data)
            (root / "complete.json").write_text(json.dumps({"status": "ok", "plan": plan(config), "streams": streams}))
            with patch("pcc.joint.JointSettings", return_value=small):
                a, _, fa = load_inputs(config, root, 0)
                b, _, fb = load_inputs(config, root, 1)
                self.assertNotEqual(fa["train"], fb["train"])
                self.assertEqual(fa["dev"], fb["dev"])
                self.assertEqual(sorted(map(bytes, a.ids)), sorted(map(bytes, b.ids)))
                with np.load(root / "train.npz", allow_pickle=False) as archive:
                    modified = {k: archive[k] for k in archive.files}
                modified["input_ids"][0, 0] += 1
                np.savez(root / "train.npz", **modified)
                with self.assertRaisesRegex(ValueError, "inputs changed"):
                    load_inputs(config, root, 0)

    def test_joint_report_uses_both_seeds_and_rejects_wrong_counts(self):
        from argparse import Namespace
        from pcc.joint import report_results
        from pcc.joint_config import ARMS, SEEDS
        dev = contexts(4)
        config, fingerprints = {"microbatch": 1}, {"train": "t", "dev": "d"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in range(2):
                for arm, nll in (("Base", 3.), ("Shallow", 2.9), ("Deep", 2.8 if seed == 0 else 2.95)):
                    path = root / f"seed-{seed}-{arm}"
                    path.mkdir()
                    identity = {"inputs": fingerprints, "config": config, "settings": asdict(settings()),
                        "arm": arm, "adapter_seed": SEEDS[seed][0], "data_order_seed": SEEDS[seed][1],
                        "code": "fixed", "weights_sha256": "fixed", "purpose": "scientific_training"}
                    (path / "identity.json").write_text(json.dumps(identity))
                    (path / "complete.json").write_text(json.dumps({"status": "ok", "updates": 4,
                        "input_tokens": 64, "arm": arm, "nll": nll}))
                    (path / "final.pt").write_bytes(b"fixture")
                    counts = np.full(4, 7)
                    np.savez(path / "eval.npz", target_counts=counts, loss_sums=counts * nll, sequence_indices=np.arange(4))
                    (path / "validation.jsonl").write_text(json.dumps({"update": 4, "nll": nll, "target_tokens": 28}) + "\n")
                    (path / "train.jsonl").write_text("".join(json.dumps({"update": u, "input_tokens": u * 16}) + "\n" for u in range(1, 5)))
            args = Namespace(output=root / "report", data_dir=root / "inputs", runs_dir=root)
            def audit(path, identity):
                return {"history": {"train": [json.loads(line) for line in (Path(path).parent / "train.jsonl").read_text().splitlines()]}}
            with patch("pcc.joint.JointSettings", return_value=settings()), patch("pcc.joint.load_inputs", return_value=(None, dev, fingerprints)), patch("pcc.joint_training.audit_final_checkpoint", side_effect=audit):
                report = report_results(args, config)
                self.assertEqual(report["decision"], "no_consistent_joint_advantage")
                self.assertTrue(report["seeds"][0]["passed"])
                self.assertFalse(report["seeds"][1]["passed"])
                self.assertTrue((args.output / "validation-curves.png").is_file())
                np.savez(root / "seed-1-Deep/eval.npz", target_counts=np.full(4, 6),
                         loss_sums=np.full(4, 2.95 * 6), sequence_indices=np.arange(4))
                args.output = root / "bad-report"
                with self.assertRaisesRegex(ValueError, "target counts"):
                    report_results(args, config)
                self.assertFalse((args.output / "complete.json").exists())

    def test_final_checkpoint_audit_reads_real_weights_and_moments(self):
        from pcc.joint_training import audit_final_checkpoint
        identity = {"settings": asdict(settings()), "arm": "Deep"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            train(tiny(), contexts(), contexts(4), root, identity, mixed_precision=False)
            self.assertEqual(audit_final_checkpoint(root / "final.pt", identity)["status"], "ok")
            state = torch.load(root / "final.pt", weights_only=True)
            next(iter(state["optimizer"]["state"].values()))["exp_avg"].fill_(float('nan'))
            torch.save(state, root / "broken.pt")
            with self.assertRaisesRegex(ValueError, "optimizer moments"):
                audit_final_checkpoint(root / "broken.pt", identity)


if __name__ == "__main__":
    unittest.main()
