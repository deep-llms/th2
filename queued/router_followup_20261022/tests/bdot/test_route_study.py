"""Route accounting, evaluation isolation, export identity and distributed coverage."""
import copy
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from accelerate import Accelerator
from datasets import Dataset
from transformers import TrainingArguments

from test_model import model
from test_data_and_trainer import tokenizer_assets
from train import DataArguments
from training_data import prepare_training_data, save_data_recipe
from packed_training import PackedCollator
from packed_evaluation import evaluate_packed
from eval.route_study import summarize_routes, run_route_study, write_route_sample, compare_route_samples

ROOT = Path(__file__).resolve().parents[2]


def stable_model(**options):
    m = model(curriculum="none", default_prior=0., **options)
    with torch.no_grad():
        for s, layer in enumerate(m.routed_steps):
            layer.router.weight.zero_()
            layer.router.visits.zero_()
            layer.router.bias.zero_()
            layer.router.bias[[1, 1, 3, 2][s]] = 2.
    return m


def rows():
    return [dict(input_ids=[2, 3, 4, 1, 5, 1], labels=[2, 3, 4, 1, 5, 1],
                 attention_mask=[1]*6, segments=[0, 0, 0, 0, 1, 1]),
            dict(input_ids=[1]*6, labels=[1]*6, attention_mask=[1]*6, segments=list(range(6))),
            dict(input_ids=[3, 2, 5, 4, 2, 1], labels=[3, 2, 5, 4, 2, 1],
                 attention_mask=[1]*6, segments=[0]*6)]


class RouteStudyTests(unittest.TestCase):
    def test_controls_no_reuse_and_fa4_reference_adapter(self):
        acc = Accelerator(cpu=True, mixed_precision="no")
        fixed = model(routing="fixed", curriculum="none", r=0)
        calls = []
        fixed.register_forward_hook(lambda *args: calls.append(1))
        _, report, _ = run_route_study(fixed, rows(), PackedCollator(), acc, 2,
            route_sequences=3, artifact_sequences=3)
        self.assertEqual(len(calls), 2)  # One pass; identical interventions reuse it.
        self.assertEqual(report["default_path_loss_increase"], 0.)
        self.assertEqual(report["route_diagnostics"]["off_default_share"], 0.)
        _, report, _ = run_route_study(stable_model(reuse=False), rows(), PackedCollator(), acc,
            route_sequences=3, artifact_sequences=3)
        self.assertEqual(report["route_diagnostics"]["reuse_rate"], 0.)
        import packed_fa4
        def kernel(q, k, v, *, cu_seqlens_q, cu_seqlens_k, causal, softmax_scale, **kwargs):
            self.assertTrue(torch.equal(cu_seqlens_q, cu_seqlens_k))
            pieces = []
            for start, end in zip(cu_seqlens_q[:-1].tolist(), cu_seqlens_q[1:].tolist()):
                out = torch.nn.functional.scaled_dot_product_attention(
                    q[start:end].transpose(0, 1)[None], k[start:end].transpose(0, 1)[None],
                    v[start:end].transpose(0, 1)[None], is_causal=causal, enable_gqa=True, scale=softmax_scale)
                pieces.append(out[0].transpose(0, 1))
            return torch.cat(pieces), None
        m = stable_model()
        _, dense, dense_sample = run_route_study(m, rows(), PackedCollator(), acc, 2,
            route_sequences=3, artifact_sequences=3)
        # Tests FA4 integration only, not the actual CUDA kernel.
        with patch.object(packed_fa4, "validate_kernel_inputs"), patch.object(packed_fa4, "load_kernel", return_value=kernel):
            packed_fa4.register_backend()
            m.set_attn_implementation(packed_fa4.BACKEND)
            _, fa4, fa4_sample = run_route_study(m, rows(), PackedCollator(attention_backend="fa4"), acc, 2,
                route_sequences=3, artifact_sequences=3)
        self.assertEqual(dense["route_diagnostics"], fa4["route_diagnostics"])
        np.testing.assert_array_equal(dense_sample["routes"], fa4_sample["routes"])
        np.testing.assert_array_equal(dense_sample["token_sha256"], fa4_sample["token_sha256"])
        for mode in ("as_trained", "fixed", "random"):
            self.assertAlmostEqual(dense["interventions"][mode]["loss"], fa4["interventions"][mode]["loss"], places=6)

    def test_hand_counted_routes_and_diversity_windows(self):
        a = np.array([[[0, 1, 2, 3], [0, 0, 2, 3], [1, 0, 3, 2]]])
        result = summarize_routes(a)
        self.assertEqual(result["routing_decisions"], 12)
        self.assertAlmostEqual(result["off_default_share"], 5/12)
        self.assertAlmostEqual(result["reuse_rate"], 1/3)
        self.assertEqual(result["distinct_paths"], 3)
        self.assertIsNone(result["distinct_paths_per_10000_tokens"])
        self.assertEqual(result["module_counts_by_step"], [[2, 1, 0, 0], [2, 1, 0, 0], [0, 0, 2, 1], [0, 0, 1, 2]])
        big = np.concatenate([np.repeat(a[:, i:i+1], n, axis=1)
                              for i, n in enumerate((15000, 5000, 3000))], axis=1)
        result = summarize_routes(big)
        self.assertEqual(result["distinct_paths_per_10000_tokens"], 1.5)
        self.assertEqual(result["diversity_excluded_tail_tokens"], 3000)
        self.assertEqual(result["distinct_paths"], 3)
        for invalid in (a.astype(float), a-1, a+4, a[:, :0]):
            with self.assertRaises(ValueError):
                summarize_routes(invalid)

    def test_interventions_match_existing_scoring_preserve_state_and_include_boundaries(self):
        m = stable_model()
        acc = Accelerator(cpu=True, mixed_precision="no")
        expected = {}
        for name, override in (("as_trained", None), ("fixed", "fixed"), ("random", "random")):
            m.routing_override = override
            m.eval_seed = 23
            expected[name] = evaluate_packed(m, rows(), PackedCollator(), acc, 2)
        m.train()
        m.model.layers[0].eval()
        m.routing_override, m.capture_routes, m.eval_seed, m.eval_counter = "fixed", False, 19, 101
        m.last_routes = torch.tensor([7])
        original = copy.deepcopy(m.state_dict())
        modes = [x.training for x in m.modules()]
        rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        primary, study, sample = run_route_study(m, rows(), PackedCollator(), acc, 2,
            route_sequences=3, artifact_sequences=2, routing_seed=23)
        for key in expected:
            self.assertEqual(study["interventions"][key]["loss"], expected[key]["loss"])
            self.assertEqual(study["interventions"][key]["num_tokens"], 9)
            self.assertEqual(study["interventions"][key]["routing_decisions"], 72)
        self.assertEqual(primary["num_sequences"], 3)
        self.assertEqual(study["route_diagnostics"]["input_tokens"], 18)
        self.assertEqual(study["route_diagnostics"]["reuse_rate"], 1.)
        self.assertEqual(study["route_diagnostics"]["off_default_share"], .75)
        self.assertEqual(study["default_path_loss_increase"], expected["fixed"]["loss"]-expected["as_trained"]["loss"])
        np.testing.assert_array_equal(sample["routes"], np.tile([1, 1, 3, 2], (2, 6, 1)))
        for k, v in m.state_dict().items():
            torch.testing.assert_close(v, original[k], rtol=0, atol=0)
        self.assertEqual(modes, [x.training for x in m.modules()])
        self.assertEqual((m.routing_override, m.capture_routes, m.eval_seed, m.eval_counter), ("fixed", False, 19, 101))
        torch.testing.assert_close(m.last_routes, torch.tensor([7]))
        self.assertEqual(rng[0], random.getstate())
        np.testing.assert_equal(rng[1], np.random.get_state())
        torch.testing.assert_close(rng[2], torch.get_rng_state(), rtol=0, atol=0)
        # Also cover a wholly unscorable batch: its route rows must not disappear.
        _, study, _ = run_route_study(m, rows(), PackedCollator(), acc, 1,
            route_sequences=3, artifact_sequences=3)
        self.assertEqual(study["route_diagnostics"]["input_tokens"], 18)
        for mode, override in (("as_trained", None), ("fixed", "fixed"), ("random", "random")):
            m.routing_override = override
            m.eval_seed = 0
            m.capture_routes = True
            standalone = evaluate_packed(m, rows(), PackedCollator(), acc, 1)
            for key in ('loss', 'off_default_share', 'routing_input_tokens', 'routing_decisions'):
                self.assertEqual(standalone[key], study['interventions'][mode][key])
        with self.assertRaisesRegex(ValueError, "explicit smaller counts"):
            run_route_study(m, rows(), PackedCollator(), acc)

    def test_artifact_comparison_rejects_wrong_tokens_boundaries_and_contract(self):
        _, _, sample = run_route_study(stable_model(), rows(), PackedCollator(),
            Accelerator(cpu=True), 1, route_sequences=3, artifact_sequences=3)
        metadata = dict(format_version=1, layout=dict(S=4), evaluation_contract=dict(backend="sdpa"))
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            write_route_sample(p/"a.npz", sample, metadata)
            write_route_sample(p/"b.npz", sample, metadata)
            self.assertEqual(compare_route_samples(p/"a.npz", p/"b.npz")["agreement"], 1.)
            changed = copy.deepcopy(sample)
            changed["routes"][0, 0, 0] = 0
            write_route_sample(p/"one.npz", changed, metadata)
            result = compare_route_samples(p/"a.npz", p/"one.npz")
            self.assertEqual(result["agreement"], 71/72)
            self.assertEqual(result["identical_path_fraction"], 17/18)
            for key in ("sequence_indices", "input_ids", "token_sha256"):
                bad = copy.deepcopy(sample)
                bad[key].flat[0] = "x" if key == "token_sha256" else 999
                write_route_sample(p/f"{key}.npz", bad, metadata)
                with self.assertRaisesRegex(ValueError, "different validation"):
                    compare_route_samples(p/"a.npz", p/f"{key}.npz")
            bad_meta = dict(metadata, evaluation_contract=dict(backend="fa4"))
            write_route_sample(p/"contract.npz", sample, bad_meta)
            with self.assertRaisesRegex(ValueError, "incompatible"):
                compare_route_samples(p/"a.npz", p/"contract.npz")
            with self.assertRaises(FileExistsError):
                write_route_sample(p/"a.npz", sample, metadata)

    def test_real_checkpoint_cli_single_distributed_uneven_and_empty_rank(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            tok = tokenizer_assets(p/"checkpoint")
            stable_model().save_pretrained(p/"checkpoint")
            Dataset.from_dict({"id": [str(i) for i in range(20)], "text": ["a b c d "*8]*20}).save_to_disk(p/"raw")
            data = DataArguments(data_dir=str(p/"raw"), data_order="document_stream", dataset_revision="fixture",
                validation_manifest=str(p/"manifest.json"), validation_tokens=40, block_size=8,
                data_cache_dir=str(p/"cache"))
            prep = TrainingArguments(output_dir=str(p/"unused"), use_cpu=True, report_to="none")
            _, recipe = prepare_training_data(data, prep, tok, p/"checkpoint", 8)
            save_data_recipe(p/"checkpoint", recipe)
            def run(name, distributed=False, limit=None, comparison=None):
                config = dict(checkpoint=str(p/"checkpoint"), eval_data_dir=str(p/"raw"),
                    output_file=str(p/name/"final_eval.json"), use_cpu=True, batch_size=2,
                    data_cache_dir=str(p/"cache"), route_study=True,
                    route_sequences=5 if limit is None else 1, route_artifact_sequences=3 if limit is None else 1)
                if limit is not None:
                    config["max_sequences"] = limit
                if comparison is not None:
                    config["compare_routes"] = str(comparison)
                (p/f"{name}.json").write_text(json.dumps(config))
                cmd = [sys.executable]
                if distributed:
                    cmd += ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2"]
                cmd += [str(ROOT/"eval/packed_ppl.py"), str(p/f"{name}.json")]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=180, cwd=ROOT,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1"))
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                return json.loads((p/name/"final_eval.json").read_text())
            single = run("single")
            multi = run("multi", True, comparison=p/"single/routes_sample.npz")
            for key in ("num_tokens", "num_sequences"):
                self.assertEqual(single[key], multi[key])
            self.assertAlmostEqual(single["loss"], multi["loss"], places=6)
            self.assertEqual(single["route_study"]["route_diagnostics"], multi["route_study"]["route_diagnostics"])
            self.assertEqual(multi["route_study"]["cross_seed_agreement"]["agreement"], 1.)
            self.assertEqual(multi["route_study"]["route_diagnostics"]["num_sequences"], 5)
            empty = run("empty", True, limit=1)
            self.assertEqual(empty["num_sequences"], 1)
            self.assertEqual(empty["route_study"]["route_diagnostics"]["num_sequences"], 1)


if __name__ == "__main__":
    unittest.main()
