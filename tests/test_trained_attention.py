"""Checkpoint replay and failure reporting for same-weight attention checks."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from safetensors.torch import save_model, save_file
from scripts.check_trained_attention import acceptable, fingerprint, summarize, restore, MODES, CASES
from scripts.benchmark_document_training import BenchmarkModel
from tests.test_deep_kv import config


class TrainedAttentionTests(unittest.TestCase):
    def test_saved_tied_checkpoint_roundtrip_is_exact(self):
        torch.set_num_threads(1)
        a = BenchmarkModel.from_scratch(config(), 'A', consumer=2, deep_target=4)
        b = BenchmarkModel.from_scratch(config(), 'A', consumer=2, deep_target=4)
        with torch.no_grad():
            next(a.parameters()).add_(.25)
        self.assertNotEqual(fingerprint(a), fingerprint(b))
        with tempfile.TemporaryDirectory() as directory:
            file = str(Path(directory)/'model.safetensors')
            save_model(a, file)
            restore(b, file)
        self.assertEqual(fingerprint(a), fingerprint(b))

    def test_trainer_cloned_tied_weights_and_conflict(self):
        torch.set_num_threads(1)
        model = BenchmarkModel.from_scratch(config(), 'A', consumer=2, deep_target=4)
        model.backbone.lm_head.weight = model.backbone.model.embed_tokens.weight
        before = fingerprint(model)
        state = {name: value.clone() for name, value in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            file = str(Path(directory)/'model.safetensors')
            save_file(state, file)
            restore(model, file)
            self.assertEqual(fingerprint(model), before)
            state['backbone.lm_head.weight'].add_(1)
            save_file(state, file)
            with self.assertRaisesRegex(ValueError, 'tied weights disagree'):
                restore(model, file)
        self.assertEqual(fingerprint(model), before)

    def test_failed_gate_keeps_complete_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for i in range(8):
                row = dict(index=i, status='completed', parameters_unchanged=True,
                    split=CASES[i%4][0], batch=CASES[i%4][1],
                    checkpoint=f'/source/{MODES[i//4]}/final_model/model.safetensors',
                    input_sha256=f'input-{i%4}', parameter_sha256=f'weights-{i//4}',
                    backend_gate_passed=(i!=7), reference_gates_passed=True)
                (root/f'case-{i}.json').write_text(json.dumps(row))
            with self.assertRaisesRegex(RuntimeError, 'numerical gate failed'):
                summarize(SimpleNamespace(source='/source', output=directory))
            summary = json.loads((root/'summary.json').read_text())
            self.assertEqual(summary['status'], 'failed')
            self.assertEqual(len(summary['cases']), 8)

    def test_gradient_failure_not_hidden_by_matching_loss(self):
        row = dict(targets=[10,10], loss_abs_diff=0., gradient_relative_l2=.04,
                   hidden={'relative_l2':0.}, logits={'relative_l2':0.})
        self.assertFalse(acceptable(row))


if __name__ == '__main__':
    unittest.main()
