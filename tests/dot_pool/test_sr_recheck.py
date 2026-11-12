import tempfile
import unittest
import json
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch

from efficient_llm.dot_pool.data import sha
from scripts.dot_pool_sr_recheck import verify_accuracy, run, check_inputs


class SRCorrectionContract(unittest.TestCase):
    def test_actual_two_job_manifest_is_evaluation_only(self):
        from scripts.launch_dot_pool_sr_recheck import job_config
        from run_experiments import load_jobs
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'jobs.json';p.write_text(json.dumps(job_config('/frozen')))
            jobs = load_jobs(p)
            self.assertEqual([j['gpus'] for j in jobs], [[0], [1]])
            for j in jobs:
                self.assertIn('scripts.dot_pool_sr_recheck', j['argv'])
                self.assertNotIn('efficient_llm.dot_pool.train', j['argv'])
                self.assertEqual(j['required_outputs'][0]['json_equals'], {'verified': True, 'training_performed': False})

    def test_additive_worker_preserves_inputs_and_exports_verified_diagnostics(self):
        from dataclasses import replace
        from efficient_llm.dot_pool.model import Model
        from efficient_llm.dot_pool.train import evaluate
        from tests.dot_pool.test_core import tiny
        with tempfile.TemporaryDirectory() as td:
            root = Path(td);data = root / 'data';data.mkdir();original = root / 'E1-SR-s0';original.mkdir()
            torch.manual_seed(71);model = Model(replace(tiny('SR'), middle=8)).eval()
            rows = np.array([[1, 2, 3, 4], [5, 6, 7, 8]] * 2)
            splits = ['id_dev', 'ood_dev', 'id_test', 'ood_test']
            for name in splits + ['ood_bridge', 'ood_dev_bridge']:
                np.save(data / (name + '.npy'), rows)
            members = {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in data.iterdir()}
            manifest = data / 'manifest.json';manifest.write_text(json.dumps({'files': members}))
            cfg = {'study': 'E1', 'arm': 'SR', 'seed': 0, 'steps': 154000, 'data': str(data)}
            (original / 'config.json').write_text(json.dumps(cfg))
            final = {'step': 154000, 'metrics': {s: evaluate(model, rows, 'E1', torch.device('cpu')) for s in splits}}
            (original / 'final_eval.json').write_text(json.dumps(final))
            inputs = {str(p): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in original.iterdir()}
            expected = {'run': str(original), 'inputs': inputs, 'data_manifest': str(manifest), 'data_manifest_sha256': sha(manifest)}
            out = root / 'corrected'
            with patch('scripts.dot_pool_sr_recheck.load', return_value=(original, cfg, model.eval())):
                run(expected, out)
            proof = json.loads((out / 'verification.json').read_text())
            self.assertTrue(proof['verified'] and proof['inputs_unchanged'])
            self.assertFalse(proof['training_performed'])
            self.assertEqual(len(proof['split_accuracy']), 4)
            complete = json.loads((out / 'complete.json').read_text())
            self.assertEqual(sha(complete['path']), complete['sha256'])
            self.assertEqual(set(p.name for p in original.iterdir()), {'config.json', 'final_eval.json'})
            check_inputs(expected)

    def test_changed_physical_data_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td);data = p / 'data.npy';data.write_bytes(b'original')
            manifest = p / 'manifest.json'
            manifest.write_text(json.dumps({'files': {'data.npy': {'bytes': 8, 'sha256': sha(data)}}}))
            expected = {'inputs': {}, 'data_manifest': str(manifest), 'data_manifest_sha256': sha(manifest)}
            check_inputs(expected)
            data.write_bytes(b'changed!')
            with self.assertRaisesRegex(ValueError, 'member changed'):
                check_inputs(expected)

    def test_all_four_split_masks_must_match_headline_accuracy(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'routes.npz'
            splits = ['id_dev', 'ood_dev', 'id_test', 'ood_test']
            final = {'metrics': {s: {'targets': 4, 'accuracy': .5} for s in splits}}
            values = {s + '_correct': np.array([True, False, True, False]) for s in splits}
            np.savez(p, **values)
            self.assertEqual(len(verify_accuracy(final, p)), 4)
            values['ood_test_correct'][0] = False
            np.savez(p, **values)
            with self.assertRaisesRegex(ValueError, 'ood_test'):
                verify_accuracy(final, p)

    def test_existing_output_refused_before_loading_or_mutating(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td);sentinel = p / 'original';sentinel.write_text('keep')
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                run({}, p)
            self.assertEqual(sentinel.read_text(), 'keep')
