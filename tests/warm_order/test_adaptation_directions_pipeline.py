import copy
import json
import tarfile
import unittest
from unittest.mock import patch
from efficient_llm.warm_order.model import Config
from pathlib import Path

import numpy as np
import torch
import test_adaptation_pipeline as fixture
from efficient_llm.warm_order.adaptation_pipeline import load_snapshot_panel
from efficient_llm.warm_order.adaptation_directions import CONTROL_SEEDS
from efficient_llm.warm_order.adaptation_directions_pipeline import (
    FLAGS, calibrate, replay_calibration, evaluate_stream, replay, selections)
from efficient_llm.warm_order.adaptation_directions_node import prepare, run_prepared, finish
from efficient_llm.warm_order.data import sha


class DirectionsPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch('test_adaptation_train.Config', side_effect=lambda **kw: Config(**dict(kw, d=16, heads=2, ff=32))):
            fixture.AdaptationPipeline.setUpClass()
        cls.f = fixture.AdaptationPipeline
        f = cls.f
        ix = f.root/'directions_eval.npy'; cal = f.root/'directions_cal.npy'
        np.save(ix, np.array([192, 193])); np.save(cal, np.array([194, 195]))
        stem = f.config['data']
        cls.c = dict(schema=1, domain='dclm', target=2, path=[0, 2, 1, 3], measurement_device='cpu',
            parent=str(f.root/'parent'), parent_hashes=f.pins, phase_files=f.phase, phase_step=4,
            data=stem, indices=str(ix), expected_inputs={p: sha(p) for p in (stem+'.bin', stem+'.starts', str(ix))},
            calibration_indices=str(cal), calibration_sha256=sha(cal), batch_size=1, rank=8,
            control_seeds=list(CONTROL_SEEDS), positions='all_predictor_positions_0_to_length_minus_2', flags=FLAGS)

    @classmethod
    def tearDownClass(cls):
        fixture.AdaptationPipeline.tearDownClass()

    def test_calibration_streaming_scores_and_node_replay(self):
        f = self.f; models, proof = load_snapshot_panel(f.root/'parent', f.pins, f.phase)
        rng = torch.get_rng_state().clone(); before = {k: v.clone() for k, v in models['parent'].state_dict().items()}
        reports = []
        for batch in (1, 2):
            c = dict(self.c, batch_size=batch); root = f.root/f'direction-stream{batch}'
            cal = root/'calibration'; calibrate(models, c, proof, cal)
            fitted = replay_calibration(cal)
            self.assertEqual(fitted['A']['observations'], 14)
            c['calibration_artifact_sha256'] = sha(cal/'calibration.json')
            report = evaluate_stream(models, c, proof, cal, root/'study')
            self.assertEqual(report['sequences'], 2); self.assertEqual(report['tokens'], 12)
            self.assertEqual(replay(root/'study')['cases'], 32)
            reports.append(root)
        for name in ('scores.npz', 'diagnostics.npz'):
            with np.load(reports[0]/'study'/name) as a, np.load(reports[1]/'study'/name) as b:
                for k in a.files:
                    np.testing.assert_allclose(a[k], b[k], atol=2e-6, rtol=1e-5)
        for k, v in models['parent'].state_dict().items():
            torch.testing.assert_close(v, before[k], atol=0, rtol=0)
        torch.testing.assert_close(rng, torch.get_rng_state(), atol=0, rtol=0)
        path = reports[0]/'calibration/A-basis.pt'; path.write_bytes(path.read_bytes()+b'tamper')
        with self.assertRaisesRegex(ValueError, 'hash'): replay_calibration(reports[0]/'calibration')
        # Portable replay intentionally needs no basis or original model/data files.
        self.assertEqual(replay(reports[0]/'study')['cases'], 32)
        path = reports[1]/'study/REPORT.json'; r = json.loads(path.read_text()); r['cases']['B/full'] += 1
        path.write_text(json.dumps(r))
        with self.assertRaisesRegex(ValueError, 'hash'): replay(reports[1]/'study')

    def test_selection_and_checkpoint_calibration_binding(self):
        c = dict(self.c, calibration_indices=self.c['indices'], calibration_sha256=sha(self.c['indices']))
        with self.assertRaisesRegex(ValueError, 'overlap'): selections(c, 8)
        for change in (dict(rank=7), dict(control_seeds=[1, 2, 3]), dict(flags=[])):
            with self.assertRaisesRegex(ValueError, 'Unregistered'): selections(dict(self.c, **change), 8)
        f = self.f; models, proof = load_snapshot_panel(f.root/'parent', f.pins, f.phase)
        cal = f.root/'bound-direction-cal'; calibrate(models, self.c, proof, cal)
        c = dict(self.c, calibration_artifact_sha256=sha(cal/'calibration.json'))
        bad = copy.deepcopy(proof); bad['phase_step'] = 3
        with self.assertRaisesRegex(ValueError, 'Checkpoint'): evaluate_stream(models, c, bad, cal, f.root/'bad-direction-proof')
        bad = copy.deepcopy(proof); bad['paired_recipe_sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'Calibration/checkpoint'): evaluate_stream(models, c, bad, cal, f.root/'bad-direction-basis')
        with self.assertRaisesRegex(ValueError, 'registration hash'):
            evaluate_stream(models, dict(c, calibration_artifact_sha256='wrong'), proof, cal, f.root/'bad-direction-hash')

    def test_prepared_cpu_gate_archive_and_backup(self):
        f = self.f; base = f.root/'directions-node'; plan = f.root/'directions-plan.json'
        plan.write_text(json.dumps(dict(seeds={'0': self.c, '1': copy.deepcopy(self.c)})))
        receipt = prepare(plan, sha(plan), base/'prepared')
        for seed in ('0', '1'):
            root = base/'prepared'/f's{seed}'; pin = receipt['seeds'][seed]['config_sha256']
            self.assertEqual(receipt['seeds'][seed]['reference_cells'], 32)
            with self.assertRaisesRegex(ValueError, 'config mismatch'): run_prepared(root, 'wrong', 'cpu')
            self.assertEqual(run_prepared(root, pin, 'cpu'), dict(cases=32, sequences=2, gate=True))
            with self.assertRaisesRegex(ValueError, 'Fresh'): run_prepared(root, pin, 'cpu')
        (base/'execution').mkdir()
        for name in ('source.json', 'admission.json'): (base/'execution'/name).write_text('{}')
        proof = finish(base)
        self.assertEqual(len(proof['node_basis_moment_backups']), 8)
        with tarfile.open(base/'tier1.tar.gz') as tar:
            names = tar.getnames(); self.assertEqual(len(names), 23)
            self.assertFalse(any('.pt' in n or 'moments' in n or 'basis' in n for n in names))
        self.assertEqual(proof['archive_sha256'], sha(base/'tier1.tar.gz'))
        from scripts.verify_adaptation_directions_export import verify
        self.assertEqual(verify(base/'tier1.tar.gz', proof['archive_sha256'], f.root/'directions-export')['members'], 23)
        for seed in ('0', '1'): replay_calibration(base/'basis-backup'/f's{seed}')


if __name__ == '__main__': unittest.main()
