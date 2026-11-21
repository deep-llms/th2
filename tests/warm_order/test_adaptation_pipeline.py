import copy
import json
import unittest

import numpy as np
import torch

import test_adaptation_train as fixture
from efficient_llm.warm_order.adaptation_train import train
from efficient_llm.warm_order.adaptation_pipeline import load_snapshot_panel, evaluate_stream
from efficient_llm.warm_order.adaptation_eval import evaluate, replay
from efficient_llm.warm_order.data import Rows, sha
from efficient_llm.warm_order.adaptation_timecourse import summarize


class AdaptationPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.AdaptationTrain.setUpClass()
        cls.root = fixture.AdaptationTrain.root
        cls.pins = fixture.AdaptationTrain.pins
        cls.config = fixture.AdaptationTrain.config
        cls.phase = {}
        for branch in ('A', 'B'):
            out = cls.root/branch; train(dict(cls.config, branch=branch), out)
            cls.phase[branch] = dict(path=str(out/'fullstate.pt'), sha256=sha(out/'fullstate.pt'))

    @classmethod
    def tearDownClass(cls): fixture.AdaptationTrain.tearDownClass()

    def test_authenticated_fork_panel_streaming_and_replay(self):
        rng = torch.get_rng_state().clone()
        models, proof = load_snapshot_panel(self.root/'parent', self.pins, self.phase)
        torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
        self.assertEqual(proof['phase_step'], 4)
        indices = self.root/'eval_rows.npy'; np.save(indices, np.array([192, 194, 193]))
        stem = self.config['data']; pins = {p: sha(p) for p in (stem+'.bin', stem+'.starts', str(indices))}
        x = Rows(stem, 8).batch(np.load(indices), 'cpu')
        one = self.root/'single'; stream = self.root/'stream'
        evaluate(models, x, np.load(indices), one)
        result = evaluate_stream(models, stem, indices, pins, 2, stream, proof)
        self.assertEqual(replay(stream)['cases'], 30)
        self.assertEqual(result['tokens_per_case'], 18)
        with np.load(one/'scores.npz') as a, np.load(stream/'scores.npz') as b:
            self.assertEqual(set(a.files), set(b.files))
            for key in a.files:
                if a[key].dtype.kind in 'iu': np.testing.assert_array_equal(a[key], b[key])
                else: np.testing.assert_allclose(a[key], b[key], atol=2e-6, rtol=1e-6)
        manifest = json.loads((stream/'manifest.json').read_text())
        self.assertTrue(manifest['provenance']['checkpoints']['checkpoint_bytes_and_pairing_verified'])
        self.assertFalse(manifest['production_ready'])
        self.assertFalse(manifest['provenance']['whole_document_freshness_verified'])

    def test_hash_pairing_and_registration_guards(self):
        bad = copy.deepcopy(self.phase); bad['A']['sha256'] = 'bad'
        with self.assertRaisesRegex(ValueError, 'hash'): load_snapshot_panel(self.root/'parent', self.pins, bad)
        bad = copy.deepcopy(self.phase)
        p = self.root/'B/snapshots/000001/fullstate.pt'; bad['B'] = dict(path=str(p), sha256=sha(p))
        with self.assertRaisesRegex(ValueError, 'Unmatched'): load_snapshot_panel(self.root/'parent', self.pins, bad)
        state = torch.load(self.phase['B']['path'], weights_only=False)
        state['recipe']['parameter_names'][0:2] = list(reversed(state['recipe']['parameter_names'][0:2]))
        p = self.root/'bad_names.pt'; torch.save(state, p)
        bad['B'] = dict(path=str(p), sha256=sha(p))
        with self.assertRaisesRegex(ValueError, 'identity'): load_snapshot_panel(self.root/'parent', self.pins, bad)
        models, proof = load_snapshot_panel(self.root/'parent', self.pins, self.phase)
        indices = self.root/'guard_rows.npy'; np.save(indices, np.array([192]))
        stem = self.config['data']; pins = {p: sha(p) for p in (stem+'.bin', stem+'.starts', str(indices))}
        pins[str(indices)] = 'bad'
        with self.assertRaisesRegex(ValueError, 'identity'):
            evaluate_stream(models, stem, indices, pins, 1, self.root/'bad_rows', proof)
        self.assertFalse((self.root/'bad_rows').exists())

    def test_timecourse_exact_fork_matched_rows_and_guards(self):
        indices = self.root/'trajectory_rows.npy'; np.save(indices, np.array([192, 193]))
        stem = self.config['data']; pins = {p: sha(p) for p in (stem+'.bin', stem+'.starts', str(indices))}
        outputs = []
        for step in (0, 4):
            phase = {}
            for branch in ('A', 'B'):
                path = self.root/branch/'snapshots'/f'{step:06d}'/'fullstate.pt'
                phase[branch] = dict(path=str(path), sha256=sha(path))
            models, proof = load_snapshot_panel(self.root/'parent', self.pins, phase)
            out = self.root/f'timecourse_{step}'
            evaluate_stream(models, stem, indices, pins, 2, out, proof)
            outputs.append(out)
        result = summarize(outputs)
        self.assertEqual([p['step'] for p in result['points']], [0, 4])
        p0, p1 = result['points']
        self.assertAlmostEqual(p1['excess_gain_over_ordinary_nat'],
                               p0['remaining_gap_nat']-p1['remaining_gap_nat'], places=12)
        with self.assertRaisesRegex(ValueError, 'Step-zero'): summarize(outputs[::-1])
        with self.assertRaisesRegex(ValueError, 'increasing'): summarize([outputs[0], outputs[0]])
        path = outputs[1]/'manifest.json'; manifest = json.loads(path.read_text())
        manifest['provenance']['checkpoints']['paired_recipe_sha256'] = 'changed'
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'Unmatched'): summarize(outputs)


if __name__ == '__main__': unittest.main()
