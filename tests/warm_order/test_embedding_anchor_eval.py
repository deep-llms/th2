import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from efficient_llm.warm_order.data import sha
from efficient_llm.warm_order.embedding_anchor import EmbeddingAnchor
from efficient_llm.warm_order.embedding_anchor_data import BoundRows
from efficient_llm.warm_order.embedding_anchor_eval import LABELS, evaluate_fixture, evaluate_checkpoints, replay, summarize
from efficient_llm.warm_order.embedding_anchor_train import CPUStreamTrainer
from efficient_llm.warm_order.model import Config


class AnchorDataEvaluationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.c = Config(d=8, ff=16, heads=1, kv_heads=1, head_dim=8,
                        vocab=17, prefix=2, length=4)
        rng = np.random.default_rng(3)
        self.stem = str(self.root / 'corpus')
        rng.integers(0, 17, (64768, 4), dtype=np.uint32).tofile(self.stem + '.bin')
        starts = np.zeros((64768, 4), dtype=np.uint8); starts[:, 0] = 1
        starts.tofile(self.stem + '.starts')
        self.cfg = dict(seed=0, domain='dclm', steps=2000, batch=32, data=self.stem)
        for key, lo, hi in [('indices', 0, 64000), ('calibration_indices', 64000, 64256),
                            ('confirmation_indices', 64256, 64768)]:
            path = self.root / (key + '.npy'); np.save(path, np.arange(lo, hi, dtype=np.int64))
            self.cfg[key] = str(path)
        self.recipe = self.root / 'recipe.json'; self.repin()

    def tearDown(self): self.tmp.cleanup()

    def repin(self):
        paths = [self.stem + ext for ext in ('.bin', '.starts')]
        paths += [self.cfg[k] for k in ('indices', 'calibration_indices', 'confirmation_indices')]
        self.cfg['expected_inputs'] = {p: sha(p) for p in paths}
        self.recipe.write_text(json.dumps(dict(config=self.cfg)))
        self.pin = dict(path=str(self.recipe), sha256=sha(self.recipe))

    def load(self): return BoundRows(self.pin, seed=0, length=4, fixture_only=True)

    def test_binding_and_row_guards(self):
        data = self.load()
        for step in (0, 767, 768, 1151, 1152, 1919):
            ids, x = data.training_batch(step)
            np.testing.assert_array_equal(ids, np.arange(step * 32, (step + 1) * 32))
            assert x.shape == (32, 4)
        assert data.binding['train_prefix_rows'] == 61440
        assert not data.binding['document_freshness_verified']
        with self.assertRaises(ValueError):
            BoundRows(dict(self.pin, sha256='0' * 64), seed=0, length=4, fixture_only=True)
        starts = np.memmap(self.stem + '.starts', mode='r+', dtype='u1', shape=(64768, 4))
        starts[0, 1] = 1; starts.flush()
        with self.assertRaisesRegex(ValueError, 'hash'): self.load()
        self.repin()
        with self.assertRaisesRegex(ValueError, 'boundary'): self.load()
        starts[0, 1] = 0; starts.flush()
        cal = np.load(self.cfg['calibration_indices']); cal[0] = 0
        np.save(self.cfg['calibration_indices'], cal); self.repin()
        with self.assertRaisesRegex(ValueError, 'overlap'): self.load()

    def test_historical_nested_seed_binding(self):
        del self.cfg['seed']
        self.cfg['row_reservation'] = dict(seed=0)
        self.repin()
        assert self.load().binding['seed'] == 0
        self.cfg['row_reservation']['seed'] = 1; self.repin()
        with self.assertRaisesRegex(ValueError, 'recipe'): self.load()
        self.cfg['row_reservation']['seed'] = 0
        self.cfg['seed'] = 1; self.repin()
        with self.assertRaisesRegex(ValueError, 'recipe'): self.load()

    def test_streaming_replay_and_no_mutation(self):
        torch.manual_seed(4)
        a = EmbeddingAnchor(self.c); c = copy.deepcopy(a); c.eval()
        data = self.load(); ids = data.measurement_ids[:3]
        rng = torch.get_rng_state().clone()
        first = evaluate_fixture(dict(A=a, C=c), data, self.root / 'one', ids=ids, batch=1)
        assert a.training and not c.training and torch.equal(rng, torch.get_rng_state())
        second = evaluate_fixture(dict(A=a, C=c), data, self.root / 'two', ids=ids, batch=2)
        self.assertEqual(replay(self.root / 'one')['cases_per_arm'], 36)
        for name in LABELS:
            self.assertAlmostEqual(first['nll'][name], second['nll'][name], places=6)
        # Shared single-block computation is independent of arm's inter-block alpha.
        for i in range(4):
            self.assertEqual(first['nll']['A/single' + str(i)], first['nll']['C/single' + str(i)])
        with self.assertRaisesRegex(ValueError, 'measurement'):
            evaluate_fixture(dict(A=a, C=c), data, self.root / 'bad', ids=data.train_ids[:1])
        path = self.root / 'one' / 'scores.npz'; path.write_bytes(path.read_bytes() + b'tamper')
        with self.assertRaisesRegex(ValueError, 'hash'): replay(self.root / 'one')

    def test_token_weighting_and_skip_sign(self):
        counts = np.array([1, 3], dtype=np.int64)
        raw = dict(row_ids=np.array([7, 9], dtype=np.int64), token_count=counts)
        for label in LABELS:
            v = np.array([1., 9.]) if label.startswith('A/') else np.array([2., 12.])
            if '-skip' in label: v = v + np.array([-.5, 1.5])
            raw[label] = v
        report = summarize(raw)
        self.assertEqual(report['nll']['A/path0123'], 2.5)
        self.assertEqual(report['paths']['A_gain_over_C/seen_even']['nll_gain'], 1.)
        result = report['skip']['A/path0123-skip0']
        self.assertEqual(result['loss_cost'], .25)
        self.assertEqual((result['positive_rows'], result['negative_rows']), (1, 1))

    def test_stream_trainer_resume(self):
        data = self.load()
        full = CPUStreamTrainer(self.c, data); full.advance(8)
        part = CPUStreamTrainer(self.c, data); part.advance(3)
        path = self.root / 'short.pt'; digest = part.save(path)
        resumed = CPUStreamTrainer.restore(path, digest, self.c, data); resumed.advance(8)
        self.assertEqual(full.trace, resumed.trace)
        for k, v in full.model.state_dict().items():
            self.assertTrue(torch.equal(v, resumed.model.state_dict()[k]))
        self.assertFalse(hasattr(full, 'tokens'))
        self.assertEqual(full.recipe['data_binding'], data.binding)

    def test_paired_checkpoint_evaluation(self):
        data = self.load()
        t = CPUStreamTrainer(self.c, data); t.advance(768)
        path = self.root / 'warm.pt'; warm = dict(path=str(path), sha256=t.save(path))
        runs = {}
        for arm in ('A', 'C'):
            branch = CPUStreamTrainer.restore(path, warm['sha256'], self.c, data, fork_arm=arm)
            branch.advance(1152)
            dest = self.root / (arm + '.pt'); runs[arm] = dict(path=str(dest), sha256=branch.save(dest))
        out = self.root / 'joined'
        evaluate_checkpoints(runs, warm, self.c, data, out, step=1152, ids=data.measurement_ids[:2])
        self.assertTrue(replay(out)['checkpoint_binding_verified'])
        meta = json.loads((out / 'manifest.json').read_text())
        self.assertEqual(meta['checkpoint_proofs']['arms']['A']['fork_sha256'], warm['sha256'])
        self.assertTrue(meta['fixture_only'])
        with self.assertRaisesRegex(ValueError, 'arm/step/ancestor'):
            evaluate_checkpoints(dict(A=runs['C'], C=runs['A']), warm, self.c, data,
                                 self.root / 'wrong', step=1152, ids=data.measurement_ids[:1])
        with self.assertRaisesRegex(ValueError, 'arm/step/ancestor'):
            evaluate_checkpoints(runs, warm, self.c, data, self.root / 'not-terminal')
        meta['checkpoint_proofs']['arms']['A']['fork_sha256'] = '0' * 64
        (out / 'manifest.json').write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, 'ancestor'): replay(out)


if __name__ == '__main__': unittest.main()
