import copy
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from efficient_llm.warm_order.model import Config
from efficient_llm.warm_order.data import sha, Rows
from efficient_llm.warm_order.horizon_train import train as horizon_train
from efficient_llm.warm_order.exposure_recovery import restore_rng
from efficient_llm.warm_order.adaptation_train import (
    train, load_parent, validate_adam, PARENT_FILES, POLICY, CANONICAL,
)


def same(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: same(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b): same(x, y)
    else:
        assert a == b, (a, b)


class AdaptationTrain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory(); cls.root = Path(cls.temp.name)
        r = cls.root; cls.model_config = Config(d=8, heads=1, kv_heads=1, head_dim=8,
                                               ff=16, vocab=19, prefix=2, length=8)
        stem = str(r/'data')
        np.random.default_rng(21).integers(0, 19, (200, 8), dtype=np.uint32).tofile(stem+'.bin')
        starts = np.zeros((200, 8), dtype='u1'); starts[:, 0] = 1; starts.tofile(stem+'.starts')
        np.save(r/'parent_ids.npy', np.arange(160)); np.savez(r/'probe.npz', valid=np.arange(192, 196))
        c = dict(model=asdict(cls.model_config), arm='D', seed=0, batch=2, steps=80,
                 lr=.0003, warmup=4, data=stem, indices=str(r/'parent_ids.npy'),
                 probe_indices=str(r/'probe.npz'), probe_size=2, probe_every=80,
                 threads=1, log_every=80, save_every=80)
        c['expected_inputs'] = {p: sha(p) for p in (stem+'.bin', stem+'.starts', c['indices'], c['probe_indices'])}
        horizon_train(c, r/'parent', 'cpu')
        cls.pins = {name: sha(r/'parent'/name) for name in PARENT_FILES}
        np.save(r/'phase_ids.npy', np.arange(160, 168)); np.save(r/'confirmation.npy', np.arange(192, 196))
        cls.config = dict(branch='A', parent=str(r/'parent'), parent_hashes=cls.pins,
            steps=4, batch=2, threads=1, domain='dclm', lr_policy=POLICY, snapshots=[0, 1, 4],
            data=stem, indices=str(r/'phase_ids.npy'), confirmation_indices=str(r/'confirmation.npy'))
        cls.config['expected_inputs'] = {p: sha(p) for p in (stem+'.bin', stem+'.starts', cls.config['indices'], cls.config['confirmation_indices'])}

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def test_both_branches_exact_fullstate_resume_and_matched_rows(self):
        traces = []
        for branch in ('A', 'B'):
            c = dict(self.config, branch=branch)
            whole = self.root/('whole'+branch); part = self.root/('part'+branch)
            train(c, whole); train(c, part, stop_after=2); train(c, part, resume=True)
            a = torch.load(whole/'fullstate.pt', weights_only=False)
            b = torch.load(part/'fullstate.pt', weights_only=False)
            same(a, b)
            ta = [json.loads(s) for s in (whole/'trace.jsonl').read_text().splitlines()]
            tb = [json.loads(s) for s in (part/'trace.jsonl').read_text().splitlines()]
            self.assertEqual(ta, tb); traces.append(ta)
            self.assertEqual(a['optimizer']['state'][0]['step'].item(), 84)
            self.assertAlmostEqual(a['optimizer']['param_groups'][0]['lr'], .00003)
        self.assertEqual([x['rows'] for x in traces[0]], [x['rows'] for x in traces[1]])
        self.assertNotEqual(traces[0][0]['path'], traces[1][0]['path'])
        self.assertNotEqual(traces[0][0]['loss'], traces[1][0]['loss'])
        self.assertEqual(self.pins, {name: sha(self.root/'parent'/name) for name in PARENT_FILES})

    def test_initial_forks_and_independent_first_adam_update(self):
        states = []
        for branch in ('A', 'B'):
            out = self.root/('fork'+branch)
            train(dict(self.config, branch=branch), out, stop_after=1)
            states.append(torch.load(out/'snapshots/000000/fullstate.pt', weights_only=False))
        parent = torch.load(self.root/'parent/fullstate.pt', weights_only=False)
        for state in states:
            for key in ('model', 'optimizer', 'torch_rng', 'numpy_rng', 'python_rng'):
                same(state[key], parent[key])
        same(states[0]['recipe']['paired_common'], states[1]['recipe']['paired_common'])
        self.assertEqual(states[0]['recipe']['parameter_names'], states[1]['recipe']['parameter_names'])
        m, opt, s, names = load_parent(self.root/'parent', self.pins)
        restore_rng(s, 'cpu'); m.train()
        x = Rows(self.config['data'], 8).batch(np.array([160, 161]), 'cpu')
        opt.zero_grad(set_to_none=True); loss = m.nll(x, CANONICAL).mean(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1., error_if_nonfinite=True); opt.step()
        actual = torch.load(self.root/'forkA/fullstate.pt', weights_only=False)
        same(m.state_dict(), actual['model']); same(opt.state_dict(), actual['optimizer'])

    def test_phase_tamper_overlap_branch_change_and_cuda_rejected(self):
        c = copy.deepcopy(self.config); c['expected_inputs'][c['indices']] = 'bad'
        with self.assertRaisesRegex(ValueError, 'identity'): train(c, self.root/'bad-hash')
        self.assertFalse((self.root/'bad-hash').exists())
        np.save(self.root/'overlap.npy', np.array([160, 193]))
        c = copy.deepcopy(self.config); c['confirmation_indices'] = str(self.root/'overlap.npy')
        c['expected_inputs'] = {p: sha(p) for p in (c['data']+'.bin', c['data']+'.starts', c['indices'], c['confirmation_indices'])}
        with self.assertRaisesRegex(ValueError, 'overlap'): train(c, self.root/'bad-overlap')
        out = self.root/'resume-guard'; train(self.config, out, stop_after=2)
        with self.assertRaisesRegex(ValueError, 'recipe'): train(dict(self.config, branch='B'), out, resume=True)
        with self.assertRaisesRegex(ValueError, 'GPU admission'): train(self.config, self.root/'gpu', device='cuda')

    def test_parent_hash_id_order_and_adam_counter_guards(self):
        bad = dict(self.pins); bad['fullstate.pt'] = 'bad'
        with self.assertRaisesRegex(ValueError, 'hash'): load_parent(self.root/'parent', bad)
        m, o, s, names = load_parent(self.root/'parent', self.pins)
        bad = copy.deepcopy(s['optimizer']); bad['param_groups'][0]['params'][0:2] = [1, 0]
        with self.assertRaisesRegex(ValueError, 'ID order'): validate_adam(m, bad, 80, .00003)
        bad = copy.deepcopy(s['optimizer']); bad['state'][0]['step'].zero_()
        with self.assertRaisesRegex(ValueError, 'counter'): validate_adam(m, bad, 80, .00003)
        bad = copy.deepcopy(s['optimizer']); bad['state'][0]['exp_avg_sq'].fill_(-1)
        with self.assertRaisesRegex(ValueError, 'Negative'): validate_adam(m, bad, 80, .00003)


if __name__ == '__main__': unittest.main()
