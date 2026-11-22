import copy
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from efficient_llm.warm_order.data import sha
from efficient_llm.warm_order.embedding_anchor_train import CPUFixtureTrainer
from efficient_llm.warm_order.model import Config


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: equal_tree(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): equal_tree(x, y)
    else:
        assert a == b


class AnchorTrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.c = Config(d=8, ff=16, heads=1, kv_heads=1, head_dim=8,
                       vocab=17, prefix=2, length=4)
        g = torch.Generator().manual_seed(19)
        cls.tokens = torch.randint(0, 17, (61440, 4), generator=g)
        cls.ids = torch.randperm(61440, generator=g)
        t = CPUFixtureTrainer(cls.c, cls.tokens, cls.ids)
        t.advance(768)
        cls.warm = cls.root / 'warm.pt'
        cls.digest = t.save(cls.warm)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def restore(self, path=None, digest=None, arm=None):
        return CPUFixtureTrainer.restore(path or self.warm, digest or self.digest,
                                         self.c, self.tokens, self.ids, fork_arm=arm)

    def test_shared_fork_and_donor_resume(self):
        a, c = self.restore(arm='A'), self.restore(arm='C')
        equal_tree(a.model.state_dict(), c.model.state_dict())
        equal_tree(a.optimizer.state_dict(), c.optimizer.state_dict())
        assert a.fork_sha256 == c.fork_sha256 == self.digest
        for arm in ('A', 'C'):
            full = self.restore(arm=arm); full.advance(776)
            part = self.restore(arm=arm); part.advance(771)
            path = self.root / (arm + '.pt'); digest = part.save(path)
            expected_rng = (torch.rand(3), np.random.random(3), random.random())
            resumed = self.restore(path, digest)
            equal_tree(expected_rng, (torch.rand(3), np.random.random(3), random.random()))
            # Restore again, so the comparison does not consume restored RNG.
            resumed = self.restore(path, digest); resumed.advance(776)
            equal_tree(full.model.state_dict(), resumed.model.state_dict())
            equal_tree(full.optimizer.state_dict(), resumed.optimizer.state_dict())
            equal_tree(full.trace, resumed.trace)
            assert sha(self.warm) == self.digest

    def test_phase_transition_and_stack_resume(self):
        for arm in ('A', 'C'):
            t = self.restore(arm=arm); t.advance(1152)
            phase = self.root / ('phase-' + arm + '.pt'); digest = t.save(phase)
            full = self.restore(phase, digest); full.advance(1156)
            part = self.restore(phase, digest); part.advance(1154)
            path = self.root / ('stack-' + arm + '.pt'); short = part.save(path)
            resumed = self.restore(path, short); resumed.advance(1156)
            equal_tree(full.model.state_dict(), resumed.model.state_dict())
            equal_tree(full.optimizer.state_dict(), resumed.optimizer.state_dict())
            equal_tree(full.trace, resumed.trace)
            assert full.trace[1151]['exposure']['phase'] == 'donor'
            assert full.trace[1152]['exposure']['phase'] == 'stack'

    def test_binding_counter_and_fork_rejection(self):
        bad_ids = self.ids.flip(0)
        with self.assertRaises(ValueError):
            CPUFixtureTrainer.restore(self.warm, self.digest, self.c, self.tokens, bad_ids)
        with self.assertRaises(ValueError): self.restore(digest='0' * 64)
        shared = self.restore()
        with self.assertRaises(ValueError): shared.advance(769)
        with self.assertRaises(FileExistsError): shared.save(self.warm)
        changed = torch.load(self.warm, weights_only=False)
        changed['optimizer']['state'][0]['step'] += 1
        path = self.root / 'tamper.pt'; torch.save(changed, path)
        with self.assertRaises(ValueError): self.restore(path, sha(path))
        a = self.restore(arm='A'); a.advance(769)
        path = self.root / 'branch.pt'; digest = a.save(path)
        with self.assertRaises(ValueError): self.restore(path, digest, arm='C')


if __name__ == '__main__':
    unittest.main()
