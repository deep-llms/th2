import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from efficient_llm.warm_order.multi_exit_prepare import prepare
from efficient_llm.warm_order.data import sha


class Tests(unittest.TestCase):
    def test_both_prior_reservations_excluded_and_hash_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stem = str(root / 'data')
            np.zeros((100, 8), dtype=np.uint32).tofile(stem + '.bin')
            starts = np.zeros((100, 8), dtype='u1')
            starts[:, 0] = 1
            starts[90:, 3] = 1
            starts.tofile(stem + '.starts')
            np.savez(root / 'eval.npz', fit=np.arange(2), valid=np.arange(2, 4))
            parent = root / 'parent.pt'
            parent.write_bytes(b'identity-only fixture; never loaded as model')
            previous = []
            for i in range(2):
                directory = root / f'previous{i}'
                directory.mkdir()
                for seed in (0, 1):
                    np.save(directory / f'train-s{seed}.npy', np.arange(4 + i * 4 + seed * 2, 6 + i * 4 + seed * 2))
                manifest = directory / 'complete.json'
                manifest.write_text(json.dumps({'files': {p.name: sha(p) for p in directory.iterdir()}}))
                previous.append(str(manifest))
            hashes = {p: sha(p) for p in [stem + '.bin', stem + '.starts', str(root / 'eval.npz'), str(parent), *previous]}
            training = {str(seed): dict(data=stem, model={'length': 8}, steps=8, batch=2) for seed in (0, 1)}
            plan = dict(hashes=hashes, previous_reservations=previous, training=training,
                        weights={str(seed): str(parent) for seed in (0, 1)}, eval={'indices': str(root / 'eval.npz')})
            prepare(plan, root / 'prepared')
            for seed in (0, 1):
                selected = np.load(root / 'prepared' / f'train-s{seed}.npy')
                self.assertEqual(len(np.unique(selected)), 16)
                self.assertTrue(np.all((selected >= 12) & (selected < 90)))
            (root / 'previous1' / 'train-s0.npy').write_bytes(b'changed reservation')
            with self.assertRaisesRegex(ValueError, 'Parent reservation changed'):
                prepare(plan, root / 'refused')
