import copy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from efficient_llm.warm_order.data import sha
from efficient_llm.warm_order.embedding_anchor_prepare import prepare, validate_plan, REFERENCE_CASES
from efficient_llm.warm_order.model import Config


class AnchorPreparationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.c = Config(d=8, ff=16, heads=1, kv_heads=1, head_dim=8, vocab=17, prefix=2, length=4)
        stem = str(self.root / 'corpus')
        np.random.default_rng(3).integers(0, 17, (128768, 4), dtype=np.uint32).tofile(stem + '.bin')
        starts = np.zeros((128768, 4), dtype=np.uint8); starts[:, 0] = 1
        starts.tofile(stem + '.starts')
        self.plan = json.loads(Path('resources/embedding_anchor_pilot_20261122.json').read_text())
        self.plan['model'] = asdict(self.c)
        panels = {}
        for seed in (0, 1):
            cfg = dict(seed=seed, domain='dclm', steps=2000, batch=32, data=stem)
            for key, lo, hi in [('indices', seed * 64000, (seed + 1) * 64000),
                                ('calibration_indices', 128000, 128256),
                                ('confirmation_indices', 128256, 128768)]:
                path = self.root / (str(seed) + key + '.npy')
                np.save(path, np.arange(lo, hi, dtype=np.int64)); cfg[key] = str(path)
            cfg['expected_inputs'] = {p: sha(p) for p in [stem + '.bin', stem + '.starts']
                                      + [cfg[k] for k in ('indices', 'calibration_indices', 'confirmation_indices')]}
            path = self.root / ('recipe' + str(seed) + '.json'); path.write_text(json.dumps(dict(config=cfg)))
            panels[str(seed)] = dict(seed=seed, data_recipe=dict(path=str(path), sha256=sha(path)),
                                     expected_inputs=cfg['expected_inputs'], row_reservation={})
        self.plan['seed_panels'] = panels
        prior = self.root / 'prior.json'; prior.write_text(json.dumps(dict(seed_panels=panels)))
        self.plan['data_provenance_source'] = dict(path=str(prior), sha256=sha(prior))
        self.path = self.root / 'plan.json'; self.path.write_text(json.dumps(self.plan))

    def tearDown(self): self.tmp.cleanup()

    def test_two_seed_prepare_zero_updates(self):
        result = prepare(self.path, sha(self.path), self.root / 'out', fixture_only=True)
        assert result['train_updates'] == 0 and not result['gpu_admitted'] and result['fixture_only']
        fingerprints = []
        for seed in (0, 1):
            root = self.root / 'out' / ('s' + str(seed))
            meta = json.loads((root / 'config.json').read_text())
            ref = torch.load(root / 'cpu-reference.pt', weights_only=True)
            state = torch.load(root / 'initial-fullstate.pt', weights_only=False)
            assert set(ref) == set(REFERENCE_CASES)
            assert state['step'] == 0 and not state['optimizer']['state']
            assert all(v == 0 for v in meta['repeat_max_abs'].values())
            assert any(n.startswith('blocks.1.') for n in ref['single0']['inactive'])
            assert not any(n.startswith('blocks.0.') for n in ref['single0']['inactive'])
            fingerprints.append(meta['initial_fingerprint'])
        assert fingerprints[0] != fingerprints[1]
        with self.assertRaises(ValueError): prepare(self.path, sha(self.path), self.root / 'out', fixture_only=True)

    def test_recipe_and_fixture_fail_closed(self):
        bad = copy.deepcopy(self.plan); bad['phase_updates'][1] += 1
        with self.assertRaisesRegex(ValueError, 'registration'): validate_plan(bad, fixture_only=True)
        with self.assertRaisesRegex(ValueError, 'dimensions'): validate_plan(self.plan)
        bad = copy.deepcopy(self.plan); bad['seed_panels']['0']['expected_inputs'] = {}
        with self.assertRaisesRegex(ValueError, 'provenance'): validate_plan(bad, fixture_only=True)
        with self.assertRaises(ValueError): prepare(self.path, '0' * 64, self.root / 'bad', fixture_only=True)


if __name__ == '__main__': unittest.main()
