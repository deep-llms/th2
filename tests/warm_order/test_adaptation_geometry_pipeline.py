import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from efficient_llm.warm_order.adaptation_geometry import GeometryMoments, aligned_observations
from efficient_llm.warm_order.adaptation_geometry_pipeline import evaluate_geometry, replay_geometry, COMPARISONS
from efficient_llm.warm_order.data import Rows, sha
from efficient_llm.warm_order.model import Model, Config


class GeometryPipeline(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(126)
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        model = Model(Config(d=8, heads=2, kv_heads=1, head_dim=4, ff=16,
                             vocab=19, prefix=2, length=8)).eval().requires_grad_(False)
        self.models = {name: copy.deepcopy(model) for name in ('parent', 'A', 'B')}
        with torch.no_grad():
            for i, name in enumerate(('A', 'B')):
                for param in self.models[name].parameters(): param.add_(torch.randn_like(param)*(.001*(i+1)))
        self.stem = str(self.root/'data')
        np.random.default_rng(126).integers(0, 19, (4, 8), dtype=np.uint32).tofile(self.stem+'.bin')
        starts = np.zeros((4, 8), dtype='u1'); starts[:, 0] = 1; starts.tofile(self.stem+'.starts')
        self.ids = self.root/'ids.npy'; np.save(self.ids, np.array([3, 0, 2]))
        self.pins = {p: sha(p) for p in (self.stem+'.bin', self.stem+'.starts', str(self.ids))}
        self.keys = ('slot/2/input', 'block/2/input', 'block/2/attention_update', 'block/3/ffn_update')
        self.positions = (1, 6)

    def tearDown(self): self.temp.cleanup()

    def run_panel(self, comparison, out, **kwargs):
        return evaluate_geometry(self.models, comparison, self.stem, self.ids, self.pins,
                                 2, self.positions, self.keys, out, **kwargs)

    def test_all_comparisons_streamed_vs_direct_and_no_mutation(self):
        saved = {name: copy.deepcopy(m.state_dict()) for name, m in self.models.items()}
        rng = torch.get_rng_state().clone()
        tokens = Rows(self.stem, 8).batch(np.load(self.ids), 'cpu')
        for comparison, (left, pl, right, pr) in COMPARISONS.items():
            out = self.root/comparison; report = self.run_panel(comparison, out)
            replay = replay_geometry(out)
            self.assertEqual(replay['observations_per_key'], 6)
            samples = aligned_observations(self.models[left], self.models[right], tokens,
                                           self.positions, path_x=pl, path_y=pr)
            for key in self.keys:
                m = GeometryMoments(8); m.update(samples[key]['x'], samples[key]['y'])
                direct = m.report()
                for field in ('centered_linear_cka', 'mean_squared_distance', 'mean_cosine', 'rms_x', 'rms_y'):
                    self.assertAlmostEqual(report[key][field], direct[field], delta=2e-6)
            manifest = json.loads((out/'manifest.json').read_text())
            self.assertFalse(manifest['raw_moments_exportable'])
            self.assertFalse(manifest['production_ready'])
            np.testing.assert_array_equal(np.load(out/'row_ids.npy'), np.load(self.ids))
        torch.testing.assert_close(rng, torch.get_rng_state(), atol=0, rtol=0)
        for name, model in self.models.items():
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, saved[name][key], atol=0, rtol=0)

    def test_report_hash_then_numeric_tamper_rejected(self):
        out = self.root/'tamper'; self.run_panel('immediate', out)
        p = out/'REPORT.json'; report = json.loads(p.read_text())
        report[self.keys[0]]['rms_x'] += .1; p.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'hash'): replay_geometry(out)
        p = out/'manifest.json'; manifest = json.loads(p.read_text())
        manifest['files']['REPORT.json'] = sha(out/'REPORT.json'); p.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'numeric'): replay_geometry(out)

    def test_budget_input_and_moment_schema_guards(self):
        with self.assertRaisesRegex(ValueError, 'budget'):
            self.run_panel('immediate', self.root/'budget', max_moment_bytes=1)
        self.assertFalse((self.root/'budget').exists())
        self.pins[str(self.ids)] = 'bad'
        with self.assertRaisesRegex(ValueError, 'identity'): self.run_panel('immediate', self.root/'bad')
        self.assertFalse((self.root/'bad').exists())
        m = GeometryMoments(8); m.update(torch.randn(3, 8), torch.randn(3, 8))
        raw = m.arrays(); raw['counts'][2] = 4
        with self.assertRaisesRegex(ValueError, 'counts'): GeometryMoments.from_arrays(raw)
        raw = m.arrays(); raw['xy'][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'matrix'): GeometryMoments.from_arrays(raw)


if __name__ == '__main__': unittest.main()
