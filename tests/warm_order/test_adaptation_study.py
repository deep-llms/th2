import copy
import json
import shutil
import unittest

import numpy as np

import test_adaptation_pipeline as fixture
from efficient_llm.warm_order.adaptation_study import run, replay, join
from efficient_llm.warm_order.data import sha


class AdaptationStudy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.AdaptationPipeline.setUpClass()
        cls.root = fixture.AdaptationPipeline.root
        indices = cls.root/'study_ids.npy'; np.save(indices, np.array([192, 193]))
        stem = fixture.AdaptationPipeline.config['data']
        cls.config = dict(schema=1, domain='dclm', parent=str(cls.root/'parent'),
            parent_hashes=fixture.AdaptationPipeline.pins, data=stem, indices=str(indices),
            expected_inputs={p: sha(p) for p in (stem+'.bin', stem+'.starts', str(indices))},
            batch_size=2, positions=[1, 6], geometry_keys=['slot/2/input', 'block/2/ffn_update'],
            max_moment_bytes=1<<20, snapshots=[])
        for step in (0, 4):
            files = {}
            for branch in ('A', 'B'):
                path = cls.root/branch/'snapshots'/f'{step:06d}'/'fullstate.pt'
                files[branch] = dict(path=str(path), sha256=sha(path))
            cls.config['snapshots'].append(dict(step=step, phase_files=files))
        cls.out = cls.root/'study'
        cls.report = run(cls.config, cls.out)

    @classmethod
    def tearDownClass(cls): fixture.AdaptationPipeline.tearDownClass()

    def test_paired_study_and_portable_raw_replay(self):
        self.assertEqual(replay(self.out)['snapshots'], 2)
        self.assertEqual([p['step'] for p in self.report['points']], [0, 4])
        for point in self.report['points']:
            self.assertEqual(set(point['geometry']), {'immediate', 'ordinary_drift', 'swapped_drift', 'remaining'})
        at_zero = self.report['points'][0]['geometry']
        for comparison in ('ordinary_drift', 'swapped_drift'):
            for value in at_zero[comparison]['metrics'].values():
                self.assertEqual(value['mean_squared_distance'], 0.)
        moved = self.root/'moved_study'; shutil.copytree(self.out, moved)
        self.assertEqual(replay(moved)['status'], 'PASS')
        with self.assertRaisesRegex(ValueError, 'Fresh'): run(self.config, self.out)

    def test_cross_snapshot_geometry_mixup_and_report_tampering(self):
        out = self.root/'mixed'; shutil.copytree(self.out, out)
        target = out/'step_000004/geometry/remaining'; shutil.rmtree(target)
        shutil.copytree(out/'step_000000/geometry/remaining', target)
        with self.assertRaisesRegex(ValueError, 'pairing'):
            join([out/'step_000000', out/'step_000004'])
        out = self.root/'tampered'; shutil.copytree(self.out, out)
        p = out/'REPORT.json'; report = json.loads(p.read_text())
        report['points'][1]['remaining_gap_nat'] += .1; p.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'hash'): replay(out)
        p = out/'manifest.json'; manifest = json.loads(p.read_text())
        manifest['files']['REPORT.json'] = sha(out/'REPORT.json'); p.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'numeric'): replay(out)

    def test_declared_registration_and_preflight_guards(self):
        bad = copy.deepcopy(self.config); bad['snapshots'][1]['phase_files']['A']['sha256'] = 'bad'
        with self.assertRaisesRegex(ValueError, 'hash'): run(bad, self.root/'bad')
        self.assertFalse((self.root/'bad').exists())
        bad = copy.deepcopy(self.config); bad['domain'] = 'fwe'
        with self.assertRaisesRegex(ValueError, 'DCLM'): run(bad, self.root/'wrong_domain')
        out = self.root/'wrong_registration'; shutil.copytree(self.out, out)
        p = out/'config.json'; config = json.loads(p.read_text()); config['positions'] = [0, 7]
        p.write_text(json.dumps(config))
        p = out/'manifest.json'; manifest = json.loads(p.read_text())
        manifest['files']['config.json'] = sha(out/'config.json'); p.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'registration'): replay(out)


if __name__ == '__main__': unittest.main()
