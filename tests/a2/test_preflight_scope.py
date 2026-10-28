import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from efficient_llm.a2.protocol import (
    DATASETS, PARENTS, FIXED_SETS, ORIGINAL_DATA_SPEC, PREFLIGHT_SCOPE_SPEC, DEPENDENCY_SPEC,
    compatible_data_spec, shared_check_sets, validate_shared_check, layers_for, validate_run_scope, Run,
)
from scripts.a2_preflight_inputs import shared_numerical_jobs
from run_experiments import load_jobs


def fixture():
    assets = {f'{ds}/{p}': dict(code_files={
        'modeling_opensci.py': 'a'*64, 'configuration_opensci.py': 'b'*64})
        for ds in DATASETS for p in PARENTS}
    sets = {ds: dict(FIXED_SETS, S2=groups) for ds, groups in
        zip(DATASETS, ([2, 4, 8], [2, 4, 8], [3, 5, 7]))}
    return assets, sets


class SharedScopeTests(unittest.TestCase):
    def test_data_reuse_is_exact_and_data_spec_unchanged(self):
        spec = Path(__file__).resolve().parents[2]/'docs/A2_pilot_spec.md'
        content = spec.read_bytes()
        original = content.split(b'\n\n## Approved preflight scope amendment', 1)[0]
        self.assertEqual(hashlib.sha256(original).hexdigest(), ORIGINAL_DATA_SPEC)
        previous=content.split(b'\n\n## Approved dependency-based launch amendment',1)[0]
        self.assertEqual(hashlib.sha256(previous).hexdigest(), PREFLIGHT_SCOPE_SPEC)
        self.assertEqual(hashlib.sha256(content).hexdigest(), DEPENDENCY_SPEC)
        self.assertTrue(compatible_data_spec(ORIGINAL_DATA_SPEC, DEPENDENCY_SPEC))
        self.assertTrue(compatible_data_spec(PREFLIGHT_SCOPE_SPEC, DEPENDENCY_SPEC))
        self.assertTrue(compatible_data_spec(ORIGINAL_DATA_SPEC, PREFLIGHT_SCOPE_SPEC))
        self.assertTrue(compatible_data_spec(PREFLIGHT_SCOPE_SPEC, PREFLIGHT_SCOPE_SPEC))
        self.assertFalse(compatible_data_spec(PREFLIGHT_SCOPE_SPEC, ORIGINAL_DATA_SPEC))
        self.assertFalse(compatible_data_spec(ORIGINAL_DATA_SPEC, 'f'*64))
        self.assertFalse(compatible_data_spec('f'*64, PREFLIGHT_SCOPE_SPEC))

    def test_union_of_all_s2_choices_and_fail_closed(self):
        assets, sets = fixture()
        patterns = shared_check_sets(sets, assets)
        self.assertEqual(len(patterns), 5)  # Three fixed sets plus two unique S2s.
        expected = {tuple(layers_for(g)) for s in sets.values() for g in s.values()}
        self.assertEqual({tuple(layers_for(g)) for g in patterns.values()}, expected)
        bad = copy.deepcopy(assets)
        bad['nem/S188']['code_files']['modeling_opensci.py'] = 'c'*64
        with self.assertRaisesRegex(ValueError, 'identical'):
            shared_check_sets(sets, bad)
        with self.assertRaisesRegex(ValueError, 'every dataset'):
            shared_check_sets({'dclm': sets['dclm']}, assets)
        bad = copy.deepcopy(sets); bad['nem']['S1'] = [1, 2, 3]
        with self.assertRaisesRegex(ValueError, 'Fixed removal'):
            shared_check_sets(bad, assets)

    def test_heal_gate_requires_complete_current_shared_receipt(self):
        assets, sets = fixture()
        patterns = [list(layers_for(g)) for g in shared_check_sets(sets, assets).values()]
        patterns += [[0], [11], [23]]
        prov = {'software': {'torch': 'runtime'}, 'code_commit': 'a'*40}
        receipt = dict(passed=True, scope='shared_mechanism', representative='dclm/P300',
            provenance=prov, sets_sha256='frozen', removal_patterns=patterns)
        validate_shared_check(receipt, sets, assets, prov, 'frozen')
        for key, value in [('passed', False), ('representative', 'nem/P300'),
                           ('provenance', {}), ('sets_sha256', 'stale'),
                           ('removal_patterns', patterns[:-1])]:
            bad = dict(receipt, **{key: value})
            with self.assertRaisesRegex(ValueError, 'shared numerical'):
                validate_shared_check(bad, sets, assets, prov, 'frozen')

    def test_bounded_gate_cannot_unlock_other_runs(self):
        assets, _ = fixture()
        scope='dclm_P300_S1_grid'; gate={'launch_scope':scope}
        sets={'dclm':{'S1':FIXED_SETS['S1']}}
        self.assertEqual(len(shared_check_sets(sets,assets,scope=scope)),1)
        for lr in (3e-5,1e-4,3e-4):
            run=Run('dclm','P300','S1','0.03B',lr)
            validate_run_scope(run,gate,scope)
            retry=Run('dclm','P300','S1','0.03B',lr/2,'-rerun')
            validate_run_scope(retry,gate,scope,run.run_id)
            with self.assertRaises(ValueError):validate_run_scope(retry,gate,scope)
        for run in (Run('dclm','P1T','S1','0.03B',1e-4),
                    Run('dclm','P300','S2','0.2B',1e-4),
                    Run('dclm','P300','S1','0.2B',1e-4)):
            with self.assertRaisesRegex(ValueError,'not covered'):
                validate_run_scope(run,gate,scope)
        with self.assertRaisesRegex(ValueError,'scope mismatch'):
            validate_run_scope(Run('dclm','P300','S1','0.03B',1e-4),gate,'all')

    def test_historical_numerical_code_lock(self):
        from efficient_llm.a2.early_grid import ast_hash
        root=Path(__file__).resolve().parents[2]
        lock=json.loads((root/'resources/a2_early_grid_reuse.json').read_text())
        for file,digest in lock['unchanged_files'].items():
            self.assertEqual(hashlib.sha256((root/file).read_bytes()).hexdigest(),digest,file)
        for file,names in lock['unchanged_definitions'].items():
            for name,digest in names.items():
                self.assertEqual(ast_hash(root/file,name),digest,(file,name))

    def test_bounded_manifest(self):
        from scripts.a2_preflight_inputs import early_grid_jobs
        with tempfile.TemporaryDirectory() as d:
            jobs=early_grid_jobs('/inputs','/prior',29537)
            path=Path(d)/'jobs.json';path.write_text(json.dumps({'jobs':jobs}))
            load_jobs(path)
            self.assertEqual(len(jobs),1)
            self.assertEqual(jobs[0]['timeout_seconds'],3700)
            self.assertEqual(jobs[0]['gpus'],list(range(8)))

    def test_post_s2_job_runs_mechanism_once_and_never_repeats_t8(self):
        assets, sets = fixture()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'sets.json'; path.write_text(json.dumps({'sets': sets}))
            jobs = shared_numerical_jobs('/source', '/inputs', 29537, path, assets)
            manifest = Path(d)/'jobs.json'; manifest.write_text(json.dumps({'jobs': jobs}))
            load_jobs(manifest)
            self.assertEqual(len(jobs), 1)
            job = jobs[0]
            self.assertEqual(job['name'], 'check')
            self.assertEqual(job['gpus'], list(range(8)))
            for unused in ('throughput', 'survey', 'heal'):
                self.assertNotIn(unused, job['argv'])
            self.assertEqual(job['argv'][job['argv'].index('--parent')+1], 'P300')
            self.assertEqual(job['argv'][job['argv'].index('--dataset')+1], 'dclm')
            self.assertIn(str(path), job['argv'])


if __name__ == '__main__':
    unittest.main()
