from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from efficient_llm.a2.data import write_json
from efficient_llm.diagnostic_data import sha256
from scripts.a2_preflight_inputs import initial_jobs, verify_prepared, initial_asset_plan
from run_experiments import load_jobs


class PreflightInputTests(unittest.TestCase):
    def test_initial_parent_scope_and_missing_hash_refusal(self):
        asset = dict(weights_dir='/model', code_dir='/code',
            files={'config.json':'a', 'model.safetensors':'b'},
            code_files={'modeling_opensci.py':'c', 'configuration_opensci.py':'d'})
        downloads = [dict(path=directory+'/'+name, sha256=digest)
            for directory, files in (('/model', asset['files']), ('/code', asset['code_files']))
            for name, digest in files.items()]
        plan = dict(assets={'dclm/P300':asset}, downloads=downloads+[
            dict(path='/other/download-in-progress', sha256='e')])
        self.assertEqual(initial_asset_plan(plan)['downloads'], downloads)
        self.assertEqual(len(plan['downloads']), 5)
        plan['downloads'] = plan['downloads'][1:]
        with self.assertRaisesRegex(ValueError, 'lock incomplete'):
            initial_asset_plan(plan)

    def test_only_bounded_initial_stages(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'jobs.json'
            jobs = initial_jobs('/frozen', '/inputs', 29537)
            write_json(path, dict(jobs=jobs))
            load_jobs(path)
            self.assertEqual([j['name'] for j in jobs], ['throughput', 'survey'])
            for job in jobs:
                self.assertEqual(job['gpus'], list(range(8)))
                self.assertNotIn('heal', job['argv'])
                self.assertIn('accelerate.commands.launch', job['argv'])
                self.assertLessEqual(job['timeout_seconds'], 3600)

    def test_contract_refuses_changed_or_failed_inputs(self):
        # PackedRows byte/shape verification has its own tests; exercise the
        # launch contract independently of multi-GB production token files.
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root/'data').mkdir()
            spec = root/'spec'; spec.write_text('preregistration')
            plan = dict(assets={'model':'locked'}, pool_contract=dict(revision='a'*40,
                manifest_sha256='b'*64))
            write_json(root/'assets.json', plan['assets'])
            write_json(root/'verified_inputs.json', {'status':'passed'})
            write_json(root/'secondary_map.json', {ds:dict(skip_reason='fixture') for ds in ('dclm','fwe','nem')})
            for name in ('documents.jsonl','indices.npz'):
                (root/'data'/name).write_bytes(b'fixed document order')
            manifest = dict(status='complete', provenance=dict(spec_sha256=sha256(spec),
                **plan['pool_contract']), disjoint_by_id_and_content=True,
                streams=dict(heal=dict(tokens=850000000), tuning=dict(rows=1024)),
                small_indices=list(range(256)),
                documents_sha256=sha256(root/'data/documents.jsonl'),
                indices_sha256=sha256(root/'data/indices.npz'))
            write_json(root/'data/manifest.json', manifest)
            receipt = dict(status='passed',data_manifest_sha256=sha256(root/'data/manifest.json'),
                verified_inputs_sha256=sha256(root/'verified_inputs.json'),
                secondary_map_sha256=sha256(root/'secondary_map.json'))
            write_json(root/'complete.json', receipt)
            with patch('scripts.a2_preflight_inputs.PackedRows') as rows:
                rows.return_value.length = 4096
                self.assertEqual(verify_prepared(root, spec, plan), receipt)
                self.assertEqual(rows.call_count, 3)
                (root/'data/indices.npz').write_bytes(b'changed order')
                with self.assertRaisesRegex(ValueError, 'metadata changed'):
                    verify_prepared(root, spec, plan)
                (root/'failure.json').write_text('{}')
                with self.assertRaisesRegex(ValueError, 'preparation receipt'):
                    verify_prepared(root, spec, plan)
                (root/'failure.json').unlink()
                write_json(root/'assets.json', {'model':'different'})
                with self.assertRaisesRegex(ValueError, 'asset contract'):
                    verify_prepared(root, spec, plan)


if __name__ == '__main__':
    unittest.main()
