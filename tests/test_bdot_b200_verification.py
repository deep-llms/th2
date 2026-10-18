"""CPU-only checks that verification preserves failures and does not report false passes."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from scripts import bdot_b200_verify as verify


class VerificationTests(unittest.TestCase):
    def test_failure_is_retained_and_independent_case_runs(self):
        for required in (True, False):
            with self.subTest(required=required), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                output = root/'checks'
                config = root/'accelerate.yaml'
                config.write_text('fixture')
                cases = [dict(name='first', argv=['fixture'], gpus=[], receipt=None, required=required),
                         dict(name='second', argv=['fixture'], gpus=[], receipt=None, required=True)]
                with patch.object(verify,'build_cases',return_value=cases), \
                     patch.object(verify,'require_free',return_value=[]), \
                     patch.object(verify.time,'sleep'), \
                     patch.object(verify.subprocess,'run',side_effect=[SimpleNamespace(returncode=7),SimpleNamespace(returncode=0)]) as run, \
                     patch('sys.argv',['verify','--output',str(output),'--accelerate-config',str(config)]):
                    code = verify.main()
                self.assertEqual(run.call_count,2)
                self.assertEqual(code,1 if required else 0)
                summary=json.loads((output/'summary.json').read_text())
                self.assertFalse(summary['results'][0]['passed'])
                self.assertTrue(summary['results'][1]['passed'])
                self.assertEqual(summary['failed_required'],['first'] if required else [])

    def test_zero_exit_without_required_receipt_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); config=root/'accelerate.yaml'; config.write_text('fixture')
            output=root/'checks'
            cases=[dict(name='missing',argv=['fixture'],gpus=[],receipt=str(output/'missing.json'),required=True)]
            with patch.object(verify,'build_cases',return_value=cases), \
                 patch.object(verify,'require_free',return_value=[]), patch.object(verify.time,'sleep'), \
                 patch.object(verify.subprocess,'run',return_value=SimpleNamespace(returncode=0)), \
                 patch('sys.argv',['verify','--output',str(output),'--accelerate-config',str(config)]):
                self.assertEqual(verify.main(),1)


if __name__=='__main__':
    unittest.main()
