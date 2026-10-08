import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.proxy_10k_fresh_queue import make
from scripts import stop_proxy_followup


class Proxy10KQueueTests(unittest.TestCase):
    def test_fresh_matched_pair_and_validation(self):
        recipe=Path(__file__).resolve().parents[1]/'proxy_heads.b200.json'
        with tempfile.TemporaryDirectory() as temporary:
            items=make(recipe,Path(temporary)/'jobs.json')
            self.assertEqual([item['name'] for item in items],[
                'seed-1042-arm-A','seed-1042-validate-A',
                'seed-1042-arm-P6-iso','seed-1042-validate-P6-iso','seed-1042-compare'])
            for arm,job in zip(('A','P6-iso'),(items[0],items[2])):
                argv=job['argv']
                self.assertEqual(job['gpus'],list(range(8)))
                self.assertEqual(argv[:2],['env','PYTHONHASHSEED=1042'])
                for flag,expected in [('--arm',arm),('--seed','1042'),('--data_seed','1042'),
                                      ('--max_steps','28600'),('--warmup_steps','1430'),
                                      ('--stop_after','10000'),('--save_steps','250'),
                                      ('--save_total_limit','2')]:
                    self.assertEqual(argv[argv.index(flag)+1],expected)
                self.assertNotIn('--resume_from_checkpoint',argv)
            self.assertNotIn('--proxy_module_seed',items[0]['argv'])
            self.assertEqual(items[2]['argv'][items[2]['argv'].index('--proxy_module_seed')+1],'1043')
            self.assertEqual(items[-1]['required_outputs'][0]['json_equals']['compared_update'],10000)

    def test_handoff_requires_successful_burn_and_guard_release(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            supervised=root/'supervised'
            supervised.mkdir()
            (supervised/'supervisor.json').write_text(json.dumps({
                'finished_at':'done','burn':{'session':stop_proxy_followup.BURN_SESSION}}))
            (supervised/'burn-verified.json').write_text(json.dumps({
                'session':stop_proxy_followup.BURN_SESSION,
                'collective_progress_verified':True}))
            original_exists=Path.exists
            with patch.object(stop_proxy_followup,'ROOT',root):
                with patch.object(Path,'exists',autospec=True,
                                  side_effect=lambda p: False if str(p)=='/mnt/local/_gpu_guard/DISABLED' else original_exists(p)):
                    self.assertTrue(stop_proxy_followup.handed_off())


if __name__=='__main__':
    unittest.main()
