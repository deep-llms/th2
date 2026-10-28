import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from scripts.launch_a2_preflight import verify_failed_handoff

ROOT=Path(__file__).resolve().parents[2]

class FailedHandoffTests(unittest.TestCase):
    def test_actual_failed_cpu_stage_and_burn_receipts(self):
        archive=ROOT/'docs/results/a2_branches_20261028/final_surveys'
        with tempfile.TemporaryDirectory() as d:
            parent=Path(d);runroot=parent/'old';burn=runroot/'source/resources/llm_pretrain_burn.py'
            lock=dict(schema=1,scope='failed_cpu_freeze_after_completed_surveys')
            for key,local,rel in [('run','run.json','supervised/run/run.json'),('supervisor','supervisor.json','supervised/supervisor.json'),('burn','burn-verified.json','supervised/burn-verified.json')]:
                path=runroot/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((archive/local).read_bytes())
                lock[key]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            lp=parent/'lock.json';lp.write_text(json.dumps(lock))
            self.assertEqual(verify_failed_handoff(parent,burn,lp)['root'],str(runroot))
            with self.assertRaisesRegex(ValueError,'root'):verify_failed_handoff(parent/'wrong',burn,lp)
            bad=copy.deepcopy(lock);bad['run']['path']=str(parent/'wrong');lp.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError,'receipt path'):verify_failed_handoff(parent,burn,lp)
            lp.write_text(json.dumps(lock));Path(lock['burn']['path']).write_text('{}')
            with self.assertRaisesRegex(ValueError,'changed'):verify_failed_handoff(parent,burn,lp)
            lock['burn']['sha256']=hashlib.sha256(b'{}').hexdigest();lp.write_text(json.dumps(lock))
            with self.assertRaisesRegex(ValueError,'incomplete'):verify_failed_handoff(parent,burn,lp)

if __name__=='__main__':unittest.main()
