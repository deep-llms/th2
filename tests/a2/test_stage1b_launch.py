import json
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest

from scripts.a2_stage1b_budget import budget
from scripts.launch_a2_stage1b_heals import phase_estimate, heals_jobs, admit_part2, admission_budget
from efficient_llm.a2_stage1b.protocol import REPLICATION_SPEC_SHA
from efficient_llm.diagnostic_data import sha256


class Stage1bLaunchTests(unittest.TestCase):
    def test_budget_and_deadline_reserve(self):
        now=datetime(2026,10,30,tzinfo=timezone.utc)
        end='2026-10-31T00:00:00+00:00'
        self.assertFalse(budget(1000,20,10000,end,now)['report_due'])
        self.assertTrue(budget(71000,20,1000,end,now)['report_due'])
        for spent,requested,deadline in [(85500,1000,end),(0,1000,'2026-10-30T00:20:00+00:00'),(float('nan'),1,end)]:
            with self.assertRaises(ValueError):budget(spent,0,requested,deadline,now)

    def test_reported_projection_preserves_hard_cap_and_deadline(self):
        now=datetime(2026,10,30,tzinfo=timezone.utc)
        end='2026-10-31T00:00:00+00:00'
        with self.assertRaisesRegex(ValueError,'Report over20h'):
            admission_budget(41000,10,32000,end,False,now)
        receipt=admission_budget(41000,10,32000,end,True,now)
        self.assertTrue(receipt['report_due'])
        self.assertTrue(receipt['over20h_reported'])
        for spent,requested,deadline in [(60000,30000,end),(41000,32000,'2026-10-30T08:00:00+00:00')]:
            with self.assertRaisesRegex(ValueError,'24h cap or continuation deadline'):
                admission_budget(spent,10,requested,deadline,True,now)

    def test_exact_part_job_contracts(self):
        gate=dict(step_seconds=1,eligible_x72=True)
        self.assertEqual(phase_estimate(gate,'part1'),8*762+2*12207+10*180)
        self.assertEqual(phase_estimate(gate,'part2'),12207+3*3051+762+5*180)
        for part in ('part1','part2'):
            job=heals_jobs(Path('/inputs'),Path('/pre'),1234,part,1000,20000)['jobs'][0]
            self.assertEqual(job['gpus'],list(range(8)))
            contract=job['required_outputs'][0]['json_equals']
            self.assertEqual(contract['phase'],part)
            self.assertEqual(contract['training_complete'],part=='part2')

    def test_interim_and_actual_predecessor_binding(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'downstream';root.mkdir()
            def save(p,v):
                p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))
                return dict(path=str(p),sha256=sha256(p))
            interim=save(root/'interim.json',dict(status='verified',report_sent=True,
                report=save(root/'report.json',{}),decision=save(root/'decision.json',{})))
            ds={k:save(root/rel,dict(status='ok',models=4)) for k,rel in [
                ('supervisor','supervised/supervisor.json'),('run_complete','supervised/run/complete.json'),
                ('burn','supervised/burn-verified.json'),('complete','supervised/run/downstream/complete.json')]}
            prior=save(root/'prior.json',dict(stage1b_spec_sha256=REPLICATION_SPEC_SHA,interim=interim,downstream=ds))
            burn=root/'source/resources/llm_pretrain_burn.py'
            admit_part2(prior,burn)
            with self.assertRaisesRegex(ValueError,'predecessor'):admit_part2(prior,root/'other/source/resources/llm_pretrain_burn.py')
            Path(interim['path']).write_text('{}')
            with self.assertRaises(ValueError):admit_part2(prior,burn)
