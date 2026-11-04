import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from efficient_llm.a2_why.common import write,read
from scripts.a2_why_run import execute
from efficient_llm.a2_why.protocol import descriptor
from scripts.a2_why_report import part1,part2,part3

class WhyOrchestrationTests(unittest.TestCase):
    def test_nonfinite_retained_without_json_abort(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.json';write(p,dict(loss=float('nan'),trace=[float('inf')]))
            v=read(p);self.assertIsNone(v['loss']);self.assertTrue(v['unverified']);self.assertEqual(len(v['nonfinite_values']),2)

    def test_order_extensions_and_terminal(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=dict(results_root=d);seen=[]
            def run(cfg,item,events):seen.append(item);events.append(dict(run_id=item['run_id'],returncode=0))
            with patch('scripts.a2_why_run.run_one',side_effect=run),patch('scripts.a2_why_run.part1',return_value={'tau':.003}),patch('scripts.a2_why_run.part2',return_value={'rdsc':{'extension':True},'avg':{'extension':False}}),patch('scripts.a2_why_run.part3',return_value={'extensions':{'dclm':{'trigger':True},'fwe':{'trigger':False}}}),patch('scripts.a2_why_run.final',return_value={'coverage_complete':True}):
                execute(cfg,'rest',str(Path(d)/'terminal.json'))
            self.assertEqual(len(seen),47)
            self.assertEqual([x['phase'] for x in seen[:9]],['YT1']+['Y1']*8)
            self.assertTrue(all(x['variant']=='rdsc' and x['budget']=='0.8B' for x in seen[17:21]))
            self.assertTrue(all(x['removed']==list(range(10,22)) and x['dataset']=='dclm' and x['budget']=='3.2B' for x in seen[-2:]))
            self.assertTrue(read(Path(d)/'terminal.json')['coverage_complete'])

    def test_missing_results_are_unverified_not_pass(self):
        with tempfile.TemporaryDirectory() as d:
            ref={ds:{b:dict(E=.02,P300_delta=.2) for b in ('0.2B','0.8B')} for ds in ('dclm','fwe')}
            cfg=dict(results_root=d,historical=dict(datasets=ref));n=part1(cfg);self.assertIsNone(n['tau']);self.assertTrue(n['unverified'])
            self.assertTrue(all(v['reading']=='unverified' for v in part2(cfg,n).values()))
            self.assertEqual(part3(cfg,n)['reading'],'unverified')

    def test_saved_results_noise_uses_historical_seed_and_new_pairs(self):
        with tempfile.TemporaryDirectory() as d:
            ref={ds:{b:dict(E=.02,P300_delta=.2) for b in ('0.2B','0.8B')} for ds in ('dclm','fwe')}
            cfg=dict(results_root=d,historical=dict(datasets=ref))
            for ds in ('dclm','fwe'):
                for seed,e in ((2031,.03),(2032,.04)):
                    for arm,delta in (('P300',.2),('P1T',.2+e)):
                        item=descriptor('Y1',ds,arm,seed=seed)
                        write(Path(d)/'heals'/item['run_id']/'result.json',dict(run_id=item['run_id'],delta_post=delta,unverified=False,flags=[]))
            n=part1(cfg);self.assertAlmostEqual(n['sigma'],.01);self.assertAlmostEqual(n['tau'],.02);self.assertAlmostEqual(n['datasets']['dclm']['mean'],.03)

if __name__=='__main__':unittest.main()
