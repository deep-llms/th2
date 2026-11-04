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

    def test_dclm_stage_then_rest_reuses_without_duplicate_fits(self):
        from scripts.a2_why_run import reuse_run
        from scripts.a2_quant_io import lock
        from efficient_llm.a2_why.protocol import base_runs
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);old=base/'old';partial=base/'partial';rest=base/'rest'
            partial.mkdir();rest.mkdir();yt1=base_runs()[0]
            src=old/'heals'/yt1['run_id'];write(src/'result.json',dict(registered_run=yt1,status='complete'))
            source=dict(root=str(src),result=lock(src/'result.json'))
            seen=[]
            def run(cfg,item,events):
                seen.append(item['run_id'])
                p=Path(cfg['results_root'])/'heals'/item['run_id']/'result.json'
                write(p,dict(registered_run=item,status='complete'))
                events.append(dict(run_id=item['run_id'],returncode=0,result=lock(p)))
            cfg=dict(results_root=str(partial),reused_yt1=source)
            with patch('scripts.a2_why_run.run_one',side_effect=run):execute(cfg,'y1-dclm',str(partial/'terminal.json'))
            expected=[x for x in base_runs() if x['phase']=='Y1' and x['dataset']=='dclm']
            self.assertEqual(seen,[x['run_id'] for x in expected]);self.assertFalse((partial/'Y1_NOTE.md').exists())
            reused={x['run_id']:dict(root=str(partial/'heals'/x['run_id']),result=lock(partial/'heals'/x['run_id']/'result.json')) for x in expected}
            cfg=dict(results_root=str(rest),reused_yt1=source,reused_y1=reused);seen.clear()
            with patch('scripts.a2_why_run.run_one',side_effect=run),patch('scripts.a2_why_run.part1',return_value={'tau':.003}),patch('scripts.a2_why_run.part2',return_value={}),patch('scripts.a2_why_run.part3',return_value={'extensions':{}}),patch('scripts.a2_why_run.final',return_value={'coverage_complete':True}):execute(cfg,'rest',str(rest/'terminal.json'))
            self.assertEqual(len(seen),36)
            self.assertFalse(set(reused)&set(seen));self.assertNotIn(yt1['run_id'],seen)
            # Corrupt identity, while keeping a valid content lock, must not be reused.
            bad=base/'bad';write(bad/'result.json',dict(registered_run=yt1,status='incomplete'))
            with self.assertRaises(ValueError):reuse_run(dict(results_root=str(base/'reject')),yt1,dict(root=str(bad),result=lock(bad/'result.json')),[])

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
