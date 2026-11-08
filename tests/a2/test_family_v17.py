import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from efficient_llm.a2_family_v17.protocol import runs,capacity
from efficient_llm.a2_family_v17.common import SPEC_SHA,write,read
from efficient_llm.a2_family_v17.readout import analyze,report
from scripts.a2_quant_io import lock


class V17(unittest.TestCase):
    def test_exact_registration(self):
        r=runs();self.assertEqual(len(r),6);self.assertEqual([i['phase'] for i in r],['F8']*4+['F7']*2)
        self.assertEqual([i['arm'] for i in r],['P300','P1T','P300','P1T','S72','S188'])
        self.assertEqual([i['removed'] for i in r],[list(range(19,22))]*2+[list(range(13,22))]*2+[list(range(16,22))]*2)
        for item in r:self.assertEqual((item['steps'],item['seed'],item['stream_seed'],item['lr']),(12207,0,2030,.001))

    def test_reading_order_and_missing(self):
        self.assertEqual(capacity({3:.01,6:.011,9:.012,12:.0125}),'A supported')
        self.assertEqual(capacity({3:.02,6:.01,9:.022,12:.021}),'A not supported')
        self.assertEqual(capacity({3:.01,6:.009,9:.02,12:.03}),'mixed')
        self.assertEqual(capacity({}),'unverified')

    def fixture(self):
        history=dict(cuts32={'6':dict(E=.02,G=.1),'12':dict(E=.04,G=.1)},E08={'3':.012,'9':.034},
            stable_series={'dclm':{'0.2B':dict(E=.02),'0.8B':dict(E=.012)},'fwe':{}})
        results=[]
        for i,item in enumerate(runs()):
            par=np.array([2.,3.,4.])-(.1 if i%2 else 0);post=par+np.array([.1,.2,.3])+(i//2+1)*.01*(i%2)
            score=lambda x:dict(per_sequence_nll=x.tolist(),loss=float(x.mean()),sequences=3,tokens=3*4095)
            results.append(dict(run_id=item['run_id'],registered_run=item,spec_sha256=SPEC_SHA,baseline=score(par),pre={'final':score(par+.5)},scores={'final':score(post)},delta_post=float((post-par).mean()),flags=[]))
        return results,history

    def test_arrays_bootstrap_and_preserved_readings(self):
        r,h=self.fixture();d=analyze(r,h,100)
        self.assertTrue(d['coverage_complete']);self.assertAlmostEqual(d['F8']['pairs']['F8-L3']['E'],.01)
        self.assertAlmostEqual(d['F7']['pair']['E'],.03)
        self.assertEqual(d['F7']['FE6_DCLM_32B']['label'],'rises')
        self.assertAlmostEqual(d['F8']['pairs']['F8-L3']['parent_gain_kept'],.9)
        ci=d['F8']['pairs']['F8-L3']['bootstrap']['E_ci95'];self.assertTrue(np.allclose(ci,[.01,.01]))
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);old={'F6':{'joint_reading':'mixed','FE6':{'dclm':'flat'}},'spec_sha256':'old'}
            (root/'old.json').write_text(json.dumps(old));(root/'old.md').write_text('original report')
            write(root/'pause.json',dict(state='paused'))
            cfg=dict(results_root=str(root),historical_v17=h,previous_decision=lock(root/'old.json'),previous_report=lock(root/'old.md'),depth_pause=lock(root/'pause.json'),FT10={'required':False})
            for v in r:write(root/'heals'/v['run_id']/'result.json',v)
            report(cfg,[],final=True);combined=read(root/'decision_family.json')
            for k,v in old.items():self.assertEqual(combined[k],v)
            self.assertTrue((root/'REPORT.md').read_text().startswith('original report'))

    def test_corrupt_and_failed_run_flagged_not_fabricated(self):
        r,h=self.fixture();r[0]['scores']['final']['loss']+=1
        d=analyze(r,h,0);self.assertFalse(d['coverage_complete']);self.assertEqual(len(d['missing_runs']),1)
        self.assertTrue(d['F8']['unverified']);self.assertTrue(d['missing_or_invalid'])

    def test_queue_orders_report_backup_before_exact_depth(self):
        from scripts.launch_a2_family_v17 import jobs
        from run_experiments import load_jobs
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'jobs.json';p.write_text(json.dumps(jobs('config.json')));load_jobs(p)
        j=jobs('config.json')['jobs'];self.assertEqual(len(j),2)
        self.assertIn('scripts.a2_family_v17_run',j[0]['argv']);self.assertIn('scripts.a2_family_v17_depth_bridge',j[1]['argv'])
        self.assertTrue(j[0]['required_outputs'][0]['json_equals']['backup_verified'])

    def test_markdown_report_is_hashed_not_parsed_as_json(self):
        from scripts.a2_family_v17_prepare import verify_previous
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'REPORT.md').write_text('# Family report\nActual Markdown, not JSON.\n')
            write(root/'decision.json',{});write(root/'backup.json',{'verified':True})
            cfg=dict(previous_report=lock(root/'REPORT.md'),previous_decision=lock(root/'decision.json'),previous_backup=lock(root/'backup.json'))
            verify_previous(cfg)
            (root/'REPORT.md').write_text('changed')
            with self.assertRaisesRegex(ValueError,'Markdown report changed'):verify_previous(cfg)

    def test_bad_model_prevents_launch(self):
        from scripts.a2_family_v17_prepare import verify_inputs
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'w').write_bytes(b'corrupt');write(root/'assets.json',{'dclm/P300':dict(weights_dir=t,files={'w':'0'*64})})
            with patch('efficient_llm.a2.protocol.parent_ref',return_value={}):
                with self.assertRaisesRegex(ValueError,'Input SHA mismatch'):verify_inputs(dict(assets=lock(root/'assets.json')))

    def test_failure_continues_all_six_before_backup(self):
        from scripts import a2_family_v17_run as runner
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);events=[]
            def launch(cfg,config,item,ev):events.append(item['run_id']);ev.append(dict(name=item['run_id'],returncode=1 if len(events)==1 else 0))
            def report(cfg,ev,final=False):
                if final:(root/'REPORT.md').write_text('failed first run retained');write(root/'decision_family.json',dict(events=ev))
                return dict(coverage_complete=False)
            with patch.object(runner,'owner'),patch.object(runner,'launch',side_effect=launch),patch.object(runner,'report',side_effect=report),patch('scripts.a2_family_v17_prepare.verify_inputs',return_value=[]):
                runner.execute(dict(results_root=t),'config',root/'receipt.json')
            self.assertEqual(events,[r['run_id'] for r in runs()]);self.assertTrue(read(root/'receipt.json')['backup_verified']);self.assertFalse(read(root/'receipt.json')['coverage_complete'])


class Numerics(unittest.TestCase):
    def test_traced_trainer_identical_to_a2_s(self):
        import torch
        from transformers import default_data_collator,set_seed
        from test_model import tiny_model
        from test_train import Rows
        from efficient_llm.a2.model import KDModel,prune
        from efficient_llm.a2.train import make_arguments
        from efficient_llm.a2_why.heal import ReportTrainer as Old,Trace
        from efficient_llm.a2_family_v17.heal import ReportTrainer as New
        teacher=tiny_model().requires_grad_(False);base=copy.deepcopy(teacher).requires_grad_(True);prune(base,[1,2]);states=[];orders=[];lrs=[]
        with tempfile.TemporaryDirectory() as d:
            for cls in (Old,New):
                set_seed(0);model=copy.deepcopy(base);trace=Trace()
                kwargs=dict(trace_path=Path(d)/'trace.jsonl') if cls is New else {}
                trainer=cls(model=KDModel(model,3),teacher=teacher,
                            args=make_arguments(Path(d)/cls.__module__,12,.001,4,1,cuda=False,global_sequences=8),
                            train_dataset=Rows(96),data_collator=default_data_collator,callbacks=[trace],**kwargs)
                trainer.train();states.append(copy.deepcopy(model.state_dict()));orders.append(trainer.seen);lrs.append(trace.lrs)
                if cls is New:
                    trainer.trace_file.close()
                    saved=[json.loads(x)['sample_ids'] for x in (Path(d)/'trace.jsonl').read_text().splitlines()]
                    self.assertEqual(saved,trainer.seen)
            self.assertEqual(lrs[0],lrs[1]);self.assertEqual(orders[0],orders[1])
            for k in states[0]:self.assertTrue(torch.equal(states[0][k],states[1][k]),k)
            self.assertTrue(all(p.grad is None for p in teacher.parameters()))
