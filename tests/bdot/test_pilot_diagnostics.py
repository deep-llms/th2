import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import MethodType

import numpy as np
import torch
from accelerate import Accelerator
from datasets import Dataset

from test_model import model
from packed_training import PackedCollator
from eval.route_study import run_route_study, summarize_routes
from eval.pilot_diagnostics import RouterObserver, router_gradient_check


def rows():
    return [dict(input_ids=[1,2,3,4,5,6], labels=[1,2,3,4,5,6],
                 attention_mask=[1]*6, segments=[0,0,0,1,1,1]) for _ in range(4)]


class PilotDiagnosticsTests(unittest.TestCase):
    def test_observer_preserves_results_and_collects_exact_prefix(self):
        m=model(curriculum='A');m.eval()
        data=Dataset.from_list(rows());acc=Accelerator(cpu=True)
        before={k:v.clone() for k,v in m.state_dict().items()}
        expected=run_route_study(m,data,PackedCollator(),acc,2,route_sequences=3,artifact_sequences=1)
        with RouterObserver(m,acc,3) as observer:
            actual=run_route_study(m,data,PackedCollator(),acc,2,route_sequences=3,artifact_sequences=1)
        self.assertEqual(actual[:2],expected[:2])
        health=observer.finish()
        self.assertTrue(health['passed'])
        self.assertEqual(health['raw_gap']['count'],3*6*4)
        self.assertTrue(all(s['tokens']==18 for s in health['steps']))
        for k,v in m.state_dict().items():torch.testing.assert_close(v,before[k],atol=0,rtol=0)
        self.assertTrue(all('select' not in l.__dict__ for l in m.routed_steps))

    def test_no_reuse_single_candidate_masks_and_exception_cleanup(self):
        m=model(reuse=False,curriculum='A');m.eval();acc=Accelerator(cpu=True)
        with RouterObserver(m,acc,4) as observer:
            run_route_study(m,Dataset.from_list(rows()),PackedCollator(),acc,2,route_sequences=4,artifact_sequences=1)
        health=observer.finish();self.assertTrue(health['passed'])
        self.assertLess(health['raw_gap']['count'],4*6*4)
        with self.assertRaisesRegex(RuntimeError,'test'):
            with RouterObserver(m,acc,4):raise RuntimeError('test')
        self.assertTrue(all('select' not in l.__dict__ for l in m.routed_steps))

    def test_usage_distribution(self):
        r=np.array([[[0,0,2,3],[1,0,2,3]]],dtype=np.int32)
        s=summarize_routes(r)
        self.assertEqual(s['distinct_modules_histogram'],[0,0,0,1,1])
        self.assertEqual(s['mean_distinct_modules'],3.5)
        self.assertEqual(s['module_load_share'],[3/8,1/8,2/8,2/8])

    def test_full_gradient_surrogate_and_state_restoration(self):
        m=model(curriculum='A');m.eval()
        before={k:v.clone() for k,v in m.state_dict().items()}
        flags=[p.requires_grad for p in m.parameters()]
        result=router_gradient_check(m,rows()[0],length=6)
        self.assertTrue(result['passed'],result)
        self.assertEqual(len(result['steps']),4)
        self.assertEqual([p.requires_grad for p in m.parameters()],flags)
        for k,v in m.state_dict().items():torch.testing.assert_close(v,before[k],atol=0,rtol=0)
        self.assertEqual(m.config._attn_implementation,'sdpa')
        self.assertTrue(all('select' not in l.__dict__ for l in m.routed_steps))

    def test_gradient_check_rejects_reversed_gate_derivative(self):
        m=model(curriculum='A');m.eval()
        for layer in m.routed_steps:
            original=layer.select
            def reversed_select(this,h,context,original=original):
                choice,weight=original(h,context)
                return choice,2-weight
            layer.select=MethodType(reversed_select,layer)
        result=router_gradient_check(m,rows()[0],length=6)
        self.assertFalse(result['passed'])
        self.assertTrue(all(not r['local_pass'] for r in result['steps']))

    def test_actual_balance_update_signs(self):
        from eval.pilot_log_analysis import replay_balance_sign
        for counts in ([10,10,10,10],[0,20,10,10],[30,1,5,4]):
            actual,expected=replay_balance_sign(counts,.001)
            torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-9)

    def test_centered_gradient_preserves_forward_and_rejects_wrong_sign(self):
        m=model(curriculum='A');m.eval()
        ordinary=router_gradient_check(m,rows()[0],length=6)
        centered=router_gradient_check(m,rows()[0],length=6,center_logits=True)
        self.assertTrue(centered['passed'],centered)
        self.assertEqual(centered['loss'],ordinary['loss'])
        for layer in m.routed_steps:
            original=layer.select
            def wrong(this,h,context,original=original):
                choice,weight=original(h,context)
                return choice,2-weight
            layer.select=MethodType(wrong,layer)
        bad=router_gradient_check(m,rows()[0],length=6,center_logits=True)
        self.assertFalse(bad['passed'])

    def test_checkpoint_cli_distributed_diagnostics(self):
        from test_data_and_trainer import tokenizer_assets
        from train import DataArguments
        from training_data import prepare_training_data,save_data_recipe
        from transformers import TrainingArguments
        root=Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);tok=tokenizer_assets(p/'checkpoint')
            model(curriculum='A').save_pretrained(p/'checkpoint')
            Dataset.from_dict({'id':[str(i) for i in range(20)],'text':['a b c d '*8]*20}).save_to_disk(p/'raw')
            data=DataArguments(data_dir=str(p/'raw'),data_order='document_stream',dataset_revision='fixture',
                validation_manifest=str(p/'manifest.json'),validation_tokens=40,block_size=8,data_cache_dir=str(p/'cache'))
            prep=TrainingArguments(output_dir=str(p/'unused'),use_cpu=True,report_to='none')
            _,recipe=prepare_training_data(data,prep,tok,p/'checkpoint',8)
            save_data_recipe(p/'checkpoint',recipe)
            results=[]
            for name,multi in [('single',False),('multi',True)]:
                config=dict(checkpoint=str(p/'checkpoint'),eval_data_dir=str(p/'raw'),
                    output_file=str(p/name/'final.json'),use_cpu=True,batch_size=2,
                    data_cache_dir=str(p/'cache'),route_study=True,route_sequences=5,
                    route_artifact_sequences=1,pilot_diagnostics=True,router_gradient_check=True,
                    router_gradient_center_logits=True)
                file=p/(name+'.json');file.write_text(json.dumps(config))
                cmd=[sys.executable]
                if multi:cmd+=['-m','torch.distributed.run','--standalone','--nproc_per_node=2']
                cmd += [str(root/'eval/packed_ppl.py'),str(file)]
                done=subprocess.run(cmd,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1'},
                    cwd=root,capture_output=True,text=True,timeout=180)
                self.assertEqual(done.returncode,0,done.stdout+done.stderr)
                result=json.loads((p/name/'final.json').read_text())
                self.assertTrue(result['route_study']['router_health']['passed'])
                self.assertTrue(result['route_study']['gradient_check']['passed'])
                self.assertTrue(result['route_study']['centered_gradient_check']['passed'])
                results.append(result)
            self.assertEqual(results[0]['route_study']['route_diagnostics'],results[1]['route_study']['route_diagnostics'])
            for a,b in zip(results[0]['route_study']['router_health']['steps'],results[1]['route_study']['router_health']['steps']):
                self.assertEqual(a['tokens'],b['tokens'])
                self.assertAlmostEqual(a['mean_logsumexp'],b['mean_logsumexp'],places=6)


if __name__=='__main__':unittest.main()
