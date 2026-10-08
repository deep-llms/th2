import importlib
import json
import os
from pathlib import Path
import tempfile
import subprocess
import shutil
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from scripts.proxy_speed_queue import make
from scripts.proxy_speed_validation import implementation, exact_tensors, rounding_violations, deterministic_fa4, gate, gate_all, diagnose, NEW_ARMS, BENCH_ARMS
from tests.test_proxy_memory import config, batch, objective


class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_reference_restores_modules_and_matches_tiny_model(self):
        current=importlib.import_module('deep_kv.proxy')
        captures=[]
        for name in ('previous','optimized'):
            with implementation(name):
                module=importlib.import_module('deep_kv.proxy')
                self.assertEqual(module is current,name=='optimized')
                m=module.ProxyModel.from_scratch(config(),'P6-iso-weighted',consumer=2,deep_target=8,
                    proxy_settings=module.ProxySettings(width=8,features=9,chunk_size=2),
                    checkpoint_layers=True,checkpoint_aux=True,checkpoint_lm=True,lm_chunk=3)
                out=m(batch(),collect_target_statistics=True);objective(out).backward()
                captures.append((out,{k:p.grad for k,p in m.named_parameters() if p.grad is not None}))
            self.assertIs(importlib.import_module('deep_kv.proxy'),current)
        for a,b in zip(*captures):self.assertEqual(exact_tensors(a,b),[])
        self.assertTrue(exact_tensors({'a':torch.ones(1)},{'a':torch.zeros(1)}))

    def test_gate_pairs_same_checkpoint_mode_and_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);recipe=root/'recipe.json';rows=root/'rows.json'
            recipe.write_text('{}');rows.write_text('[]')
            args=SimpleNamespace(arm='P6-iso-short',recipe=recipe,rows=rows,output=root/'passed.json')
            def capture(arm,backend,checkpoint,*unused):
                value=torch.tensor(float(checkpoint))
                return dict(initial='same-weights',outputs={'loss':value},gradients={'p':value},loss=0.,peak_gib=1.)
            with patch('scripts.proxy_speed_validation.capture',side_effect=capture):gate(args)
            self.assertEqual(len(json.loads(args.output.read_text())['cases']),8)
            counter=0
            def failing(*a):
                nonlocal counter
                counter+=1;result=capture(*a)
                result['gradients']['p']=torch.tensor(float(counter))
                return result
            args.output=root/'failed.json'
            with patch('scripts.proxy_speed_validation.capture',side_effect=failing):
                with self.assertRaises(AssertionError):gate(args)
            self.assertEqual(json.loads(args.output.read_text())['status'],'failed')

    def test_rounding_bound_rejects_nonfinite_and_substantive_changes(self):
        a={'p':torch.tensor([1.,2.])}
        for b,allowed in (({'p':torch.tensor([1.+2**-23,2.])},True),
                          ({'p':torch.tensor([1.001,2.])},False),
                          ({'p':torch.tensor([float('nan'),2.])},False)):
            self.assertEqual(not rounding_violations(a,exact_tensors(a,b)),allowed)
        zeros={'p':torch.zeros(2)}
        self.assertTrue(rounding_violations(zeros,exact_tensors(zeros,{'p':torch.full((2,),1e-20)})))

    def test_deterministic_kernel_override_is_scoped(self):
        from deep_kv import fa4
        def kernel(*args,deterministic=False):return deterministic
        loader=lambda:(kernel,{'version':'test'})
        with patch.object(fa4,'load_kernel',loader):
            with deterministic_fa4(True):
                fn,metadata=fa4.load_kernel()
                self.assertTrue(fn());self.assertTrue(metadata['deterministic'])
                self.assertTrue(metadata['validation_only'])
            self.assertIs(fa4.load_kernel,loader)
            self.assertFalse(fa4.load_kernel()[0]())

    def test_manifest_scope_and_identical_training_recipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            items=make(root,Path('proxy_heads.b200.json'),Path('/real/rows.json'))
            numerics=[x for x in items if x['name'].startswith('numerics-')]
            self.assertEqual(len(numerics),1)
            self.assertEqual(len(numerics[0]['required_outputs']),10)
            training=[x for x in items if x.get('gpus')==list(range(8)) and '--implementation' in x['argv']]
            self.assertEqual(len(training),20)  # Six fresh, six resume, eight benchmark.
            for x in training:
                a=x['argv']
                self.assertEqual(a[a.index('--stop_after')+1],'25')
                self.assertEqual(a[a.index('--max_steps')+1],'28600')
                self.assertEqual(a[a.index('--save_steps')+1],'24')
                self.assertEqual(a[a.index('--per_device_train_batch_size')+1],'16')
                self.assertEqual(a[a.index('--gradient_accumulation_steps')+1],'4')
                self.assertEqual(a[a.index('--logging_steps')+1],'10')
                self.assertIn('--module',a)
                self.assertEqual('--audit-update25' in a,'--profile' not in a)
                self.assertEqual('--deterministic-fa4' in a,'--profile' not in a)
            for arm in NEW_ARMS:
                names=[x['name'] for x in items]
                self.assertLess(names.index('smoke-'+arm),names.index('copy-resume-'+arm))
                self.assertLess(names.index('copy-resume-'+arm),names.index('resume-'+arm))
                self.assertLess(names.index('resume-'+arm),names.index('check-resume-'+arm))
            self.assertEqual(len([x for x in training if '--profile' in x['argv']]),2*len(BENCH_ARMS))
            self.assertEqual(items[-1]['name'],'summarize')

    def test_parallel_gates_cover_all_arms_and_stop_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            args=SimpleNamespace(recipe='recipe',rows='rows',output=root/'summary.json')
            def run(command,**kwargs):
                arm=command[command.index('--arm')+1]
                self.assertEqual(len(kwargs['env']['CUDA_VISIBLE_DEVICES'].split(',')),1)
                Path(command[-1]).write_text(json.dumps({'status':'failed' if arm=='P7' else 'passed'}))
                return SimpleNamespace(returncode=int(arm=='P7'))
            with patch.dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'),patch('subprocess.run',side_effect=run):
                with self.assertRaises(AssertionError):gate_all(args)
            report=json.loads(args.output.read_text())
            self.assertEqual(report['status'],'failed')
            self.assertEqual({x['arm'] for x in report['cases']},set((*NEW_ARMS,'P6-iso','P7-simple','P7')))

    def test_repeatability_is_measurement_not_acceptance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            items=make(root,Path('proxy_heads.b200.json'),Path('/real/rows.json'),diagnostic=True)
            self.assertEqual(len(items),1)
            args=SimpleNamespace(arm='P6-iso-sparse',recipe=root/'validation-recipe.json',
                                 rows=root/'rows.json',output=root/'measured.json')
            args.rows.write_text('[]')
            calls=[]
            def capture(arm,backend,checkpoint,*unused):
                calls.append((backend,checkpoint))
                return dict(initial='same',outputs={'x':torch.zeros(1)},
                            gradients={'p':torch.tensor([float(len(calls))])},loss=0.,peak_gib=1.)
            with patch('scripts.proxy_speed_validation.capture',side_effect=capture):diagnose(args)
            report=json.loads(args.output.read_text())
            self.assertEqual(report['status'],'measured')
            self.assertEqual(calls,[('sdpa',False)]*6+[('fa4',False)]*3)
            self.assertTrue(report['cases'][1]['differences']['gradients'])

    def test_profile_uses_existing_step_timer(self):
        from scripts.profile_proxy_training import Capture
        from deep_kv.proxy_training import ProxyCallback
        with tempfile.TemporaryDirectory() as tmp:
            cb=ProxyCallback(25,1048576);cb.seconds_per_update=1.25
            trainer=SimpleNamespace(callback_handler=SimpleNamespace(callbacks=[cb]))
            capture=Capture(trainer)
            args=SimpleNamespace(process_index=0,output_dir=tmp)
            for step in range(10,26):capture.on_step_end(args,SimpleNamespace(global_step=step),None)
            capture.on_train_end(args,None,None)
            result=json.loads((Path(tmp)/'step-profile.json').read_text())
            self.assertEqual(len(result['steps']),16)
            self.assertTrue(all(x['seconds']==1.25 for x in result['steps']))

    def test_real_training_wrapper_both_implementations(self):
        from safetensors.torch import load_file
        from tests.test_proxy_training import proxy_fixture
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=proxy_fixture(root)
            values=[]
            for name in ('previous','optimized'):
                cfg={**base,'arm':'P6-iso-weighted','stop_after':1,'output_dir':str(root/name)}
                path=root/(name+'.json');path.write_text(json.dumps(cfg))
                result=subprocess.run([sys.executable,'-m','scripts.proxy_speed_validation','train',
                    '--implementation',name,'--',str(path)],capture_output=True,text=True,
                    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},timeout=120)
                self.assertEqual(result.returncode,0,result.stdout[-3000:]+result.stderr[-3000:])
                self.assertEqual(json.loads((root/name/'result.json').read_text())['global_step'],1)
                values.append(load_file(root/name/'model.safetensors'))
            self.assertEqual(exact_tensors(*values),[])

    def test_real_wrapper_audits_resumed_data(self):
        from datasets import Dataset
        from safetensors.torch import load_file
        from tests.test_proxy_training import proxy_fixture
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=proxy_fixture(root)
            shutil.rmtree(root/'train')
            Dataset.from_dict({'text':[' '.join(str((row+j)%30) for j in range(7)) for row in range(250)]}).save_to_disk(root/'train/en')
            values=[];records=[]
            for name in ('full','resumed'):
                folder=root/name
                if name=='resumed':
                    folder.mkdir()
                    shutil.copy2(root/'full/train_config.json',folder/'train_config.json')
                    shutil.copytree(root/'full/checkpoint-24',folder/'checkpoint-24')
                cfg={**base,'arm':'P6-iso-weighted','max_steps':30,'stop_after':25,
                     'save_steps':24,'gradient_accumulation_steps':4,'output_dir':str(folder)}
                path=root/(name+'.json');path.write_text(json.dumps(cfg))
                result=subprocess.run([sys.executable,'-m','scripts.proxy_speed_validation','train',
                    '--implementation','optimized','--audit-update25','--',str(path)],
                    capture_output=True,text=True,
                    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},timeout=180)
                self.assertEqual(result.returncode,0,result.stdout[-3000:]+result.stderr[-3000:])
                records.append(json.loads((folder/'update25-rank0.json').read_text()))
                values.append(load_file(folder/'model.safetensors'))
            self.assertEqual(records[0],records[1])
            self.assertEqual(len(records[0]['microbatches']),4)
            self.assertEqual(exact_tensors(*values),[])
