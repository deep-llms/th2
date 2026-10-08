import importlib
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from scripts.proxy_speed_queue import make
from scripts.proxy_speed_validation import implementation, exact_tensors, gate, NEW_ARMS, BENCH_ARMS
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

    def test_manifest_scope_and_identical_training_recipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            items=make(root,Path('proxy_heads.b200.json'),Path('/real/rows.json'))
            numerics=[x for x in items if x['name'].startswith('numerics-')]
            self.assertEqual(len(numerics),9)
            training=[x for x in items if x.get('gpus')==list(range(8))]
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
            for arm in NEW_ARMS:
                names=[x['name'] for x in items]
                self.assertLess(names.index('smoke-'+arm),names.index('copy-resume-'+arm))
                self.assertLess(names.index('copy-resume-'+arm),names.index('resume-'+arm))
                self.assertLess(names.index('resume-'+arm),names.index('check-resume-'+arm))
            self.assertEqual(len([x for x in training if '--profile' in x['argv']]),2*len(BENCH_ARMS))
            self.assertEqual(items[-1]['name'],'summarize')

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
