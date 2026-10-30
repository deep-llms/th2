import copy
import unittest
import numpy as np
import torch
import tempfile
from pathlib import Path
import json
from unittest.mock import patch
from efficient_llm.a2.quant import rtn, gptq, grid_test, quantize_gptq, matrix_manifest, weight_digest
from efficient_llm.a2.quant_stats import label, pooled, nll_report, PARENTS

torch.set_num_threads(1)


class QuantTests(unittest.TestCase):
    def test_grid_formula_and_identity_hessian(self):
        torch.manual_seed(17)
        w=torch.randn(12,256).bfloat16().float()
        for bits in (3,4):
            q=rtn(w,bits)
            parts=w.reshape(12,2,128)
            scale=(parts.amax(-1,keepdim=True)-parts.amin(-1,keepdim=True))/(2**bits-1)
            zero=torch.round(-parts.amin(-1,keepdim=True)/scale)
            expected=((torch.round(parts/scale)+zero).clamp(0,2**bits-1)-zero)*scale
            self.assertTrue(torch.equal(q,expected.reshape_as(w)))
            self.assertTrue(torch.equal(q,gptq(w,torch.eye(256),bits)))
            self.assertTrue(grid_test(w,bits)['cardinality'])

    def test_nontrivial_hessian_and_constant_groups(self):
        torch.manual_seed(3)
        w=torch.randn(3,16);x=torch.randn(16,50);h=2*x@x.T/50
        q=gptq(w,h,3,group_size=8)
        self.assertTrue(torch.isfinite(q).all())
        self.assertFalse(torch.equal(q,rtn(w,3,8)))
        self.assertTrue(torch.equal(rtn(torch.full((2,128),.25),3),torch.full((2,128),.25)))

    def test_true_sequential_native_and_repeat(self):
        from test_model import tiny_model
        model=tiny_model(num_hidden_layers=2);other=copy.deepcopy(model)
        original={k:v.clone() for k,v in model.state_dict().items()}
        rows=[{'input_ids':torch.tensor([2,4,7,3,1,8,9,5])} for _ in range(3)]
        events=[]
        quantize_gptq(model,rows,3,lambda *args:events.append(args))
        quantize_gptq(other,rows,3,lambda *args:None)
        self.assertEqual(weight_digest(model),weight_digest(other))
        names={x['name'] for x in matrix_manifest(model)}
        self.assertEqual(len(names),14)
        for name,value in model.state_dict().items():
            if name not in names:self.assertTrue(torch.equal(value,original[name]),name)
        self.assertTrue(all(e[1] for e in events if e[0]!='QT3'))
        grid_events=[e for e in events if e[0]=='QT3']
        self.assertTrue(all(e[2]['cardinality'] and e[2]['fixed_grid_idempotent'] for e in grid_events))
        # Tiny GPTQ groups may no longer attain both original endpoints. Refit
        # idempotence is honestly flagged, without aborting later matrices/layers.
        self.assertTrue(any(not e[1] and e[2]['repeat_max_abs']>0 for e in grid_events))
        self.assertEqual([e[2]['matrix'] for e in events if e[0]=='GPTQ_matrix'][:7],
            ['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj'])
        self.assertTrue(torch.isfinite(model(rows[0]['input_ids'][None]).logits).all())

    def test_label_precedence_and_paired_sequences(self):
        self.assertEqual(label(.01,[.005,.02],.04)['label'],'more damage')
        self.assertEqual(label(-.01,[-.02,-.005],.04)['label'],'less damage')
        self.assertEqual(label(.002,[-.01,.01],.04)['label'],'no effect')
        self.assertEqual(label(.01,[-.01,.02],.04)['label'],'grey')
        base={p:[2.,3.,4.] for p in PARENTS}
        compressed={p:np.array(base[p])+(i+1)*.01 for i,p in enumerate(PARENTS)}
        report=nll_report({'dclm':{'parent':base,'rtn3':compressed}},resamples=200)
        value=report['datasets']['dclm']['rtn3']['contrasts']['L_ann']
        self.assertAlmostEqual(value['estimate'],.02)
        self.assertAlmostEqual(value['ci95'][0],.02)

    def test_pooled_identical_datasets_preserve_covariance(self):
        from eval.a2_stage1_stats import MODELS
        from eval.downstream_suite import COUNTS
        rng=np.random.default_rng(15)
        tasks={t:{m:rng.integers(0,2,20) for m in MODELS} for t in COUNTS}
        report=pooled({'dclm':tasks,'fwe':tasks},resamples=300)
        for metric in report['metrics'].values():
            for row in metric.values():
                self.assertEqual(row['dclm'],row['pooled'])
                self.assertAlmostEqual(row['correlation'],1.)

    def test_calibration_boundary_overlap_is_explicit(self):
        from scripts.a2_quant_inputs import calibrations
        with tempfile.TemporaryDirectory() as folder:
            rows=[dict(split='train',index=1,group='heal',text_sha256='a',gpt_neox_tokens=128*4096+3),
                  dict(split='train',index=2,group='heal',text_sha256='b',gpt_neox_tokens=128*4096),
                  dict(split='train',index=3,group='tuning',text_sha256='c',gpt_neox_tokens=4096),
                  dict(split='validation',index=1,group='final',text_sha256='d',gpt_neox_tokens=4096)]
            Path(folder,'documents.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            result=calibrations(folder)
            self.assertFalse(result['passed'])
            self.assertEqual(result['overlaps']['cal0:cal1']['ids'],[('train',1)])
            self.assertEqual(result['sequence_ids']['cal1'],list(range(128,256)))

    def test_matrix_failure_continues_and_preserves_flag(self):
        from test_model import tiny_model
        model=tiny_model(num_hidden_layers=2);events=[];calls=0
        def fail_once(*args,**kw):
            nonlocal calls
            calls+=1
            if calls==1:raise RuntimeError('injected Cholesky error')
            return gptq(*args,**kw)
        with patch('efficient_llm.a2.quant.gptq',side_effect=fail_once):
            quantize_gptq(model,[{'input_ids':torch.tensor([2,3,4,5])}],3,lambda *a:events.append(a))
        self.assertEqual(calls,14)
        failed=[r for r in events if r[0]=='GPTQ_matrix' and not r[1]]
        self.assertEqual(len(failed),1)
        self.assertIn('RTN',failed[0][2]['fallback'])

    def test_pinned_reference_gptq(self):
        import importlib.util
        import sys
        import os
        import hashlib
        folder=Path(os.environ['A2_GPTQ_REFERENCE'])
        expected=json.loads(Path('resources/a2_quant_gptq_reference_20261031.json').read_text())
        for name in ('gptq.py','quant.py'):
            self.assertEqual(hashlib.sha256((folder/name).read_bytes()).hexdigest(),expected['files'][name])
        sys.path.insert(0,str(folder))
        try:
            spec=importlib.util.spec_from_file_location('a2_pinned_reference_gptq',folder/'gptq.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        finally:sys.path.pop(0)
        from efficient_llm.a2.quant import grid
        class Grid:
            def __init__(self,bits):self.bits=bits;self.maxq=2**bits-1
            def ready(self):return True
            def find_params(self,w,weight=True):self.scale,self.zero,_,_=grid(w,self.bits)
        torch.manual_seed(5)
        layer=torch.nn.Linear(256,8,bias=False);x=torch.randn(256,400);h=2*x@x.T/400
        for bits in (3,4):
            native=copy.deepcopy(layer);reference=module.GPTQ(native)
            reference.H=h.clone();reference.quantizer=Grid(bits)
            with patch('torch.cuda.synchronize',lambda:None):reference.fasterquant(groupsize=128)
            self.assertTrue(torch.equal(native.weight,gptq(layer.weight,h,bits)))

    def test_report_partial_coverage_is_not_success(self):
        from scripts.a2_quant_report import render
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'complete.json').write_text(json.dumps({'coverage_complete':False,'execution_complete':True}))
            render(root,{'spec_sha256':'a'*64,'code_commit':'b'*40})
            report=(root/'REPORT.md').read_text()
            self.assertIn('Coverage complete: False',report)
            self.assertIn('chưa có artifact hợp lệ',report)

    def test_real_worker_baseline_artifact_flow_on_cpu(self):
        from scripts.a2_quant_worker import main
        from scripts.a2_quant_io import write, lock
        from test_model import tiny_model
        from contextlib import redirect_stdout
        import io
        model=tiny_model(num_hidden_layers=24)
        rows=[{'input_ids':torch.tensor([2,3,4,5,6,7,8,9])} for _ in range(2)]
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);assets=write(root/'assets.json',{})
            cfg=write(root/'config.json',dict(spec_sha256='a'*64,assets=assets,code_commit='b'*40,
                sets={'dclm':{name:dict(root='unused',stream='final',manifest={}) for name in ('final','own_corpus')}},references={}))
            args=['worker','--config',cfg['path'],'--config-sha256',cfg['sha256'],
                  '--dataset','dclm','--parent','S188','--phase','baseline','--output',str(root/'worker')]
            with patch('sys.argv',args),patch('efficient_llm.a2.model.load_parent',return_value=model),\
                 patch('efficient_llm.a2.data.PackedRows',return_value=rows),redirect_stdout(io.StringIO()):
                main()
            done=json.loads((root/'worker/complete.json').read_text())
            self.assertTrue(done['execution_complete'])
            self.assertEqual(len(done['runs']),4)
            tests=json.loads((root/'worker/tests.json').read_text())
            self.assertFalse(tests[0]['passed'])  # tiny shape is flagged; all four vals still exist.
            for item in done['runs']:
                result=json.loads(Path(item['per_sequence_file']['path']).read_text())
                self.assertEqual(len(result['per_sequence_nll']),2)


if __name__=='__main__':unittest.main()
