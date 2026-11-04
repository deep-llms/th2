import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch
from torch import nn
from efficient_llm.a2_why.protocol import base_runs,noise,merge_reading,capacity_reading,extend_capacity,spearman
from efficient_llm.a2_why.merge import merge_and_prune
from scripts.a2_why_streams import prefix_copy
from scripts.a2_stage1_data import INDEX
import numpy as np
import tempfile
from pathlib import Path

class Checks:
    def record(self,name,passed,detail=None):
        if not passed:raise AssertionError(name)

class WhyTests(unittest.TestCase):
    def test_matrix_and_boundaries(self):
        runs=base_runs();self.assertEqual(len(runs),41);self.assertEqual(len({r['run_id'] for r in runs}),41)
        self.assertEqual([sum(r['phase']==s for r in runs) for s in ('YT1','Y1','Y2','Y3')],[1,8,8,24])
        self.assertTrue(all(22 not in r['removed'] and 23 not in r['removed'] for r in runs))
        self.assertFalse(extend_capacity(1,1.2));self.assertTrue(extend_capacity(1,1.201))
        self.assertEqual(merge_reading(dict(dclm=.01,fwe=.01),dict(dclm=.02,fwe=.02),.01),'B supported')
        self.assertEqual(merge_reading(dict(dclm=.02,fwe=.02),dict(dclm=.02,fwe=.02),.003),'no effect')
        increasing={d:{3:1.,6:2.,9:3.,12:4.} for d in ('dclm','fwe')}
        self.assertEqual(capacity_reading(increasing,.003),'A supported');self.assertEqual(spearman([1,2,3,4]),1)
        result=noise(dict(dclm=[1.,2.,3.],fwe=[1.,2.,3.]));self.assertEqual(result['sigma'],1);self.assertEqual(result['tau'],2)

    def test_merges_preserve_norms_and_unrelated_layers(self):
        for variant in ('rdsc','avg'):
            layers=nn.ModuleList()
            for i in range(24):
                l=nn.Module();l.self_attn=nn.Module();l.self_attn.q_proj=nn.Linear(2,2)
                l.self_attn.q_norm=nn.LayerNorm(2);l.mlp=nn.Linear(2,3);l.input_layernorm=nn.LayerNorm(2)
                for j,p in enumerate(l.parameters()):p.data.fill_(float(i+j/10))
                layers.append(l)
            model=SimpleNamespace(model=SimpleNamespace(layers=layers))
            original=[{k:v.clone() for k,v in l.named_parameters()} for l in layers]
            with patch('efficient_llm.a2_why.merge.prune',return_value=list(range(16))+[22,23]):merge_and_prune(model,variant,Checks())
            for i,l in enumerate(layers):
                for name,p in l.named_parameters():
                    if i in (15,22) and 'norm' not in name:
                        group=[15,16,17,18] if i==15 else [19,20,21,22]
                        expected=sum(original[k][name].double() for k in group)/4 if variant=='avg' else sum(original[k][name].double() for k in group)-3*original[i][name].double()
                        self.assertTrue(torch.allclose(p.double(),expected,atol=1e-5,rtol=0))
                    else:self.assertTrue(torch.equal(p,original[i][name]))

    def test_prefix_matches_full_stream(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);src=root/'src';src.mkdir();out=root/'out';out.mkdir()
            np.array([11,0,21,22,0,31,0],dtype='<u4').tofile(src/'heal.bin')
            np.array([1,0,1,0,0,1,0],dtype='u1').tofile(src/'heal.starts')
            docs=np.zeros(3,dtype=INDEX);docs['offset']=[0,2,5];docs['tokens']=[2,3,2]
            prefix_copy(docs,np.array([2,0,1]),src,out,6)
            self.assertEqual(np.fromfile(out/'heal.bin',dtype='<u4').tolist(),[31,0,11,0,21,22])
            self.assertEqual(np.fromfile(out/'heal.starts',dtype='u1').tolist(),[1,0,1,0,1,0])

if __name__=='__main__':unittest.main()
