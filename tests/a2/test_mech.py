import copy
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from test_model import tiny_model
from efficient_llm.a2.mech import weight_statistics,perturb,noise_seed,profile_hooks,token_score,token_buckets
from efficient_llm.a2.measure import score
from efficient_llm.a2.model import bypass
from scripts.a2_mech_report import relative,patterns

class MechTests(unittest.TestCase):
    def test_weight_stats_saved_error(self):
        w=torch.arange(256,dtype=torch.float32).reshape(2,128)-127
        out=weight_statistics(w,{'rtn3_weight_error':w.clone(),'rtn4_weight_error':w+1})
        self.assertEqual(out['rtn3_weight_error'],0)
        self.assertAlmostEqual(out['rtn4_weight_error'],float(256/w.double().square().sum()))
        self.assertEqual(out['weights'],256)
    def test_noise_groups_and_seeds(self):
        torch.manual_seed(5);w=torch.randn(1024,256).bfloat16().float()
        zero,_=perturb(w,'iso',0,0,0.);self.assertTrue(torch.equal(zero,w))
        a,c=perturb(w,'rtn3',0,0);b,_=perturb(w,'rtn3',0,0);other,_=perturb(w,'rtn3',1,0)
        self.assertTrue(c['passed']);self.assertTrue(torch.equal(a,b));self.assertFalse(torch.equal(a,other))
        self.assertEqual(len({noise_seed(s,i) for s in (0,1) for i in range(168)}),336)
    def test_identity_profile(self):
        m=tiny_model(num_hidden_layers=24).eval();ids=torch.tensor([[1,2,3,4]])
        with torch.no_grad(),bypass(m,list(range(16,22))),profile_hooks(m) as (v,b):m.model(ids,use_cache=False)
        self.assertEqual(float(v[16:22,:2].abs().max()),0)
        self.assertEqual(float(b[:2].abs().max()),0)
    def test_physical_single_removal_equals_hook(self):
        m=tiny_model(num_hidden_layers=24).eval();ids=torch.tensor([[1,2,3,4]])
        original=m.model.layers
        with torch.no_grad():
            for i in (0,11,23):
                with bypass(m,[i]):expected=m(ids).logits
                try:
                    m.model.layers=torch.nn.ModuleList([v for j,v in enumerate(original) if j!=i])
                    actual=m(ids).logits
                finally:m.model.layers=original
                torch.testing.assert_close(actual,expected,rtol=0,atol=1e-4)

    def test_token_storage_matches_original_score(self):
        m=tiny_model().eval();rows=[{'input_ids':torch.tensor([1,2,3,4,5])},{'input_ids':torch.tensor([2,4,6,8,10])}]
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'x.npy';a=token_score(m,rows,path);b=score(m,rows,distributed=False)
            self.assertEqual(a['per_sequence_nll'],b['per_sequence_nll']);self.assertEqual(a['storage_replay_max_abs'],0)
    def test_buckets_and_gap_signs(self):
        target=np.arange(100).reshape(10,10)%7
        v={n:np.ones((10,10),dtype=np.float32)*z for n,z in [('P300',3),('P1T',2),('S72',4),('S188',2),('pre-P300',4),('pre-P1T',4),('student-P300',3),('student-P1T',2.5),('student-X72',3.2)]}
        a=token_buckets(v,target)
        for b in a['buckets']['frequency']:
            self.assertEqual(b['count'],10);self.assertEqual(b['retention'],.5);self.assertEqual(b['pre_extra'],1)
        self.assertEqual(a,token_buckets(v,target))
    def test_frequency_counts_include_unscored_first_tokens(self):
        target=np.tile(np.arange(10),(10,1))
        names=('P300','P1T','S72','S188','pre-P300','pre-P1T','student-P300','student-P1T','student-X72')
        values={n:np.ones_like(target,dtype=np.float32)*(3 if n=='P300' else 2) for n in names}
        counts=np.bincount(target.ravel());counts[0]+=100
        out=token_buckets(values,target,frequency_counts=counts)
        self.assertEqual(out['buckets']['frequency'][-1]['key_min'],110)
        self.assertEqual(out['buckets']['frequency'][0]['key_min'],10)

    def test_pattern_all_datasets_required(self):
        vals={d:dict(S72=1,P300=1.1,S188=1.5,P1T=1.8) for d in ('dclm','fwe','nem')}
        self.assertTrue(patterns(relative(vals))['pruning']);self.assertFalse(patterns(relative(vals))['ptq'])
        del vals['nem'];self.assertIsNone(patterns(relative(vals))['pruning'])

    def test_full_readout_and_replay(self):
        import json
        from unittest.mock import patch
        from scripts.a2_mech_report import summarize
        from scripts.a2_quant_io import write,lock
        from efficient_llm.a2.quant_stats import PARENTS
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);q={'datasets':{}}
            for ds in ('dclm','fwe','nem'):
                q['datasets'][ds]={c:{'damage':{p:{'damage':.1+i*.01} for i,p in enumerate(PARENTS)}} for c in ('pre-S1','rtn3','gptq3')}
                for i,p in enumerate(PARENTS):
                    for phase in ('weights','layers','noise'):
                        folder=root/'workers'/phase/ds/p
                        r={'matrices':[{}]*168,'means':{'rtn3_weight_error':.1+i*.02}} if phase=='weights' else {'BI_S1':.1+i*.02,'S1_sum':.2+i*.02,'rho':2+i*.1,'single_damage':[.01]*24} if phase=='layers' else {}
                        write(folder/'result.json',{'results':r,'checks':[]})
                        if phase=='noise':
                            write(folder/'parent-final.json',{'per_sequence_nll':[2.]*10})
                            for name in ('N-iso-0.1','N-iso-0.2','N-iso-0.3','N-rtn3'):
                                for seed in (0,1):write(folder/(name+'-seed'+str(seed)+'.json'),{'per_sequence_nll':[2.1+i*.01+seed*.001]*10})
                    if ds!='nem':
                        folder=root/'workers/tokens'/ds/p;write(folder/'result.json',{'results':{},'checks':[]})
                        names=[p]+(['pre-'+p,'student-'+p] if p in ('P300','P1T') else ['student-X72'] if p=='S72' else [])
                        for name in names:
                            value={'P300':3.,'P1T':2.,'student-P300':3.,'student-P1T':2.5}.get(name,3.5)
                            np.save(folder/(name+'.npy'),np.full((10,10),value,dtype=np.float32))
            cfg={'spec_sha256':'0'*64,'quant_decision':write(root/'q.json',q),'data_root':'unused'}
            class Rows:
                ids=np.tile(np.arange(11),(10,1))
            with patch('efficient_llm.a2.data.PackedRows',return_value=Rows()):
                first=summarize(cfg,root);second=summarize(cfg,root)
            self.assertEqual(first,second);self.assertTrue(first['coverage_complete'])
            self.assertEqual(len(first['H_E_readout']),2)

if __name__=='__main__':unittest.main()
