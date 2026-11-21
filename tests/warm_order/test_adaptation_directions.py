import copy
import unittest

import torch

from efficient_llm.warm_order.adaptation_directions import (
    DeltaMoments, update_cases, vo_boundary, finish_update, CONTROL_SEEDS)
from efficient_llm.warm_order.adaptation_factors import factors_crossovers
from efficient_llm.warm_order.model import Model, Config


class Directions(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(419)
        self.p = Model(Config(d=16, heads=2, kv_heads=1, head_dim=8,
                              ff=24, vocab=31, prefix=3, length=12)).eval().requires_grad_(False)
        self.a = copy.deepcopy(self.p)
        with torch.no_grad():
            for v in self.a.parameters():
                v.add_(torch.randn_like(v)*.003)
        self.x = torch.randint(0,31,(2,12))

    def test_streamed_basis_matches_explicit_svd_and_replay(self):
        d = torch.randn(47,16,dtype=torch.float64)*torch.arange(1,17)+3
        m = DeltaMoments(16)
        for x in d.split(7):m.update(x)
        f = m.fit()
        _,s,v = torch.linalg.svd(d-d.mean(0),full_matrices=False)
        torch.testing.assert_close(f['mean'],d.mean(0))
        torch.testing.assert_close(f['basis']@f['basis'].T,v[:8].T@v[:8])
        torch.testing.assert_close(f['eigenvalues'],s.square())
        replay=DeltaMoments.from_arrays(m.arrays()).fit()
        torch.testing.assert_close(f['basis'],replay['basis'],atol=0,rtol=0)

    def test_native_vo_graft_and_old_factor_endpoints(self):
        b = vo_boundary(self.p,self.a,self.x)
        old = factors_crossovers(self.p,self.a,self.x)['cells']
        graft = copy.deepcopy(self.p)
        graft.middle[2].self_attn.v_proj.load_state_dict(self.a.middle[2].self_attn.v_proj.state_dict())
        graft.middle[2].self_attn.o_proj.load_state_dict(self.a.middle[2].self_attn.o_proj.state_dict())
        for value,ref in [(b['base'],old['attn/000']), (b['donor'],old['attn/001'])]:
            torch.testing.assert_close(finish_update(self.p,b,value),ref,atol=2e-6,rtol=1e-5)
        torch.testing.assert_close(finish_update(self.p,b,b['donor']),graft.hidden(self.x,(0,2,1,3)),atol=2e-6,rtol=1e-5)

    def test_projector_algebra_and_matched_controls(self):
        m=DeltaMoments(16);m.update(torch.randn(100,16));f=m.fit()
        u=torch.randn(2,12,16);v=u+torch.randn_like(u)*.1
        rng=torch.get_rng_state().clone();c=update_cases(u,v,f)
        self.assertEqual(len(c),16)
        torch.testing.assert_close(c['projected']+c['complement']-u,v)
        torch.testing.assert_close(c['affine'],c['mean']+c['centered']-u)
        torch.testing.assert_close(c['norm_restored'].norm(dim=-1),u.norm(dim=-1))
        torch.testing.assert_close(c['radial'].norm(dim=-1),c['projected'].norm(dim=-1))
        for seed in CONTROL_SEEDS:
            torch.testing.assert_close((c[f'random_energy/{seed}']-u).norm(dim=-1),(c['projected']-u).norm(dim=-1),atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(rng,torch.get_rng_state(),atol=0,rtol=0)

    def test_frozen_basis_causality_and_no_mutation(self):
        calibration=vo_boundary(self.p,self.a,torch.randint(0,31,(3,12)))
        m=DeltaMoments(16);m.update((calibration['donor']-calibration['base'])[:,:-1].reshape(-1,16).cpu());f=m.fit()
        states=[{k:v.clone() for k,v in model.state_dict().items()} for model in (self.p,self.a)]
        rng=torch.get_rng_state().clone()
        x2=self.x.clone();x2[:,7:]=(x2[:,7:]+1)%31
        b1=vo_boundary(self.p,self.a,self.x);b2=vo_boundary(self.p,self.a,x2)
        c1=update_cases(b1['base'],b1['donor'],f);c2=update_cases(b2['base'],b2['donor'],f)
        for k in c1:
            torch.testing.assert_close(finish_update(self.p,b1,c1[k])[:,:7],finish_update(self.p,b2,c2[k])[:,:7],atol=0,rtol=0)
        for model,state in zip((self.p,self.a),states):
            for k,v in model.state_dict().items():torch.testing.assert_close(v,state[k],atol=0,rtol=0)
        torch.testing.assert_close(rng,torch.get_rng_state(),atol=0,rtol=0)

    def test_self_patch_and_constant_delta_degeneracy(self):
        b=vo_boundary(self.p,self.p,self.x)
        torch.testing.assert_close(b['base'],b['donor'],atol=0,rtol=0)
        m=DeltaMoments(16);m.update(torch.zeros(20,16));f=m.fit();self.assertEqual(f['actual_rank'],0)
        for value in update_cases(b['base'],b['donor'],f).values():
            torch.testing.assert_close(value,b['base'],atol=0,rtol=0)
        m=DeltaMoments(16);m.update(torch.ones(20,16));f=m.fit();self.assertEqual(f['actual_rank'],0)
        u=torch.ones(1,2,16);c=update_cases(u,u+1,f)
        torch.testing.assert_close(c['projected'],u)
        torch.testing.assert_close(c['mean'],u+1)

    def test_reject_invalid_basis_and_degenerate_norm_match(self):
        m=DeltaMoments(16)
        with self.assertRaises(ValueError):m.fit()
        m.update(torch.randn(20,16));f=m.fit()
        with self.assertRaisesRegex(ValueError,'Degenerate'):
            update_cases(torch.zeros(1,2,16),torch.ones(1,2,16),f)
        bad=dict(f,basis=f['basis']*2)
        with self.assertRaisesRegex(ValueError,'Orthonormal'):
            update_cases(torch.ones(1,2,16),torch.ones(1,2,16)*2,bad)


if __name__=='__main__':unittest.main()
