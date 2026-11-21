import unittest
import torch
from efficient_llm.warm_order.model import Model,Config
from efficient_llm.warm_order.adaptation_analysis import CANONICAL,SWAPPED
from efficient_llm.warm_order.path_background import boundaries,suffix,score,anchors,fit,transform,random_basis,transfer,overlap

class PathBackground(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(2111)
        self.m=Model(Config(d=8,heads=2,kv_heads=1,head_dim=4,ff=16,vocab=23,prefix=2,length=8)).eval().requires_grad_(False)
        self.x=torch.randint(0,23,(3,8))

    def test_exact_boundary_native_nll_causality_and_immutability(self):
        m,x=self.m,self.x;state={k:v.clone() for k,v in m.state_dict().items()};rng=torch.get_rng_state().clone()
        a,b=boundaries(m,x)
        for h,path in ((a,CANONICAL),(b,SWAPPED)):
            torch.testing.assert_close(suffix(m,x,h),m.hidden(x,path),rtol=0,atol=0)
            sums,counts=score(m,x,h);torch.testing.assert_close((sums/counts).float(),m.nll(x,path),atol=1e-6,rtol=1e-6)
        other=x.clone();other[:,5:]=(other[:,5:]+1)%23
        aa,bb=boundaries(m,other)
        torch.testing.assert_close(a[:,:5],aa[:,:5],atol=0,rtol=0);torch.testing.assert_close(b[:,:5],bb[:,:5],atol=0,rtol=0)
        for k,v in m.state_dict().items():torch.testing.assert_close(v,state[k],atol=0,rtol=0)
        torch.testing.assert_close(torch.get_rng_state(),rng,atol=0,rtol=0)

    def test_known_lowrank_and_unseen_transfer_not_automatic_bias(self):
        g=torch.Generator().manual_seed(4);d=torch.zeros(50,8,dtype=torch.float64);d[:,1]=torch.randn(50,generator=g);d[:,2]=2*torch.randn(50,generator=g);d[:,0]=3
        f=fit(d,2);self.assertEqual(f['actual_rank'],2)
        torch.testing.assert_close(f['basis']@f['basis'].T,torch.diag(torch.tensor([0.,1,1,0,0,0,0,0],dtype=torch.float64)),atol=1e-10,rtol=1e-10)
        held=d[:10].clone();held[:,3]=7
        self.assertGreater(transfer(held,f)['after_mean_correction_mean_square'],0)
        self.assertAlmostEqual(overlap(f,f),1.)
        z=fit(torch.ones(5,8),2);self.assertEqual(z['actual_rank'],0);self.assertIsNone(overlap(z,z))
        with self.assertRaises(ValueError):fit(d,8)

    def test_projection_controls_and_mean_are_separate(self):
        f=fit(torch.randn(30,8),2);h=torch.randn(3,8,8);r=random_basis(8,2,91)
        q=transform(h,f,'project');delta=h-q
        torch.testing.assert_close(q@f['basis'].float(),torch.zeros(3,8,2),atol=1e-6,rtol=1e-6)
        matched=transform(h,f,'random_energy_matched',r)
        torch.testing.assert_close((h-matched).norm(dim=-1),delta.norm(dim=-1),atol=1e-6,rtol=1e-6)
        normed=transform(h,f,'project_norm_restored');torch.testing.assert_close(normed.norm(dim=-1),h.norm(dim=-1),atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(transform(transform(h,f,'minus_mean'),f,'plus_mean'),h,atol=1e-6,rtol=1e-6)
        bad=dict(f,basis=f['basis']*2)
        with self.assertRaises(ValueError):transform(h,bad,'project')

    def test_anchor_families_rng_and_nonmutation(self):
        x=self.x.clone();rng=torch.get_rng_state().clone()
        r=anchors(x,'repeat',4);p=anchors(x,'periodic',4);s=anchors(x,'shuffle',4)
        self.assertTrue((r==x[:,:1]).all());self.assertTrue((p[:,::2]==x[:,:1]).all())
        torch.testing.assert_close(s.sort(1).values,x.sort(1).values,atol=0,rtol=0)
        torch.testing.assert_close(s,anchors(x,'shuffle',4),atol=0,rtol=0)
        torch.testing.assert_close(x,self.x,atol=0,rtol=0);torch.testing.assert_close(torch.get_rng_state(),rng,atol=0,rtol=0)

if __name__=='__main__':unittest.main()
