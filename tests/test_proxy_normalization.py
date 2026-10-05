"""Revision-7 T9: shifted FP32 statistics, clipping, freeze and target versioning."""
import copy
import io
import unittest

import torch

from tests.test_proxy_heads import model, batch
from train import resume_performance_changes


class ProxyNormalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_t9_stationary_fp64_reference_and_heldout_distribution(self):
        m=model('P1-block')
        generator=torch.Generator().manual_seed(701)
        means=torch.linspace(2.,5.,32);means[0]=1000.
        scales=torch.linspace(.8,1.2,32)
        def sample(n):return torch.randn(n,32,generator=generator)*scales+means
        def moments(x,mean):
            shifted=x-mean
            return shifted.sum(0).expand_as(m.mu).clone(),shifted.square().sum(0).expand_as(m.mu).clone(),torch.full((len(m.mu),),len(x))
        # Same synthetic first batch, two passes. Never E[x^2]-E[x]^2.
        x=sample(4096)
        sums,squares,counts=moments(x,torch.zeros(32))
        m.update_statistics(sums,squares,counts,initialize='mean')
        sums,squares,counts=moments(x,m.mu[0])
        m.update_statistics(sums,squares,counts,initialize='variance')
        reference_mu=x.double().mean(0)
        reference_variance=(x.double()-reference_mu).square().mean(0)
        max_mean_error=max_variance_error=0.
        for step in range(1000):
            x=sample(4096)
            sums,squares,counts=moments(x,m.mu[0])
            m.update_statistics(sums,squares,counts)
            shifted=x.double()-reference_mu
            delta=shifted.mean(0)
            variance=(shifted.square().mean(0)-delta.square()).clamp_min(0)
            reference_mu=reference_mu+.01*delta
            reference_variance=.99*reference_variance+.01*variance
            if step<500:
                max_mean_error=max(max_mean_error,float(((m.mu[0]-reference_mu).abs()/reference_mu.abs()).max()))
                max_variance_error=max(max_variance_error,float(((m.sigma2[0]-reference_variance).abs()/reference_variance).max()))
        self.assertLessEqual(max_mean_error,1e-3)
        self.assertLessEqual(max_variance_error,1e-3)
        fresh=sample(100000)
        normalized=m.normalize_target(fresh,m.mean_layers[0])
        self.assertLessEqual(normalized.mean(0).abs().max().item(),.05)
        variance=normalized.var(0,unbiased=False)
        self.assertTrue(bool(((variance>=.9)&(variance<=1.1)).all()))
        self.assertLess(normalized.abs().max().item(),10.)
        print('T9 FP64 relative errors',max_mean_error,max_variance_error,
              'held mean',normalized.mean(0).abs().max().item(),'variance range',variance.min().item(),variance.max().item())

    def test_step_variance_pools_microbatches_but_not_history(self):
        m=model('P1-block');m.mu.fill_(1000.);m.sigma2.fill_(3.)
        # Unequal microbatch sizes and different means: averaging their variances is wrong.
        chunks=[torch.tensor([997.,999.]),torch.tensor([1001.,1003.,1005.])]
        raw=torch.cat(chunks).double()
        sums=sum((x-1000.).sum() for x in chunks).expand_as(m.mu)
        squares=sum((x-1000.).square().sum() for x in chunks).expand_as(m.mu)
        counts=torch.full((len(m.mu),),len(raw))
        m.update_statistics(sums,squares,counts)
        expected_mu=.99*1000.+.01*raw.mean()
        expected_variance=.99*3.+.01*raw.var(unbiased=False)
        torch.testing.assert_close(m.mu,torch.full_like(m.mu,expected_mu),rtol=0,atol=1e-4)
        torch.testing.assert_close(m.sigma2,torch.full_like(m.sigma2,expected_variance),rtol=0,atol=1e-6)
        # A subsequent constant-valued step changes mean but contributes zero variance.
        before=m.sigma2.clone()
        delta=1010.-m.mu
        m.update_statistics(5*delta,5*delta.square(),counts)
        torch.testing.assert_close(m.sigma2,.99*before,rtol=0,atol=1e-6)

    def test_floor_clip_detachment_and_prediction_gradient(self):
        m=model('P1-block');key=m.mean_layers[0]
        m.sigma2[0,0]=0
        raw=torch.ones(1,1,32,requires_grad=True);raw.data[0,0,0]=.2;raw.data[0,0,1]=100
        target,clipped=m.normalize_target(raw,key,return_clipped=True)
        self.assertEqual(clipped.item(),1);self.assertEqual(target[0,0,1].item(),10)
        self.assertAlmostEqual(target[0,0,0].item(),.2/(.01+1e-6)**.5,places=5)
        self.assertFalse(target.requires_grad)
        pred=torch.arange(32,dtype=torch.float32).reshape_as(raw).requires_grad_()
        clipped_loss=1-torch.nn.functional.cosine_similarity(pred,target,dim=-1)
        unbounded=target.clone();unbounded[:,:,1]=100/(1+1e-6)**.5
        unbounded_loss=1-torch.nn.functional.cosine_similarity(pred,unbounded,dim=-1)
        self.assertFalse(torch.equal(torch.autograd.grad(clipped_loss,pred)[0],torch.autograd.grad(unbounded_loss,pred)[0]))
        self.assertIsNone(raw.grad)
        m.sigma2.zero_()
        diagnostic=m.update_statistics(torch.ones_like(m.mu),2*torch.ones_like(m.mu),torch.ones(len(m.mu)))
        self.assertTrue(all(torch.isfinite(value).all() for value in diagnostic.values()))

    def test_frozen_statistics_and_exact_next_targets_after_save(self):
        for arm in ('P1-flow','P3-block','P3-lambda0'):
            m=model(arm,checkpoint=True);ctx=batch()
            with torch.no_grad():
                for phase in ('mean','variance'):
                    out=m(ctx,compute_auxiliary_losses=False,collect_target_statistics=True,statistics_mode=phase)
                    m.update_statistics(out['center_sums'],out['center_squares'],out['center_counts'],initialize=phase)
            buffers={k:v.clone() for k,v in m.named_buffers()}
            # Multiple forwards and checkpoint backward cannot update running buffers.
            first=m(ctx,collect_target_statistics=True)
            for _ in range(2):
                out=m(ctx,collect_target_statistics=True)
                (out['lm_sum']+.1*out['aux_sum']).backward();m.zero_grad(set_to_none=True)
                torch.testing.assert_close(out['center_sums'],first['center_sums'],rtol=0,atol=0)
                torch.testing.assert_close(out['center_squares'],first['center_squares'],rtol=0,atol=0)
            m.eval()
            with torch.no_grad():m(ctx)
            for key,value in m.named_buffers():torch.testing.assert_close(value,buffers[key],rtol=0,atol=0)
            m.update_statistics(first['center_sums'],first['center_squares'],first['center_counts'])
            saved=io.BytesIO();torch.save(m.state_dict(),saved);saved.seek(0)
            resumed=model(arm,checkpoint=True).eval();resumed.load_state_dict(torch.load(saved,weights_only=True))
            with torch.no_grad():
                a=m(ctx,collect_target_statistics=True);b=resumed(ctx,collect_target_statistics=True)
                for field in ('center_sums','center_squares','statistics','aux_sum'):
                    torch.testing.assert_close(a[field],b[field],rtol=0,atol=0)
                for current,out in ((m,a),(resumed,b)):
                    current.update_statistics(out['center_sums'],out['center_squares'],out['center_counts'])
                for key,value in m.state_dict().items():torch.testing.assert_close(value,resumed.state_dict()[key],rtol=0,atol=0)

    def test_bf16_forward_preserves_fp32_target_statistics(self):
        for arm in ('P1-flow','P3-block'):
            m=model(arm)
            with torch.autocast('cpu',dtype=torch.bfloat16):
                out=m(batch(),collect_target_statistics=True)
            for field in ('center_sums','center_squares','center_counts','aux_sum'):
                self.assertEqual(out[field].dtype,torch.float32,field)
            self.assertEqual(m.mu.dtype,torch.float32);self.assertEqual(m.sigma2.dtype,torch.float32)
            self.assertFalse(out['center_sums'].requires_grad)
            self.assertFalse(out['center_squares'].requires_grad)

    def test_smooth_l1_optional_and_no_legacy_resume(self):
        for arm in ('P1-block','P3-flow'):
            m=model(arm,loss_form='smooth_l1')
            out=m(batch());self.assertTrue(torch.isfinite(out['aux_sum']))
            out['aux_sum'].backward()
            self.assertGreater(m.heads['2'].w1.weight.grad.abs().sum().item(),0)
        for key,value in [('proxy_target_version','r6'),('proxy_target_clip',9.),('proxy_momentum',.9),('proxy_variance_floor',.02)]:
            requested=dict(pilot=dict(proxy_target_version='r7',proxy_target_clip=10.,proxy_momentum=.99,proxy_variance_floor=.01),training={})
            previous=copy.deepcopy(requested);previous['pilot'][key]=value
            with self.assertRaises(ValueError):resume_performance_changes(previous,requested,allow=True)
        with self.assertRaisesRegex(ValueError,'r7'):model('P1-block',target_version='r6')


if __name__=='__main__':unittest.main()
