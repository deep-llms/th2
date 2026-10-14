"""Combined target/LM gradients: independent graph, Trainer and DDP acceptance."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from deep_kv import followup_metadata
from deep_kv.proxy import compute_budget
from tests.test_proxy_screen24 import model, config, fixture
from tests.test_proxy_heads import batch, objective

ARM = 'P7-simple-targetgrad-lm'


class CombinedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_independent_combined_graph_and_forward_identity(self):
        actual = model(ARM)
        ref = model('P7-simple-lm')
        parent = model('P7-simple-targetgrad')
        for name, value in ref.state_dict().items():
            torch.testing.assert_close(actual.state_dict()[name], value, rtol=0, atol=0)
        self.assertEqual(followup_metadata(ARM), dict(auxiliary_routes=dict(shallow=False,target=True),lm_to_predictor=True))
        self.assertEqual(compute_budget(config(),actual.settings,8)[ARM],compute_budget(config(),actual.settings,8)['P7-simple'])
        ctx=batch(); mlps={}; predictions={}; handles=[]; sources=[]
        for i, block in enumerate(ref.backbone.model.layers,1):
            handles.append(block.mlp.register_forward_hook(lambda m,a,o,i=i:mlps.__setitem__(i,o)))
            if i in ref.layers:
                handles.append(ref.heads[str(i)].w2.register_forward_hook(lambda m,a,o,i=i:predictions.__setitem__(i,o)))
        for head in actual.heads.values():
            handles.append(head.w1.register_forward_pre_hook(lambda m,a:sources.append(a[0].requires_grad)))
        out=actual(ctx,collect_target_statistics=True)
        expected=ref(ctx,compute_auxiliary_losses=False)
        torch.testing.assert_close(out['lm_sum'],expected['lm_sum'],rtol=0,atol=0)
        torch.testing.assert_close(out['lm_sum'],parent(ctx)['lm_sum'],rtol=0,atol=0)
        terms=[]
        for layer in ref.layers:
            raw=sum(mlps[i].float() for i in range(layer,layer+4))
            idx=ref.mean_index[layer];v=ref.sigma2[idx]
            target=((raw-ref.mu[idx])*torch.rsqrt(v.clamp_min(.01*v.median())+1e-6)).clamp(-10,10)
            terms.append(((1-F.cosine_similarity(predictions[layer].float(),target,dim=-1,eps=1e-6))*ctx.valid).sum())
        aux=torch.stack(terms).mean()
        torch.testing.assert_close(out['aux_sum'],aux,rtol=1e-6,atol=1e-6)
        # The independent teacher graph retains target -> memory -> predictor
        # paths as well as target -> backbone; no part of the target is detached.
        objective(actual,out).backward()
        (expected['lm_sum']/out['lm_count']+.1*aux/out['aux_count']).backward()
        self.assertTrue(sources);self.assertFalse(any(sources))
        self.assertFalse(out['center_sums'].requires_grad)
        for (name,p),(_,q) in zip(actual.named_parameters(),ref.named_parameters()):
            self.assertEqual(p.grad is None,q.grad is None,name)
            if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=4e-5,atol=3e-6,msg=name)
        for h in handles:h.remove()

    def test_separate_losses_and_backend_checkpoint_equivalence(self):
        from tests.test_fa4_baseline import reference_kernel
        for loss in ('lm_sum','aux_sum'):
            m=model(ARM);m(batch())[loss].backward()
            self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()) for p in m.backbone.parameters()))
            self.assertTrue(all(h.w1.weight.grad is not None and bool(h.w1.weight.grad.abs().sum()) for h in m.heads.values()))
        for bf16 in (False,True):
            dense=model(ARM);checked=model(ARM,checkpoint=True)
            with patch('deep_kv.fa4.load_kernel',return_value=(reference_kernel,{'version':'CPU-reference'})):
                flash=model(ARM,backend='fa4')
            for m in (dense,checked,flash):
                with torch.autocast('cpu',dtype=torch.bfloat16,enabled=bf16):objective(m,m(batch())).backward()
            for other in (checked,flash):
                for (name,p),(_,q) in zip(dense.named_parameters(),other.named_parameters()):
                    self.assertEqual(p.grad is None,q.grad is None,name)
                    if p.grad is not None and (not bf16 or other is checked):
                        torch.testing.assert_close(p.grad,q.grad,rtol=4e-4 if not bf16 else 0,atol=3e-6 if not bf16 else 0,msg=name)
                if bf16 and other is flash:
                    # The CPU FA4 oracle and native SDPA round intermediate BF16
                    # products differently. Use the existing CUDA acceptance's
                    # 3% gradient relative-L2 bound, separately for each branch,
                    # instead of relative per-element errors near zero.
                    for group in ('backbone.', 'predictor', 'projections'):
                        def select(m):
                            return torch.cat([p.grad.flatten() for n,p in m.named_parameters() if p.grad is not None and
                                (n.startswith(group) if group=='backbone.' else n.startswith('heads.') and
                                 (('.w1.' in n or '.w2.' in n) == (group=='predictor')))])
                        a,b=select(dense),select(other)
                        self.assertLess(float((a-b).norm()/a.norm().clamp_min(1e-8)),.03,group)

    def test_trainer_resume_reload_and_report(self):
        from tests.test_train import invoke
        from safetensors.torch import load_file
        from eval.models import load_checkpoint
        from deep_kv.report import report
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);recipe=fixture(root)
            args={**recipe,'arm':ARM,'output_dir':str(root/ARM)}
            invoke(root,{**args,'stop_after':1});invoke(root,args)
            invoke(root,{**args,'output_dir':str(root/'full')})
            resumed=load_file(root/ARM/'model.safetensors');full=load_file(root/'full/model.safetensors')
            for name in resumed:torch.testing.assert_close(resumed[name],full[name],rtol=0,atol=0,msg=name)
            loaded,_,_=load_checkpoint(root/ARM,device='cpu',attention_backend='sdpa')
            self.assertTrue(loaded.wrapped.aux_target_grad)
            self.assertTrue(all(h.lm_predictor for h in loaded.wrapped.heads.values()))
            report(root,[ARM])

    def test_two_rank_real_trainer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);recipe=fixture(root,world=2)
            recipe.update(arm=ARM,ddp_backend='gloo',ddp_find_unused_parameters=False,logging_steps=2,eval_on_start=False)
            path=root/'recipe.json';path.write_text(json.dumps(recipe));log=root/'ddp.log'
            with log.open('w') as out:
                p=subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
                    '--module','tests.test_proxy_screen24','--ddp-worker',str(path)],
                    cwd=Path(__file__).resolve().parents[1],env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1'},
                    stdout=out,stderr=subprocess.STDOUT,timeout=180)
            self.assertEqual(p.returncode,0,log.read_text()[-15000:])
            self.assertEqual(json.loads((Path(recipe['output_dir'])/'result.json').read_text())['global_step'],3)


if __name__=='__main__':unittest.main()
