"""Matched router semantics and real native-Trainer integration, offline CPU."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',HF_HUB_OFFLINE='1',
                  HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from efficient_llm.routers import RouterPair,RouterSelection,TeacherFeatures,subset_costs,save_routers,load_routers
from efficient_llm.group_selection import subset_masks,contributions,apply_mask
from efficient_llm.router_pilot import FeatureDataset
from efficient_llm.router_check import cached_check
from efficient_llm.decode_profile import cache_check, SDPATimer
from efficient_llm.diagnostic_data import sha256
torch.set_num_threads(1)


def tiny_model():
    torch.manual_seed(19)
    return Qwen3ForCausalLM._from_config(Qwen3Config(vocab_size=47,hidden_size=32,
        intermediate_size=48,num_hidden_layers=2,num_attention_heads=8,num_key_value_heads=4,
        head_dim=8,max_position_embeddings=64,attention_dropout=0.,attention_bias=True),
        attn_implementation='sdpa').eval().requires_grad_(False)


class RouterTests(unittest.TestCase):
    def test_static_cache_and_profile_wrapper_preserve_output(self):
        self.assertLess(cache_check(tiny_model()), 2e-3)
        original = torch.nn.functional.scaled_dot_product_attention
        q = torch.randn(1,2,4,8)
        expected = original(q,q,q,is_causal=True)
        class Event:
            def __init__(self, **kwargs): pass
            def record(self): pass
            def elapsed_time(self, end): return 1.
        with patch('torch.cuda.Event', Event), patch('torch.cuda.synchronize'):
            with self.assertRaisesRegex(RuntimeError, 'deliberate'):
                with SDPATimer() as timer:
                    actual = torch.nn.functional.scaled_dot_product_attention(q,q,q,is_causal=True)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertEqual(timer.milliseconds(), 1.)
                    raise RuntimeError('deliberate')
        self.assertIs(torch.nn.functional.scaled_dot_product_attention, original)

    def test_costs_and_independent_target(self):
        c=torch.tensor([[[10.,0.],[-10.,0.],[0.,1.],[0.,1.]]])
        masks=subset_masks(4,2); costs=subset_costs(c,masks)
        self.assertEqual(masks[costs[0,0].argmin()].tolist(),[False,False,True,True])
        self.assertEqual(masks[costs[0,1].argmin()].tolist(),[True,True,False,False])
        for i,mask in enumerate(masks):
            torch.testing.assert_close(costs[:,1,i],(c[:,~mask].square().sum(-1)).sum(-1))

    def test_teacher_alignment_identity_and_frozen_gradients(self):
        m=tiny_model();ids=torch.tensor([[1,3,5,7,9,11,13,15]])
        before={k:v.clone() for k,v in m.state_dict().items()}
        seen={}
        h=m.model.layers[0].input_layernorm.register_forward_hook(lambda mod,args,out:seen.update(x=out.detach().clone()))
        with torch.no_grad():
            expected=m(ids).logits
            with TeacherFeatures(m,3,stride=2) as teacher:
                actual=m(ids).logits;x,costs=teacher.take()
            torch.testing.assert_close(expected,actual,rtol=0,atol=0)
            torch.testing.assert_close(x[:,0],seen['x'][:,1::2].reshape(-1,32),rtol=0,atol=0)
        h.remove()
        r=RouterPair(2,32,4,3,torch.ones(2))
        r(x,costs)['loss'].backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in r.parameters()))
        self.assertTrue(all(p.grad is None for p in m.parameters()))
        for k,v in m.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)

    def test_bank_gradient_independence_and_batch_normalization(self):
        torch.manual_seed(3)
        r=RouterPair(2,8,4,3,torch.ones(2))
        x,cost=torch.randn(6,2,8),torch.rand(6,2,2,4)
        torch.testing.assert_close(r.weight[0],r.weight[1],rtol=0,atol=0)
        r(x,cost)['loss'].backward();full=r.weight.grad.clone();r.zero_grad()
        changed=cost.clone();changed[:,:,0]=torch.flip(cost[:,:,0],[-1])
        r(x,changed)['loss'].backward()
        torch.testing.assert_close(r.weight.grad[1],full[1],rtol=0,atol=0)
        r.zero_grad()
        for part in (slice(0,3),slice(3,6)):
            (r(x[part],cost[part])['loss']/2).backward()
        torch.testing.assert_close(r.weight.grad,full)

    def test_pre_attention_decisions_keep_all_cache_and_reload(self):
        m=tiny_model();r=RouterPair(2,32,4,3,torch.ones(2)).eval()
        ids=torch.tensor([[1,3,5,7,9,11,13,15]])
        for bank in (0,1):
            with torch.no_grad(),RouterSelection(m,r,bank,collect=True,stride=2) as selection:
                full=m(ids,use_cache=False).logits
            for row in selection.stats.values():self.assertEqual(row['totals'][-1],4)
            self.assertFalse(selection.pending)
        cached_check(m,r,ids)
        all_groups=RouterPair(2,32,4,4,torch.ones(2))
        with torch.no_grad():
            expected=m(ids).logits
            with RouterSelection(m,all_groups,0):
                torch.testing.assert_close(m(ids).logits,expected,rtol=0,atol=0)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'r.pt';save_routers(r,path);loaded=load_routers(path)
            with torch.no_grad(),RouterSelection(m,loaded,1):
                torch.testing.assert_close(m(ids,use_cache=False).logits,full,rtol=0,atol=0)

    def test_feature_order_and_duplicate_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths=[]
            for rank in (0,1):
                path=Path(tmp)/f'{rank}.pt';paths.append(path)
                ids=torch.arange(rank,6,2)
                torch.save(dict(example_ids=ids,features=ids[:,None,None].float(),costs=ids[:,None,None,None].float()),path)
            ds=FeatureDataset(paths)
            self.assertEqual([ds[i]['features'].item() for i in range(6)],list(range(6)))
            with self.assertRaisesRegex(ValueError,'Missing or duplicated'):FeatureDataset(paths[:1])

    def test_real_hf_trainer_two_rank_pilot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model=root/'model';probe=root/'probe';probe.mkdir()
            tiny_model().save_pretrained(model)
            torch.manual_seed(4)
            torch.save(dict(calibration=torch.randint(0,47,(8,17)),evaluation=torch.randint(0,47,(4,17))),probe/'tokens.pt')
            (probe/'manifest.json').write_text(json.dumps(dict(status='passed',length=16,seed=42,
                calibration_documents=8,evaluation_documents=4,tokens_sha256=sha256(probe/'tokens.pt'),
                model_files={p.name:sha256(p) for p in model.iterdir() if p.is_file()})))
            cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
                '-m','efficient_llm.router_pilot','--model',str(model),'--probe',str(probe),
                '--output',str(root/'run'),'--device','cpu','--keep','3','--stride','2','--steps','8','--feature-batch','4']
            done=subprocess.run(cmd,capture_output=True,text=True,timeout=240)
            self.assertEqual(done.returncode,0,done.stdout+done.stderr)
            train=json.loads((root/'run/training.json').read_text())
            self.assertEqual(train['steps'],8);self.assertEqual(train['feature_examples'],64)
            for bank in ('joint','independent'):
                self.assertLess(train['final'][bank]['cross_entropy'],train['initial'][bank]['cross_entropy'])
            ev=json.loads((root/'run/evaluation.json').read_text())
            self.assertEqual(ev['status'],'passed');self.assertEqual(ev['world_size'],2)
            self.assertEqual(set(ev['methods']),{'dense','static','raw_norm','projected_norm','joint','joint_router','independent_router'})
            for row in ev['methods'].values():self.assertEqual(row['scored_tokens'],64)
            checkpoints=list((root/'run/trainer').glob('checkpoint-*'))
            self.assertEqual(len(checkpoints),2)
            for c in checkpoints:
                self.assertTrue((c/'optimizer.pt').is_file());self.assertTrue((c/'scheduler.pt').is_file())


if __name__=='__main__':unittest.main()
