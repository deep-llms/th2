import copy
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
import torch
from torch.nn import functional as F
from efficient_llm.a2.model import forward_kl,prune,bypass,save_student,reload_student,round_to_saved_precision
from efficient_llm.a2.measure import residual_profile,score,document_mask

torch.set_num_threads(1)


def tiny_model():
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    code=os.environ.get('A2_OPENSCI_CODE')
    if not code or not Path(code,'modeling_opensci.py').is_file():
        raise unittest.SkipTest('Set A2_OPENSCI_CODE to pinned main source; no automatic network in tests')
    expected={'modeling_opensci.py':'f1a826eed8897a4df548c26d37741fb7b5d5cf3d170db2a7b079cd9f30da512d','configuration_opensci.py':'194d816263ec0279f5804f41041ecd29a645c330b5893636d38dc7cb94747885'}
    for name,digest in expected.items():
        if hashlib.sha256(Path(code,name).read_bytes()).hexdigest()!=digest:
            raise AssertionError('Test must use the exact pinned OpenSci main source')
    cls=get_class_from_dynamic_module('modeling_opensci.OpensciForCausalLM',code,local_files_only=True)
    config=cls.config_class(hidden_size=16,intermediate_size=32,num_hidden_layers=4,
        num_attention_heads=2,num_key_value_heads=2,head_dim=8,vocab_size=32,
        attention_dropout=0.,attention_bias=True,mlp_bias=True,qk_layernorm=True,
        tie_word_embeddings=True,use_cache=False)
    config._attn_implementation='sdpa'
    torch.manual_seed(0)
    return cls(config).float().eval()


class KDTests(unittest.TestCase):
    def test_chunk_loss_and_gradients(self):
        torch.manual_seed(3)
        h=torch.randn(3,11,7,requires_grad=True); w=torch.randn(17,7,requires_grad=True)
        th=torch.randn_like(h); tw=torch.randn_like(w)
        p=F.linear(th,tw).log_softmax(-1)
        full=F.kl_div(F.linear(h,w).log_softmax(-1),p,reduction='sum',log_target=True)/(3*11)
        grads=torch.autograd.grad(full,(h,w))
        for chunk in (1,4,20):
            result=forward_kl(h,w,th,tw,chunk)
            got=torch.autograd.grad(result,(h,w))
            torch.testing.assert_close(result,full,rtol=1e-6,atol=1e-6)
            for a,b in zip(grads,got):torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
        equal=forward_kl(h,w,h.detach(),w.detach(),4)
        self.assertLess(abs(equal.item()),1e-6)
        self.assertTrue(all(torch.isfinite(g).all() for g in torch.autograd.grad(equal,(h,w))))

    def test_accumulation_64_vs_8(self):
        torch.manual_seed(3)
        x=torch.randn(64,5,7); th=torch.randn_like(x); tw=torch.randn(19,7)
        weight=torch.randn(7,7,requires_grad=True); head=torch.randn(19,7,requires_grad=True)
        loss=forward_kl(x@weight,head,th,tw,2)
        reference=torch.autograd.grad(loss,(weight,head))
        for start in range(0,64,8):
            (forward_kl(x[start:start+8]@weight,head,th[start:start+8],tw,2)/8).backward()
        for p,expected in zip((weight,head),reference):torch.testing.assert_close(p.grad,expected,atol=1e-5,rtol=1e-5)
        self.assertIsNone(tw.grad)


class NativeModelTests(unittest.TestCase):
    def test_native_prune_hooks_and_reload(self):
        parent=tiny_model(); ids=torch.tensor([[1,2,3,0,5]])
        same=copy.deepcopy(parent); prune(same,[])
        self.assertTrue(torch.equal(parent(ids).logits,same(ids).logits))
        for remove in ([0],[1],[3],[0,2]):
            student=copy.deepcopy(parent); keep=prune(student,remove)
            with bypass(parent,remove):expected=parent(ids).logits
            torch.testing.assert_close(student(ids).logits,expected,rtol=0,atol=1e-6)
            self.assertEqual(keep,[i for i in range(4) if i not in remove])
        rows=[{'input_ids':ids[0],'sample_id':0}]
        round_to_saved_precision(student)
        with tempfile.TemporaryDirectory() as d:
            save_student(student,Path(d)/'final'); restored=reload_student(Path(d)/'final')
            self.assertEqual(score(student,rows)['loss'],score(restored,rows)['loss'])
            self.assertEqual(restored.config.a2_kept_layers,[1,3])
            self.assertEqual(restored.lm_head.weight.data_ptr(),restored.model.embed_tokens.weight.data_ptr())

    def test_residual_not_final_norm(self):
        m=tiny_model(); rows=[dict(input_ids=torch.tensor([1,2,3,4]),sample_id=0)]
        with bypass(m,[0,1,2,3]):
            out=residual_profile(m,rows)
        for key in ('layer_bi','group_bi','relative_update'):
            self.assertTrue(all(abs(v)<1e-6 for v in out[key]),out)

    def test_mask_and_score(self):
        mask,pos=document_mask(torch.tensor([[1,0,1,0,0]],dtype=torch.bool),torch.float32)
        self.assertEqual(pos.tolist(),[[0,1,0,1,2]])
        self.assertEqual(mask[0,0,3,2].item(),0)
        self.assertLess(mask[0,0,3,1].item(),-1e20)
        m=tiny_model(); ids=torch.tensor([[1,2,0,5],[2,3,4,5]])
        rows=[dict(input_ids=row,sample_id=i) for i,row in enumerate(ids)]
        result=score(m,rows,batch=2,chunk=1)
        ref=F.cross_entropy(m(ids).logits[:,:-1].reshape(-1,32),ids[:,1:].reshape(-1))
        self.assertAlmostEqual(result['loss'],ref.item(),places=6)
        self.assertEqual(result['tokens'],6)

class NativeCheckTests(unittest.TestCase):
    def test_runtime_numerical_probe(self):
        from efficient_llm.a2.check import numerical_checks
        model=tiny_model()
        rows=[dict(input_ids=torch.tensor([1,2,3,4,5,6,7,8]),sample_id=0)]
        with tempfile.TemporaryDirectory() as d:
            result=numerical_checks(model,rows,{'S1':[1]},d)
        self.assertTrue(result['passed'],result)

    def test_gaussian_exact_restore(self):
        from efficient_llm.a2.measure import gaussian
        model=tiny_model()
        before={k:v.clone() for k,v in model.state_dict().items()}
        rows=[dict(input_ids=torch.tensor([1,2,3,4]),sample_id=0)]
        result=gaussian(model,rows)
        self.assertEqual(set(result['sigmas']),{'0.005','0.01','0.02'})
        for key,value in model.state_dict().items():self.assertTrue(torch.equal(value,before[key]),key)


class ReleasedShapeTests(unittest.TestCase):
    def test_released_parameter_count_on_meta(self):
        import json
        tiny=tiny_model()
        config_path=Path(os.environ['A2_OPENSCI_CODE'])/'config.json'
        config=type(tiny.config).from_dict(json.loads(config_path.read_text()))
        config.attention_dropout=0.;config.use_cache=False;config._attn_implementation='sdpa'
        with torch.device('meta'): model=type(tiny)(config)
        count=sum(p.numel() for p in model.parameters())
        self.assertEqual(count,1714377728)
        self.assertEqual(count-model.model.embed_tokens.weight.numel(),1611355136)
