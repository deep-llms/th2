import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from transformers import SmolLM3Config,SmolLM3ForCausalLM
from efficient_llm.a2_breadth.common import blocks,relative_bootstrap,spearman,weighted,write,Checks
from efficient_llm.a2_breadth.smol import removed_layers,score_docs,encode_document
from efficient_llm.a2_breadth.heal import Trace,ReportTrainer
from efficient_llm.a2.train import HealTrainer,make_arguments

class BreadthTests(unittest.TestCase):
    def model(self):
        torch.manual_seed(4)
        c=SmolLM3Config(vocab_size=64,pad_token_id=0,bos_token_id=1,eos_token_id=2,hidden_size=32,intermediate_size=64,num_hidden_layers=8,num_attention_heads=4,num_key_value_heads=2,head_dim=8,no_rope_layers=[1,1,1,0,1,1,1,0],max_position_embeddings=64,attention_dropout=0.)
        c._attn_implementation='sdpa';return SmolLM3ForCausalLM(c).float().eval()
    def test_blocks(self):
        self.assertEqual(blocks(24),dict(B_early=list(range(2,8)),B_S1=list(range(16,22)),B_end=list(range(18,24))))
        self.assertEqual(blocks(36),dict(B_early=list(range(3,12)),B_S1=list(range(24,33)),B_end=list(range(27,36))))
    def test_native_bitwise_rope_preservation_and_restore(self):
        m=self.model();ids=torch.tensor([[1,2,3,4,5]])
        with torch.no_grad():
            ref=m(ids,use_cache=False).logits
            with removed_layers(m,[]):self.assertTrue(torch.equal(ref,m(ids,use_cache=False).logits))
            with removed_layers(m,[1,2,3]):
                self.assertEqual([l.self_attn.layer_idx for l in m.model.layers],[0,4,5,6,7])
                self.assertEqual([l.self_attn.use_rope for l in m.model.layers],[1,1,1,1,0])
                self.assertTrue(torch.isfinite(m(ids,use_cache=False).logits).all())
            self.assertTrue(torch.equal(ref,m(ids,use_cache=False).logits))
    def test_document_nll_exact_independent(self):
        m=self.model();tokens=np.array([1,2,3,4,5,6,7]);offset=np.array([0,3,7]);got=score_docs(m,tokens,offset)
        values=[]
        with torch.no_grad():
            for start,end in zip(offset[:-1],offset[1:]):
                ids=torch.tensor(tokens[start:end])[None];logits=m(ids,use_cache=False).logits
                values.append(torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,64),ids[:,1:].flatten(),reduction='none').double().mean().item())
        np.testing.assert_array_equal(values,got['per_sequence_nll']);self.assertEqual(got['tokens'],5)
        altered=tokens.copy();altered[:3]=[9,10,11]
        self.assertEqual(got['per_sequence_nll'][1],score_docs(m,altered,offset)['per_sequence_nll'][1])
    def test_tokenization_policy(self):
        class T:
            eos_token_id=2
            def __call__(self,text,**kw):
                assert kw==dict(add_special_tokens=False,truncation=False)
                return dict(input_ids=[3]*int(text))
        self.assertEqual(encode_document(T(),'3'),[3,3,3,2]);self.assertEqual(len(encode_document(T(),'5000')),4096)
    def test_paired_bootstrap_token_weighted(self):
        a=np.array([1.,2.,4.]);b=1.5*a;c=[1,10,100]
        r=relative_bootstrap(a,b,c,repeats=100)
        self.assertAlmostEqual(r['value'],.5);np.testing.assert_allclose(r['ci95'],[.5,.5]);self.assertEqual(spearman([1,2,3],[7,8,9]),1.)
        self.assertNotEqual(weighted(a,c),a.mean())
    def test_report_flags_continue_and_atomic(self):
        with tempfile.TemporaryDirectory() as d:
            c=Checks(Path(d)/'checks.json');c.record('fail',False);c.record('pass',True)
            self.assertEqual([x['passed'] for x in json.loads(c.path.read_text())['checks']],[False,True])
    def test_kd_forward_and_gradients_unchanged(self):
        import copy
        from efficient_llm.a2.model import KDModel
        student=self.model();other=copy.deepcopy(student);teacher=copy.deepcopy(student).requires_grad_(False)
        a=object.__new__(HealTrainer);b=object.__new__(ReportTrainer)
        a.teacher=b.teacher=teacher;a.seen=[];b.seen=[]
        inputs=dict(input_ids=torch.tensor([[1,2,3,4]]),sample_id=torch.tensor([0]))
        la=a.compute_loss(KDModel(student,2),inputs);lb=b.compute_loss(KDModel(other,2),inputs)
        self.assertTrue(torch.equal(la,lb));la.backward();lb.backward()
        for x,y in zip(student.parameters(),other.parameters()):
            if x.grad is not None:self.assertTrue(torch.equal(x.grad,y.grad))
    def test_coordinator_continues_after_worker_failure(self):
        import scripts.a2_breadth_run as runner
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cfg=root/'plan.json'
            write(cfg,dict(results_root=d,downloads=dict(checkpoints=[dict(path=d,files=[],revision=str(i)) for i in range(11)])))
            receipt=root/'complete.json';calls=[]
            def fake(cmd,**kw):
                calls.append((cmd,kw['env']['CUDA_VISIBLE_DEVICES']))
                return SimpleNamespace(returncode=1 if len(calls)==1 else 0)
            with patch('sys.argv',['runner','--config',str(cfg),'--receipt',str(receipt)]),patch.object(runner.subprocess,'run',side_effect=fake),patch('scripts.a2_breadth_report.finalize',return_value=dict(coverage_complete=False)):
                runner.main()
            self.assertEqual(len(calls),16)
            self.assertTrue(json.loads(receipt.read_text())['execution_complete'])
            self.assertFalse(json.loads(receipt.read_text())['coverage_complete'])
            self.assertEqual([c[1] for c in calls[:4]],['0,1,2,3,4,5,6,7']*4)
    def test_complete_saved_readout_and_missing_coverage(self):
        from scripts.a2_breadth_report import compute
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);entries=[dict(revision='r'+str(i),tokens=(i+1)*100,step=i,commit='fixture') for i in range(11)]
            cfg=dict(downloads=dict(checkpoints=entries),annealed_reference={ds:dict(E_ann=.025,P300_delta=.2) for ds in ('dclm','fwe')},deviations=[])
            for i,e in enumerate(entries):
                p=np.array([5.,6.,7.])-.01*i
                for b,g in [('parent',0),('B_early',.001*(i+1)**2),('B_S1',.01*(i+1)),('B_end',.02*(i+1))]:
                    write(root/'perseq'/e['revision']/(b+'.json'),dict(per_sequence_nll=(p+g).tolist(),per_sequence_tokens=[3,7,11],loss=weighted(p+g,[3,7,11])))
            for ds in ('dclm','fwe'):
                for arm,delta in [('S72',.1),('S188',.12)]:
                    name=f'ST1-{ds}-{arm}-S1-0.2B-lr1e-3'
                    write(root/'heals'/name/'result.json',dict(delta_post=delta,training_seconds=1,baseline=dict(per_sequence_nll=[3.]*3),scores=dict(final=dict(per_sequence_nll=[3.+delta]*3))))
            decision,tests,runs=compute(cfg,root)
            self.assertTrue(decision['coverage_complete']);self.assertTrue(all(decision['clauses'].values()));self.assertEqual(len(runs),15)
            self.assertAlmostEqual(decision['stable_healing']['fwe']['E_stab'],.02)
            self.assertTrue(all(t['passed'] for t in tests))
            (root/'perseq'/'r4'/'B_S1.json').unlink()
            decision,_,_=compute(cfg,root)
            self.assertFalse(decision['coverage_complete']);self.assertIsNone(decision['clauses']['SM1'])
    def test_training_recipe_numerical_core(self):
        self.assertIs(ReportTrainer.create_optimizer,HealTrainer.create_optimizer)
        self.assertIs(ReportTrainer.create_scheduler,HealTrainer.create_scheduler)
        a=make_arguments('/tmp/breadth-test',762,.001,4,8,cuda=False)
        self.assertEqual(a.gradient_accumulation_steps,2);self.assertFalse(a.gradient_checkpointing)
        self.assertEqual(a.max_steps,762);self.assertEqual(762*64*4096,199753728)
        t=Trace();self.assertFalse(hasattr(t,'limit'))

if __name__=='__main__':unittest.main()
