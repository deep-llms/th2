"""No GPU management; exercise exact routing and actual compiler autograd."""
import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import torch
from scripts.p6_compile_trial import implementation, compare
from deep_kv import proxy_estimators
from tests.test_proxy_memory import model, batch, objective


class CompileTrialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_compiled_cosine_epsilon_mask_and_detached_target(self):
        reference = proxy_estimators.cosine_loss
        torch.manual_seed(17)
        with implementation('cosine'):
            compiled = proxy_estimators.cosine_loss
        self.assertIs(proxy_estimators.cosine_loss,reference)
        for scale in (0.,1e-8,1e-6,1.):
            x=(torch.randn(2,4,32)*scale).requires_grad_()
            y=torch.randn_like(x,requires_grad=True)
            valid=torch.tensor([[1,1,0,1],[1,0,1,1]],dtype=torch.bool)
            a=reference(x,y,valid);a[1].sum().backward();expected=x.grad.clone();x.grad=None
            b=compiled(x,y,valid);b[1].sum().backward()
            self.assertIsNone(y.grad)
            for v,w in zip(a,b):torch.testing.assert_close(v,w,rtol=2e-6,atol=2e-6)
            torch.testing.assert_close(x.grad,expected,rtol=2e-5,atol=max(2e-6,float(expected.abs().max())*2e-6))

    def test_full_model_loss_routing_and_statistics(self):
        for checkpoint in (False,True):
            reference=model('P6-iso',checkpoint=checkpoint)
            with implementation('cosine',backend='aot_eager'):
                candidate=model('P6-iso',checkpoint=checkpoint)
            captures=[]
            for m in (reference,candidate):
                with torch.no_grad():
                    for head in m.heads.values():head.alpha.fill_(.2)
                out=m(batch(),collect_target_statistics=True)
                objective(out).backward()
                captures.append(dict(outputs={k:v.detach() for k,v in out.items()},
                    gradients={k:p.grad for k,p in m.named_parameters() if p.grad is not None},
                    buffers=dict(m.named_buffers())))
            for name in captures[0]:
                result=compare(captures[0][name],captures[1][name],exact=True)
                self.assertFalse(result['failures'],result)

    def test_summary_requires_matching_batches_and_no_timed_recompilation(self):
        from scripts.p6_compile_trial import summary
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            names=('A-first','P6-eager-first','P6-compiled-first','P6-compiled-second','P6-eager-second','A-second')
            for name in names:
                arm='A' if name.startswith('A-') else 'P6-iso'
                folder=root/name/'seed-1042'/arm;folder.mkdir(parents=True)
                (folder/'train_config.json').write_text(json.dumps(dict(train_fingerprint='same')))
                (folder/'result.json').write_text(json.dumps(dict(global_step=100,evaluation={},training_cost={})))
                for rank in range(8):
                    report=dict(steps=[dict(step=i,seconds=2.) for i in range(1,101)],data_sha256=[str(rank)]*8,
                        graph_counts={'20':{},'90':{}},peak_gib=100,compile_counters={},implementation='eager')
                    (folder/f'compile-trial-rank{rank}.json').write_text(json.dumps(report))
            summary(SimpleNamespace(run_dir=root))
            self.assertEqual(json.loads((root/'summary.json').read_text())['p6_speedup'],1.)
            (root/'summary.json').unlink()
            path=root/'P6-compiled-first/seed-1042/P6-iso/compile-trial-rank3.json'
            report=json.loads(path.read_text());report['data_sha256'][0]='wrong';path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'batch order'):summary(SimpleNamespace(run_dir=root))
            report['data_sha256'][0]='3';report['graph_counts']['90']={'unique_graphs':1};path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'new graphs'):summary(SimpleNamespace(run_dir=root))

    def test_gate_rejects_nonfinite_missing_and_bad_gradient(self):
        x={'x':torch.ones(4)}
        self.assertFalse(compare(x,x)['failures'])
        self.assertTrue(compare(x,{'x':torch.ones(4)*1.1})['failures'])
        self.assertTrue(compare(x,{'x':torch.full((4,),float('nan'))})['failures'])
        with self.assertRaises(ValueError):compare(x,{})
        original=proxy_estimators.cosine_loss
        with self.assertRaises(RuntimeError):
            with implementation('cosine',backend='aot_eager'):raise RuntimeError('test')
        self.assertIs(proxy_estimators.cosine_loss,original)


if __name__=='__main__':unittest.main()
