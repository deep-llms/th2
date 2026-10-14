"""Independent teacher and single-site downstream counterfactual checks."""
import copy
import unittest
from unittest.mock import patch

import torch

from eval.p7_teacher import ARMS, capture, intervention, validate_capture
from tests.test_proxy_screen24 import model
from tests.test_proxy_heads import batch
from tests.test_p7_consumer import response
from tests.test_fa4_baseline import reference_kernel


class TeacherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    @torch.no_grad()
    def test_independent_targets_teacher_and_downstream(self):
        for backend in ('sdpa','fa4'):
            for arm in ARMS:
                with patch('deep_kv.fa4.load_kernel',return_value=(reference_kernel,{'version':'CPU-reference'})):
                    m=model(arm,backend=backend).eval()
                # Exercise a genuinely nonzero deep extension, not its zero init.
                if m.extensions:
                    from tests.test_proxy_extensions import activate
                    activate(m)
                before=copy.deepcopy(m.state_dict());ctx=batch();mlps={};hooks=[]
                for i,layer in enumerate(m.backbone.model.layers,1):
                    hooks.append(layer.mlp.register_forward_hook(lambda mod,a,o,i=i:mlps.__setitem__(i,o)))
                plain=m.hidden_states(ctx)[0]
                for h in hooks:h.remove()
                hidden,sites=capture(m,ctx)
                torch.testing.assert_close(hidden,plain,rtol=0,atol=0)
                validate_capture(m,sites,ctx)
                for site,entry in sites.items():
                    raw=sum(mlps[j].float() for j in range(site,site+4))
                    torch.testing.assert_close(entry['target'],m.normalize_target(raw,site),atol=0,rtol=0)
                    layer,u,head,pred,mask,rotary,plan=entry['args']
                    teacher=response(layer,head,u,entry['target'],rotary,ctx.allowed())
                    torch.testing.assert_close(teacher,entry['teacher'],atol=1e-6,rtol=1e-5)
                    for alpha in (0,.1,-.1,1):
                        with intervention(m,sites,site,alpha):actual=m.hidden_states(ctx)[0]
                        if alpha==0:torch.testing.assert_close(actual,plain,atol=0,rtol=0)
                        def replace(mod,args):
                            heads=args[0].reshape(*u.shape[:-1],-1,layer.self_attn.head_dim).clone()
                            heads[:,:,-4:]+=alpha*(teacher-heads[:,:,-4:])
                            return (heads.flatten(-2),)
                        hook=layer.self_attn.o_proj.register_forward_pre_hook(replace)
                        expected=m.hidden_states(ctx)[0];hook.remove()
                        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
                for k,v in m.state_dict().items():torch.testing.assert_close(v,before[k],atol=0,rtol=0)
                self.assertNotIn('block',m.__dict__);self.assertNotIn('memory_attention',m.__dict__)

    @torch.no_grad()
    def test_future_and_other_document_do_not_change_teacher(self):
        m=model('P7-simple-deep').eval();ctx=batch()
        from tests.test_proxy_extensions import activate
        activate(m)
        _,first=capture(m,ctx)
        altered=copy.deepcopy(ctx);altered.input_ids[0,-1]=(altered.input_ids[0,-1]+1)%m.backbone.config.vocab_size
        _,second=capture(m,altered)
        for site in first:
            torch.testing.assert_close(first[site]['teacher'][0,:-1],second[site]['teacher'][0,:-1],atol=0,rtol=0)
        altered=copy.deepcopy(ctx)
        other=ctx.segments[0]!=ctx.segments[0,-1]
        altered.input_ids[0,other]=(altered.input_ids[0,other]+1)%m.backbone.config.vocab_size
        _,second=capture(m,altered)
        for site in first:
            torch.testing.assert_close(first[site]['teacher'][0,~other],second[site]['teacher'][0,~other],atol=0,rtol=0)

    @torch.no_grad()
    def test_restoration_on_failure_and_stale_capture_rejection(self):
        m=model('P7-simple').eval();ctx=batch();_,sites=capture(m,ctx)
        with self.assertRaisesRegex(RuntimeError,'deliberate'):
            with intervention(m,sites,2,.1):raise RuntimeError('deliberate')
        self.assertNotIn('memory_attention',m.__dict__)
        ctx.input_ids[0,0]=(ctx.input_ids[0,0]+1)%m.backbone.config.vocab_size
        with self.assertRaisesRegex(RuntimeError,'source changed'):
            with intervention(m,sites,2,.1):m.hidden_states(ctx)

    def test_selection_and_summary_fail_closed(self):
        from scripts.probe_p7_teacher import selected_rows,summarize
        indices=selected_rows(4882,512)
        self.assertEqual(len(set(indices)),512)
        self.assertFalse(set(indices[::2])&set(indices[1::2]))
        with self.assertRaises(ValueError):selected_rows(2,8)
        scope=dict(checkpoints=[dict(label='test',checkpoint='ck',sha256='weights')],
            rows_sha256='data',sites=[2],alphas=[.1])
        rows=[dict(index=i) for i in range(8)]
        reports=[dict(status='passed',case=0,half=h,stage='full',checkpoint_sha256='weights',
            rows_sha256='data',model_state_unchanged=True,checks=[dict(identity_exact=True)],
            rows=[dict(row=i,targets=10,nll={'baseline':30.,'2:0.1':29.}) for i in range(h,8,2)]) for h in (0,1)]
        result=summarize(scope,reports,rows,'full')['results']['test']
        self.assertEqual(result['target_tokens'],80)
        self.assertAlmostEqual(result['interventions']['2:0.1']['delta_nats_per_token'],-.1)
        reports[1]['rows'][0]['row']=reports[0]['rows'][0]['row']
        with self.assertRaises(ValueError):summarize(scope,reports,rows,'full')


if __name__=='__main__':unittest.main()
