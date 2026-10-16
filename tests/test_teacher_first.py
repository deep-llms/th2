"""Independent causality, gradient and actual frozen-tail tests."""
import copy
import tempfile
import unittest
from pathlib import Path

import torch
from deep_kv.teacher_first import TeacherFirst, MemoryBranch, make_trainer_class
from tests.test_proxy_screen24 import model
from tests.test_proxy_heads import batch


class TeacherFirstTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_zero_identity_full_tail_and_gradients(self):
        ctx = batch()
        for source in ('deep', 'shallow'):
            m = TeacherFirst(model('A'), source, consumer=2, target=8, width=8)
            before = copy.deepcopy(m.base.state_dict())
            result = m(ctx, compare=True)
            native = m.base.lm_statistics(ctx, m.base.hidden_states(ctx)[0])[0]
            torch.testing.assert_close(result['nll'], native, atol=0, rtol=0)
            torch.testing.assert_close(result['baseline'], native, atol=0, rtol=0)
            opt = torch.optim.AdamW(m.branch.parameters(), lr=.01)
            for _ in range(3):
                opt.zero_grad()
                r = m(ctx)
                (r['nll'].sum()/r['counts'].sum()).backward()
                self.assertTrue(all(p.grad is None for p in m.base.parameters()))
                opt.step()
            self.assertGreater(m.branch.extract.weight.grad.abs().sum().item(), 0)
            self.assertGreater(m.branch.query.weight.grad.abs().sum().item(), 0)
            for name, value in m.base.state_dict().items():
                torch.testing.assert_close(value, before[name], atol=0, rtol=0)
            # Save/reload ONLY the branch; frozen model remains the reference.
            other = TeacherFirst(copy.deepcopy(m.base), source, consumer=2, target=8, width=8)
            other.branch.load_state_dict(m.branch.state_dict(), strict=True)
            torch.testing.assert_close(other(ctx)['nll'], m(ctx)['nll'], atol=0, rtol=0)

    def test_independent_strict_past_and_empty_document_starts(self):
        ctx = batch(); h = torch.randn(*ctx.input_ids.shape, 16); src = torch.randn_like(h)
        m = MemoryBranch(16, 8)
        torch.nn.init.normal_(m.out.weight, std=.1)
        out = m(h, src, ctx)
        z = torch.nn.functional.rms_norm(m.extract(torch.nn.functional.rms_norm(src, (16,), eps=1e-6)), (8,), eps=1e-6)
        q = m.query(torch.nn.functional.rms_norm(h, (16,), eps=1e-6)).reshape(*h.shape[:2], 2, 4)
        k = m.key(z).reshape_as(q); v = m.value(z).reshape_as(q)
        expected = torch.zeros_like(out)
        for b in range(h.shape[0]):
            for t in range(h.shape[1]):
                js = [j for j in range(t) if ctx.segments[b,j] == ctx.segments[b,t]]
                if js:
                    score = torch.einsum('hd,jhd->hj', q[b,t], k[b,js])/2
                    expected[b,t] = m.out(torch.einsum('hj,jhd->hd',score.softmax(-1),v[b,js]).flatten())
        torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-5)
        changed = src.clone(); changed[:, -1] += 100
        torch.testing.assert_close(m(h,changed,ctx),out,atol=0,rtol=0)
        changed=src.clone(); other=ctx.segments[0]!=ctx.segments[0,-1];changed[0,other]+=100
        torch.testing.assert_close(m(h,changed,ctx)[0,~other],out[0,~other],atol=0,rtol=0)

    def test_full_model_future_and_document_isolation(self):
        m=TeacherFirst(model('A'), consumer=2,target=8,width=8).eval()
        torch.nn.init.normal_(m.branch.out.weight,std=.1)
        ctx=batch(); original=[]
        # Observe the real final hidden states, before CE changes its labels.
        hook=m.base.backbone.model.norm.register_forward_hook(lambda mod,a,o:original.append(o.detach().clone()))
        with torch.no_grad():
            m(ctx); first=original[-1]
            changed=copy.deepcopy(ctx);changed.input_ids[0,-1]=(changed.input_ids[0,-1]+1)%m.base.backbone.config.vocab_size
            m(changed);torch.testing.assert_close(original[-1][0,:-1],first[0,:-1],atol=0,rtol=0)
            changed=copy.deepcopy(ctx);other=ctx.segments[0]!=ctx.segments[0,-1]
            changed.input_ids[0,other]=(changed.input_ids[0,other]+1)%m.base.backbone.config.vocab_size
            m(changed);torch.testing.assert_close(original[-1][0,~other],first[0,~other],atol=0,rtol=0)
        hook.remove()

    def test_real_hf_trainer_uses_branch_and_global_denominator(self):
        from transformers import TrainingArguments
        from deep_kv.packing import isolated_data_collator
        ctx=batch()
        rows=[dict(input_ids=ctx.input_ids[i],attention_mask=ctx.valid[i].long(),
                   segments=ctx.segments[i],labels=ctx.input_ids[i]) for i in range(len(ctx.input_ids))]*4
        with tempfile.TemporaryDirectory() as tmp:
            m=TeacherFirst(model('A'),consumer=2,target=8,width=8)
            trainer=make_trainer_class()(model=m,args=TrainingArguments(output_dir=tmp,
                use_cpu=True,max_steps=2,per_device_train_batch_size=1,gradient_accumulation_steps=2,
                save_strategy='no',report_to=[],remove_unused_columns=False,ddp_find_unused_parameters=False),
                train_dataset=rows,eval_dataset=rows,data_collator=isolated_data_collator)
            trainer.train()
            self.assertEqual(trainer.state.global_step,2)
            self.assertGreater(m.branch.out.weight.abs().sum().item(),0)
            result=trainer.predict(rows)
            self.assertEqual(result.predictions.shape,(len(rows),3))
            self.assertTrue(torch.isfinite(torch.from_numpy(result.predictions)).all())

    def test_uneven_rank_token_normalization_matches_full_batch(self):
        from types import SimpleNamespace
        from deep_kv.packing import isolated_data_collator
        ctx=batch();rows=[]
        for i in range(2):
            labels=ctx.input_ids[0].clone()
            if i==0:labels[1:3]=-100
            rows.append(dict(input_ids=ctx.input_ids[0],attention_mask=ctx.valid[0].long(),
                segments=ctx.segments[0],labels=labels))
        full=TeacherFirst(model('A'),consumer=2,target=8,width=8)
        torch.nn.init.normal_(full.branch.out.weight,std=.1)
        from deep_kv.training import DeepKVTrainer
        inputs=isolated_data_collator(rows);context=DeepKVTrainer.context(inputs)
        result=full(context);count=result['counts'].sum()
        (result['nll'].sum()/count).backward()
        expected=[p.grad.clone() for p in full.branch.parameters()]
        rank_grads=[]
        for row in rows:
            local=copy.deepcopy(full);local.zero_grad(set_to_none=True)
            fake=SimpleNamespace(accelerator=SimpleNamespace(num_processes=2))
            loss=make_trainer_class().compute_loss(fake,local,isolated_data_collator([row]),num_items_in_batch=count)
            loss.backward();rank_grads.append([p.grad.clone() for p in local.branch.parameters()])
        for a,b,want in zip(*rank_grads,expected):
            torch.testing.assert_close((a+b)/2,want,atol=2e-7,rtol=2e-5)

    def test_scientific_gate_requires_both_backends_and_matching_rows(self):
        import json
        from types import SimpleNamespace
        from scripts.teacher_first_pilot import summarize,CHECKPOINT_HASH
        from scripts.train_then_burn import digest
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'scope.json').write_text('{}')
            for agree in (True,False):
                out=root/str(agree)
                for source in ('shallow','deep'):
                    folder=out/('full-'+source);folder.mkdir(parents=True)
                    value=29.9 if source=='shallow' else 29.8
                    dense=value if agree or source=='shallow' else 30.1
                    result=dict(status='passed',source=source,steps=256,scope_sha256=digest(root/'scope.json'),
                        checkpoint_sha256=CHECKPOINT_HASH,frozen_backbone_unchanged=True,baseline_identity_exact=True,
                        reload_exact=True,rows=[[value,30,10]]*512,sdpa_rows=[[dense,30,10]]*512,
                        elapsed_seconds=1,baseline_nll=3,nll=value/10)
                    (folder/'result.json').write_text(json.dumps(result))
                summarize(SimpleNamespace(root=root,output=out))
                result=json.loads((out/'summary.json').read_text())
                self.assertEqual(result['scientific_gate'],'promising_teacher' if agree else 'stop_no_teacher_advantage')
            out=root/'False';p=out/'full-deep/result.json'
            changed=json.loads(p.read_text());changed['sdpa_rows'][0][2]=9;p.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'Paired evaluation differs'):
                summarize(SimpleNamespace(root=root,output=out))


if __name__=='__main__':unittest.main()
