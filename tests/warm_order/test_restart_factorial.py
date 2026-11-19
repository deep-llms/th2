import copy
import json
from pathlib import Path
import torch
from tests.warm_order.test_frozen_router_train import FrozenRouterTrainTests
from efficient_llm.warm_order.soft_route import Model as SoftModel
from efficient_llm.warm_order.exposure_recovery import optimizer_for
from efficient_llm.warm_order.pair_order import load_initial
from efficient_llm.warm_order.restart_factorial import transfer_body_adam,initialize_optimizer
from efficient_llm.warm_order.restart_factorial_train import train
from efficient_llm.warm_order.soft_exploration_train import write_json
from efficient_llm.warm_order.data import sha


class RestartFactorialTests(FrozenRouterTrainTests):
    def setUp(self):
        super().setUp()
        p=Path(self.cfg['parent']);m=SoftModel(self.c,'dense',.999)
        opt=optimizer_for(m,1e-4,1e-3)
        self.x=torch.randint(self.c.vocab,(2,self.c.length))
        for i in range(4):
            opt.zero_grad();m.nll(self.x).mean().backward();opt.step()
        opt.param_groups[0]['lr']=1e-5
        pc=dict(name='C-s0',arm='C',seed=0,model=self.cfg['model'],steps=2,parent_steps=2,schedule='inherit-terminal-lr-constant-no-floor')
        recipe=dict(config=pc,parent_recipe=dict(config=dict(arm='C',steps=2,lr=1e-4,canonical_probability=.999)))
        torch.save(dict(config=self.cfg['model'],arm='C',seed=0,canonical_probability=.999,model=m.state_dict()),p/'weights.pt')
        self.parent=dict(recipe=recipe,step=2,model=m.state_dict(),optimizer=opt.state_dict())
        torch.save(self.parent,p/'fullstate.pt');write_json(p/'recipe.json',recipe)
        write_json(p/'complete.json',dict(name='C-s0',arm='C',seed=0,step=2,weights_sha256=sha(p/'weights.pt'),fullstate_sha256=sha(p/'fullstate.pt')))
        self.cfg.update(arm='C',name='IL-s0',factor='IL',schedule='dense-restart-factorial',terminal_lr=1e-5,parent_steps=2,
                        parent_hashes={f:sha(p/f) for f in ('weights.pt','fullstate.pt','complete.json','recipe.json')})

    def test_named_transfer_reordered_target_and_direct_update(self):
        m=load_initial(self.cfg)
        target=torch.optim.AdamW([dict(params=list(reversed([p for p in m.parameters() if p.requires_grad])),lr=1e-5,weight_decay=.1)],betas=(.9,.95))
        proof=transfer_body_adam(m,target,self.parent,phase_steps=2,terminal_lr=1e-5)
        self.assertEqual(proof['cumulative_adam_steps'],4)
        ref=SoftModel(self.c,'dense',.999);ref.load_state_dict(self.parent['model'])
        opt=optimizer_for(ref,1e-4,1e-3);opt.load_state_dict(self.parent['optimizer'])
        target.zero_grad();opt.zero_grad();m.nll(self.x).mean().backward();ref.nll(self.x).mean().backward()
        target.step();opt.step()
        for n,p in ref.named_parameters():
            if p.requires_grad:
                torch.testing.assert_close(dict(m.named_parameters())[n],p,atol=1e-7,rtol=1e-6)
                self.assertEqual(float(target.state[dict(m.named_parameters())[n]]['step']),5)

    def test_factor_changes_only_state_or_policy_and_rejects_tamper(self):
        m=load_initial(self.cfg)
        for factor in ('FH','FL','IH','IL'):
            before=torch.get_rng_state().clone();o,pr=initialize_optimizer(m,{**self.cfg,'factor':factor},'cpu')
            self.assertTrue(torch.equal(before,torch.get_rng_state()))
            self.assertEqual(bool(o.state),factor.startswith('I'))
            self.assertEqual(pr['cumulative_adam_steps'],4)
        for change in ('step','variance','coverage','weight','lr'):
            bad=copy.deepcopy(self.parent);entry=next(iter(bad['optimizer']['state'].values()))
            if change=='step':entry['step'].fill_(2)
            elif change=='variance':entry['exp_avg_sq'].flatten()[0]=-1
            elif change=='coverage':bad['optimizer']['state'].pop(next(iter(bad['optimizer']['state'])))
            elif change=='weight':bad['model']['embed.weight'].add_(1)
            else:bad['optimizer']['param_groups'][0]['lr']=1e-3
            o=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-5,betas=(.9,.95),weight_decay=.1)
            with self.assertRaises(ValueError):transfer_body_adam(m,o,bad,phase_steps=2,terminal_lr=1e-5)
            self.assertFalse(o.state)

    def test_actual_entrypoint_all_arms_exact_resume_and_immutable_body(self):
        for factor in ('FH','FL','IH','IL'):
            c={**self.cfg,'factor':factor,'name':factor+'-s0'}
            m,o=train(c,self.root/(factor+'whole'),'cpu')
            train(c,self.root/(factor+'part'),'cpu',stop_after=1)
            n,no=train(c,self.root/(factor+'part'),'cpu',resume=True)
            self.assert_same(m.state_dict(),n.state_dict());self.assert_same(o.state_dict(),no.state_dict())
            expected=10 if factor.startswith('I') else 6
            self.assertTrue(all(float(st['step'])==expected for st in o.state.values()))
            tr=[json.loads(x) for x in (self.root/(factor+'whole')/'trace.jsonl').read_text().splitlines()]
            if factor.endswith('L'):self.assertTrue(all(r['lr']==1e-5 for r in tr))
            else:self.assertAlmostEqual(tr[-1]['lr'],.0003)
            for name,digest in c['parent_hashes'].items():self.assertEqual(sha(Path(c['parent'])/name),digest)

    def test_fresh_high_matches_original_pair_control(self):
        from efficient_llm.warm_order.pair_order_train import train as old_train
        old={k:v for k,v in self.cfg.items() if k not in ('factor','terminal_lr')}
        old.update(name='C-s0',schedule='paired-body-router-warmup-cosine',router_lr=.01)
        new={**self.cfg,'factor':'FH','name':'FH-s0'}
        a,oa=old_train(old,self.root/'original','cpu')
        b,ob=train(new,self.root/'factorial','cpu')
        self.assert_same(a.state_dict(),b.state_dict());self.assert_same(oa.state_dict(),ob.state_dict())

    def test_resume_rejects_changed_recipe_data_and_checkpoint(self):
        out=self.root/'partial';train(self.cfg,out,'cpu',stop_after=1)
        for field,value in [('factor','FL'),('terminal_lr',2e-5)]:
            with self.assertRaises(ValueError):train({**self.cfg,field:value},out,'cpu',resume=True)
        p=out/'fullstate.pt';p.write_bytes(p.read_bytes()+b'bad')
        with self.assertRaises(ValueError):train(self.cfg,out,'cpu',resume=True)
