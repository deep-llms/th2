"""v3 scientific options, compatibility, queue and retention (CPU, offline)."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import AutoConfig
from bdot.model import BDOTForCausalLM, RouteContext
from bdot.routing import RoutingConfig, quota_table
from bdot.trainer import BDOTTrainer, MilestoneCheckpointCallback
from bdot import grouped
from train import BaselineTrainingArguments
from test_model import model, batch
from test_grouped import reference_mm, enable_grouped

ROOT = Path(__file__).resolve().parents[2]


class V3Tests(unittest.TestCase):
    def test_predeclared_selection_and_two_seed_gate(self):
        from scripts.bdot_v3_report import select_candidate,gate
        row=dict(loss=3.,spike=False,off_default=.2,default_gain=.02,throughput_ratio=.8)
        rows={arm:dict(row) for arm in ['V1','V2','V3','V4','V5','V6','V7']}
        rows['V5']['loss']=1. # Exploratory arm never enters the official selection.
        rows['V3']['loss']=2.995
        self.assertEqual(select_candidate(rows),('V2',False))
        rows['V3']['loss']=2.98
        self.assertEqual(select_candidate(rows),('V3',False))
        rows['V1']['loss']=2.98
        self.assertEqual(select_candidate(rows),('V1',True))
        self.assertEqual(gate([row],[dict(loss=3.02)],[dict(loss=3.)])['status'],
                         'pending_two_seed_confirmation')
        self.assertEqual(gate([row,row],[dict(loss=3.02)]*2,[dict(loss=3.)]*2)['status'],'pass')
        self.assertEqual(gate([dict(row,spike=True),row],[dict(loss=3.02)]*2,[dict(loss=3.)]*2)['status'],'fail')

    def test_cap_full_logit_and_gradients(self):
        m=model(logit_cap=2.,select_on='probs',curriculum='none')
        r=m.routed_steps[1].router
        h=torch.randn(5,16,requires_grad=True); visits=torch.randn(5,4)
        allowed=torch.tensor([[True,True,True,False]]*5)
        with torch.no_grad():
            r.visits.fill_(.4); r.bias.copy_(torch.tensor([1.,-2.,3.,4.]))
        z=r(h,visits,allowed)
        x=h.float(); x=x*torch.rsqrt(x.square().mean(-1,keepdim=True)+r.norm.variance_epsilon)
        raw=torch.nn.functional.linear(x*r.norm.weight,r.weight,r.bias)+visits@r.visits.T
        expected=(2*torch.tanh(raw/2)).masked_fill(~allowed,-torch.inf)
        torch.testing.assert_close(z,expected,atol=0,rtol=0)
        self.assertTrue((z[allowed].abs()<2).all())
        z[allowed].sum().backward()
        self.assertTrue(torch.isfinite(h.grad).all());self.assertGreater(float(h.grad.abs().sum()),0)
        with torch.no_grad():r.bias.fill_(1e6)
        self.assertTrue((r(h,visits,allowed)[allowed].abs()<=2).all())
        for options in [dict(logit_cap=0),dict(logit_cap=float('nan')),dict(select_on='bad'),
                        dict(placement_film_scope='query')]:
            with self.assertRaises(ValueError):RoutingConfig(**options)

    def test_probability_selection_and_impossible_candidates(self):
        m=model(curriculum='none',select_on='probs'); step=m.routed_steps[0]
        h=torch.zeros(1,1,16)
        def context(beta):
            return RouteContext(torch.zeros(1,4),torch.zeros(1,dtype=torch.long),
                torch.Generator().manual_seed(0),1,False,0.,torch.tensor(beta),[],[],[],[])
        # Raw-logit selection would choose 0; p+beta must choose 1. A huge
        # disallowed bias must never select 2/3 (masked p alone is insufficient).
        with torch.no_grad():
            step.router.weight.zero_();step.router.bias.copy_(torch.tensor([2.,1.,0.,0.]))
        choice,gate=step.select(h,context([0.,.5,100.,100.]))
        self.assertEqual(choice.item(),1);self.assertEqual(gate.item(),1)
        with torch.no_grad():step.router.bias.zero_()
        self.assertEqual(step.select(h,context([0.,0.,100.,100.]))[0].item(),0)

    def test_pf_counts_identity_home_and_optimizer_group(self):
        for reach,count in [(1,86016),(2,159744)]:
            cfg=AutoConfig.from_pretrained(ROOT/'configs/bdot/V2.json',local_files_only=True)
            cfg.bdot['r']=reach;cfg._attn_implementation='sdpa'
            with torch.device('meta'):m=BDOTForCausalLM(cfg)
            self.assertEqual(sum(p.numel() for n,p in m.named_parameters() if 'placement_film' in n),count)
        a=model(curriculum='none'); b=model(curriculum='none',placement_film=True)
        common=dict(a.named_parameters())
        for name,p in b.named_parameters():
            if name in common:torch.testing.assert_close(p,common[name],atol=0,rtol=0)
        torch.testing.assert_close(a(**batch()).logits,b(**batch()).logits,atol=0,rtol=0)
        for step in b.routed_steps:
            for p in step.placement_film.parameters():p.data.fill_(.3)
        b.eval(); a.eval(); b.routing_override=a.routing_override='fixed'
        torch.testing.assert_close(a(**batch()).logits,b(**batch()).logits,atol=0,rtol=0)
        with tempfile.TemporaryDirectory() as tmp:
            t=BDOTTrainer(model=b,args=BaselineTrainingArguments(output_dir=tmp,use_cpu=True,
                report_to='none',max_steps=10,learning_rate=.001,weight_decay=.1,remove_unused_columns=False))
            t.create_optimizer()
            pf={id(p) for step in b.routed_steps for p in step.placement_film.parameters()}
            for group in t.optimizer.param_groups:
                if any(id(p) in pf for p in group['params']):
                    self.assertEqual(group['weight_decay'],0.);self.assertEqual(group['lr'],.001)

    def test_pf_gradients_only_used_pairs_and_placement_formulas(self):
        m=model(curriculum='none',placement_film=True,logit_cap=2.,select_on='probs')
        paths=torch.tensor([1,0,2,3])[:,None,None].expand(4,2,6)
        m(**batch(),bdot_forced_routes=paths).loss.backward()
        for s,step in enumerate(m.routed_steps):
            for key,pair in step.placement_film.items():
                active=(s,int(key)) in [(0,1),(1,0)]
                for p in pair.parameters():
                    if active:self.assertIsNotNone(p.grad);self.assertGreater(float(p.grad.abs().sum()),0)
                    else:self.assertIsNone(p.grad)
        pair=m.routed_steps[0].placement_film['1']
        for kind,dim in [('query',32),('mlp',16)]:
            getattr(pair,kind+'_gain').data.fill_(.2)
            getattr(pair,kind+'_bias').data.fill_(.3)
            x=torch.randn(3,dim)
            torch.testing.assert_close(pair(x,kind),x*1.2+.3,atol=0,rtol=0)

    def test_grouped_parity_reload_and_checkpointing_all_scopes(self):
        for scope in ['both','query','mlp']:
            for reuse,reach in [(True,1),(False,2)]:
                with self.subTest(scope=scope,reuse=reuse):
                    a=model(curriculum='none',placement_film=True,placement_film_scope=scope,
                        logit_cap=2.,select_on='probs',reuse=reuse,r=reach,default_prior=0.)
                    for name,p in a.named_parameters():
                        if 'placement_film' in name:p.data.uniform_(-.1,.1)
                    b=enable_grouped(copy.deepcopy(a)); b.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={'use_reentrant':False})
                    a.capture_routes=b.capture_routes=True
                    with patch.object(grouped,'_grouped_mm',side_effect=reference_mm):
                        x=a(**batch());y=b(**batch());x.loss.backward();y.loss.backward()
                    torch.testing.assert_close(x.logits,y.logits,atol=1e-6,rtol=1e-5)
                    torch.testing.assert_close(a.last_routes,b.last_routes,atol=0,rtol=0)
                    for (name,p),(_,q) in zip(a.named_parameters(),b.named_parameters()):
                        if p.grad is None:self.assertIsNone(q.grad,name)
                        else:torch.testing.assert_close(p.grad,q.grad,atol=1e-6,rtol=1e-4,msg=name)
                    for m in [a,b]:
                        torch.optim.AdamW(m.parameters(),lr=.001).step();m.finish_update()
                    if not reuse:self.assertTrue(a.balance_bias.eq(0).all())
                    with tempfile.TemporaryDirectory() as tmp:
                        a.save_pretrained(tmp)
                        loaded=BDOTForCausalLM.from_pretrained(tmp,attn_implementation='sdpa')
                        for n,p in a.state_dict().items():torch.testing.assert_close(p,loaded.state_dict()[n],atol=0,rtol=0)
                        a.eval();loaded.eval()
                        torch.testing.assert_close(a(**batch()).logits,loaded(**batch()).logits,atol=0,rtol=0)

    def test_r2_boundaries_and_exact_paths(self):
        c=RoutingConfig(r=2,reuse=False,curriculum='A')
        for step,reach in [(476,0),(477,1),(953,1),(954,2),(1431,2)]:
            self.assertEqual(c.phase(step/4768,True)[0],reach)
        for step in [477,954]:self.assertGreater(c.phase(step/4768,True)[2],.049)
        self.assertEqual(c.phase(1431/4768,True)[2],0)
        allowed,nxt=quota_table(8,2)
        def count(i):
            return sum(count(int(nxt[i,m])) for m in allowed[i].nonzero().flatten()) if allowed[i].any() else 1
        self.assertEqual(count(0),400)

    def test_logging_definitions(self):
        m=model(routing='fixed',curriculum='none');m(**batch());m.finish_update()
        d=m.routing_diagnostic_metrics()
        self.assertEqual(d['mean_distinct_modules_last_update'],4)
        self.assertAlmostEqual(sum(v for k,v in d.items() if k.startswith('placement_')),1.)
        for s in range(4):
            self.assertEqual(d[f'module_{s}_load_last_update'],.25)
            self.assertEqual(d[f'module_{s}_beta_after_update'],0)

    def test_v3_queue_seeds_scopes_and_isolated_outputs(self):
        from run_experiments import load_jobs
        from scripts.make_bdot_pilot import V3_RUNS
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'plan'
            subprocess.run([sys.executable,str(ROOT/'scripts/make_bdot_pilot.py'),
                '--study','v3','--tokenizer','/assets/tok','--data','/assets/train',
                '--eval-data','/assets/val','--revision','a'*40,'--expert-backend','grouped',
                '--accelerate-config','/assets/accelerate.yaml','--output',str(out)],check=True,capture_output=True)
            jobs=load_jobs(out/'jobs.json');self.assertEqual(len(jobs),26)
            self.assertEqual(len(V3_RUNS),13)
            outputs=[]
            for index,(arm,seed) in enumerate(V3_RUNS):
                name=f'{arm}-s{seed}'
                self.assertEqual(jobs[2*index]['name'],name)
                a=json.loads((out/f'{name}.train.json').read_text())
                c=json.loads((out/f'{name}.model.json').read_text())
                RoutingConfig(**c['bdot'])
                self.assertEqual((a['seed'],c['bdot']['seed'],a['data_seed']),(seed,seed,0))
                self.assertEqual(a['keep_checkpoint_steps'],[1000,2500,4768])
                self.assertEqual(a['max_steps'],4768);self.assertEqual(a['warmup_steps'],96)
                self.assertEqual(a['block_size'],1024)
                outputs.append(jobs[2*index+1]['required_outputs'][0]['path'])
            self.assertEqual(len(set(outputs)),13)

    def test_retention_fails_closed_and_keeps_milestones_latest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); c=MilestoneCheckpointCallback()
            args=SimpleNamespace(output_dir=tmp,should_save=True,world_size=1,keep_checkpoint_steps=[2,4,6])
            state=SimpleNamespace(global_step=5)
            for step in range(1,6):
                p=root/f'checkpoint-{step}';p.mkdir()
                (p/'trainer_state.json').write_text(json.dumps(dict(global_step=step)))
            with self.assertRaisesRegex(RuntimeError,'incomplete'):c.on_save(args,state,None)
            self.assertEqual(len(list(root.glob('checkpoint-*'))),5)
            for name in ['optimizer.pt','scheduler.pt','config.json','train_config.json','data_recipe.json',
                         'training_recipe.json','rng_state.pth','model.safetensors']:
                (root/'checkpoint-5'/name).write_text('fixture')
            c.on_save(args,state,None)
            self.assertEqual(sorted(p.name for p in root.glob('checkpoint-*')),
                             ['checkpoint-2','checkpoint-4','checkpoint-5'])

    def test_pipeline_gates_all_training_and_preserves_recipe(self):
        from scripts.bdot_v3_pipeline import prepare
        from run_experiments import load_jobs
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pilot=root/'pilot';pilot.mkdir();out=root/'v3';out.mkdir()
            subprocess.run([sys.executable,str(ROOT/'scripts/make_bdot_pilot.py'),
                '--tokenizer','/assets/tok','--data','/assets/train','--eval-data','/assets/val',
                '--revision','a'*40,'--expert-backend','grouped',
                '--accelerate-config',str(ROOT/'resources/accelerate_b200.yaml'),
                '--output',str(pilot/'prepared')],check=True,capture_output=True)
            path=prepare(out,pilot,root/'N',root/'R');jobs=load_jobs(path)
            self.assertEqual(len(jobs),28)
            self.assertEqual(jobs[0]['name'],'v3_acceptance')
            self.assertEqual(jobs[-1]['name'],'v3_report')
            self.assertTrue(all(job['gpus']==list(range(8)) for job in jobs[:-1]))
            a=pilot/'prepared/P1-A.train.json';cfg=json.loads(a.read_text())
            cfg['data_seed']=7;a.write_text(json.dumps(cfg))
            another=root/'wrong';another.mkdir()
            with self.assertRaisesRegex(ValueError,'unrelated training settings'):
                prepare(another,pilot,root/'N',root/'R')


if __name__=='__main__':unittest.main()
