"""Disposable full-Qwen FA4 proxy checks and validation of real Trainer smokes.

Uses existing packed diagnostic rows for numerical probes, and the normal full
training dataset/cache for Trainer jobs. No custom optimizer or training loop.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from deep_kv import PROXY_ARMS, ALL_PROXY_ARMS, MEMORY_ARMS, SIMPLE_MEMORY_ARMS, simple_memory_layout


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as handle:json.dump(value,handle,indent=2,allow_nan=False)


def worker(args):
    import torch
    from transformers import AutoConfig
    from deep_kv.proxy import ProxyModel
    from deep_kv.proxy_training import ProxyTrainer
    from deep_kv.packing import isolated_data_collator
    from scripts.benchmark_document_training import compare
    from scripts.check_trained_attention import acceptable, fingerprint
    from scripts.check_proxy_trained_attention import file_hash
    torch.set_num_threads(2);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    arm=args.arm if args.arm is not None else PROXY_ARMS[args.index];recipe=read(args.recipe)
    cfg=AutoConfig.from_pretrained(recipe['config_name'],local_files_only=True)
    cfg._attn_implementation='sdpa';cfg.use_cache=False
    m=ProxyModel.from_scratch(cfg,arm,attention_backend='fa4',seed=42,
        checkpoint_layers=False,checkpoint_lm=False,checkpoint_aux=False,lm_chunk=128).cuda().train()
    rows=read(args.rows)[:2]
    assert len(rows)==2 and all(len(r['input_ids'])==2048 for r in rows)
    assert len(set(rows[0]['segments']))>1
    ctx=ProxyTrainer.context({k:v.cuda() for k,v in isolated_data_collator(rows).items()})
    # Nonzero gates exercise proxy K/V injection and its gradients, not only the
    # vanilla-equivalent zero-gate initialization. Bootstrap on one fixed backend.
    with torch.no_grad():
        if not m.anticipatory and not m.memory_proxy:
            for head in m.heads.values():head.alpha.normal_(0,.05)
        if m.family:
            for mode in ('mean','variance'):
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    out=m(ctx,compute_auxiliary_losses=False,auxiliary_grad=False,
                          collect_target_statistics=True,statistics_mode=mode)
                m.update_statistics(out['center_sums'],out['center_squares'],out['center_counts'],initialize=mode)
    initial=fingerprint(m)
    buffers={k:v.clone() for k,v in m.named_buffers()}
    def capture(backend):
        print('CAPTURE',arm,backend,flush=True)
        m.attention_backend=backend;m.zero_grad(set_to_none=True);saved={}
        def hook(module,inputs,out):
            saved['hidden']=out.detach().float().cpu()
            saved['logits']=m.backbone.lm_head(out[:,::128]).detach().float().cpu()
        handle=m.backbone.model.norm.register_forward_hook(hook)
        try:
            with torch.autocast('cuda',dtype=torch.bfloat16):
                out=m(ctx,auxiliary_grad=not arm.endswith('lambda0'),collect_target_statistics=True)
                lm=out['lm_sum']/out['lm_count'];aux=out['aux_sum']/out['aux_count'].clamp_min(1)
                if m.relational_proxy:aux=aux+.5*out['rel_sum']/out['rel_count'].clamp_min(1)
                loss=lm+m.auxiliary_weight(250)*aux
            loss.backward()
        finally:handle.remove()
        grads={n:p.grad.detach().float().cpu().clone() for n,p in m.named_parameters() if p.grad is not None}
        assert len(grads)==len(list(m.named_parameters()))
        assert all(torch.isfinite(g).all() for g in grads.values()) and torch.isfinite(loss)
        for name,value in m.named_buffers():assert torch.equal(value,buffers[name]),name
        return dict(loss=float(loss.detach()),lm_loss=float(lm.detach()),aux_loss=float(aux.detach()),
                    targets=int(out['lm_count']),grads=grads,**saved)
    dense,flash=capture('sdpa'),capture('fa4')
    comparison=compare(dense,flash,False)
    proxy_comparison=None
    if m.family:
        # The much larger backbone must not hide a proxy-only gradient error.
        proxy_comparison=compare({**dense,'grads':{n:g for n,g in dense['grads'].items() if n.startswith('heads.')}},
                                 {**flash,'grads':{n:g for n,g in flash['grads'].items() if n.startswith('heads.')}},False)
        assert all(g.abs().sum()>0 for n,g in flash['grads'].items() if n.startswith('heads.'))
    losses={k:[dense[k],flash[k]] for k in ('lm_loss','aux_loss')}
    del dense,flash
    # End-to-end document isolation includes P3's EMS and nonzero proxy gates.
    m.zero_grad(set_to_none=True)
    one=copy.deepcopy(ctx)
    for field in ('input_ids','valid','position_ids','segments','labels'):
        setattr(one,field,getattr(ctx,field)[:1])
    boundary=int((one.segments[0,1:]!=one.segments[0,:-1]).nonzero()[0])+1
    captured=[]
    handle=m.backbone.model.embed_tokens.register_forward_hook(lambda mod,inp,out:captured.append(out))
    try:
        with torch.autocast('cuda',dtype=torch.bfloat16):hidden=m.hidden_states(one)[0]
        grad=torch.autograd.grad(hidden[:,boundary:].float().square().sum(),captured[-1])[0]
        forbidden=float(grad[:,:boundary].abs().max())
        changed=copy.deepcopy(one);changed.input_ids=one.input_ids.clone();changed.input_ids[:,:boundary]=17
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):after=m.hidden_states(changed)[0]
        difference=float((hidden[:,boundary:]-after[:,boundary:]).abs().max())
    finally:handle.remove()
    passed=(acceptable(comparison) and (proxy_comparison is None or acceptable(proxy_comparison))
            and all(abs(a-b)<.01 for a,b in losses.values()) and forbidden==difference==0)
    unchanged=fingerprint(m)==initial
    result=dict(status='passed' if passed and unchanged else 'failed',arm=arm,index=args.index,
        kernel=m.attention_runtime,rows_sha256=file_hash(args.rows),batch=2,sequence_length=2048,
        active_gate_std=None if m.anticipatory or m.memory_proxy else .05,
        active_gate_init=m.settings.alpha_init if m.anticipatory else None,
        auxiliary_weight=m.auxiliary_weight(250),comparison=comparison,
        proxy_comparison=proxy_comparison,losses=losses,parameters_unchanged=unchanged,
        cross_document_gradient_max=forbidden,cross_document_output_max=difference,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    write(args.output,result)
    print('PROXY_FA4_NUMERICS',json.dumps(result),flush=True)
    assert result['status']=='passed','Numerical/isolation check failed; evidence saved'


def run(args):
    root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    visible=os.environ['CUDA_VISIBLE_DEVICES'].split(',');assert len(visible)==8
    children=[];logs=[]
    try:
        for index,gpu in enumerate(visible):
            log=(root/f'case-{index}.log').open('x');logs.append(log)
            command=[sys.executable,'-u','-m','scripts.check_fa4_proxy','worker','--recipe',args.recipe,
                     '--rows',args.rows,'--index',str(index),'--output',str(root/f'case-{index}.json')]
            children.append(subprocess.Popen(command,env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu},stdout=log,stderr=subprocess.STDOUT))
        codes=[p.wait() for p in children]
    finally:
        for p in children:
            if p.poll() is None:p.terminate()
        for p in children:
            try:p.wait(timeout=20)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for log in logs:log.close()
    results=[read(root/f'case-{i}.json') for i in range(8) if (root/f'case-{i}.json').exists()]
    passed=codes==[0]*8 and len(results)==8 and all(r['status']=='passed' and r['arm']==PROXY_ARMS[i] for i,r in enumerate(results))
    write(root/'summary.json',dict(status='passed' if passed else 'failed',exit_codes=codes,cases=results,
        tolerances=dict(loss_absolute=.01,gradient_relative_l2=.03,proxy_gradient_relative_l2=.03,output_relative_l2=.02)))
    assert passed,codes


def validate(args):
    """Fail closed on actual backend, full recipe, final state, or normalization errors."""
    from safetensors import safe_open
    root=Path(args.run_dir);results=[]
    assert 1 <= args.steps < 28600 and args.arms and len(set(args.arms))==len(args.arms)
    for arm in args.arms:
        folder=root/f'seed-{args.seed}'/arm
        result,state,config=(read(folder/name) for name in ('result.json','trainer_state.json','train_config.json'))
        assert result['arm']==arm and result['global_step']==state['global_step']==args.steps
        assert result['status']=='stopped' and result['schedule_steps']==state['max_steps']==28600
        assert result['input_tokens']==args.steps*1048576
        backend=getattr(args,'attention_backend','fa4')
        assert config['pilot'].get('attention_backend','sdpa')==backend and config['data']['isolate_documents']
        assert config['training']['per_device_train_batch_size']==16 and config['training']['gradient_accumulation_steps']==4
        assert config['training']['warmup_steps']==1430
        assert not any(config['pilot'][key] for key in ('checkpoint_layers','checkpoint_lm','checkpoint_aux'))
        if backend=='fa4':
            assert result['attention_runtime']['version']=='4.0.0b33' and not result['sdpa_receipts']
        logs=[row for row in state['log_history'] if 'loss' in row]
        interval=int(config['training']['logging_steps'])
        assert interval>=1 and interval==config['training']['logging_steps']
        expected=set(range(interval,args.steps+1,interval))
        observed={row['step'] for row in logs}
        assert logs and expected.issubset(observed)
        # A forced cutoff need not coincide with a scheduled loss log. The
        # final step is independently checked in result/state/checkpoint above.
        assert logs[-1]['step'] in {args.steps,args.steps//interval*interval}
        assert all(math.isfinite(row['loss']) and math.isfinite(row['grad_norm']) for row in logs)
        assert config['training']['seed']==config['training']['data_seed']==args.seed
        assert config['world_size']==8
        assert result['evaluation']['eval_rows']==config['data']['eval_rows']
        assert math.isfinite(result['evaluation']['eval_lm_loss'])
        receipt_names=result[backend+'_receipts']
        receipts=[read(folder/name) for name in receipt_names]
        assert {r['phase'] for r in receipts}=={'train','eval'}
        memory_layers = simple_memory_layout(arm,28)[0] if arm in SIMPLE_MEMORY_ARMS else tuple(range(2,25,2))
        attention_calls=28+len(memory_layers) if arm in MEMORY_ARMS else 28
        for r in receipts:
            assert r['arm']==arm and r['world_size']==8
            if backend=='fa4':
                assert r['calls']==attention_calls
                assert r['kernel_dtype']=='torch.bfloat16' and r['positions']=='reset_per_document'
                assert r['document_isolation'] and r['causal'] and not r['dense_mask']
                if r['phase']=='train':assert all(r[k+'_gradient_calls']==attention_calls for k in ('query','key','value'))
            else:
                assert len(r['calls'])==attention_calls and not r['has_math']
                assert not r['unattributed_calls'] and not r['unattributed_backward']
                assert all(c['mask']['dtype']=='torch.bool' and not c['is_causal'] for c in r['calls'])
                assert not any(r['checkpointing'].values())
                if r['phase']=='train':assert all(c['backward_operators'] for c in r['calls'])
        checkpoint=folder/f'checkpoint-{args.steps}'
        for name in ['model.safetensors','optimizer.pt','scheduler.pt','trainer_state.json',*[f'rng_state_{i}.pth' for i in range(8)]]:
            assert (checkpoint/name).stat().st_size>0,name
        assert read(checkpoint/'trainer_state.json')['global_step']==args.steps
        with safe_open(checkpoint/'model.safetensors',framework='pt',device='cpu') as weights:
            import torch
            assert bool(weights.get_tensor('mu_initialized'))
            if arm.startswith('P'):
                for name in ('mu','sigma2'):
                    value=weights.get_tensor(name)
                    assert torch.isfinite(value).all() and value.abs().sum()>0
                assert (weights.get_tensor('sigma2')>=0).all()
                gates=[weights.get_tensor(n) for n in weights.keys() if n.startswith('heads.') and n.endswith('.alpha')]
                if arm in MEMORY_ARMS:
                    assert not gates and config['pilot']['proxy_target_version']==('p4p6-r1' if arm in SIMPLE_MEMORY_ARMS else 'p7-r1')
                    for layer in memory_layers:
                        estimator=('w1.weight','w2.weight') if arm in SIMPLE_MEMORY_ARMS else ('conv','w_in.weight','w_out.weight')
                        for suffix in estimator+('k_proj.weight','k_norm.weight')+(() if arm=='P7-kq' else ('v_proj.weight',)):
                            value=weights.get_tensor(f'heads.{layer}.{suffix}')
                            assert torch.isfinite(value).all() and value.abs().sum()>0
                        mass=result['evaluation'][f'eval_proxy_layer_{layer}_attention_mass']
                        assert math.isfinite(mass) and 0<=mass<=1
                    assert all(math.isfinite(result['evaluation'][key]) for key in (('eval_cos_loss','eval_aux_loss') if arm in SIMPLE_MEMORY_ARMS else ('eval_cos_loss','eval_rel_loss','eval_aux_loss')))
                else:
                    assert gates and all(torch.isfinite(g).all() and g.abs().sum()>0 for g in gates)
        peak=result['training_cost']['peak_cuda_allocated_bytes'];assert peak<160*2**30
        results.append(dict(arm=arm,peak_gib=peak/2**30,eval_lm_loss=result['evaluation']['eval_lm_loss'],receipts=receipt_names))
    write(args.output,dict(status='passed',steps=args.steps,arms=results))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('run','worker','validate'))
    p.add_argument('--recipe');p.add_argument('--rows');p.add_argument('--run-dir')
    p.add_argument('--output',required=True);p.add_argument('--index',type=int,choices=range(8))
    p.add_argument('--arm',choices=ALL_PROXY_ARMS,help='Explicit arm for a numerical worker')
    p.add_argument('--steps',type=int,default=3);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--arms',nargs='+',choices=('A',)+ALL_PROXY_ARMS,default=('A',)+PROXY_ARMS)
    p.add_argument('--attention-backend',choices=('fa4','sdpa'),default='fa4',help='Backend expected by validate mode')
    args=p.parse_args();globals()[args.mode](args)


if __name__=='__main__':main()
