"""Offline launch gates for a fresh FA4 A matched to a completed dense A."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False)


def validate(arm,reference,steps,output):
    """CPU-only artifact, exact recipe/data matching and actual-backend checks."""
    arm,reference=Path(arm),Path(reference)
    result,state,config=(read(arm/name) for name in ('result.json','trainer_state.json','train_config.json'))
    old=read(reference/'train_config.json')
    assert config['pilot']['attention_backend']=='fa4'
    matched=copy.deepcopy(config);matched['pilot'].pop('attention_backend')
    if steps==3:matched['training']['logging_steps']=old['training']['logging_steps']
    differences={k:[old.get(k),matched.get(k)] for k in old.keys()|matched.keys() if old.get(k)!=matched.get(k)}
    assert not differences, differences
    assert result['arm']=='A' and result['status']=='stopped'
    assert state['global_step']==result['global_step']==steps and state['max_steps']==result['schedule_steps']==28600
    assert result['input_tokens']==steps*1048576
    assert result['evaluation']==read(arm/'eval_results.json')
    assert math.isfinite(result['evaluation']['eval_lm_loss'])
    assert (arm/'model.safetensors').stat().st_size>0
    checkpoint=arm/f'checkpoint-{steps}'
    for name in ['optimizer.pt','scheduler.pt','model.safetensors','trainer_state.json',*[f'rng_state_{i}.pth' for i in range(8)]]:
        assert (checkpoint/name).stat().st_size>0,name
    assert read(checkpoint/'trainer_state.json')['global_step']==steps
    logs=[row for row in state['log_history'] if 'loss' in row]
    assert logs and all(math.isfinite(row['loss']) and math.isfinite(row['grad_norm']) for row in logs)
    if steps==3:assert len(logs)==3
    assert not result['sdpa_receipts']
    assert result['attention_runtime']['backend']=='fa4' and result['attention_runtime']['version']=='4.0.0b33'
    receipts=[read(arm/name) for name in result['fa4_receipts']]
    assert {r['phase'] for r in receipts}=={'train','eval'}
    for receipt in receipts:
        assert receipt['calls']==28 and receipt['world_size']==8
        assert receipt['kernel_dtype']=='torch.bfloat16'
        assert receipt['document_isolation'] and receipt['causal'] and not receipt['dense_mask']
        assert receipt['positions']=='reset_per_document'
        if receipt['phase']=='train':assert receipt['query_gradient_calls']==28
    peak=result['training_cost']['peak_cuda_allocated_bytes']
    assert peak is not None and peak<160*2**30,peak
    write(output,dict(status='passed',steps=steps,reference=str(reference),
        config_matches_except_attention=True,peak_cuda_allocated_bytes=peak,
        eval_lm_loss=result['evaluation']['eval_lm_loss'],fa4_receipts=result['fa4_receipts']))
    print('MATCHED_FA4_BASELINE_VALIDATED',steps,flush=True)


def numerics(recipe_path,output):
    """Real full-model CUDA kernels, same weights and real packed text, no updates."""
    import torch
    from transformers import AutoConfig,AutoTokenizer,TrainingArguments
    from deep_kv.proxy import ProxyModel
    from deep_kv.proxy_training import ProxyTrainer
    from deep_kv.packing import preprocess_dataset,isolated_data_collator
    from deep_kv.fa4 import load_kernel
    from scripts.benchmark_document_training import compare
    from train import load_text
    torch.set_num_threads(4)
    recipe=read(recipe_path)
    cfg=AutoConfig.from_pretrained(recipe['config_name'],local_files_only=True)
    cfg._attn_implementation='sdpa';cfg.use_cache=False
    tokenizer=AutoTokenizer.from_pretrained(recipe['tokenizer_name'],local_files_only=True)
    raw=load_text(recipe['data_dir']);raw=raw.select(range(min(512,len(raw))))
    args=TrainingArguments(output_dir=str(Path(output).parent),use_cpu=True,report_to=[])
    data=preprocess_dataset(raw,tokenizer,2048,args,num_proc=1,isolate_documents=True)
    rows=[]
    for row in data:
        if len(set(row['segments']))>1:rows.append(row)
        if len(rows)==2:break
    assert len(rows)==2
    inputs={k:v.to('cuda') for k,v in isolated_data_collator(rows).items()}
    ctx=ProxyTrainer.context(inputs)
    model=ProxyModel.from_scratch(cfg,'A',seed=42,checkpoint_layers=False,checkpoint_lm=False,
                                 checkpoint_aux=False,lm_chunk=128).cuda().train()
    model.fa4_kernel,metadata=load_kernel()
    def capture(backend):
        model.attention_backend=backend;model.zero_grad(set_to_none=True);saved={}
        def hook(module,args,out):
            saved['hidden']=out.detach().float().cpu()
            saved['logits']=model.backbone.lm_head(out[:,::128]).detach().float().cpu()
        handle=model.backbone.model.norm.register_forward_hook(hook)
        try:
            with torch.autocast('cuda',dtype=torch.bfloat16):
                result=model(ctx);loss=result['lm_sum']/result['lm_count']
            loss.backward()
        finally:handle.remove()
        gradients={name:p.grad.detach().float().cpu().clone() for name,p in model.named_parameters()}
        assert all(torch.isfinite(g).all() for g in gradients.values())
        return dict(loss=float(loss.detach()),targets=int(result['lm_count']),grads=gradients,**saved)
    dense=capture('sdpa');flash=capture('fa4')
    comparison=compare(dense,flash,True)
    del dense,flash
    model.zero_grad(set_to_none=True)
    # Full-model isolation on document 2: no document-1 embedding gradients.
    one=ProxyTrainer.context({k:v[:1] for k,v in inputs.items()})
    boundary=int((one.segments[0,1:]!=one.segments[0,:-1]).nonzero()[0])+1
    captured=[]
    handle=model.backbone.model.embed_tokens.register_forward_hook(lambda module,args,out:captured.append(out))
    try:
        with torch.autocast('cuda',dtype=torch.bfloat16):hidden=model.hidden_states(one)[0]
        grad=torch.autograd.grad(hidden[:,boundary:].square().sum(),captured[-1])[0]
        assert grad[:,:boundary].abs().max().item()==0
        altered=copy.deepcopy(one);altered.input_ids[:,:boundary]=17
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):changed=model.hidden_states(altered)[0]
        assert torch.equal(hidden[:,boundary:],changed[:,boundary:])
    finally:handle.remove()
    checksum=hashlib.sha256(inputs['input_ids'].cpu().numpy().tobytes()+inputs['segments'].cpu().numpy().tobytes()).hexdigest()
    write(output,dict(status='passed',kernel=metadata,comparison=comparison,cross_document_gradient_max=0.,
                      cross_document_output_max=0.,packed_batch_sha256=checksum,batch_size=2,sequence_length=2048))
    print('PRODUCTION_FA4_NUMERICS_PASSED',json.dumps(comparison),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    n=sub.add_parser('numerics');n.add_argument('--recipe',required=True);n.add_argument('--output',required=True)
    v=sub.add_parser('validate');v.add_argument('--arm',required=True);v.add_argument('--reference',required=True)
    v.add_argument('--steps',required=True,type=int);v.add_argument('--output',required=True)
    args=p.parse_args()
    if args.mode=='numerics':numerics(args.recipe,args.output)
    else:validate(args.arm,args.reference,args.steps,args.output)


if __name__=='__main__':main()
