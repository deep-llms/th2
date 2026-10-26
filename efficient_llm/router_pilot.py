"""Frozen teacher → native HF Trainer for two router banks → own-trajectory LM eval."""
import argparse
import bisect
import json
import math
import os
from pathlib import Path
import time

from efficient_llm.diagnostic import load_probe, load_model, evaluate, paired_summary, write
from efficient_llm.diagnostic_data import sha256
from efficient_llm.group_selection import subset_masks
from efficient_llm.routers import TeacherFeatures, RouterPair, RouterSelection, save_routers, load_routers
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, default_data_collator


class FeatureDataset(Dataset):
    def __init__(self, paths):
        self.shards = [torch.load(p, map_location="cpu", weights_only=True, mmap=True) for p in paths]
        self.ends, total = [], 0
        for s in self.shards:
            n = len(s['features'])
            if len(s['costs']) != n or len(s['example_ids']) != n:
                raise ValueError("Feature shard row mismatch")
            total += n; self.ends.append(total)
        ids = torch.cat([s['example_ids'] for s in self.shards])
        self.order = ids.argsort()
        if not torch.equal(ids[self.order], torch.arange(total)):
            raise ValueError("Missing or duplicated teacher examples")

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        flat = int(self.order[index])
        shard = bisect.bisect_right(self.ends, flat)
        row = flat - (self.ends[shard-1] if shard else 0)
        return {k: self.shards[shard][k][row] for k in ('features', 'costs')}


@torch.no_grad()
def collect(model, tokens, directory, keep, stride, rank, world, batch_size=2, teacher=None):
    features, costs, example_ids = [], [], []
    indices = list(range(rank, len(tokens), world))
    positions = (tokens.shape[1]-1)//stride
    if positions < 1 or not indices:
        raise ValueError("Each rank needs teacher examples")
    device = next(model.parameters()).device
    with (teacher if teacher is not None else TeacherFeatures(model, keep, stride)) as teacher:
        for start in range(0, len(indices), batch_size):
            ids = indices[start:start+batch_size]
            model.model(tokens[ids, :-1].to(device), use_cache=False)
            x, errors = teacher.take()
            if not torch.isfinite(x).all() or not torch.isfinite(errors).all():
                raise ValueError("Nonfinite teacher examples")
            features.append(x.cpu()); costs.append(errors.cpu())
            example_ids.append((torch.tensor(ids)[:,None]*positions + torch.arange(positions)).flatten())
    shard = {'features':torch.cat(features), 'costs':torch.cat(costs), 'example_ids':torch.cat(example_ids)}
    path = directory/f'rank-{rank}.pt'
    torch.save(shard, path)
    # A single common scale per layer for both targets; estimated on training only.
    values = shard['costs'].to(device)
    scale_sum = values[:,:,0].std(-1, unbiased=False).double().sum(0)
    count = torch.tensor(len(values), device=device, dtype=torch.float64)
    cost_sum = values.double().sum(0)
    if world > 1:
        for value in (scale_sum, count, cost_sum): dist.all_reduce(value)
    scales = (scale_sum/count).float().clamp_min(1e-8)
    return path, scales, cost_sum, int(count.item())


@torch.no_grad()
def supervised_metrics(routers, shard, world):
    device = routers.weight.device
    totals = torch.zeros(2, 5, dtype=torch.float64, device=device)
    for start in range(0, len(shard['features']), 256):
        x, costs = (shard[k][start:start+256].to(device) for k in ('features','costs'))
        scores, q = routers.scores(x), routers.targets(costs)
        ce = -(q*scores.log_softmax(-1)).sum(-1)
        entropy = -(q*q.clamp_min(1e-30).log()).sum(-1)
        choice = scores.argmax(-1)
        target = costs.permute(0,2,1,3).argmin(-1)
        regret = costs.permute(0,2,1,3).gather(-1, choice[...,None]).squeeze(-1) - costs.permute(0,2,1,3).min(-1).values
        normalized = regret / routers.scales[None,None,:]
        totals += torch.stack((ce.double().sum((0,2)), entropy.double().sum((0,2)),
            (choice==target).double().sum((0,2)), normalized.double().sum((0,2)),
            torch.full((2,), ce.shape[0]*ce.shape[2], device=device, dtype=torch.float64)), dim=1)
    if world > 1: dist.all_reduce(totals)
    output = {}
    for name, row in zip(('joint','independent'), totals.cpu().tolist()):
        ce, ent, agree, regret, n = row
        output[name] = dict(cross_entropy=ce/n, target_entropy=ent/n, kl= (ce-ent)/n,
                           top1_agreement=agree/n, normalized_cost_regret=regret/n, layer_positions=int(n))
    return output


def regret_report(stats):
    rows = []
    for layer, data in sorted(stats.items()):
        actual, oracle, energy, correct, count = data['totals'].cpu().tolist()
        if energy <= 0 or count <= 0:
            raise ValueError("Missing routed regret data")
        rows.append(dict(layer=layer, positions=int(count), error_over_energy=actual/energy,
                         oracle_error_over_energy=oracle/energy, regret_over_energy=(actual-oracle)/energy,
                         top1_joint_agreement=correct/count))
    return rows


def fit_routers(model, tokens, *, output, probe, keep, stride, steps, feature_batch, rank, world, teacher=None):
    """Shared native Trainer recipe for frozen teacher features; no backbone updates."""
    device = next(model.parameters()).device
    start=time.monotonic()
    path,scales,cost_sum,n=collect(model,tokens['calibration'],output/'features',keep,stride,rank,world,teacher=teacher)
    if world>1: dist.barrier()
    dataset=FeatureDataset([output/'features'/f'rank-{i}.pt' for i in range(world)])
    if n!=len(dataset) or n%(world*feature_batch):
        raise ValueError('Feature count must divide into complete global batches')
    routers=RouterPair(model.config.num_hidden_layers,model.config.hidden_size,
        model.config.num_key_value_heads,keep,scales.cpu()).to(device)
    if not torch.equal(routers.weight[0],routers.weight[1]): raise ValueError('Initial banks differ')
    shard=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    before=supervised_metrics(routers,shard,world)
    collection_seconds=time.monotonic()-start
    if rank==0: print('TEACHER_READY',json.dumps({'examples':n,'seconds':collection_seconds,'initial':before}),flush=True)
    args=TrainingArguments(output_dir=str(output/'trainer'),per_device_train_batch_size=feature_batch,
        max_steps=steps,learning_rate=1e-3,lr_scheduler_type='cosine',warmup_steps=max(1,round(.05*steps)),
        optim='adamw_torch',weight_decay=.01,max_grad_norm=0.,gradient_accumulation_steps=1,
        bf16=device.type=='cuda',use_cpu=device.type=='cpu',seed=42,data_seed=42,report_to='none',
        dataloader_drop_last=True,dataloader_num_workers=0,dataloader_pin_memory=device.type=='cuda',
        save_steps=max(1,steps//2),logging_steps=max(1,steps//8),logging_first_step=True,
        ddp_find_unused_parameters=False,ddp_broadcast_buffers=False,disable_tqdm=True)
    trainer=Trainer(model=routers,args=args,train_dataset=dataset,data_collator=default_data_collator)
    if trainer.model_accepts_loss_kwargs: raise ValueError('Unexpected loss denominator injection')
    result=trainer.train()
    if trainer.state.global_step!=steps: raise ValueError('Incomplete router fit')
    routers=trainer.accelerator.unwrap_model(trainer.model).eval()
    after=supervised_metrics(routers,shard,world)
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise ValueError('Frozen backbone received gradients')
    if rank==0:
        save_routers(routers,output/'routers.pt')
        write(output/'training.json',dict(status='passed',steps=steps,world_size=world,
            training_context_tokens=len(tokens['calibration'])*(tokens['calibration'].shape[1]-1),
            feature_examples=n,feature_passes=steps*feature_batch*world/n,
            scales=scales.cpu().tolist(),temperature=routers.temperature,router_spec=routers.spec,
            initial=before,final=after,trainer_metrics=result.metrics,trainer_args=args.to_dict(),
            probe_manifest_sha256=sha256(probe/'manifest.json'),routers_sha256=sha256(output/'routers.pt'),
            feature_shards=[{'name':f'rank-{i}.pt','sha256':sha256(output/'features'/f'rank-{i}.pt')} for i in range(world)]))
    if world>1: dist.barrier()
    loaded=load_routers(output/'routers.pt',device)
    for key,value in routers.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key],value,rtol=0,atol=0)
    return loaded, cost_sum


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True); p.add_argument('--probe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--keep',type=int,default=6); p.add_argument('--stride',type=int,default=16)
    p.add_argument('--steps',type=int,default=512); p.add_argument('--feature-batch',type=int,default=256)
    a=p.parse_args()
    rank,world=int(os.environ.get('RANK',0)),int(os.environ.get('WORLD_SIZE',1))
    device=torch.device('cuda',int(os.environ.get('LOCAL_RANK',0))) if a.device=='cuda' else torch.device('cpu')
    if device.type=='cuda': torch.cuda.set_device(device)
    if world>1: dist.init_process_group('nccl' if device.type=='cuda' else 'gloo')
    try:
        if min(a.stride,a.steps,a.feature_batch)<1:
            raise ValueError('Invalid training size')
        if rank==0:
            a.output.mkdir(parents=True,exist_ok=False); (a.output/'features').mkdir()
        if world>1: dist.barrier()
        manifest,tokens=load_probe(a.probe)
        if rank==0:
            for name,digest in manifest['model_files'].items():
                if sha256(a.model/name)!=digest: raise ValueError('Model changed: '+name)
        if world>1: dist.barrier()
        torch.manual_seed(42); torch.set_float32_matmul_precision('highest')
        model=load_model(a.model,device,torch.bfloat16 if device.type=='cuda' else torch.float32)
        loaded, cost_sum = fit_routers(model,tokens,output=a.output,probe=a.probe,keep=a.keep,
            stride=a.stride,steps=a.steps,feature_batch=a.feature_batch,rank=rank,world=world)
        from efficient_llm.router_check import cached_check
        cached = cached_check(model.float(), loaded, tokens['evaluation'][:1,:24].to(device))
        if device.type=='cuda': model.bfloat16()
        report=dict(status='running',world_size=world,keep=a.keep,methods={},regret={},
                    trained_router_cache_check=cached,
                    routers_sha256=sha256(a.output/'routers.pt'),probe_manifest_sha256=sha256(a.probe/'manifest.json'))
        choices=cost_sum[:,0].argmin(-1)
        static=subset_masks(model.config.num_key_value_heads,a.keep,device=device)[choices]
        for policy in ('dense','static','raw_norm','projected_norm','joint','joint_router','independent_router'):
            selector=RouterSelection(model,loaded,0 if policy=='joint_router' else 1,collect=True,stride=a.stride) if policy.endswith('_router') else None
            metrics,stats=evaluate(model,tokens['evaluation'],keep=a.keep,policy=policy,
                static_masks=static,selection=selector,rank=rank,world=world)
            report['methods'][policy]=metrics
            if selector is not None: report['regret'][policy]=regret_report(stats)
            if rank==0:
                print('ROUTER_EVAL',policy,json.dumps({k:v for k,v in metrics.items() if k!='per_document_nll'}),flush=True)
                write(a.output/'evaluation.partial.json',report)
        report['comparisons']={ref:paired_summary(report['methods']['joint_router']['per_document_nll'],
            report['methods'][ref]['per_document_nll']) for ref in ('dense','independent_router','joint','projected_norm')}
        report['status']='passed'
        if rank==0:
            write(a.output/'evaluation.json',report)
            print('ROUTER_PILOT_COMPLETE',flush=True)
    finally:
        if dist.is_initialized(): dist.destroy_process_group()


if __name__=='__main__': main()
