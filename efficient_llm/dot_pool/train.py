"""Plain PyTorch/DDP registered training, full-state checkpoints, seeded eval."""
import argparse,contextlib,dataclasses,json,math,os,random,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from .model import Config,Model
from .data import Stream,e1_batch,write,sha
from . import SPEC_SHA

def seed(n):
    random.seed(n);np.random.seed(n);torch.manual_seed(n)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(n)
def rng_state():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
def restore_rng(s):
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch'].cpu())
    if s['cuda']:torch.cuda.set_rng_state_all([x.cpu() for x in s['cuda']])
def autocast(device):return torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else contextlib.nullcontext()
def lr_at(step,total,lr,study):
    warm=1000 if study=='E1' else 200
    if step<=warm:return lr*step/warm
    if study=='E1':return lr
    start=math.floor(total*.8)
    return lr if step<=start else lr*max(0,(total-step)/(step-start))
def save(root,m,opt,step,cfg,rank,world):
    root=Path(root)
    if rank==0:root.mkdir(parents=True,exist_ok=False)
    if world>1:dist.barrier()
    state=dict(rng=rng_state(),pending=m.pending,step=step,rank=rank,world=world,spec_sha256=SPEC_SHA)
    torch.save(state,root/f'rank{rank}.pt')
    if rank==0:torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),step=step,config=cfg,scheduler=dict(step=step,total=cfg['steps'],lr=cfg['lr'],study=cfg['study']),spec_sha256=SPEC_SHA),root/'state.pt')
    if world>1:dist.barrier()
    if rank==0:write(root/'complete.json',dict(step=step,world=world,files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in root.glob('*.pt')}))
    if world>1:dist.barrier()
def load(root,m,opt,cfg,rank,world,device):
    root=Path(root);receipt=json.loads((root/'complete.json').read_text())
    if receipt['world']!=world:raise ValueError('Resume world size differs')
    for name in ['state.pt',f'rank{rank}.pt']:
        if sha(root/name)!=receipt['files'][name]['sha256']:raise ValueError('Checkpoint hash mismatch')
    state=torch.load(root/'state.pt',map_location=device,weights_only=False)
    for k in ['study','arm','seed','lr','batch','model','data','microbatch','data_ready_sha256','source_manifest_sha256']:
        if state['config'].get(k)!=cfg.get(k):raise ValueError('Resume config differs '+k)
    if cfg['study']=='E2' and state['config']['steps']!=cfg['steps']:raise ValueError('E2 schedule mutation')
    m.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);r=torch.load(root/f'rank{rank}.pt',map_location=device,weights_only=False);m.pending=r['pending'];restore_rng(r['rng']);return state['step']

@torch.no_grad()
def evaluate(m,data,study,device,rank=0,world=1,limit=None,batch=32):
    saved=rng_state();seed(9182);m.eval();sums=torch.zeros(3,device=device,dtype=torch.float64)
    n=min(len(data),limit) if limit else len(data)
    for start in range(rank*batch,n,world*batch):
        ix=np.arange(start,min(start+batch,n));inp=e1_batch(data[ix],device) if study=='E1' else data.batch(ix,device)
        with autocast(device):
            if study=='E1':
                logits=m(**{k:v for k,v in inp.items() if k!='labels'});mask=inp['labels'][:,1:]!=-100;tar=inp['labels'][:,1:][mask];pred=logits[:,:-1][mask];loss=torch.nn.functional.cross_entropy(pred.float(),tar,reduction='sum');sums[0]+=loss;sums[1]+=len(tar);sums[2]+=(pred.argmax(-1)==tar).sum()
            else:
                out=m(**inp);sums[0]+=out['nll_sum'];sums[1]+=out['count']
    if world>1:dist.all_reduce(sums)
    result=dict(nll=float(sums[0]/sums[1]),targets=int(sums[1]),accuracy=float(sums[2]/sums[1]) if study=='E1' else None)
    restore_rng(saved);m.train();return result

def run(cfg):
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    if world>1:dist.init_process_group('nccl' if torch.cuda.is_available() else 'gloo')
    device=torch.device('cuda',local) if torch.cuda.is_available() else torch.device('cpu')
    if device.type=='cuda':torch.cuda.set_device(device)
    torch.set_num_threads(int(cfg.get('cpu_threads',2)));seed(cfg['seed'])
    root=Path(cfg['output']);mc=Config(**cfg['model']);m=Model(mc).to(device)
    wrapped=DDP(m,device_ids=[local] if device.type=='cuda' else None,find_unused_parameters=False,broadcast_buffers=False) if world>1 else m
    seed(1000000+cfg['seed']*1000+rank)
    opt=torch.optim.AdamW(m.parameter_groups(),lr=cfg['lr'],betas=(.9,.95),eps=1e-8,fused=device.type=='cuda')
    data=Path(cfg['data']);study=cfg['study']
    train=np.load(data/'train.npy',mmap_mode='r') if study=='E1' else Stream(data,'train')
    vals={name:np.load(data/(name+'.npy'),mmap_mode='r') for name in ['id_dev','ood_dev','id_test','ood_test','atoms','ood_atoms']} if study=='E1' else {'val':Stream(data,'val')}
    total=cfg['steps'];stop_after=cfg.get('stop_after',total);batch=cfg['batch'];micro=cfg.get('microbatch',batch//world)
    if batch%world:raise ValueError('Global batch not divisible by world')
    if rank==0:
        root.mkdir(parents=True,exist_ok=bool(cfg.get('resume')));write(root/'config.json',dict(cfg,actual_world=world,counts=m.counts(),runtime=dict(torch=torch.__version__,device=str(device))))
    if world>1:dist.barrier()
    start=load(cfg['resume'],m,opt,cfg,rank,world,device) if cfg.get('resume') else 0
    seed_state=rng_state();metrics=(root/'metrics.jsonl').open('a') if rank==0 else None
    taylor_log=(root/'taylor_stats.jsonl').open('a') if rank==0 and mc.arm=='DOT' else None
    epoch_order=None;epoch_previous=-1;epoch_steps=math.ceil(len(train)/batch) if study=='E1' else None
    from collections import deque
    usage=deque(maxlen=100)
    flags=[];t0=time.monotonic();period=cfg.get('eval_every',50*epoch_steps if study=='E1' else 250);pre=math.floor(total*.8)
    for step in range(start+1,min(total,stop_after)+1):
        if study=='E1':
            epoch=(step-1)//epoch_steps
            if epoch!=epoch_previous:epoch_order=np.random.default_rng(10000+epoch).permutation(len(train));epoch_previous=epoch
            global_ix=epoch_order[((step-1)%epoch_steps)*batch:((step-1)%epoch_steps+1)*batch]
        else:global_ix=np.arange((step-1)*batch,step*batch)
        if len(global_ix)==0 or global_ix.max()>=len(train):raise ValueError('Training stream exhausted')
        local_ix=global_ix[rank::world];batches=[];local_targets=0
        for off in range(0,len(local_ix),micro):
            ix=local_ix[off:off+micro];inp=e1_batch(train[ix],device) if study=='E1' else train.batch(ix,device);batches.append(inp);local_targets+=int((inp['labels'][:,1:]!=-100).sum())
        denominator=torch.tensor(local_targets,device=device,dtype=torch.float64)
        if world>1:dist.all_reduce(denominator)
        if denominator<=0:raise ValueError('No training targets')
        opt.zero_grad(set_to_none=True);lr=lr_at(step,total,cfg['lr'],study)
        for group in opt.param_groups:group['lr']=lr
        loss_total=torch.zeros((),device=device);balance_total=0.
        for i,inp in enumerate(batches):
            context=wrapped.no_sync() if world>1 and i<len(batches)-1 else contextlib.nullcontext()
            with context:
              with autocast(device):
                m.gradient_factor=float(denominator)/world
                out=wrapped(**inp);frac=len(inp['tokens'])*world/len(global_ix)
                loss=out['nll_sum']*world/denominator+(out['balance_loss']+out['taylor_loss'])*frac
              loss.backward()
            loss_total+=out['nll_sum'].detach();balance_total+=float(out['balance_loss'].detach())*frac
        norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
        if not torch.isfinite(norm):flags.append(dict(step=step,flag='nonfinite_gradient'));opt.zero_grad(set_to_none=True)
        else:opt.step()
        m.finish_update();m.promote_labels()
        if hasattr(m,'last_loads'):usage.append(m.last_loads.numpy())
        if world>1:dist.all_reduce(loss_total)
        summary=None
        if mc.arm=='DOT':
            stats=torch.cat([z['std'] for z in m.pending]) if m.pending else torch.empty(0,device=device)
            if world>1:
                n=torch.tensor([len(stats)],device=device);sizes=[torch.zeros_like(n) for _ in range(world)];dist.all_gather(sizes,n);largest=max(int(x) for x in sizes)
                if largest:
                    padded=torch.nn.functional.pad(stats,(0,largest-len(stats)));parts=[torch.zeros_like(padded) for _ in range(world)];dist.all_gather(parts,padded);stats=torch.cat([v[:int(n)] for v,n in zip(parts,sizes)])
            if len(stats):summary=dict(median=float(stats.median()),p05=float(torch.quantile(stats,.05)),below_1e4=float((stats<1e-4).float().mean()))
            if taylor_log is not None and summary:taylor_log.write(json.dumps(dict(step=step,**summary),separators=(',',':'))+'\n');taylor_log.flush()
        if step%10==0 or step==1 or step==total:
            if device.type=='cuda':torch.cuda.synchronize()
            item=dict(step=step,epoch=step/epoch_steps if epoch_steps else None,nll=float(loss_total/denominator),lr=lr,grad_norm=float(norm) if torch.isfinite(norm) else None,elapsed=time.monotonic()-t0,seconds_per_update=(time.monotonic()-t0)/(step-start),balance_loss=balance_total,spec_sha256=SPEC_SHA)
            if summary:item.update(taylor_std_median=summary['median'],taylor_std_p05=summary['p05'],taylor_std_below_1e4=summary['below_1e4'])
            if rank==0:metrics.write(json.dumps(item)+'\n');metrics.flush();print(json.dumps(item),flush=True)
        if not cfg.get('benchmark') and (step%period==0 or step==total or (study=='E2' and step==pre)):
            result={name:evaluate(m,v,study,device,rank,world,limit=None if study=='E1' or step in (pre,total) else 1000000//512,batch=cfg.get('eval_batch',32)) for name,v in vals.items() if name in ('id_dev','ood_dev','val')}
            if rank==0:write(root/f'eval-{step}.json',dict(step=step,metrics=result))
        if not cfg.get('benchmark') and (study=='E2' and step==pre or study=='E1' and step%(1000*epoch_steps)==0):
            save(root/('pre_decay' if study=='E2' else f'checkpoint-{step}'),m,opt,step,cfg,rank,world)
            if study=='E1' and rank==0:
                import shutil
                saved=sorted(root.glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))
                for old in saved[:-2]:
                    if old.is_symlink() or old.parent.resolve()!=root.resolve() or not (old/'complete.json').exists():raise ValueError('Unsafe checkpoint retention path')
                    shutil.rmtree(old)
            if world>1:dist.barrier()
    if not cfg.get('benchmark'):
        save(root/'final_state',m,opt,step,cfg,rank,world)
        final={name:evaluate(m,v,study,device,rank,world,batch=cfg.get('eval_batch',32)) for name,v in vals.items()}
        if rank==0:
            torch.save({k:v.detach().cpu().to(torch.bfloat16) if v.is_floating_point() else v.detach().cpu() for k,v in m.state_dict().items()},root/'final_weights.pt')
            write(root/'final_eval.json',dict(metrics=final,flags=flags,step=step,elapsed=time.monotonic()-t0))
    if rank==0:
        if usage:
            counts=np.sum(usage,axis=0);means=counts.mean(-1,keepdims=True);write(root/'usage.json',dict(updates=len(usage),counts=counts.tolist(),dead=(counts<.1*means).tolist(),max_over_mean=(counts.max(-1)/means[:,0]).tolist()))
        write(root/'train_complete.json',dict(step=step,elapsed=time.monotonic()-t0,seconds_per_update=(time.monotonic()-t0)/(step-start),flags=flags,benchmark=cfg.get('benchmark',False)));metrics.close()
    if taylor_log is not None:taylor_log.close()
    if world>1:dist.barrier();dist.destroy_process_group()

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();cfg=json.loads(Path(a.config).read_text())
    if cfg['spec_sha256']!=SPEC_SHA:raise ValueError('Spec SHA mismatch')
    run(cfg)
if __name__=='__main__':main()
