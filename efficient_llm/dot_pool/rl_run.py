"""Single-GPU DOT-RL main/FR workers with exact resume and final evaluations."""
import argparse, dataclasses, json, math, os, shutil, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .rl import *
from .train import seed

@torch.no_grad()
def evaluate(m,p,rows,device,mode,draws=1,batch_size=256,capture=False):
    saved=rng_state();seed(9182);m.eval()
    correct=[];rewards=[];paths=[]
    for start in range(0,len(rows),batch_size):
        x,pos,y=batch(rows[start:start+batch_size],device);cc=[];rr=[];pp=[]
        for _ in range(draws):
            with autocast(device):r=m(x,pos,p,mode)
            cc.append((r['logits'].argmax(-1)==y).cpu().numpy())
            rr.append(r['logits'].log_softmax(-1).gather(1,y[:,None]).squeeze(1).cpu().numpy())
            if capture:pp.append(r['paths'].cpu().numpy().astype('uint8'))
        correct.append(np.stack(cc,1));rewards.append(np.stack(rr,1))
        if capture:paths.append(np.stack(pp,1))
    c=np.concatenate(correct);r=np.concatenate(rewards)
    result=dict(accuracy=float(c.mean()),reward=float(r.mean()),oracle=float(c.any(1).mean()),items=len(rows),draws=draws)
    if capture:result.update(correct=c,paths=np.concatenate(paths))
    restore_rng(saved);m.train();return result

def scalar_logs(info):return {k:(v.cpu().tolist() if v.ndim else float(v)) for k,v in info.items()}
def append(path,value):
    with Path(path).open('a') as f:f.write(json.dumps(dict(value,spec_sha256=SPEC_SHA),allow_nan=False)+'\n')

def curve(m,p,data,device,arm,epoch,root):
    modes=['random','loop'] if arm=='P-rand' else ['loop' if arm=='L' else 'greedy']
    records={}
    for mode in modes:
        records[mode]={k:evaluate(m,p,data[k],device,mode,draws=8 if mode=='random' else 1) for k in ['id_dev','ood_dev']+(['rl_dev'] if p else [])}
    append(root/'curves.jsonl',dict(epoch=epoch,modes=records))

def path_analysis(m,p,data,device,epoch,root):
    out={};histograms={}
    for name in ['atoms','train_compositions','rl_train','ood_test']:
        r=evaluate(m,p,data[name],device,'greedy',capture=True)
        paths=r.pop('paths')[:,0];correct=r.pop('correct')[:,0]
        unique,counts=np.unique(paths,axis=0,return_counts=True)
        histograms[name]={tuple(path):float(n/len(paths)) for path,n in zip(unique,counts)}
        out[name]=path_summary(paths,correct)
        np.savez_compressed(root/f'paths-epoch{epoch}-{name}.npz',paths=paths,correct=correct)
    comparisons=[]
    names=list(histograms)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            aa,bb=histograms[a],histograms[b];keys=set(aa)|set(bb)
            comparisons.append(dict(a=a,b=b,total_variation=.5*sum(abs(aa.get(k,0)-bb.get(k,0)) for k in keys),shared_path_mass=sum(min(aa.get(k,0),bb.get(k,0)) for k in keys)))
    write(root/f'paths-epoch{epoch}.json',dict(epoch=epoch,groups=out,comparisons=comparisons))

def path_summary(paths,correct):
    def top(mask):
        v,n=np.unique(paths[mask],axis=0,return_counts=True);order=np.argsort(-n,kind='stable')[:10]
        return [dict(path=(v[i]+1).tolist(),count=int(n[i]),share=float(n[i]/max(1,len(paths[mask])))) for i in order]
    return dict(items=len(paths),correct=int(correct.sum()),loop_share=float((paths==np.tile(np.arange(4),2)).all(1).mean()),repeat_share=float((paths[:,:4]==paths[:,4:]).all(1).mean()),top=top(np.ones(len(paths),bool)),top_correct=top(correct),top_wrong=top(~correct))

def final_eval(m,p,data,device,arm,root,tag='final'):
    modes=['loop'] if arm=='L' else ['random','loop'] if arm=='P-rand' else ['greedy','sampled']
    result={};arrays={}
    for mode in modes:
        draws=8 if mode in ['random','sampled'] else 1;results={}
        for name in ['id_test','ood_test','rl_dev','train','ood_rl_atoms','ood_test_atoms']:
            capture=mode in ['loop','greedy'] or name in ['id_test','ood_test','rl_dev']
            r=evaluate(m,p,data[name],device,mode,draws=draws,capture=capture)
            if capture:
                arrays[mode+'_'+name+'_paths']=r.pop('paths');arrays[mode+'_'+name+'_correct']=r.pop('correct')
            results[name]=r
        result[mode]=results
    if p:
        path_analysis(m,p,data,device,tag,root)
        # Save tiny router states separately; never put model/fullstate in Tier1.
        torch.save(p.state_dict(),root/(tag+'_router.pt'))
    np.savez_compressed(root/(tag+'_paths.npz'),**arrays)
    write(root/(tag+'_eval.json'),dict(arm=arm,modes=result))
    return result

def optim(m,p,device,fr=False):
    opt=None if fr else torch.optim.AdamW(m.parameter_groups(),lr=.001,betas=(.9,.95),eps=1e-8,fused=device.type=='cuda')
    po=torch.optim.AdamW([v for v in p.parameters() if v.requires_grad],lr=.001 if fr else .0001,betas=(.9,.95),weight_decay=0.,eps=1e-8,fused=device.type=='cuda') if p else None
    return opt,po

def update(m,p,opt,po,rows,reward_rows,device,step,mode='random',active=False,group=4,paths=None,warmup=True):
    if opt:opt.zero_grad(set_to_none=True)
    if po:po.zero_grad(set_to_none=True)
    info={}
    if opt:
        x,pos,y=batch(rows,device)
        with autocast(device):out=m(x,pos,p if active else None,mode,paths=paths);ce=F.cross_entropy(out['logits'],y)
        ce.backward();info['ce']=ce.detach()
    if active:
        loss,stats=rollout(m,p,reward_rows,device,group);loss.backward();info.update(stats);info['router_loss']=loss.detach()
    # No optimizer changes until BOTH passes have used the same layer weights.
    if opt:
        norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True)
        for g in opt.param_groups:g['lr']=.001*min(1,step/1000) if warmup else g['lr']
        opt.step();info['grad_norm']=norm.detach()
    if active:po.step()
    return info

def main_run(cfg):
    torch.set_num_threads(2);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');seed(cfg['seed'])
    m=Library(Config(**cfg['model'])).to(device);p=Policy(m.c.d).to(device) if cfg['arm'] in ['J-RL','WS-RL'] else None
    opt,po=optim(m,p,device);seed(1000000+1000*cfg['seed'])
    root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=bool(cfg.get('resume')))
    assert sha(Path(cfg['data'])/'manifest.json')==cfg['data_sha256']
    manifest=json.loads((Path(cfg['data'])/'manifest.json').read_text())
    for name,r in manifest['files'].items():assert sha(Path(cfg['data'])/name)==r['sha256']
    data={p.stem:np.load(p,mmap_mode='r') for p in Path(cfg['data']).glob('*.npy')}
    stream=RewardStream(len(data['rl_train']),6000+cfg['seed']);step=0;extra={};flags=[]
    if cfg.get('resume'):step,extra=load(cfg['resume'],m,p,opt,po,cfg,stream,device)
    write(root/'config.json',dict(cfg,runtime=dict(torch=torch.__version__,device=str(device)),nonembedding=m.counts(),policy_parameters=sum(v.numel() for v in p.parameters()) if p else 0))
    steps_epoch=math.ceil(len(data['train'])/cfg['batch']);start_epoch=step//steps_epoch
    assert step%steps_epoch==0,'Resume only at complete epoch checkpoints'
    start=time.monotonic();previous_elapsed=extra.get('elapsed',0);switch_actual=extra.get('switch_actual');N=extra.get('epochs');budget_path=Path(cfg['budget_path'])
    for epoch in range(start_epoch+1,4501):
        if budget_path.exists():
            budget=json.loads(budget_path.read_text());N=budget['epochs']
            if cfg.get('development_only'):
                assert device.type=='cpu' and m.c.d<=32 and N>=1
            else:assert 2000<=N<=4500 and N%500==0
            if epoch>N:break
        active=cfg['arm']=='J-RL' or (cfg['arm']=='WS-RL' and N is not None and epoch>budget['switch_epoch'])
        if active and switch_actual is None:
            switch_actual=epoch-1
            if cfg['arm']=='WS-RL' and switch_actual!=budget['switch_epoch']:flags.append('Late WS switch: '+str(switch_actual))
        order=np.random.default_rng(10000+epoch-1).permutation(len(data['train']))
        for off in range(0,len(order),cfg['batch']):
            step+=1;rows=data['train'][order[off:off+cfg['batch']]]
            rr=data['rl_train'][stream.next(512)] if active else None
            info=update(m,p,opt,po,rows,rr,device,step,mode='loop' if cfg['arm']=='L' else 'random',active=active)
            if step%100==0:
                stats=scalar_logs(info)
                if active:
                    saved=rng_state()
                    with torch.no_grad(),autocast(device):
                        xx,pp,_=batch(rr,device);greedy=m(xx,pp,p,'greedy')['paths']
                    stats['greedy_loop_share']=float((greedy==torch.arange(8,device=device)%4).all(1).float().mean());restore_rng(saved)
                    if np.mean(stats['entropy'])<.2 and epoch-switch_actual<=500 and 'policy collapse within500epochs' not in flags:flags.append('policy collapse within500epochs')
                append(root/'metrics.jsonl',dict(step=step,epoch=epoch,elapsed=previous_elapsed+time.monotonic()-start,**stats))
        if epoch%50==0:curve(m,p,data,device,cfg['arm'],epoch,root)
        elapsed=previous_elapsed+time.monotonic()-start
        if epoch==200:
            write(root/'first200.json',dict(epoch=200,step=step,elapsed=elapsed,seconds_per_epoch=elapsed/200))
            # Seed0 is predeclared speed reference; no barrier/restart for other arms.
            if cfg['arm']=='J-RL' and cfg['seed']==0:
                assert not budget_path.exists();budget=choose_budget(elapsed/200);budget.update(reference='J-RL seed0 first200 includes periodic dev',fixed_utc=time.time());write(budget_path,budget);N=budget['epochs']
        if epoch%500==0:
            if p:path_analysis(m,p,data,device,epoch,root)
            extra=dict(elapsed=previous_elapsed+time.monotonic()-start,switch_actual=switch_actual,epochs=N)
            save(root/f'checkpoint-{epoch}',m,p,opt,po,step,cfg,stream,extra)
            checkpoints=sorted(root.glob('checkpoint-*'),key=lambda x:int(x.name.split('-')[-1]))
            for old in checkpoints[:-2]:shutil.rmtree(old)  # exact owned periodic states only
        write(root/'progress.json',dict(epoch=epoch,step=step,epochs=N,switch_actual=switch_actual,elapsed=previous_elapsed+time.monotonic()-start,flags=flags))
    assert N is not None and step==N*steps_epoch,('Budget/end mismatch',N,step)
    extra=dict(elapsed=previous_elapsed+time.monotonic()-start,switch_actual=switch_actual,epochs=N)
    save(root/'final_state',m,p,opt,po,step,cfg,stream,extra)
    torch.save(m.state_dict(),root/'final_weights.pt')
    write(root/'train_complete.json',dict(step=step,epochs=N,switch_actual=switch_actual,elapsed=extra['elapsed'],flags=flags,weights_sha256=sha(root/'final_weights.pt'),fullstate_sha256=sha(root/'final_state/state.pt')))
    final_eval(m,p,data,device,cfg['arm'],root)
    write(root/'complete.json',dict(complete=True,epochs=N,flags=flags))

def fr_run(cfg):
    torch.set_num_threads(2);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');seed(cfg['seed'])
    root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=False);parent=Path(cfg['parent'])
    pcfg=json.loads((parent/'config.json').read_text());done=json.loads((parent/'train_complete.json').read_text())
    assert pcfg['arm']=='P-rand' and pcfg['seed']==cfg['seed'] and sha(parent/'final_weights.pt')==done['weights_sha256']
    m=Library(Config(**cfg['model'])).to(device);m.load_state_dict(torch.load(parent/'final_weights.pt',map_location=device,weights_only=True));m.requires_grad_(False)
    before=tensor_hash(m);p=Policy(m.c.d,fixed=cfg['arm']=='FR-fixed').to(device);opt,po=optim(m,p,device,True)
    data={f.stem:np.load(f,mmap_mode='r') for f in Path(cfg['data']).glob('*.npy')};stream=RewardStream(len(data['rl_train']),7000+cfg['seed'])
    write(root/'config.json',dict(cfg,parent_weight_sha256=done['weights_sha256'],frozen_hash_before=before));best=-1.;selected=0;start=time.monotonic()
    for step in range(1,3001):
        info=update(m,p,None,po,None,data['rl_train'][stream.next(256)],device,step,active=True,group=8)
        if step%100==0:
            dev=evaluate(m,p,data['rl_dev'],device,'greedy');append(root/'metrics.jsonl',dict(step=step,elapsed=time.monotonic()-start,**scalar_logs(info),rl_dev=dev))
            if dev['accuracy']>best:
                best=dev['accuracy'];selected=step;torch.save(p.state_dict(),root/'selected_router.pt')
    final_eval(m,p,data,device,cfg['arm'],root,'last')
    p.load_state_dict(torch.load(root/'selected_router.pt',map_location=device,weights_only=True));final_eval(m,p,data,device,cfg['arm'],root,'selected')
    after=tensor_hash(m);assert after==before
    write(root/'complete.json',dict(complete=True,updates=3000,selected_update=selected,T12=dict(passed=before==after,before=before,after=after),elapsed=time.monotonic()-start))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);a=ap.parse_args();cfg=json.loads(Path(a.config).read_text());assert cfg['spec_sha256']==SPEC_SHA
    (fr_run if cfg['arm'].startswith('FR') else main_run)(cfg)
