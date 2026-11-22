"""Bounded correction: actual CPU references, CUDA parity/resume, terminal cross-path scores."""
import argparse
import copy
import gc
import shutil
import tarfile
from pathlib import Path

import torch

from .attention_shift import PATHS, freeze, fingerprint, shifted
from .attention_shift_train import read, selections, validate, train, source_hashes
from .attention_shift_eval import load_means, evaluate, replay
from .adaptation_train import load_parent
from .adaptation_gpu_probe import exact, state
from .adaptation_runtime import measurement_runtime, training_policy
from .horizon_train import write_json
from .data import sha


def references(model,x,means):
    model.eval();result={}
    vectors=dict(zero=torch.zeros(model.c.d,device=x.device),
                 test=torch.linspace(-.01,.01,model.c.d,device=x.device),
                 **{k:v.to(x.device) for k,v in means.items()})
    for branch,path in PATHS.items():
        for name,value in vectors.items():
            b=value.detach().clone().requires_grad_(name in ('zero','test'))
            with torch.no_grad():
                h=shifted(model,x,b,branch,hidden=True)
                if name=='zero':
                    try:torch.testing.assert_close(h,model.hidden(x,path),atol=0,rtol=0)
                    except AssertionError as error:
                        raise AssertionError(f'{branch}/zero native identity: {error}') from error
            loss=shifted(model,x,b,branch)
            result[branch+'/'+name]=dict(hidden=h.detach().cpu(),nll=loss.detach().cpu())
            if b.requires_grad:
                grad=torch.autograd.grad(loss.mean(),b)[0]
                if not torch.isfinite(grad).all() or not torch.any(grad!=0):raise ValueError('Bias gradient missing/nonfinite')
                result[branch+'/'+name]['gradient']=grad.detach().cpu()
    if any(p.grad is not None for p in model.parameters()):raise ValueError('Frozen body received gradient')
    return result


def prepare_seed(configs,reference,out):
    out=Path(out)
    if out.exists():raise ValueError('Fresh prepared output required')
    if set(configs)!=set(PATHS):raise ValueError('Both paths required')
    c=configs['C'];validate(c);validate(configs['S'])
    if {k:v for k,v in c.items() if k!='branch'}!={k:v for k,v in configs['S'].items() if k!='branch'}:
        raise ValueError('Unmatched prepared paths')
    model,opt,parent,_=load_parent(c['parent'],c['parent_hashes']);del opt
    if parent['recipe']['config']['seed']!=c['seed']:raise ValueError('Seed mismatch')
    del parent
    freeze(model);digest=fingerprint(model);rows,used,cal,confirm,binding=selections(c,model.c.length)
    means=load_means(reference,c,model.c.d)
    with measurement_runtime('cpu') as policy:
        ref=references(model,rows.batch(cal[:1],'cpu'),means)
        repeated=references(model,rows.batch(cal[:1],'cpu'),means);repeat_max={}
        for key,fields in ref.items():
            repeat_max[key]={}
            for name,value in fields.items():
                torch.testing.assert_close(value,repeated[key][name],atol=policy['component_atol'],rtol=policy['component_rtol'])
                repeat_max[key][name]=float((value-repeated[key][name]).abs().max())
        del repeated
    if fingerprint(model)!=digest:raise ValueError('CPU reference changed parent')
    out.mkdir(parents=True);torch.save(ref,out/'cpu-reference.pt')
    config=dict(fits=configs,mean_reference=reference,reference_row=int(cal[0]),
                reference_sha256=sha(out/'cpu-reference.pt'),source=source_hashes(),row_binding=binding,
                parent_fingerprint=digest,fixture_only=c['fixture_only'],cpu_repeat_max_abs=repeat_max)
    write_json(out/'config.json',config)
    return dict(config_sha256=sha(out/'config.json'),cpu_reference_sha256=config['reference_sha256'],
                parent_fingerprint=digest,training_rows=len(used),calibration_rows=len(cal),confirmation_rows=len(confirm),
                reference_cells=len(ref),train_updates=0,cpu_native_parity=True)


def prepare(plan,expected_sha,out):
    if sha(plan)!=expected_sha or Path(out).exists():raise ValueError('Plan hash/fresh output required')
    p=read(plan);panels=p['seed_panels']
    if set(panels)!={'0','1'}:raise ValueError('Two actual parent seeds required')
    common={k:v for k,v in p.items() if k not in ('seed_panels','flags')}
    receipts={};out=Path(out);out.mkdir(parents=True)
    for seed,panel in panels.items():
        rowcount=panel['training_prefix_rows'];reference=panel['mean_reference']
        c=dict(common,**{k:v for k,v in panel.items() if k not in ('mean_reference','training_prefix_rows')},
               seed=int(seed),threads=4,fixture_only=False)
        if rowcount!=c['steps']*c['batch']:raise ValueError('Registered prefix extent mismatch')
        receipts[seed]=prepare_seed({b:dict(c,branch=b) for b in PATHS},reference,out/f's{seed}')
        print('SHIFT_PARENT_CPU_COMPLETE',seed,receipts[seed],flush=True)
    result=dict(seeds=receipts,actual_panels_loaded=2,cpu_native_parity=True,train_updates=0,
                source=source_hashes(),registered_plan_sha256=expected_sha,gpu_admitted=False)
    write_json(out/'prepared.json',result);return result


def dispose(*objects):
    # Caller drops references before this helper; no ownership or process action.
    gc.collect()
    if torch.cuda.is_initialized():torch.cuda.empty_cache()


def run_prepared(root,expected_sha,device='cuda'):
    root=Path(root)
    if sha(root/'config.json')!=expected_sha:raise ValueError('Prepared config hash mismatch')
    c=read(root/'config.json');configs=c['fits']
    if c['source']!=source_hashes() or sha(root/'cpu-reference.pt')!=c['reference_sha256']:
        raise ValueError('Prepared source/reference mismatch')
    if device=='cuda' and c['fixture_only']:raise ValueError('Fixtures cannot admit CUDA')
    if device=='cpu' and not c['fixture_only']:raise ValueError('Production correction requires CUDA admission')
    if (root/'work').exists():raise ValueError('Existing work: reconcile instead of relaunch')
    work=root/'work';work.mkdir();base=configs['C']
    model,opt,parent,_=load_parent(base['parent'],base['parent_hashes']);del opt,parent
    freeze(model);digest=fingerprint(model)
    if digest!=c['parent_fingerprint']:raise ValueError('Frozen parent differs')
    rows,_,cal,_,binding=selections(base,model.c.length)
    if binding!=c['row_binding'] or int(cal[0])!=c['reference_row']:raise ValueError('Prepared row mismatch')
    means=load_means(c['mean_reference'],base,model.c.d);model.to(device)
    saved=torch.load(root/'cpu-reference.pt',map_location='cpu',weights_only=True)
    with measurement_runtime(device) as policy:
        now=references(model,rows.batch(cal[:1],device),means);maxima={}
        if set(now)!=set(saved):raise ValueError('Reference cells mismatch')
        for k,fields in now.items():
            if set(fields)!=set(saved[k]):raise ValueError('Reference fields mismatch')
            maxima[k]={}
            for key,v in fields.items():
                torch.testing.assert_close(v,saved[k][key],atol=policy['component_atol'],rtol=policy['component_rtol'])
                maxima[k][key]=float((v-saved[k][key]).abs().max())
    del model,rows,means,saved,now;dispose()
    runs={};qualsteps=2 if c['fixture_only'] else 40
    for branch in PATHS:
        cfg=configs[branch];d=work/branch;d.mkdir()
        numerical=dict(pass_=True,kind='numerical',config=cfg,source=source_hashes(),policy=training_policy(device),
                       fixture_only=c['fixture_only'],reference_sha256=c['reference_sha256'],
                       prepared_config_sha256=expected_sha,measurement_policy=policy,max_abs=maxima)
        write_json(d/'numerical.json',numerical);admission=dict(path=str(d/'numerical.json'),sha256=sha(d/'numerical.json'))
        for sub,stop,resume in [('whole',qualsteps,False),('split',qualsteps//2,False),('split',qualsteps,True)]:
            m,b,o=train(cfg,d/sub,stop_after=stop,resume=resume,device=device,admission=admission,qualification=True)
            del m,b,o;dispose()
        exact(state(d/'whole'),state(d/'split'))
        for name in ('trace.jsonl','validation.jsonl'):
            if (d/'whole'/name).read_bytes()!=(d/'split'/name).read_bytes():raise ValueError('Resume trace differs')
        gate=dict(numerical,kind='resume',step=qualsteps,exact_resume=True,
                  numerical_sha256=sha(d/'numerical.json'),qualification_updates_executed=qualsteps*2,
                  whole_fullstate_sha256=sha(d/'whole/fullstate.pt'),split_fullstate_sha256=sha(d/'split/fullstate.pt'))
        write_json(d/'gate.json',gate);admission=dict(path=str(d/'gate.json'),sha256=sha(d/'gate.json'))
        # Fresh zero bias/Adam/RNG; qualification is not extra production training.
        m,b,o=train(cfg,d/'fit',device=device,admission=admission)
        del m,b,o;dispose();runs[branch]=dict(path=str(d/'fit'),sha256=sha(d/'fit/fullstate.pt'))
        print('SHIFT_FIT_COMPLETE',base['seed'],branch,runs[branch]['sha256'],flush=True)
    result=evaluate(configs,runs,c['mean_reference'],root/'study',device=device,batch=2)
    write_json(root/'complete.json',dict(replay=replay(root/'study'),runs=runs,config_sha256=expected_sha,
                                       trained_parameters=base.get('fixture_only') and 8 or 768))
    return dict(cases=len(result['cases']),sequences=result['sequences'],fits=2,steps=base['steps'])


def finish(base):
    base=Path(base);prepared=read(base/'prepared/prepared.json');files=[];reports={};copies={}
    backup=base/'correction-backup'
    if backup.exists():raise ValueError('Fresh correction backup required')
    for seed in ('0','1'):
        root=base/'prepared'/f's{seed}'
        if sha(root/'config.json')!=prepared['seeds'][seed]['config_sha256']:raise ValueError('Prepared seed mismatch')
        replay(root/'study');reports[seed]=read(root/'study/REPORT.json')
        files += [root/'config.json',root/'complete.json']+[root/'study'/n for n in ('config.json','manifest.json','REPORT.json','scores.npz')]
        for branch in PATHS:
            d=root/'work'/branch;gate=read(d/'gate.json');run=d/'fit'
            if not gate['pass_'] or not gate['exact_resume'] or gate['step']!=40:raise ValueError('Qualification missing')
            if sha(run/'fullstate.pt')!=read(root/'complete.json')['runs'][branch]['sha256']:raise ValueError('Terminal state changed')
            target=backup/f's{seed}'/branch;shutil.copytree(run,target)
            for file in run.rglob('*'):
                if file.is_file():
                    cp=target/file.relative_to(run)
                    if sha(file)!=sha(cp):raise ValueError('Correction backup mismatch')
                    copies[str(cp.relative_to(base))]=sha(cp)
            files += [d/'numerical.json',d/'gate.json']+[run/n for n in ('recipe.json','checkpoint.json','complete.json','trace.jsonl','validation.jsonl')]
    write_json(base/'REPORT.json',dict(seeds=reports,scope='One donor-free frozen-body constant correction recipe; reused exploratory DCLM'))
    files += [base/'REPORT.json',base/'prepared/prepared.json',base/'execution/source.json',base/'execution/admission.json']
    inventory={str(p.relative_to(base)):sha(p) for p in files};write_json(base/'small-files.json',inventory)
    with tarfile.open(base/'tier1.tar.gz','w:gz') as tar:
        for p in sorted(files+[base/'small-files.json']):
            if p.is_symlink() or p.suffix=='.pt':raise ValueError('Forbidden export member')
            tar.add(p,arcname=str(p.relative_to(base)),recursive=False)
    if (base/'tier1.tar.gz').stat().st_size>5_000_000:raise ValueError('Archive too large')
    proof=dict(archive_sha256=sha(base/'tier1.tar.gz'),archive_bytes=(base/'tier1.tar.gz').stat().st_size,
               files=inventory,node_replay=True,node_correction_backups=copies,
               weights_or_vectors_exported=False,original_parents_unchanged=True)
    write_json(base/'backup.json',proof);return proof


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);p.add_argument('--path',required=True)
    p.add_argument('--sha',required=True);p.add_argument('--out');p.add_argument('--device',default='cuda');a=p.parse_args()
    torch.set_num_threads(4)
    result=prepare(a.path,a.sha,a.out) if a.mode=='prepare' else run_prepared(a.path,a.sha,a.device)
    print('ATTENTION_SHIFT_'+a.mode.upper()+'_COMPLETE',result,flush=True)


if __name__=='__main__':main()
