"""Pinned exploratory path-background study, one boundary, no optimizer updates.

Preparation is CPU only. CUDA requires a separate owned queue and actual CPU
parity receipt. Existing confirmation rows are explicitly reused exploratory.
"""
import argparse,json
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch

from .adaptation_train import load_parent
from .adaptation_runtime import measurement_runtime,measurement_device,measurement_policy
from .data import Rows,sha
from .horizon_train import write_json
from .path_background import boundaries,suffix,score,anchors,observations,fit,random_basis,transform,transfer,overlap
from .adaptation_analysis import CANONICAL,SWAPPED


def prepare(prepared, prepared_sha, out):
    prepared,out=Path(prepared),Path(out)
    if out.exists():raise ValueError('Fresh path-background output required')
    if sha(prepared/'manifest.json')!=prepared_sha:raise ValueError('Prepared identity mismatch')
    m=json.loads((prepared/'manifest.json').read_text())
    for f,h in m['files'].items():
        if sha(prepared/f)!=h:raise ValueError('Prepared member mismatch')
    out.mkdir(parents=True);receipts={}
    for seed in (0,1):
        c=json.loads((prepared/f'A-s{seed}.json').read_text());root=out/f's{seed}';root.mkdir()
        with torch.random.fork_rng(devices=[]):model,opt,state,_=load_parent(c['parent'],c['parent_hashes'])
        if state['recipe']['config']['seed']!=seed:raise ValueError('Parent seed mismatch')
        del opt,state;model.eval().requires_grad_(False)
        paths=(c['data']+'.bin',c['data']+'.starts',c['calibration_indices'],c['confirmation_indices'])
        inputs={p:sha(p) for p in paths}
        if any(inputs[p]!=c['expected_inputs'][p] for p in paths):raise ValueError('Input binding mismatch')
        cal=np.load(c['calibration_indices'],allow_pickle=False);evaluation=np.load(c['confirmation_indices'],allow_pickle=False)
        if min(len(cal),len(evaluation))<64 or np.intersect1d(cal,evaluation).size:raise ValueError('Insufficient/disjoint rows required')
        rows=Rows(c['data'],model.c.length);tokens=rows.batch(cal[:1],'cpu');a,b=boundaries(model,tokens)
        references={}
        for label,h,path in [('A',a,CANONICAL),('B',b,SWAPPED)]:
            hidden=suffix(model,tokens,h);torch.testing.assert_close(hidden,model.hidden(tokens,path),atol=2e-6,rtol=1e-5)
            sums,counts=score(model,tokens,h);torch.testing.assert_close((sums/counts).float(),model.nll(tokens,path),atol=2e-6,rtol=1e-5)
            references[label]=dict(boundary=h,hidden=hidden,nll_sum=sums,token_count=counts)
        torch.save(references,root/'cpu-reference.pt')
        plan=dict(schema=1,seed=seed,parent=c['parent'],parent_hashes=c['parent_hashes'],data=c['data'],
                  calibration_indices=c['calibration_indices'],evaluation_indices=c['confirmation_indices'],
                  expected_inputs=inputs,model=asdict(model.c),rank=min(8,model.c.d-1),
                  positions=[model.c.prefix-1,model.c.length//4-1,model.c.length//2-1,model.c.length-2],
                  anchor_rows=16,evaluation_rows=min(128,len(evaluation)),control_seeds=[4101,4102,4103],
                  cpu_reference_sha256=sha(root/'cpu-reference.pt'),
                  flags=['DCLM_only','frozen_parent','reused_exploratory_evaluation','historical_overlap_unknown',
                         'one_central_pair_boundary','no_training','not_intrinsic_knowledge_extraction'])
        plan['positions']=list(dict.fromkeys(plan['positions']))
        if len(plan['positions'])<2:raise ValueError('Multiple sample positions required')
        write_json(root/'plan.json',plan);receipts[str(seed)]=dict(plan_sha256=sha(root/'plan.json'),cpu_reference_sha256=plan['cpu_reference_sha256'])
        del model,references
    write_json(out/'prepared.json',dict(seeds=receipts,prior_prepared_sha256=prepared_sha,actual_parents_loaded=2,
                                      cpu_native_parity=True,gpu_admitted=False,train_updates=0))
    return receipts


def _summary(raw,labels):
    ids=raw['row_ids'];counts=raw['token_count']
    if ids.ndim!=1 or len(np.unique(ids))!=len(ids) or counts.shape!=ids.shape or (counts<=0).any():raise ValueError('Invalid row/counts')
    if set(raw)!=set(labels)|{'row_ids','token_count'}:raise ValueError('Score schema mismatch')
    den=int(counts.sum());result={}
    for label in labels:
        v=raw[label]
        if v.shape!=ids.shape or not np.isfinite(v).all() or (v<0).any():raise ValueError('Invalid NLL sums')
        branch=label.split('/')[0];base=raw[branch+'/identity'];gain=(base-v)/counts
        result[label]=dict(nll=float(v.sum()/den),gain_vs_same_path_identity=float((base-v).sum()/den),
                           sequences_better=int((gain>0).sum()),sequences_worse=int((gain<0).sum()),
                           gain_quantiles=np.quantile(gain,[.1,.5,.9]).tolist())
    return dict(cases=result,sequences=len(ids),tokens=den,
                original_gap=result['B/identity']['nll']-result['A/identity']['nll'],
                projected_gap=result['B/project']['nll']-result['A/project']['nll'],
                raw_scores_replayed=True,mechanism_verdict='No automatic bias/portability verdict; compare native cost and controls')


def replay(out):
    out=Path(out);m=json.loads((out/'manifest.json').read_text())
    for f,h in m['small_files'].items():
        if sha(out/f)!=h:raise ValueError('Small artifact hash mismatch')
    with np.load(out/'scores.npz',allow_pickle=False) as f:raw={k:f[k].copy() for k in f.files}
    report=json.loads((out/'REPORT.json').read_text())
    if report['scores']!=_summary(raw,m['labels']):raise ValueError('Score replay differs')
    return dict(status='PASS',cases=len(m['labels']),sequences=len(raw['row_ids']),basis_and_model_not_replayed_locally=True)


def run(root,plan_sha,device='cpu'):
    root=Path(root);planfile=root/'plan.json'
    if sha(planfile)!=plan_sha:raise ValueError('Plan hash mismatch')
    c=json.loads(planfile.read_text());out=root/'study'
    if out.exists():raise ValueError('Fresh study output required')
    for p,h in c['expected_inputs'].items():
        if sha(p)!=h:raise ValueError('Input hash mismatch')
    if sha(root/'cpu-reference.pt')!=c['cpu_reference_sha256']:raise ValueError('CPU parity reference changed')
    cal=np.load(c['calibration_indices'],allow_pickle=False);eval_ids=np.load(c['evaluation_indices'],allow_pickle=False)[:c['evaluation_rows']]
    if len(np.unique(cal))!=len(cal) or len(np.unique(eval_ids))!=len(eval_ids) or np.intersect1d(cal,eval_ids).size:raise ValueError('Row overlap/duplicate')
    with torch.random.fork_rng(devices=[]):model,opt,state,_=load_parent(c['parent'],c['parent_hashes'])
    del opt,state;model.eval().requires_grad_(False);model.to(device)
    if asdict(model.c)!=c['model']:raise ValueError('Model changed')
    rows=Rows(c['data'],model.c.length);out.mkdir()
    with measurement_runtime(device),torch.no_grad():
        policy=measurement_policy();ref=torch.load(root/'cpu-reference.pt',map_location='cpu',weights_only=True)
        tokens=rows.batch(cal[:1],device);a,b=boundaries(model,tokens);parity={}
        for label,h,path in [('A',a,CANONICAL),('B',b,SWAPPED)]:
            hidden=suffix(model,tokens,h);sums,counts=score(model,tokens,h)
            for name,value in [('boundary',h),('hidden',hidden),('nll_sum',sums)]:
                torch.testing.assert_close(value.cpu(),ref[label][name],atol=policy['component_atol'],rtol=policy['component_rtol']);parity[label+'/'+name]=float((value.cpu()-ref[label][name]).abs().max())
            torch.testing.assert_close(hidden,model.hidden(tokens,path),atol=policy['component_atol'],rtol=policy['component_rtol'])
        write_json(out/'parity.json',dict(runtime=policy,max_abs=parity,pass_=True))
        n=c['anchor_rows'];contrasts={}
        for family,start in [('repeat',0),('periodic',n),('shuffle',2*n),('natural',3*n)]:
            pieces=[]
            for offset,rid in enumerate(cal[start:start+n]):
                x=rows.batch(np.array([rid]),device)
                if family!='natural':x=anchors(x,family,5201+offset)
                aa,bb=boundaries(model,x);pieces.append(observations(aa,bb,c['positions']))
            if len(pieces)!=n:raise ValueError('Missing anchor rows')
            contrasts[family]=torch.cat(pieces)
        fitted=fit(torch.cat([contrasts['repeat'],contrasts['periodic']]),c['rank'])
        familyfits={f:fit(v,c['rank']) for f,v in contrasts.items()}
        diagnostic=dict(rank=fitted['actual_rank'],requested_rank=c['rank'],mean_norm=float(fitted['mean'].norm()),
                        repeat_periodic_subspace_overlap=overlap(familyfits['repeat'],familyfits['periodic']),
                        transfer={f:transfer(v,fitted) for f,v in contrasts.items()},
                        rank0_means_no_centered_subspace=fitted['actual_rank']==0,
                        no_automatic_stability_or_bias_claim=True)
        controls={s:random_basis(model.c.d,fitted['actual_rank'],s) for s in c['control_seeds']}
        # Basis chosen without scored evaluation; basis and activation-derived tensors stay node-side.
        torch.save(dict(fitted=fitted,controls=controls),out/'basis.pt');basis_sha=sha(out/'basis.pt')
        modes=[('identity',None),('minus_mean',None),('plus_mean',None),('project',None),('project_norm_restored',None)]
        modes += [(mode,s) for s in c['control_seeds'] for mode in ('random_project','random_energy_matched')]
        labels=[branch+'/'+mode+(f'/{s}' if s is not None else '') for branch in ('A','B') for mode,s in modes]
        raw={'row_ids':eval_ids.copy()};pieces={k:[] for k in labels};counts=[]
        for index,rid in enumerate(eval_ids):
            x=rows.batch(np.array([rid]),device);aa,bb=boundaries(model,x)
            for branch,h in [('A',aa),('B',bb)]:
                for mode,s in modes:
                    label=branch+'/'+mode+(f'/{s}' if s is not None else '')
                    sums,den=score(model,x,transform(h,fitted,mode,controls.get(s)))
                    pieces[label].append(sums.cpu().numpy())
            counts.append(den.cpu().numpy())
            if index%16==0:write_json(out/'progress.json',dict(scored=index+1,total=len(eval_ids)))
        raw.update({k:np.concatenate(v) for k,v in pieces.items()});raw['token_count']=np.concatenate(counts)
        np.savez_compressed(out/'scores.npz',**raw)
        write_json(out/'REPORT.json',dict(scores=_summary(raw,labels),geometry=diagnostic,flags=c['flags'],
                                        basis_sha256=basis_sha,calibration_fit_rows=cal[:2*n].tolist(),
                                        calibration_holdout_rows=cal[2*n:4*n].tolist(),measurement=policy))
        files={f:sha(out/f) for f in ('REPORT.json','scores.npz','parity.json')}
        write_json(out/'manifest.json',dict(small_files=files,labels=labels,plan_sha256=plan_sha,
                                          model_pins=c['parent_hashes'],input_pins=c['expected_inputs'],basis_sha256=basis_sha,
                                          weights_exported=False,basis_exported=False))
        result=replay(out);write_json(out/'complete.json',result)
    return result


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='mode',required=True)
    a=sub.add_parser('prepare');a.add_argument('--prepared',required=True);a.add_argument('--prepared-sha',required=True);a.add_argument('--out',required=True)
    b=sub.add_parser('run');b.add_argument('--root',required=True);b.add_argument('--plan-sha',required=True);b.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=p.parse_args();torch.set_num_threads(4)
    result=prepare(args.prepared,args.prepared_sha,args.out) if args.mode=='prepare' else run(args.root,args.plan_sha,args.device)
    print('PATH_BACKGROUND_'+args.mode.upper()+'_COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()
