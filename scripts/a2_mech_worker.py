"""A2 M worker, one owned device; failures are recorded per registered probe."""
import argparse
import json
from pathlib import Path
import time
import traceback
from scripts.a2_quant_io import write,lock
from scripts.a2_remaining import read_locked


def main():
    p=argparse.ArgumentParser()
    for name in ('config','config-sha256','dataset','parent','phase','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();cfg=read_locked(dict(path=a.config,sha256=a.config_sha256));out=Path(a.output);out.mkdir(parents=True)
    import torch
    import numpy as np
    from efficient_llm.a2.model import load_parent,round_to_saved_precision,bypass
    from efficient_llm.a2.data import PackedRows
    from efficient_llm.a2.measure import score,full_fp32
    from efficient_llm.a2.quant import matrix_manifest
    from efficient_llm.a2.mech import weight_statistics,profile_hooks,perturb,token_score
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True,warn_only=True);torch.manual_seed(0)
    start=time.monotonic();checks=[];results={};model=None
    def emit(name,passed,detail):
        checks.append(dict(test=name,passed=bool(passed),detail=detail,spec_sha256=cfg['spec_sha256'],dataset=a.dataset,parent=a.parent,phase=a.phase))
        write(out/'tests.json',checks);print('MECH_CHECK',name,bool(passed),flush=True)
    def attempt(name,fn):
        try:return fn()
        except Exception as e:
            traceback.print_exc();emit(name,False,dict(error=repr(e),action='continue remaining registered probes'));return None
    assets=read_locked(cfg['assets'])
    base=Path(cfg['quant_results'])/'workers'
    def reference(kind):return read_locked(cfg['reference_nll'][a.dataset+'/'+a.parent+'/'+kind])
    if a.phase=='weights':
        from safetensors import safe_open
        from efficient_llm.diagnostic_data import sha256
        wpath=Path(assets[a.dataset+'/'+a.parent]['weights_dir'])/'model.safetensors'
        emit('parent_weight_identity',sha256(wpath)==assets[a.dataset+'/'+a.parent]['files']['model.safetensors'],str(wpath))
        saved={k:read_locked(cfg['rtn_weights'][a.dataset+'/'+a.parent+'/'+k]) for k in ('rtn3','rtn4')}
        for k,v in saved.items():emit(k+'_identity',sha256(v['weights']['path'])==v['weights']['sha256'],v['weights'])
        rows=[]
        with safe_open(str(wpath),framework='pt',device='cpu') as parent, safe_open(saved['rtn3']['weights']['path'],framework='pt',device='cpu') as q3, safe_open(saved['rtn4']['weights']['path'],framework='pt',device='cpu') as q4:
            for matrix in saved['rtn3']['matrices']:
                name=matrix['name']
                def measure():
                    w=parent.get_tensor(name).bfloat16().float()
                    return dict(name=name,**weight_statistics(w,dict(rtn3_weight_error=q3.get_tensor(name).float(),rtn4_weight_error=q4.get_tensor(name).float())))
                value=attempt(name,measure)
                if value:rows.append(value)
        emit('168_matrix_coverage',len(rows)==168,dict(count=len(rows)))
        keys=('excess_kurtosis','group_range_over_rms','rtn3_weight_error','rtn4_weight_error')
        results=dict(matrices=rows,means={k:sum(r[k]*r['weights'] for r in rows)/sum(r['weights'] for r in rows) if rows and all(r[k] is not None for r in rows) else None for k in keys})
    else:
        model=attempt('load_parent',lambda:load_parent(assets,a.dataset,a.parent,'cuda:0').eval().requires_grad_(False))
        if model is not None:
            round_to_saved_precision(model)
            final=PackedRows(cfg['data_root'],'final',verify=False)
            small=PackedRows(cfg['data_root'],'tuning',count=256,order=cfg['small_indices'],verify=False)
            def evaluate(name,rows):
                val=score(model,rows,distributed=False)
                val.update(spec_sha256=cfg['spec_sha256'],dataset=a.dataset,parent=a.parent,probe=name)
                write(out/(name+'.json'),val);return val
            if a.phase=='layers':
                def baseline():
                    with profile_hooks(model) as (layers,block):b=evaluate('parent-small',small)
                    return dict(parent=b,BI=(layers[:,0]/layers[:,2]).cpu().tolist(),update=(layers[:,1]/layers[:,2]).cpu().tolist(),BI_S1=float(block[0]/block[2]))
                summary=attempt('profile',baseline)
                if summary:results.update(summary)
                singles=[]
                for layer in range(24):
                    def single():
                        with bypass(model,[layer]):return evaluate('single-'+str(layer),small)
                    singles.append(attempt('single:'+str(layer),single))
                results['single']=singles
                def pre():
                    with bypass(model,list(range(16,22))):return evaluate('pre-S1-small',small)
                results['pre']=attempt('pre-S1-small',pre)
                if summary and all(singles) and results['pre']:
                    ds=[v['loss']-summary['parent']['loss'] for v in singles];summed=sum(ds[16:22])
                    results.update(single_damage=ds,S1_sum=summed,rho=(results['pre']['loss']-summary['parent']['loss'])/summed if summed else None)
                    emit('rho_denominator',summed!=0,dict(S1_sum=summed))
                # Full original sequence, selected 128 positions in FP32 head, exact independent physical removal.
                def correctness():
                    ids=small[0]['input_ids'][None].to('cuda:0');original=model.model.layers
                    for layer in (0,11,23):
                        with full_fp32(),torch.no_grad():
                            with bypass(model,[layer]):
                                with profile_hooks(model) as (lp,bp):
                                    h=model.model(ids,use_cache=False,return_dict=True).last_hidden_state
                                expected_hidden=h
                            emit('MT4:'+str(layer),float(lp[layer,0])==0 and float(lp[layer,1])==0,dict(BI=float(lp[layer,0]/lp[layer,2]),update=float(lp[layer,1]/lp[layer,2])))
                            try:
                                model.model.layers=torch.nn.ModuleList([v for i,v in enumerate(original) if i!=layer])
                                actual_hidden=model.model(ids,use_cache=False,return_dict=True).last_hidden_state
                            finally:model.model.layers=original
                            error=max(float((model.lm_head(expected_hidden[:,i:i+128]).float()-model.lm_head(actual_hidden[:,i:i+128]).float()).abs().max()) for i in range(0,ids.shape[1],128))
                            emit('MT5:'+str(layer),error<=1e-4,dict(max_logit_error=error,positions=ids.shape[1],sequence_length=ids.shape[1]))
                attempt('identity_removal_tests',correctness)
            elif a.phase=='noise':
                baseline=attempt('parent-final',lambda:evaluate('parent-final',final))
                ref=reference('parent')
                if baseline:emit('MT1:parent',abs(baseline['loss']-ref['loss'])<=1e-6,dict(actual=baseline['loss'],reference=ref['loss']))
                def pre_final():
                    with bypass(model,list(range(16,22))):v=evaluate('pre-S1-final',final)
                    if baseline:emit('MT1:pre-S1',abs((v['loss']-baseline['loss'])-(reference('pre-S1')['loss']-ref['loss']))<=1e-6,dict(actual=v['loss'],reference=reference('pre-S1')['loss']))
                    return v
                attempt('pre-S1-final',pre_final)
                matrices=matrix_manifest(model);state=model.state_dict();originals={v['name']:state[v['name']].detach().cpu().clone() for v in matrices}
                def noise(kind,seed,sigma):
                    name=('N-iso-'+str(sigma) if kind=='iso' else 'N-rtn3')+'-seed'+str(seed)
                    matrix_checks=[]
                    with torch.no_grad():
                        for index,v in enumerate(matrices):
                            namew=v['name'];w=originals[namew].to('cuda:0')
                            value,check=perturb(w,kind,seed,index,sigma);state[namew].copy_(value)
                            if check:matrix_checks.append(dict(name=namew,**check))
                    if matrix_checks:emit('MT3:'+name,all(r['passed'] for r in matrix_checks),matrix_checks)
                    value=evaluate(name,final)
                    if sigma==0 and baseline:emit('MT2',abs(value['loss']-baseline['loss'])<=1e-6,dict(difference=value['loss']-baseline['loss']))
                    return value
                for kind,sigma,seed in [('iso',0.,0)]+[(k,s,z) for k,s in [('iso',.1),('iso',.2),('iso',.3),('rtn3',None)] for z in (0,1)]:
                    name=f'{kind}:{sigma}:{seed}';results[name]=attempt(name,lambda k=kind,s=sigma,z=seed:noise(k,z,s))
            else:
                def tokens(name):
                    path=out/(name+'.npy');v=token_score(model,final,path)
                    emit('MT6:'+name,v['storage_replay_max_abs']<=1e-6,dict(max_abs=v['storage_replay_max_abs']))
                    v.update(spec_sha256=cfg['spec_sha256'],tokens_file=lock(path),dataset=a.dataset,model=name)
                    write(out/(name+'.json'),v);return v
                results[a.parent]=attempt('tokens-parent',lambda:tokens(a.parent))
                if a.parent in ('P300','P1T'):
                    def pre_tokens():
                        with bypass(model,list(range(16,22))):return tokens('pre-'+a.parent)
                    results['pre-'+a.parent]=attempt('tokens-pre',pre_tokens)
                student='X72' if a.parent=='S72' else a.parent if a.parent in ('P300','P1T') else None
                if student:
                    del model;model=None;torch.cuda.empty_cache()
                    def student_tokens():
                        nonlocal model
                        from eval.a2_checkpoint import verify_identity,load_verified
                        identity=cfg['students'][a.dataset+'/'+student];verify_identity(identity)
                        model=load_verified(identity,'cuda:0').eval().requires_grad_(False)
                        return tokens('student-'+student)
                    results['student-'+student]=attempt('tokens-student',student_tokens)
    failures=[c for c in checks if not c['passed']]
    for path in out.glob('*.json'):
        if path.name=='tests.json':continue
        row=json.loads(path.read_text());row.update(unverified=bool(failures),affected_checks=failures)
        write(path,row)
    write(out/'result.json',dict(spec_sha256=cfg['spec_sha256'],dataset=a.dataset,parent=a.parent,phase=a.phase,results=results,checks=checks,elapsed_seconds=time.monotonic()-start,execution_complete=True))


if __name__=='__main__':main()
