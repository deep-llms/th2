"""Actual-parent CPU reference, gated frozen evaluation and small-only delivery."""
import argparse
import json
import tarfile
from pathlib import Path

import numpy as np
import torch

from .adaptation_pipeline import load_snapshot_panel
from .adaptation_factors import factors_crossovers, score_factors
from .adaptation_factors_pipeline import evaluate_stream, bind_config, replay
from .adaptation_runtime import measurement_runtime
from .data import Rows, sha
from .horizon_train import write_json


def read(p):return json.loads(Path(p).read_text())


def load(c):
    models,proof=load_snapshot_panel(c['parent'],c['parent_hashes'],c['phase_files'])
    if proof['phase_step']!=c['phase_step']:raise ValueError('Wrong snapshot step')
    return models,proof


def inputs(c):
    pins=dict(c['expected_inputs']);pins[c['calibration_indices']]=c['calibration_sha256']
    for p,h in pins.items():
        if sha(p)!=h:raise ValueError('Input mismatch '+p)
    cal=np.load(c['calibration_indices'],allow_pickle=False)
    evaluation=np.load(c['indices'],allow_pickle=False)
    if (cal.ndim!=1 or cal.dtype.kind not in 'iu' or not len(cal) or
            np.intersect1d(cal,evaluation).size):raise ValueError('Invalid calibration selection')
    return cal,evaluation


@torch.no_grad()
def reference(models,x):
    result={}
    for branch in ('A','B'):
        cells=factors_crossovers(models['parent'],models[branch],x)
        scores=score_factors(models['parent'],models[branch],x,cells)
        for key,h in cells['cells'].items():
            result[branch+'/'+key]=dict(hidden=h.cpu(),nll_sum=scores['cells'][key]['nll_sum'],
                                        token_count=scores['cells'][key]['token_count'])
    return result


def prepare(plan,expected_sha,out):
    plan,out=Path(plan),Path(out)
    if sha(plan)!=expected_sha or out.exists():raise ValueError('Plan hash/fresh output required')
    configs=read(plan)['seeds'];receipts={};out.mkdir(parents=True)
    if set(configs)!={'0','1'}:raise ValueError('Two registered seeds required')
    for seed,c in configs.items():
        cal,_=inputs(c);models,proof=load(c)
        rows=Rows(c['data'],models['parent'].c.length);x=rows.batch(cal[:1],'cpu')
        with measurement_runtime('cpu'):ref=reference(models,x)
        root=out/f's{seed}';root.mkdir();torch.save(ref,root/'cpu-reference.pt')
        c=dict(c,calibration_row=int(cal[0]),cpu_reference_sha256=sha(root/'cpu-reference.pt'))
        write_json(root/'config.json',c)
        receipts[seed]=dict(config_sha256=sha(root/'config.json'),cpu_reference_sha256=c['cpu_reference_sha256'],
                            checkpoints=proof,reference_cells=len(ref),calibration_row=int(cal[0]))
        del models,ref
    write_json(out/'prepared.json',dict(seeds=receipts,actual_panels_loaded=2,models_per_panel=3,
        cpu_native_parity=True,train_updates=0,registered_plan_sha256=expected_sha))
    return read(out/'prepared.json')


def run_prepared(root,expected_sha,device='cuda'):
    root=Path(root)
    if sha(root/'config.json')!=expected_sha:raise ValueError('Prepared config mismatch')
    c=read(root/'config.json')
    if c['measurement_device']!=device:raise ValueError('Runtime differs from registered config')
    if sha(root/'cpu-reference.pt')!=c['cpu_reference_sha256']:raise ValueError('CPU reference mismatch')
    if (root/'study').exists() or (root/'gate.json').exists():raise ValueError('Fresh seed output required')
    cal,_=inputs(c)
    if int(cal[0])!=c['calibration_row']:raise ValueError('Calibration row changed')
    models,proof=load(c)
    saved=torch.load(root/'cpu-reference.pt',map_location='cpu',weights_only=True)
    with measurement_runtime(device) as policy:
        for model in models.values():model.to(device)
        x=Rows(c['data'],models['parent'].c.length).batch(cal[:1],device)
        current=reference(models,x);maxima={}
        if set(current)!=set(saved):raise ValueError('Reference cells differ')
        for key,fields in current.items():
            maxima[key]={}
            for name,value in fields.items():
                torch.testing.assert_close(value,saved[key][name],atol=policy['component_atol'],rtol=policy['component_rtol'])
                maxima[key][name]=float((value-saved[key][name]).abs().max())
        write_json(root/'gate.json',dict(pass_=True,policy=policy,max_abs=maxima,config_sha256=expected_sha))
        del saved,current
        result=evaluate_stream(models,c['data'],c['indices'],c['expected_inputs'],c['batch_size'],root/'study',proof)
        bind_config(root/'study',c)
        write_json(root/'complete.json',replay(root/'study'))
    return dict(cases=len(result['cases']),sequences=result['sequences'],gate=True)


def finish(base):
    base=Path(base);files=[];reports={};prepared=read(base/'prepared/prepared.json')
    for seed in ('0','1'):
        root=base/'prepared'/f's{seed}';study=root/'study';replay(study)
        if sha(root/'config.json')!=prepared['seeds'][seed]['config_sha256']:raise ValueError('Prepared config changed')
        if read(study/'config.json')!=read(root/'config.json'):raise ValueError('Study/config mismatch')
        gate=read(root/'gate.json')
        if gate['pass_'] is not True or gate['config_sha256']!=sha(root/'config.json'):raise ValueError('Missing bound gate')
        reports[seed]=read(study/'REPORT.json')
        files += [root/'config.json',root/'gate.json',root/'complete.json']
        files += [study/name for name in ('config.json','manifest.json','REPORT.json','scores.npz')]
    write_json(base/'REPORT.json',dict(seeds=reports,scope='Frozen norm/projection-factor crossover at fixed parent input and reader; exploratory'))
    files += [base/'REPORT.json',base/'prepared/prepared.json',base/'execution/source.json',base/'execution/admission.json']
    inventory={str(p.relative_to(base)):sha(p) for p in files};write_json(base/'small-files.json',inventory)
    with tarfile.open(base/'tier1.tar.gz','w:gz') as tar:
        for p in sorted(files+[base/'small-files.json']):
            if p.suffix=='.pt' or p.is_symlink():raise ValueError('Non-small export member')
            tar.add(p,arcname=str(p.relative_to(base)),recursive=False)
    if (base/'tier1.tar.gz').stat().st_size>5_000_000:raise ValueError('Archive too large')
    proof=dict(archive_sha256=sha(base/'tier1.tar.gz'),archive_bytes=(base/'tier1.tar.gz').stat().st_size,
        files=inventory,node_replay=True,weights_exported=False,reference_activations_exported=False,
        checkpoint_policy='Existing parent/terminal fullstates and prior32backup copies unchanged; inference only')
    write_json(base/'backup.json',proof);return proof


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);p.add_argument('--path',required=True)
    p.add_argument('--sha',required=True);p.add_argument('--out');p.add_argument('--device',default='cuda');a=p.parse_args()
    result=prepare(a.path,a.sha,a.out) if a.mode=='prepare' else run_prepared(a.path,a.sha,a.device)
    print('FACTORS_'+a.mode.upper()+'_COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__':main()
