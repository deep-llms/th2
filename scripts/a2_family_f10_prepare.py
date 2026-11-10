"""CPU F10 admission: bind new inputs and original F9 models without mutations."""
import argparse
from pathlib import Path
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from eval.a2_family_f10 import SPEC_SHA,write,select_models
from eval.a2_family_f9 import read
from scripts.a2_quant_io import lock
from efficient_llm.diagnostic_data import sha256


def prepare(plan,output,commit):
    cfg=read(plan);out=Path(output)
    if cfg['spec_sha256']!=SPEC_SHA or out.exists():raise ValueError('Wrong spec or existing output')
    old=Path(cfg['f9_root'])/'results/a2_family_20261107/downstream'
    manifest=old/'backup/verified.json'
    if sha256(manifest)!='8b0b68bdc69cf6aa7532250e1f368ae37fdcf1b6cebc7bdb099814b72633de3a':raise ValueError('F9 manifest changed')
    files=read(manifest)['files']
    def checked(name):
        p=old/name;v=files[name]
        if p.stat().st_size!=v['bytes'] or sha256(p)!=v['sha256']:raise ValueError('F9 file changed: '+name)
        return read(p)
    prior=checked('config.json');entries=checked('contrasts.json')['model_metadata']
    oneshots,groups=select_models(entries,cfg)
    aroot=Path(cfg['f10a_root']);done=read(aroot/'complete.json')
    if not done['complete'] or done['tests']!=90 or done['spec_sha256']!=SPEC_SHA:raise ValueError('F10a not complete')
    for name,h in done['files'].items():
        if sha256(aroot/name)!=h:raise ValueError('F10a output changed')
    if sha256(cfg['previous_report']['path'])!=cfg['previous_report']['sha256']:raise ValueError('Prior report changed')
    # Bind the complete saved F9 artifacts used later by confidence analysis.
    for name in files:
        if name.startswith('per_item/') or name in ('portable_index.json','continuous.json','contrasts.json'):
            checked(name)
    needed={e['model_id']:e for e in groups[0]+sum(groups[2:],[])}
    needed.update({e['model_id']:e for e in entries if e['model_id'] in cfg['oneshots']})
    def check(e):
        for path,h in e['file_hashes'].items():
            if sha256(path)!=h:raise ValueError('Model/config code changed: '+path)
        return e['model_id']
    with ThreadPoolExecutor(max_workers=2) as pool:verified=list(pool.map(check,needed.values()))
    from efficient_llm.a2.data import PackedRows
    PackedRows(prior['opensci_cx_root'],'final',verify=True)
    # Smol final set has its own pinned manifest and token files in old locked_inputs.
    for item in prior['locked_inputs']:
        if sha256(item['path'])!=item['sha256']:raise ValueError('Prior input lock changed')
    for common in prior['benchmarks'].values():
        for key in ('prepared','manifest','input_spec','tokenizer_manifest'):
            if key in common and sha256(common[key]['path'])!=common[key]['sha256']:raise ValueError('F9 benchmark changed')
        for name,h in read(common['prepared']['path'])['protocol']['tokenizer_files'].items():
            if sha256(Path(common['tokenizer'])/name)!=h:raise ValueError('Tokenizer changed')
    out.mkdir(parents=True)
    command=[prior['benchmarks']['gpt_neox']['eval_python'],'-u','-m','eval.a2_family_f10_tasks',
             '--manifest',cfg['inputs'],'--tokenizer',prior['benchmarks']['gpt_neox']['tokenizer'],
             '--output',str(out/'tasks'),'--code-commit',commit]
    with (out/'prepare.log').open('x') as log:
        subprocess.run(command,env={**os.environ,'CUDA_VISIBLE_DEVICES':''},stdout=log,stderr=subprocess.STDOUT,check=True)
    cfg.update(code_commit=commit,f9_data_root=str(old),f9_config=lock(old/'config.json'),
               f9_manifest=lock(manifest),f10a=lock(aroot/'complete.json'),
               oneshot_entries=oneshots,groups=groups,benchmarks=prior['benchmarks'],batches=prior['batches'],
               prepared=lock(out/'tasks/prepared.json'),FT16=lock(out/'tasks/FT16.json'),
               fresh_model_hashes=verified,input_manifest=lock(cfg['inputs']),retained_flags=checked('flags.json')['flags'])
    for e in oneshots+sum(groups,[]):write(out/'entries'/(e['model_id']+'.json'),e)
    config=write(out/'config.json',cfg)
    write(out/'ready.json',dict(cpu_ready=True,config=config,model_hashes=len(verified),oneshots=11,new_models=42,
                              FT14_pending=True,FT15_pending=True,FT16_recorded=True))
    print('F10_CPU_READY',config,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('plan','output','code-commit'):p.add_argument('--'+k,required=True)
    a=p.parse_args();prepare(a.plan,a.output,a.code_commit)
