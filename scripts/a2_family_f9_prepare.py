"""CPU-only F9 admission: hash actual models/data and prepare both tokenizers.

Consumes a reviewed census-to-report mapping. Unknown models remain explicitly
unclassified rather than being assigned an invented parent or experiment.
"""
import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
from eval.a2_family_f9 import SPEC_SHA,read,write
from efficient_llm.diagnostic_data import sha256
from scripts.a2_quant_io import lock
from scripts.a2_family_f9_run import locked


def hash_entry(entry):
    directory=Path(entry['path']);entry=dict(entry);files={};expected=entry.get('expected_files',{})
    if entry.get('origin_recipe_path'):
        from scripts.a2_family_f9_inventory import compact
        recipe=read(entry['origin_recipe_path']);result=read(entry['origin_result_path'])
        reviewed=hashlib.sha256(json.dumps(dict(recipe=compact(recipe),result=compact(result)),sort_keys=True).encode()).hexdigest()
        if reviewed!=entry['census_metadata_sha256'] or result.get('status')!='complete':
            raise ValueError('Actual run metadata differs from reviewed census')
        for path in (entry['origin_recipe_path'],entry['origin_result_path']):files[path]=sha256(path)
        if read(directory/'config.json')!=entry['recorded_model_config']:
            raise ValueError('Actual model config differs from reviewed census')
    weights=sorted([*directory.glob('*.safetensors'),*directory.glob('pytorch_model.bin')])
    if not weights or not (directory/'config.json').is_file():raise ValueError('Model weights/config missing')
    for p in [directory/'config.json',*weights,*directory.glob('*.index.json')]:
        files[str(p)]=sha256(p)
        if p.name in expected and files[str(p)]!=expected[p.name]:raise ValueError('Known model checksum mismatch: '+str(p))
    for name in expected:
        if not (directory/name).is_file() or sha256(directory/name)!=expected[name]:
            raise ValueError('Known asset checksum mismatch: '+str(directory/name))
    if entry['family']!='smollm3':
        code=Path(entry['code_dir'])
        for name,h in entry['expected_code'].items():
            p=code/name
            if sha256(p)!=h:raise ValueError('Pinned native model code differs: '+str(p))
            files[str(p)]=h
    entry.update(file_hashes=files,spec_sha256=SPEC_SHA)
    for key in ('model_id','run_id','parent_id','dataset','arm','cut','variant','budget','lr','seed','trained_node','part','phase','batch_family'):
        if key not in entry:raise ValueError('Missing required inventory metadata: '+key)
    return entry


def prepare(plan,output):
    cfg=read(plan);root=Path(output)
    if cfg['spec_sha256']!=SPEC_SHA or root.exists():raise ValueError('Wrong spec or existing preparation')
    root.mkdir(parents=True);errors=[];models=[]
    for item in cfg['locked_inputs']:
        if sha256(item['path'])!=item['sha256']:raise ValueError('Changed input: '+item['path'])
    if cfg.get('opensci_cx_root'):
        from efficient_llm.a2.data import PackedRows
        # The manifest is locked above; verify the actual packed binary files too.
        PackedRows(cfg['opensci_cx_root'],'final',verify=True)
    alias_checks=[]
    for alias in cfg.get('aliases',[]):
        matches={name:sha256(Path(alias['path'])/name)==h for name,h in alias['expected_files'].items()}
        alias_checks.append(dict(alias=alias,checks=matches,identical=all(matches.values())))
        if not all(matches.values()):raise ValueError('Parent alias differs; inventory must be revised: '+alias['path'])
    def one(e):
        try:return hash_entry(e),None
        except Exception as error:return None,dict(model=e['model_id'],path=e['path'],error=repr(error))
    # Bound disk hashing so the host remains responsive.
    with ThreadPoolExecutor(max_workers=2) as pool:
        for entry,error in pool.map(one,cfg['models']):
            if error:errors.append(error)
            else:models.append(entry)
    if len({e['model_id'] for e in models})!=len(models):raise ValueError('Duplicate inventory model ID')
    for benchmark in cfg['benchmarks'].values():
        for key in ('manifest','input_spec'):locked(benchmark[key])
        family=benchmark.get('tokenizer_family','gpt_neox');target=root/(family+'_prepared.json')
        args=[benchmark['eval_python'],'-u','-m','eval.a2_requests','prepare',
            '--tokenizer',benchmark['tokenizer'],'--input-spec',benchmark['input_spec']['path'],
            '--dataset-root',benchmark['dataset_root'],'--manifest',benchmark['manifest']['path'],
            '--tasks',*benchmark['tasks'],'--output',str(target)]
        if family=='smollm3':
            locked(benchmark['tokenizer_manifest'])
            args+=['--tokenizer-family',family,'--tokenizer-manifest',benchmark['tokenizer_manifest']['path']]
        with (root/(family+'_prepare.log')).open('x') as stream:
            subprocess.run(args,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1'},
                stdout=stream,stderr=subprocess.STDOUT,check=True)
        write(target,read(target))
        benchmark['prepared']=lock(target)
    inventory=write(root/'FT12_inventory.json',dict(models=models,missing=cfg.get('missing',[]),
        excluded=cfg.get('excluded',[]),unclassified=cfg.get('unclassified',[]),hash_failures=errors,
        verified_parent_aliases=alias_checks,
        actual_census=cfg['census'],loads_pending=True,spot_checks_pending=True))
    cfg['inventory']=inventory;cfg.pop('models');cfg['preparation_flags']=errors
    config=write(root/'ready_config.json',cfg)
    write(root/'ready.json',dict(cpu_ready=True,config=config,models=len(models),flags=errors,
        note='CPU hashes and benchmark requests ready; FT12 GPU loads/spot checks and FT11 remain'))
    print('F9_CPU_READY',len(models),'models',len(errors),'hash failures',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();prepare(a.plan,a.output)
