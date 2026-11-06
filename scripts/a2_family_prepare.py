"""CPU-only Family admission preparation; background downloads stay independent."""
import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from pathlib import Path
from efficient_llm.a2_family.common import read, write, sha, SPEC_SHA


def wait_file(path, info, progress):
    path=Path(path)
    while not path.is_file() or path.stat().st_size!=info['bytes']:
        write(progress,dict(state='waiting_download',path=str(path),expected_bytes=info['bytes'],actual_bytes=path.stat().st_size if path.is_file() else None))
        time.sleep(30)
    actual=sha(path)
    if actual!=info['sha256']:
        raise ValueError(f'Completed-size download hash mismatch: {path}')
    return dict(path=str(path),bytes=path.stat().st_size,sha256=actual)


def work(root):
    root=Path(root);models=read('resources/a2_family_downloads_20261107.json');inputs=read('resources/a2_family_data_inputs_20261107.json')
    model_lock=sha('resources/a2_family_downloads_20261107.json');input_lock=sha('resources/a2_family_data_inputs_20261107.json')
    def verify_models():
        entries=[]
        for e in models['checkpoints']:
            for f in e['files']:entries.append(wait_file(Path(e['path'])/f['name'],f,root/'models_progress.json'))
        return write(root/'models_ready.json',dict(status='verified',files=entries,manifest_sha256=model_lock))
    def prepare_cx():
        for f in [inputs['shared_tokenizer'],*inputs['files']]:wait_file(f['local'],f,root/'cx_progress.json')
        out=root/'cx'
        with (root/'cx.log').open('x') as log:
            rc=subprocess.run([sys.executable,'-u','-m','scripts.a2_family_prepare_cx','--inputs','resources/a2_family_data_inputs_20261107.json',
                 '--output',str(out),'--reference','resources/a2_family_cx_reference_20261107.json'],stdout=log,stderr=subprocess.STDOUT).returncode
        if rc!=0:raise RuntimeError(f'CX preparation exit {rc}; retain log and outputs')
        value=read(out/'complete.json')
        if value['status']!='complete':raise ValueError('CX incomplete')
        return dict(path=str(out/'complete.json'),sha256=sha(out/'complete.json'))
    def prepare_fwe():
        f=inputs['shared_tokenizer'];wait_file(f['local'],f,root/'fwe_progress.json');out=root/'fwe'
        with (root/'fwe.log').open('x') as log:
            rc=subprocess.run([sys.executable,'-u','-m','scripts.a2_family_prepare_fwe',
                '--plan','/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-data-20261104-a01/corpus_plan.json',
                '--old-shuffle','/mnt/local/_data/deep-llms_th2/a2_why_fwe_shuffle2030_20261104',
                '--downloads','resources/a2_family_downloads_20261107.json',
                '--data-inputs','resources/a2_family_data_inputs_20261107.json','--output',str(out)],stdout=log,stderr=subprocess.STDOUT).returncode
        if rc!=0:raise RuntimeError(f'FWE preparation exit {rc}; retain log and outputs')
        if read(out/'complete.json')['status']!='complete':raise ValueError('FWE incomplete')
        return dict(path=str(out/'complete.json'),sha256=sha(out/'complete.json'))
    results={};failures={}
    with concurrent.futures.ThreadPoolExecutor(3) as pool:
        tasks={pool.submit(fn):name for name,fn in [('models',verify_models),('cx',prepare_cx),('fwe',prepare_fwe)]}
        for future in concurrent.futures.as_completed(tasks):
            name=tasks[future]
            try:results[name]=future.result()
            except Exception as error:failures[name]=repr(error)
            write(root/'progress.json',dict(stage='preparing',ready=results,failures=failures))
    write(root/'complete.json',dict(status='complete' if not failures else 'incomplete',ready=results,failures=failures,
                model_manifest_sha256=model_lock,data_inputs_sha256=input_lock,
                note='CPU preparation only; no Depth pause or Family GPU launch performed'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();work(a.root)
