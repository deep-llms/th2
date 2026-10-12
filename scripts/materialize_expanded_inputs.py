"""Verify controller-downloaded raw snapshots and normalize English XCSR offline."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from eval.benchmarks import local_dataset_paths
from scripts.prepare_expanded_benchmarks import normalize_xcsr


def materialize(root, source, output):
    from datasets import Dataset, load_dataset
    root=Path(root).resolve();output=Path(output)
    if output.exists():raise ValueError('Choose a fresh output manifest')
    mapping=local_dataset_paths(root,source)  # Verify every pinned source before writing.
    result=copy.deepcopy(json.loads(Path(source).read_text()))
    for repo in result['repositories']:
        for task in repo['tasks']:
            if task not in ('xcsqa_en','xcodah_en'):continue
            config=mapping[task]
            rows=load_dataset(config['dataset_path'],split='validation',**config['dataset_kwargs'])
            normalized=Dataset.from_list([normalize_xcsr(row) for row in rows])
            relative=str(Path(repo['path'])/'normalized'/(task+'.parquet'))
            destination=root/relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            temporary=destination.with_suffix('.pending.parquet')
            if temporary.exists():raise ValueError('Incomplete normalization exists; inspect before retrying')
            normalized.to_parquet(temporary)
            if destination.exists():
                if destination.read_bytes()!=temporary.read_bytes():
                    temporary.unlink();raise ValueError('Refusing to overwrite different XCSR output')
                temporary.unlink()
            else:temporary.rename(destination)
            repo['files'].append(dict(path=relative,bytes=destination.stat().st_size,
                sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),rows=len(normalized)))
            repo['load_configs'][task]=dict(loader='parquet',data_files=dict(validation=[relative]))
    result['raw_manifest_sha256']=hashlib.sha256(Path(source).read_bytes()).hexdigest()
    with output.open('x') as stream:json.dump(result,stream,indent=2)
    local_dataset_paths(root,output)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','source','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();materialize(a.root,a.source,a.output)
    print('EXPANDED_INPUTS_VERIFIED_AND_READY',flush=True)
