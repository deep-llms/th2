"""DEV-ONLY public Hub metadata lock; downloads small code/config files, not weights.

Do not run on B200. Model bytes must subsequently arrive through controller #d.
Output records the exact expected SHA256 and offline paths for A2's loader.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
from huggingface_hub import HfApi
from efficient_llm.a2.protocol import DATASETS,PARENTS,parent_ref
from efficient_llm.a2.data import write_json


def digest(data):return hashlib.sha256(data).hexdigest()


def branch_identity(config,main_config,*,parent,hashes,annealed72000_hash):
    """T3 failure invalidates this branch, never the annealed pair or its sibling."""
    normalized=lambda c:{k:v for k,v in c.items() if k not in ('_name_or_path','transformers_version')}
    reasons=[]
    if normalized(config)!=normalized(main_config):reasons.append('branch config differs')
    name=config.get('_name_or_path',''); main=main_config.get('_name_or_path','')
    step={'S72':'0072000','S188':'0188000'}[parent]
    if name.rsplit('/hf/',1)[0]!=main.rsplit('/hf/',1)[0] or not name.endswith('iter_'+step):
        reasons.append('branch run identity mismatch')
    if any(hashes[parent]==value for p,value in hashes.items() if p!=parent):
        reasons.append('duplicate parent weights')
    if parent=='S72' and hashes[parent]==annealed72000_hash:reasons.append('wrong S72 branch')
    return dict(passed=not reasons,reasons=reasons)


def build(root):
    api=HfApi(); infos={}; bytes_cache={}
    def info(repo,revision):
        key=repo,revision
        if key not in infos:infos[key]=api.model_info(repo,revision=revision,files_metadata=True)
        return infos[key]
    def content(repo,rev,name):
        key=repo,rev,name
        if key not in bytes_cache:
            with urllib.request.urlopen(f'https://huggingface.co/{repo}/resolve/{rev}/{name}',timeout=60) as response:
                bytes_cache[key]=response.read()
        return bytes_cache[key]
    def weight_hash(repo,rev):
        files={s.rfilename:s for s in info(repo,rev).siblings}
        w=files['model.safetensors']
        if not w.lfs or not w.lfs.sha256:raise ValueError('No published weight LFS hash')
        return w.lfs.sha256
    assets={}; tests={}
    for ds in DATASETS:
        configs={}; code_tokens={}; refs={p:parent_ref(ds,p) for p in PARENTS}
        for p,ref in refs.items():
            repo,rev,code=ref['repo'],ref['revision'],ref['code_revision']
            if info(repo,rev).sha!=rev:raise ValueError('Pinned revision did not resolve exactly')
            config=content(repo,rev,'config.json'); configs[p]=json.loads(config)
            hashes={name:digest(content(repo,code,name)) for name in
                    ('modeling_opensci.py','configuration_opensci.py','tokenizer.json','tokenizer_config.json','special_tokens_map.json')}
            code_tokens[p]=hashes
            assets[f'{ds}/{p}']=dict(ref,weights_dir=str(root/ds/p),
                code_dir=str(root/ds/('P300' if p=='P300' else 'P1T')),
                files={'config.json':digest(config),'model.safetensors':weight_hash(repo,rev)},
                code_files={k:v for k,v in hashes.items() if k.endswith('.py')})
        def normalized(config,branch=False):
            return {k:v for k,v in config.items() if k not in (('_name_or_path','transformers_version') if branch else ('_name_or_path',))}
        for parent in ('P300','P1T'):
            name=configs[parent]['_name_or_path']
            required=('global_bs-1008_context-4096_schedule-WSD_lr-4e-3_warmup-25000',
                      'tokenizer-GPT-NeoX', 'samples-'+('300B' if parent=='P300' else '1000B'))
            if not all(field in name for field in required):raise ValueError('T2 undocumented recipe/run name')
        if normalized(configs['P300'])!=normalized(configs['P1T']): raise ValueError('T2 main configs differ')
        if code_tokens['P300']!=code_tokens['P1T']:raise ValueError('T2 code/tokenizer mismatch')
        trunk={p:dict(commit=info(refs[p]['repo'],'iter_0058000').sha,
                     sha256=weight_hash(refs[p]['repo'],'iter_0058000')) for p in ('P300','P1T')}
        if trunk['P300']['sha256']!=trunk['P1T']['sha256']:raise ValueError('T2 shared trunk mismatch')
        hashes={p:assets[f'{ds}/{p}']['files']['model.safetensors'] for p in PARENTS}
        if hashes['P300']==hashes['P1T']:raise ValueError('T2 identical annealed parent weights')
        branch300=weight_hash(refs['P300']['repo'],'iter_0072000')
        branches={p:branch_identity(configs[p],configs['P1T'],parent=p,hashes=hashes,
                                   annealed72000_hash=branch300) for p in ('S72','S188')}
        tests[ds]=dict(T2=True,T3=branches,trunk=trunk,
                       annealed_step72000_commit=info(refs['P300']['repo'],'iter_0072000').sha,
                       annealed_step72000_sha256=branch300,
                       run_names={p:c['_name_or_path'] for p,c in configs.items()},
                       tokenizer_files=code_tokens['P1T'])
    return assets,tests


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--node-asset-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise ValueError('Fresh lock directory required')
    assets,tests=build(args.node_asset_root)
    args.output.mkdir(parents=True)
    write_json(args.output/'assets.json',assets)
    write_json(args.output/'identity_metadata.json',dict(status='passed' if all(
        branch['passed'] for t in tests.values() for branch in t['T3'].values()) else 'partial',tests=tests,
        note='Metadata only. Does not claim downloaded weight hashes/runtime correctness.'))

if __name__=='__main__':main()
