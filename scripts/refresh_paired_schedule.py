"""Re-freeze execution hashes before an authorized, not-yet-started paired study.

Preserves original inputs as immutable backups. Refuses any scientific-rule,
example or score change. The running fine-tuning process does not use these files.
"""
import argparse
import copy
import json
import os
from pathlib import Path


def read(path):return json.loads(Path(path).read_text())


def running_finetune(root):
    run=read(root/'supervised/run/run.json')
    if (run['status']!='running' or run['jobs'][-1]['name']!='finetune-expanded'
            or run['jobs'][-1]['status']!='running' or (root/'supervised/run/paired').exists()):
        raise ValueError('Refresh only while fine-tuning is active and paired planning has not started')


def refresh(root, expected, output):
    from eval.paired import protocol, position
    root=Path(root);output=Path(output)
    if output.exists():raise FileExistsError(output)
    running_finetune(root)
    jobs=read(root/'supervised/run/jobs.snapshot.json')['jobs']
    argv=next(j['argv'] for j in jobs if j['name']=='paired-plan')
    frozen_path=Path(argv[argv.index('--protocol')+1]);sample_path=Path(argv[argv.index('--position-sample')+1])
    old=read(frozen_path);sample=read(sample_path)
    if protocol.digest({k:v for k,v in old.items() if k!='protocol_sha256'})!=old['protocol_sha256']:
        raise ValueError('Invalid previous frozen protocol hash')
    position.read_sample(sample_path,old)
    new=copy.deepcopy(protocol.definition(old['position']['documents']))
    allowed=read(expected)
    for key in old.keys()-{'implementation','protocol_sha256'}:
        if old[key]!=new.get(key):raise ValueError('Scientific protocol changed: '+key)
    if set(new)-{'implementation'} != set(old)-{'implementation','protocol_sha256'}:
        raise ValueError('Scientific protocol fields changed')
    changed={k for k,v in old['implementation'].items() if new['implementation'].get(k)!=v}
    added=set(new['implementation'])-set(old['implementation'])
    if changed!=set(allowed['before']) or added!=set(allowed['added']):
        raise ValueError(('Unexpected implementation delta',sorted(changed),sorted(added)))
    if any(old['implementation'][k]!=v for k,v in allowed['before'].items()):
        raise ValueError('Previous implementation differs from reviewed source')
    new['protocol_sha256']=protocol.digest(new)
    new_sample=copy.deepcopy(sample);new_sample['protocol_sha256']=new['protocol_sha256']
    new_sample['sample_sha256']=protocol.digest({k:v for k,v in new_sample.items() if k!='sample_sha256'})
    pairs=[]
    for path,body in ((frozen_path,new),(sample_path,new_sample)):
        backup=path.with_name(path.name+'.before-parallel-20261013')
        temporary=path.with_name(path.name+'.parallel-pending')
        with backup.open('xb') as stream:stream.write(path.read_bytes())
        protocol.write_json(temporary,body)
        pairs.append((path,temporary,backup))
    protocol.read(pairs[0][1]);position.read_sample(pairs[1][1],new)
    running_finetune(root)
    for path,temporary,backup in pairs:
        if path.read_bytes()!=backup.read_bytes():raise ValueError('Input changed during refresh')
    for path,temporary,_ in pairs:os.replace(temporary,path)
    protocol.read(frozen_path);position.read_sample(sample_path,new)
    protocol.write_json(output,dict(status='passed',queue_root=str(root),
        previous_protocol_sha256=old['protocol_sha256'],protocol_sha256=new['protocol_sha256'],
        previous_sample_sha256=sample['sample_sha256'],sample_sha256=new_sample['sample_sha256'],
        rows_sha256=protocol.digest(sample['rows']),rows_unchanged=sample['rows']==new_sample['rows'],
        changed=sorted(changed),added=sorted(added),workers='auto: one per visible GPU',
        backups=[str(p[2]) for p in pairs]))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','expected','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();refresh(a.root,a.expected,a.output)
