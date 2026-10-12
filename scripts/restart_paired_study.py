"""Fresh paired-only retry after the nested-callable serialization repair.

The failed study and completed fine-tuning artifacts remain immutable. Freeze
the repaired implementation with identical scientific settings and token rows.
"""
import copy
import hashlib
from pathlib import Path

from eval.paired import artifacts, position, protocol
from run_experiments import load_jobs

PREVIOUS_RUN_SHA256 = 'cd0ebb7ede0c7a50add6f3667648f382a8d0f3abc5c1f510b1eaf88d2db80901'
STAGES = ['paired-plan', 'paired-evaluate', 'analyze-F', 'analyze-Z', 'analyze-probes', 'analyze-position']


def prepare(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    run = artifacts.load_json(source/'supervised/run/run.json')
    supervisor = artifacts.load_json(source/'supervised/supervisor.json')
    burn = artifacts.load_json(source/'supervised/burn-verified.json')
    if (run['status'] != 'failed' or run['jobs'][-1]['name'] != 'paired-evaluate'
            or supervisor.get('burn') != burn or not burn.get('collective_progress_verified')):
        raise ValueError('Require failed paired stage and completed verified burn handoff')
    if artifacts.load_json(source/'supervised/run/finetune/summary.json')['status'] != 'passed':
        raise ValueError('Expected completed fine-tuning')
    old_plan = artifacts.load_json(source/'supervised/run/paired/plan.json')
    if old_plan['plan_sha256'] != protocol.digest({k:v for k,v in old_plan.items() if k != 'plan_sha256'}):
        raise ValueError('Source plan changed')
    old = artifacts.load_json(old_plan['protocol'])
    if old['protocol_sha256'] != protocol.digest({k:v for k,v in old.items() if k != 'protocol_sha256'}):
        raise ValueError('Source protocol changed')
    if old['protocol_sha256'] != old_plan['protocol_sha256']:
        raise ValueError('Source plan/protocol mismatch')
    sample = position.read_sample(old_plan['position_sample'], old)
    if sample['sample_sha256'] != old_plan['sample_sha256']:
        raise ValueError('Source sample changed')
    new = copy.deepcopy(protocol.definition(old['position']['documents']))
    old_science = {k:v for k,v in old.items() if k not in ('implementation','protocol_sha256')}
    if old_science != {k:v for k,v in new.items() if k != 'implementation'}:
        raise ValueError('Scientific settings changed')
    changed = {k for k in old['implementation'].keys() | new['implementation'].keys()
               if old['implementation'].get(k) != new['implementation'].get(k)}
    if changed != {'eval/paired/run.py'} or old['implementation']['eval/paired/run.py'] != PREVIOUS_RUN_SHA256:
        raise ValueError('Unexpected implementation change')
    jobs = artifacts.load_json(source/'supervised/run/jobs.snapshot.json')['jobs']
    jobs = copy.deepcopy([j for j in jobs if j['name'] in STAGES])
    if [j['name'] for j in jobs] != STAGES:
        raise ValueError('Unexpected source queue')
    destination.mkdir(parents=True)
    frozen_path = destination/'paired-protocol.json'
    new['protocol_sha256'] = protocol.digest(new)
    protocol.write_json(frozen_path,new)
    new_sample = copy.deepcopy(sample)
    new_sample['protocol_sha256'] = new['protocol_sha256']
    new_sample['sample_sha256'] = protocol.digest({k:v for k,v in new_sample.items() if k != 'sample_sha256'})
    sample_path = destination/'position-sample.json'
    protocol.write_json(sample_path,new_sample)
    manifest = artifacts.load_json(old_plan['manifest'])
    if hashlib.sha256(Path(old_plan['manifest']).read_bytes()).hexdigest() != old_plan['manifest_sha256']:
        raise ValueError('Source dataset manifest changed')
    repaired = []
    for repo in manifest['repositories']:
        config = repo.get('load_configs',{}).get('xstorycloze_en')
        if config is not None and config['loader'] == 'csv':
            if config.get('kwargs',{}) not in ({}, {'sep':'\t'}):
                raise ValueError('Unexpected XStoryCloze parser settings')
            if not all(f.endswith('.tsv') for files in config['data_files'].values() for f in files):
                raise ValueError('Expected XStoryCloze TSV files')
            config['kwargs'] = {'sep':'\t'}
            repaired.append('xstorycloze_en: explicit TSV separator')
    manifest_path = destination/'manifest.json'
    protocol.write_json(manifest_path,manifest)
    for job in jobs:
        job['argv'] = [str(frozen_path) if arg == old_plan['protocol'] else
                       str(sample_path) if arg == old_plan['position_sample'] else
                       str(manifest_path) if arg == old_plan['manifest'] else arg for arg in job['argv']]
    protocol.write_json(destination/'jobs.json',dict(jobs=jobs))
    load_jobs(destination/'jobs.json')
    protocol.read(frozen_path);position.read_sample(sample_path,new)
    protocol.write_json(destination/'restart-receipt.json',dict(status='passed',source=str(source),
        previous_protocol_sha256=old['protocol_sha256'],protocol_sha256=new['protocol_sha256'],
        changed=sorted(changed),rows_unchanged=new_sample['rows']==sample['rows'],
        data_loading_repairs=repaired,previous_manifest_sha256=old_plan['manifest_sha256'],
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        rows_sha256=protocol.digest(sample['rows']),stages=STAGES,
        policy='fresh paired outputs; preserve all old artifacts; no fine-tuning rerun'))
