"""F6 foreground continuation under the existing Family native supervisor."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from efficient_llm.a2_family_f6.common import read, write, Checks, SPEC_SHA
from efficient_llm.a2_family_f6.protocol import initial_runs, descriptor, select_extension, joint_reading
from efficient_llm.a2_family_f6.readout import analyze
from scripts.a2_family_v14_gate import sha, read_lock, verify_ready
from scripts.gpu_status import require_free
from scripts.gpu_process import process


def owner(cfg):
    guard = read('/mnt/local/_gpu_guard/DISABLED')
    expected = cfg['native_supervisor']
    if socket.gethostname()!=cfg['host'] or guard['owner_pid']!=expected['pid'] or process(expected['pid'])!=expected:
        raise ValueError('F6 native GPU ownership changed')
    pid = os.getpid()
    while pid != expected['pid'] and pid > 1:
        pid = process(pid)['ppid']
    if pid != expected['pid']:
        raise ValueError('F6 is not a child of the authorized native supervisor')


def launch(cfg, config, item, events):
    owner(cfg)
    for attempt in range(3):
        try:
            require_free(list(range(8))); break
        except RuntimeError:
            if attempt == 2: raise
            time.sleep(5)
    name = item['run_id']; root = Path(cfg['results_root'])
    if any(e['name']==name for e in events) or (root/'heals'/name).exists():
        raise ValueError('Duplicate F6 attempt')
    write(root/'progress.json', dict(active=name, events=events))
    log = root/'logs'/(name+'.log'); log.parent.mkdir(exist_ok=True)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0)); port = str(sock.getsockname()[1])
    command = [sys.executable,'-u','-m','accelerate.commands.launch','--config_file',
               'resources/accelerate_b200.yaml','--main_process_port',port,'-m',
               'efficient_llm.a2_family_f6.heal','--config',str(config),'--run-id',name]
    started = time.monotonic()
    with log.open('x') as handle:
        rc = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                            env={**os.environ,'CUDA_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7'}).returncode
    event = dict(name=name,returncode=rc,seconds=time.monotonic()-started,log=str(log))
    events.append(event)
    with (root/'runs.jsonl').open('a') as f:
        f.write(json.dumps(dict(event,spec_sha256=SPEC_SHA))+'\n')
    write(root/'ledger.json',dict(events=events,node_hours=sum(x['seconds'] for x in events)/3600))
    print('F6_RUN_EVENT',json.dumps(event),flush=True)


def build_report(cfg, events, extension, final=False):
    root = Path(cfg['results_root'])
    results = [read(p) for p in sorted(root.glob('heals/*/result.json'))]
    decision = analyze(results, read_lock(cfg['overlap_audit']), cfg['historical'], final=final)
    old_root = Path(cfg['previous_results_root'])
    old = read(old_root/'decision_family.json')
    smol_ratios=[]
    primary=old.get('overlap',{}).get('fwe',{}).get('primary')
    for pair in old.get('pairs',[]):
        if pair['quantity']!='E' or pair['data']!='fwe' or pair['seed']!=2030 or pair['lr']!=old.get('lr_selection',{}).get('lr') or primary not in ('full','reduced'):
            continue
        short=next((r for r in old['rows'] if (r['arm'],r['data'],r['budget'],r['lr'],r['stream_seed'])==('k6','fwe',pair['budget'],pair['lr'],2030)),None)
        if short is None:continue
        delta=short['delta_post' if primary=='full' else 'reduced_delta_post']
        smol_ratios.append(dict(budget=pair['budget'],E=pair[primary]['E'],short_delta_post=delta,
                               E_over_short_delta=pair[primary]['E']/delta if abs(delta)>1e-15 else None,primary=primary))
    ft9=[r for r in results if r['registered_run']['phase']=='FT9']
    ft9_checks=[c for r in ft9 for c in r.get('flags',[]) if c['test']=='FT9']
    ft9_ok=bool(ft9) and not ft9_checks and bool(ft9[0].get('scores',{}).get('final'))
    missing=[e for e in events if e['returncode'] or not (root/'heals'/e['name']/'result.json').exists()]
    for ds,value in decision['FE6'].items():
        value['unverified']=not ft9_ok or bool(decision['missing_or_invalid']) or any(r['flags'] for r in decision['rows'] if r['dataset']==ds)
    decision.update(events=events,extension=extension,
                    historical_02B=cfg['historical'],Smol_E_over_short_delta=smol_ratios,
                    missing_or_failed_runs=missing,FT9_pass=ft9_ok,FT9_flags=ft9_checks,
                    joint_reading=joint_reading(decision['FE6'],old.get('readings',{}).get('FE2',{})),
                    joint_reading_unverified=any(v.get('unverified') for v in decision['FE6'].values()) or old.get('readings',{}).get('FE2',{}).get('unverified',True),
                    previous_decision=dict(path=str(old_root/'decision_family.json'),sha256=sha(old_root/'decision_family.json')),
                    historical_origin='A2 S on 6y3o; full .2B values reused, not replaced by FT9; node differences retained',
                    FT9_reference=cfg['ft9_reference'])
    write(root/'decision_family.json',dict(previous_F0_F5=old,F6=decision))
    write(root/'decision_f6.json',decision)
    if final:
        tests=[]
        for p in sorted(root.glob('heals/*/tests-rank*.json')):
            tests.extend(dict(x,path=str(p)) for x in read(p)['checks'])
        write(root/'tests.json',dict(checks=tests,failed=[x for x in tests if not x['passed']]))
        text=(old_root/'REPORT.md').read_text()
        text+='\n\n# Family v1.4 — F6 stable-state extension\n\n'
        text+='Original F0–F5 above retain their v1.3 provenance and scientific flags. F6 uses its own locked v1.4 source.\n\n'
        text+='| Run | Parent | NLL | Δpre | Δpost | Reduced NLL | Reduced Δpost | Flags |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
        for r in decision['rows']:
            text+=f"| {r['run_id']} | {r['parent']:.9f} | {r['NLL']:.9f} | {r['delta_pre']:+.9f} | {r['delta_post']:+.9f} | {r['reduced_NLL']:.9f} | {r['reduced_delta_post']:+.9f} | {len(r['flags'])} |\n"
        text+='\nFull numeric results, CIs, exposure and failed checks follow; no failed row is silently marked verified.\n\n```json\n'+json.dumps(decision,indent=2)+'\n```\n'
        (root/'REPORT.md').write_text(text)
    return decision


def execute(cfg, config, receipt):
    root=Path(cfg['results_root']);owner(cfg)
    ready=read(Path(cfg['transition_root'])/'ready.json');verify_ready(ready)
    # Hash actual bytes again at GPU admission, not merely at CPU preparation.
    from scripts.a2_family_f6_prepare import verify_inputs
    verify_inputs(cfg)
    previous=read(Path(cfg['previous_results_root'])/'backup/verified.json')
    if not previous['verified']:raise ValueError('Prior Family backup incomplete')
    for item in previous['files']:
        if sha(item['path'])!=item['sha256']:raise ValueError('Prior Family backup changed')
    events=[];checks=Checks(root/'coordinator_checks.json');extension=None
    for item in initial_runs():
        launch(cfg,config,item,events)
        checks.attempt('partial_readout',lambda:build_report(cfg,events,extension))
    partial=checks.attempt('extension_readout',lambda:build_report(cfg,events,None))
    if partial is None:
        choice=select_extension({}, {})
    else:
        e08={p['dataset']:p['reduced' if p['dataset']=='fwe' and partial['overlap']['primary_FWE']=='reduced' else 'full']['E']
             for p in partial['pairs'] if p['budget']=='0.8B'}
        choice=select_extension(partial['E02'],e08)
    extension=choice['dataset']
    write(root/'extension_decision.json',dict(**choice,basis=partial['overlap'] if partial else None,
                                            note='Immutable trigger-time evidence; later results do not rewrite executed selection'))
    for ds,row in choice['datasets'].items():checks.record('F6_trigger_evaluable:'+ds,row['trigger'] is not None,row)
    if extension:
        for arm in ('S72','S188'):launch(cfg,config,descriptor(extension,arm,'3.2B'),events)
    decision=build_report(cfg,events,choice,final=True)
    # All raw rank traces (including partial failed attempts), arrays and reports
    # are retained. v1.3 scientific backup is referenced and verified separately.
    from scripts.a2_family_f6_backup import backup
    saved=backup(root)
    backup_files=[dict(path=str(root/'backup/verified.json'),sha256=sha(root/'backup/verified.json')),
                  *saved['files'],*previous['files']]
    write(receipt,dict(execution_complete=True,report_complete=True,backup_verified=True,
          extension_dataset=extension,events=events,
          report=dict(path=str(root/'REPORT.md'),sha256=sha(root/'REPORT.md')),
          decision=dict(path=str(root/'decision_family.json'),sha256=sha(root/'decision_family.json')),
          backup_files=backup_files,scientific_failures_preserved=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--receipt',required=True)
    a=p.parse_args();cfg=read(a.config)
    if cfg['spec_sha256']!=SPEC_SHA:raise ValueError('Wrong F6 spec')
    root=Path(cfg['results_root']);root.mkdir(parents=True,exist_ok=True)
    with (root/'coordinator.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        execute(cfg,a.config,a.receipt)
