"""Offline actual-byte F6 admission; publishes readiness only after all gates."""
import argparse
import gzip
import json
from pathlib import Path
import shutil
import socket
import time
from efficient_llm.a2_family_f6.common import read, write, SPEC_SHA
from efficient_llm.a2_family_f6.protocol import initial_runs, descriptor
from efficient_llm.a2_family_f6.readout import adapted
from scripts.a2_family_report import pair_scores
from scripts.a2_family_v14_gate import sha, read_lock
from scripts.a2_quant_io import lock
from scripts.launch_a2_downstream import freeze_source


def verify_inputs(cfg):
    """Immutable producer manifests are the trust anchors, not new local hashes."""
    verified=[]
    def check(path, expected):
        if sha(path)!=expected:raise ValueError('F6 input bytes differ: '+str(path))
        verified.append(dict(path=str(path),sha256=expected,bytes=Path(path).stat().st_size))
    assets=read_lock(cfg['assets'])
    for ds in ('dclm','fwe'):
        for arm in ('S72','S188'):
            entry=assets[ds+'/'+arm]
            from efficient_llm.a2.protocol import parent_ref
            if any(entry[k]!=v for k,v in parent_ref(ds,arm).items()):raise ValueError('Wrong A2 S revision')
            for directory,key in [('weights_dir','files'),('code_dir','code_files')]:
                for name,digest in entry[key].items():check(Path(entry[directory])/name,digest)
    manifests=[(cfg['data_root'],cfg['data_manifest'],('final','tuning'))]
    for ds in ('dclm','fwe'):
        stream=cfg['heal_streams'][ds]['2030']
        manifests.append((stream['path'],dict(path=str(Path(stream['path'])/'manifest.json'),sha256=stream['sha256']),('heal',)))
    for directory,item,splits in manifests:
        base=Path(directory);m=read_lock(item)
        if m['status']!='complete':raise ValueError('Incomplete F6 data')
        for split in splits:
            info=m['streams'][split]
            if info['length']!=4096 or info['tokens']!=info['rows']*4096:
                raise ValueError('Wrong packed F6 shape')
            if split=='heal' and info['rows']<12207*64:raise ValueError('Missing 3.2B prefix')
            for suffix,key,scale in [('bin','token_sha256',4),('starts','starts_sha256',1)]:
                path=base/(split+'.'+suffix)
                if path.stat().st_size!=info['tokens']*scale:raise ValueError('F6 data size differs')
                check(path,info[key])
        for name,key in [('documents.bin','documents_sha256'),('document_order.bin','document_order_sha256'),('documents.jsonl','documents_sha256')]:
            if (base/name).is_file() and key in m:check(base/name,m[key])
    audit=read_lock(cfg['overlap_audit'])
    if audit['culturax_manifest']!=cfg['data_manifest'] or audit['pool_manifest']['sha256']!=cfg['heal_streams']['fwe']['2030']['sha256']:
        raise ValueError('Overlap mapping is for different packed data')
    if len(audit['overlap_hashes'])!=3:raise ValueError('Unexpected FWE overlap identity')
    for item in cfg.get('reference_nll',{}).values():read_lock(item)
    return verified


def prepare(plan, source, output, commit):
    source,output=Path(source),Path(output)
    if socket.gethostname()!=plan['host'] or plan['spec_sha256']!=SPEC_SHA:
        raise ValueError('Wrong node/spec')
    if output.exists():raise ValueError('Existing F6 preparation; inspect before any recovery')
    transition=Path(plan['transition_root']);installed=read(transition/'installed.json')
    if not installed['installed'] or installed['gate_sha256']!=sha(transition/'gate.py'):
        raise ValueError('F6 handoff gate not installed or changed')
    previous=Path(plan['family_root'])
    if sha(previous/'source/scripts/a2_family_depth_bridge.py')!=installed['patched_sha256']:
        raise ValueError('F6 bridge changed')
    proof=read(source/'resources/a2_family_f6_cpu_20261108.json')
    if proof['status']!='PASS' or not proof['launch_ready'] or proof['spec_sha256']!=SPEC_SHA:
        raise ValueError('F6 CPU validation incomplete')
    for rel,digest in proof['files'].items():
        if sha(source/rel)!=digest:raise ValueError('F6 code/proof changed: '+rel)
    output.mkdir(parents=True,exist_ok=False)
    root=output/'results/a2_family_20261107';root.mkdir(parents=True)
    cfg=dict(plan,results_root=str(root),source_manifest=None)
    # Controller transfers run independently. Wait only for missing final paths;
    # an existing file with wrong bytes is an error, never silently overwritten.
    assets=read_lock(cfg['assets'])
    required=[Path(assets[ds+'/'+arm]['weights_dir'])/name for ds in ('dclm','fwe')
              for arm in ('S72','S188') for name in assets[ds+'/'+arm]['files']]
    while any(not p.is_file() for p in required):
        write(output/'progress.json',dict(state='waiting_model_delivery',missing=[str(p) for p in required if not p.is_file()],ready=False))
        time.sleep(30)
    write(output/'progress.json',dict(state='verifying_actual_bytes',ready=False))
    verified=verify_inputs(cfg)
    write(root/'manifests/actual_inputs.json',dict(status='verified',files=verified))
    write(root/'manifests/model_revisions.json',dict(assets={ds+'/'+arm:assets[ds+'/'+arm]
          for ds in ('dclm','fwe') for arm in ('S72','S188')}))
    historical_path=source/plan['historical_file']['path']
    if sha(historical_path)!=plan['historical_file']['sha256']:raise ValueError('Historical arrays changed')
    historical=json.loads(gzip.decompress(historical_path.read_bytes()))
    # Preserve the old spec SHA inside the original records; do not relabel them.
    write(root/'historical/A2_S.json',dict(original=historical))
    refs={};summary={};audit=read_lock(cfg['overlap_audit'])
    for ds in ('dclm','fwe'):
        pair={}
        for arm in ('S72','S188'):
            old=historical['runs'][f'ST1-{ds}-{arm}-S1-0.2B-lr1e-3'];pair[arm]=adapted(old)
            refs[ds+'/'+arm+'/parent']=write(root/f'historical/{ds}-{arm}-parent.json',old['baseline'])
        full=pair_scores(pair['S72'],pair['S188'],repeats=0)
        reduced=pair_scores(pair['S72'],pair['S188'],audit['exclude_final_sequence_ids'] if ds=='fwe' else (),repeats=0)
        summary[ds]=dict(E_full=full['E'],E_reduced=reduced['E'],full=full,reduced=reduced,
                         delta_post={arm:historical['runs'][f'ST1-{ds}-{arm}-S1-0.2B-lr1e-3']['delta_post'] for arm in ('S72','S188')},
                         origin='A2 S first node 6y3o; exact saved arrays; no rerun except FT9')
        summary[ds]['E_over_short_delta']=full['E']/summary[ds]['delta_post']['S72']
    cfg.update(reference_nll=refs,historical=summary,
               ft9_reference=historical['runs']['ST1-dclm-S72-S1-0.2B-lr1e-3']['scores']['final']['loss'])
    run_list=initial_runs()+[descriptor(ds,arm,'3.2B') for ds in ('dclm','fwe') for arm in ('S72','S188')]
    cfg['runs']={r['run_id']:r for r in run_list}
    extra=['docs/A2_family_spec_v1.4.md','resources/a2_family_f6_cpu_20261108.json',
           *[f for f in proof['files'] if f.startswith(('tests/','resources/'))]]
    frozen=freeze_source(source,output,commit,extra_files=extra)
    cfg['source_manifest']=lock(output/'source-manifest.json')
    config=write(root/'config/plan.json',cfg)
    shutil.copyfile(source/'docs/A2_family_spec_v1.4.md',root/'config/spec.md')
    write(root/'overlap_audit/pool.json',dict(original=audit,source_lock=cfg['overlap_audit']))
    ready=dict(spec_sha256=SPEC_SHA,source=str(frozen),source_manifest=cfg['source_manifest'],config=config)
    from scripts.a2_family_v14_gate import verify_ready
    verify_ready(ready)
    if (transition/'ready.json').exists():raise ValueError('F6 readiness already published')
    write(transition/'ready.json',ready)
    write(output/'progress.json',dict(state='ready_for_native_F6_after_F4',ready=True,training_started=False))
    print('FAMILY_F6_READY',json.dumps(ready),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--output',required=True);p.add_argument('--code-commit',required=True)
    a=p.parse_args();prepare(read(a.plan),Path(__file__).resolve().parents[1],a.output,a.code_commit)
