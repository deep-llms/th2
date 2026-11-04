"""Build a Why launch plan from retrieved native terminal and live ownership evidence."""
import argparse
from pathlib import Path
from efficient_llm.a2_why.common import read,write,SPEC_SHA

DEPTH='/mnt/local/_outputs/deep-llms_th2/wc69-a2-depth-partc-20261103-a01'
PAUSE='/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-pause-20261104-a01/pause/terminal.json'


def build(status,predecessor_root,stage,*,data_lock=None,yt1=None):
    root=Path(predecessor_root);sup=str(root/'supervised/supervisor.json');burn=str(root/'supervised/burn-verified.json');source=str(root/'source-manifest.json')
    terminal=status['files'][sup];inspection=status['inspection']
    if not terminal.get('finished_at') or not terminal.get('burn',{}).get('collective_progress_verified') or not status.get('burn_progress'):
        raise ValueError('Native terminal/advancing burn evidence required')
    if inspection['host']!='thiennh-p6-wc69-worker-0' or inspection['guard_disabled']:raise ValueError('Wrong node/guard held')
    records=status['live_identities'];wanted=set()
    for gpu in inspection['gpus']:
        if len(gpu['pids'])!=1:raise ValueError('Unexpected physical GPU mapping')
        pid=gpu['pids'][0];wanted.update((str(pid),str(records[str(pid)]['ppid'])))
    depth=read('resources/a2_depth_wc69_plan_20261103_a01.json')
    cfg={k:depth[k] for k in ('assets','data_root','data_manifest','reference_nll')}
    cfg['reference_nll']={k:v for k,v in cfg['reference_nll'].items() if k.split('/')[0] in ('dclm','fwe') and k.split('/')[1] in ('P300','P1T')}
    cfg.update(spec_sha256=SPEC_SHA,historical=read('resources/a2_depth_stage1_references_20261101.json'),
        heal_streams={'dclm':{'2030':depth['heal_streams']['dclm']}},stage=stage,scientific_policy='flag+continue; no cap',
        predecessor=dict(root=str(root),burn_script=str(root/'source/resources/llm_pretrain_burn.py'),prior_status=terminal['training_status'],
            supervisor=status['locks'][sup],burn=status['locks'][burn],source_manifest=status['locks'][source],known_burn_identities={k:records[k] for k in wanted}))
    if stage=='yt1':
        pause=status['files'][PAUSE]
        if str(root)!=DEPTH or not pause['handoff_verified'] or pause['state']!='paused':raise ValueError('Depth pause not verified')
        cfg['depth_pause']=status['locks'][PAUSE]
    else:
        if not data_lock or not yt1:raise ValueError('Need verified completed data and reused YT1')
        cfg.update(why_data=data_lock,reused_yt1=yt1)
    return cfg

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--status',required=True);p.add_argument('--predecessor-root',required=True);p.add_argument('--stage',choices=('yt1','rest'),required=True);p.add_argument('--output',required=True);p.add_argument('--data-lock');p.add_argument('--yt1')
    a=p.parse_args();write(a.output,build(read(a.status),a.predecessor_root,a.stage,data_lock=read(a.data_lock) if a.data_lock else None,yt1=read(a.yt1) if a.yt1 else None))
