"""Verify pinned inputs, source, prior report and paused state on wc69."""
from pathlib import Path
from efficient_llm.a2_family_v17.common import read,write,SPEC_SHA
from scripts.a2_family_v14_gate import sha,read_lock
from scripts.a2_family_v17_depth_bridge import validate


def verify_inputs(cfg):
    verified=[]
    def check(p,h):
        p=Path(p)
        if sha(p)!=h:raise ValueError('Input SHA mismatch: '+str(p))
        verified.append(dict(path=str(p),sha256=h,bytes=p.stat().st_size))
    assets=read_lock(cfg['assets'])
    from efficient_llm.a2.protocol import parent_ref
    for arm in ('P300','P1T','S72','S188'):
        entry=assets['dclm/'+arm]
        if any(entry[k]!=v for k,v in parent_ref('dclm',arm).items()):raise ValueError('Wrong pinned revision: '+arm)
        for directory,key in [('weights_dir','files'),('code_dir','code_files')]:
            for name,h in entry[key].items():check(Path(entry[directory])/name,h)
    stream=cfg['heal_streams']['dclm']['2030']
    for base,lock,splits in [(Path(cfg['data_root']),cfg['data_manifest'],('final','tuning')),
        (Path(stream['path']),dict(path=str(Path(stream['path'])/'manifest.json'),sha256=stream['sha256']),('heal',))]:
        m=read_lock(lock)
        if m['status']!='complete':raise ValueError('Incomplete data')
        for split in splits:
            info=m['streams'][split]
            if info['length']!=4096 or info['tokens']!=info['rows']*4096:raise ValueError('Wrong packing')
            if split=='heal' and info['rows']<12207*64:raise ValueError('Missing consumed prefix')
            for suffix,key,scale in [('bin','token_sha256',4),('starts','starts_sha256',1)]:
                p=base/(split+'.'+suffix)
                if p.stat().st_size!=info['tokens']*scale:raise ValueError('Wrong data size')
                check(p,info[key])
        for name,key in [('documents.bin','documents_sha256'),('document_order.bin','document_order_sha256')]:
            if (base/name).exists() and key in m:check(base/name,m[key])
    for lock in cfg['reference_nll'].values():read_lock(lock)
    for key in ('previous_report','previous_decision','previous_backup'):read_lock(cfg[key])
    previous=read_lock(cfg['previous_backup'])
    if not previous['verified']:raise ValueError('Previous backup unverified')
    for item in previous['files']:check(item['path'],item['sha256'])
    validate(cfg)
    return verified
