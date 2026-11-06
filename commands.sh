#1 +120+a
#th2-wc69-why-overlap-depth-health-20261106-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'HEALTH'
from pathlib import Path
import json,hashlib,subprocess
from datetime import datetime,timezone
from scripts.gpu_process import inspect,process
from scripts.a2_remaining import read_locked
from scripts.train_then_burn import digest
root=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-depth-partc-20261103-a01');base=root/'results/a2_depth_20261031'
def artifact(p):
 if not p.is_file():return None
 raw=p.read_bytes();v=json.loads(raw) if p.suffix=='.json' else raw.decode(errors='replace')
 return dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),value=v)
def tail(p,n=4500):
 if not p.is_file():return None
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-n));v=f.read().decode(errors='replace')
 return dict(path=str(p),bytes=p.stat().st_size,text=v)
r=dict(utc=datetime.now(timezone.utc).isoformat(),inspection=inspect());chains={}
for worker in r['inspection']['workers']:
 pid=worker;rows=[]
 while pid>1 and len(rows)<12:
  try:v=process(pid)
  except (FileNotFoundError,ProcessLookupError):break
  rows.append(v);pid=v['ppid']
 chains[str(worker)]=rows
r['chains']=chains;r['continuations']=artifact(root/'continuations.json')
latest=r['continuations']['value']['attempts'][-1];attempt=Path(latest['root'])
r['native']={p:artifact(attempt/p) for p in ['admission.json','inspection.json','supervised/supervisor.json','supervised/reclaim.json','supervised/gpus-free-before-training.json','supervised/run/run.json','supervised/run/complete.json','supervised/run/depth-partc-complete.json','supervised/burn-verified.json']}
r['source_checks']=[]
for folder,manifest in [(root/'source',json.loads((root/'source-manifest.json').read_text())),(attempt/'source',read_locked(latest['execution_source_manifest']))]:
 changed=[p for p,sha in manifest['files'].items() if digest(folder/p)!=sha]
 r['source_checks'].append(dict(path=str(folder),files=len(manifest['files']),changed=changed))
r['execution_manifest']=artifact(attempt/'execution-source-manifest.json')
pf=read_locked(dict(path='/mnt/local/_outputs/deep-llms_th2/wc69-a2-depth-resume-preflight-20261106-a01.json',sha256='bbb9c3d0dc28b8637fad183d852ce75cc7c1cba06753faee2fa3186fb1575aca'))
read_locked(pf['pause_receipt']);cfg=read_locked(pf['depth_launch']['config']);stream=cfg['heal_streams']['dclm']
read_locked(dict(path=stream['path']+'/manifest.json',sha256=stream['sha256']))
r['input_locks']=dict(pause_receipt_unchanged=True,config_unchanged=True,heal_manifest_unchanged=True,last_actual_data_hash_verification=pf['utc'],last_actual_checkpoint_22sha_verification=pf['utc'],full_data_rehashed_this_health=False)
r['progress']={p:artifact(base/p) for p in ['progress.json','DT6_checks.json','DT6.json','coordinator_checks.json','attempts.json','decision_depth_partc.json','downstream/progress.json','downstream/decision_downstream.json']}
r['arms']={}
for arm in ['X72','P300','P1T']:
 folder=base/'heals'/f'ST1-dclm-{arm}-S1-12.8B-lr1e-3';tests=[]
 for p in sorted(folder.glob('tests-rank*.json')):
  a=artifact(p);c=a['value']['checks'];tests.append(dict(path=str(p),sha256=a['sha256'],checks=len(c),flags=[x for x in c if not x['passed']],resume=[x for x in c if x['test']=='resume_available']))
 result=artifact(folder/'result.json')
 if result:
  v=result['value'];result['value']={k:v.get(k) for k in ['status','run_id','flags','unverified','delta_post','roundtrip_difference','resume_events']};result['value']['scores']={k:None if x is None else x.get('loss') for k,x in v['scores'].items()};result['value']['updates']=len(v['lr_updates'])
 checkpoints=[]
 for p in sorted(folder.glob('trainer/**/depth-checkpoint.json')):
  try:
   marker=json.loads(p.read_text());actual={name:(p.parent/name).stat().st_size for name in marker['files']}
   checkpoints.append(dict(path=str(p),step=marker['global_step'],complete=marker['complete'],finite=marker.get('state_finite'),actual_sizes_match=actual==marker['files']))
  except FileNotFoundError as error:checkpoints.append(dict(path=str(p),observation='member disappeared during snapshot; possible native rotation',error=repr(error)))
 r['arms'][arm]=dict(tests=tests,resume=artifact(folder/'resume_events.json'),result=result,checkpoints=checkpoints)
r['logs']=[tail(p) for p in [attempt/'supervisor.log',attempt/'supervised/run/a2-depth-partc.log',attempt/'supervised/burn.log',*sorted(base.glob('logs/C-*.log'))]]
r['nonfinite_flags']={}
for p in base.glob('logs/C-*.log'):
 with p.open(errors='replace') as f:r['nonfinite_flags'][p.name]=sum('DEPTH_NONFINITE_KD_FLAG' in line or 'DEPTH_NONFINITE_RECOVERY' in line for line in f)
guard=Path('/mnt/local/_gpu_guard/DISABLED');r['guard']=guard.read_text() if guard.exists() else None
r['session']=subprocess.run(['tmux','display-message','-p','-t',latest['session'],'#{pane_dead} #{pane_pid}'],capture_output=True,text=True).stdout.strip()
r['cpu_processes']=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  cmd=(proc/'cmdline').read_bytes().split(b'\0')
  if b'scripts.a2_depth_partc_run' not in cmd:continue
  item=process(int(proc.name));item.update(cwd=str((proc/'cwd').resolve()),io=(proc/'io').read_text(),status=(proc/'status').read_text().splitlines()[:8]);r['cpu_processes'].append(item)
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
print('DEPTH_RESUME_HEALTH',json.dumps(r),flush=True)
HEALTH

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'OVERLAP'
from pathlib import Path
import json,hashlib
from datetime import datetime,timezone
import numpy as np
from scripts.a2_stage1_data import INDEX
from scripts.a2_remaining import read_locked
from scripts.a2_quant_io import lock,write
from scripts.train_then_burn import digest

def seqs(start,stop,total,length):
    lo=max(0,int(start));hi=min(int(stop),int(total))
    return list(range(lo//length,(hi-1)//length+1)) if hi>lo else []

def positions(docs,order,hits,budgets,chunk=1_000_000):
    found=[];offset=0;hitset=np.array(sorted(hits),dtype=np.uint64)
    for begin in range(0,len(order),chunk):
        indices=order[begin:begin+chunk];lens=docs['tokens'][indices];ends=np.cumsum(lens,dtype=np.uint64)
        for j in np.flatnonzero(np.isin(indices,hitset)):
            index=int(indices[j]);start=offset+int(ends[j])-int(lens[j]);stop=offset+int(ends[j])
            found.append(dict(pool_index=index,permutation_position=begin+int(j),token_start=start,token_stop=stop,
                text_sha256=docs[index]['hash'].tobytes().hex(),prefix_tokens={str(n):max(0,min(stop,n)-start) for n in budgets}))
        offset+=int(ends[-1]) if len(ends) else 0
    assert sorted(x['pool_index'] for x in found)==sorted(hits)
    return found,offset

def main():
    base=Path('/mnt/local/_outputs/deep-llms_th2');why=base/'wc69-a2-why-rest-20261105-a01/results/a2_why_20261104'
    target=base/'wc69-a2-why-overlap-audit-20261106-a01.json';assert not target.exists()
    cfg=read_locked(dict(path=str(why/'config/plan.json'),sha256='720ec95cec21e522df41de3d6aa92596c4d1252617f99cb01f8cc9f62a05dd6f'))
    cx=Path(cfg['data_root']);cm=read_locked(cfg['data_manifest']);assert digest(cx/'documents.jsonl')==cm['documents_sha256']
    # Verify final bytes against the registered manifest before deriving row exclusions.
    for suffix,key in [('bin','token_sha256'),('starts','starts_sha256')]:assert digest(cx/('final.'+suffix))==cm['streams']['final'][key]
    held=[];offsets={'tuning':0,'final':0}
    for line in (cx/'documents.jsonl').open():
        doc=json.loads(line);group=doc['group']
        if group not in offsets:continue
        start=offsets[group];offsets[group]+=doc['gpt_neox_tokens']
        held.append(dict(doc,token_start=start,token_stop=offsets[group],sequence_ids=seqs(start,offsets[group],cm['streams'][group]['tokens'],cm['streams'][group]['length'])))
    assert len(held)==16862
    for group in offsets:assert offsets[group]==cm['document_tokens'][group]
    hashes=np.unique(np.array([bytes.fromhex(d['text_sha256']) for d in held],dtype='V32'))
    stream=cfg['heal_streams']['fwe']['2030'];pool=Path(stream['path']);manifest=read_locked(dict(path=str(pool/'manifest.json'),sha256=stream['sha256']))
    assert digest(pool/'documents.bin')==manifest['documents_sha256']
    docs=np.memmap(pool/'documents.bin',dtype=INDEX,mode='r');assert len(docs)==manifest['document_count']
    hits=[]
    for begin in range(0,len(docs),500000):hits.extend((begin+np.flatnonzero(np.isin(docs['hash'][begin:begin+500000],hashes))).tolist())
    overlap_hashes=sorted({docs[i]['hash'].tobytes().hex() for i in hits});assert len(overlap_hashes)==3
    matched=[x for x in held if x['text_sha256'] in overlap_hashes]
    for d in matched:
        starts=np.memmap(cx/(d['group']+'.starts'),dtype='u1',mode='r')
        if d['token_start']<len(starts):assert starts[d['token_start']]==1
    runs=[json.loads(p.read_text())['registered_run'] for p in sorted(why.glob('heals/*/result.json')) if json.loads(p.read_text())['dataset']=='fwe']
    coverage={}
    for seed in (2030,2031,2032):
        item=cfg['heal_streams']['fwe'][str(seed)];folder=Path(item['path']);m=read_locked(dict(path=str(folder/'manifest.json'),sha256=item['sha256']))
        assert digest(folder/'document_order.bin')==m['document_order_sha256']
        if seed!=2030:assert m['pool_documents_sha256']==manifest['documents_sha256']
        order=np.memmap(folder/'document_order.bin',dtype='<u8',mode='r');assert len(order)==len(docs)
        # Y2/Y3 use2030; Y1 uses2031/2032. Include historical L6 budgets for comparison.
        budgets=sorted({r['steps']*64*4096 for r in runs if r['stream_seed']==seed}|({762*64*4096,3051*64*4096,12207*64*4096} if seed==2030 else set()))
        found,total=positions(docs,order,hits,budgets)
        coverage[str(seed)]=dict(order_manifest=lock(folder/'manifest.json'),order_sha256=m['document_order_sha256'],full_document_tokens=total,hits=found,budgets=budgets)
    excluded=sorted({s for d in matched if d['group']=='final' for s in d['sequence_ids']})
    result=dict(utc=datetime.now(timezone.utc).isoformat(),spec_sha256=cfg['spec_sha256'],status='PASS',config=lock(why/'config/plan.json'),culturax_manifest=cfg['data_manifest'],culturax_documents_sha256=cm['documents_sha256'],final_actual_hash_pass=True,pool_manifest=lock(pool/'manifest.json'),pool_documents_sha256=manifest['documents_sha256'],pool_document_count=len(docs),overlap_hashes=overlap_hashes,overlap_pool_indices=hits,heldout_matches=matched,prefix_coverage=coverage,exclude_final_sequence_ids=excluded,final_total_sequences=cm['streams']['final']['rows'],final_sequence_length=cm['streams']['final']['length'],policy='Conservative same final-row mask for all arms/parents; all rows touching any whole-pool overlap. Original metrics/checks remain unchanged. Actual exposure recorded separately by stream/budget.')
    write(target,result);print('WHY_OVERLAP_AUDIT',json.dumps(result),flush=True)
if __name__=='__main__':main()
OVERLAP
