#1 +30+a
#th2-6y3o-dot-router-final-status-20261023-a01
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
/mnt/local/conda-py311/envs/train/bin/python3.11 - <<'REMOTE'
import json,hashlib,math
from pathlib import Path
from scripts.gpu_process import inspect
from scripts.native_baseline_pipeline import validate_burn_handoff
from scripts.train_then_burn import burn_progress
base=Path('/mnt/local/_outputs/deep-llms_th2')
read=lambda p:json.loads(Path(p).read_text())
def tail(p,n=3000):
    if not p.is_file():return None
    with p.open('rb') as f:
        f.seek(max(0,p.stat().st_size-n));return f.read().decode(errors='replace')
result={'live':inspect(),'studies':{}}
for name in ['6y3o-dot-mlp-20261022-a01','6y3o-dot-router-20261022-a01']:
    root=base/name;s={'root':str(root)};result['studies'][name]=s
    for rel in ['wait-status.json','launch.json','supervised/supervisor.json','supervised/run/run.json','supervised/run/checks/summary.json','supervised/run/mlp-results.json','supervised/reclaim.json','supervised/gpus-free-before-training.json']:
        p=root/rel
        if p.is_file():s[rel]=read(p)
    s['log_tails']={rel:tail(root/rel) for rel in ['wait.log','supervisor.log','supervised/burn.log']}
    s['runs']={}
    for run in sorted((root/'prepared/runs').glob('*')):
        if not run.is_dir():continue
        r={};s['runs'][run.name]=r
        for rel in ['final_eval.json','train_results.json']:
            if (run/rel).is_file():r[rel]=read(run/rel)
        metrics=tail(run/'metrics.jsonl',8000)
        if metrics:
            parsed=[]
            for line in metrics.splitlines():
                try:parsed.append(json.loads(line))
                except json.JSONDecodeError:pass
            r['latest_metrics']=parsed[-1:] if parsed else []
        receipt=root/'supervised/run'/(run.name+'.verified.json')
        if receipt.is_file():r['verified']=read(receipt)
        ev=root/'supervised/run'/run.name/'result.json'
        if ev.is_file():
            value=read(ev);r['standalone_eval']={k:value.get(k) for k in ['status','global_step','loss','perplexity','num_tokens','num_sequences']}
        cps=sorted(run.glob('checkpoint-*'),key=lambda x:int(x.name.split('-')[-1]))
        r['checkpoints']=[p.name for p in cps]
        cp=run/'checkpoint-4768'
        if cp.is_dir():
            r['final_checkpoint_files']={p.name:p.stat().st_size for p in cp.iterdir() if p.is_file()}
            r['final_step']=read(cp/'trainer_state.json')['global_step']
    manifest=root/'source/SOURCE_MANIFEST.json'
    if manifest.is_file():
        s['source_manifest_sha256']=hashlib.sha256(manifest.read_bytes()).hexdigest()
        s['source_matches']=all(hashlib.sha256((root/'source'/k).read_bytes()).hexdigest()==v for k,v in read(manifest)['files'].items())
    state=root/'supervised/supervisor.json';proof=root/'supervised/burn-verified.json'
    if state.is_file() and proof.is_file():
        s['burn_proof']=read(proof)
        try:
            validate_burn_handoff(read(state),read(proof),result['live'])
            s['burn_live_matches']=True
        except Exception as exc:s['burn_live_matches']=False;s['burn_match_detail']=str(exc)
        s['burn_collective_progress']=burn_progress((root/'supervised/burn.log').read_text())
        if s['burn_live_matches']:
            s['burn_workers_identical']=all(result['live']['processes'].get(k)==v for k,v in read(proof).get('processes',{}).items())
d=read(base/'6y3o-dot-pilot-20261020-a01/prepared/runs/D/final_eval.json')
result['baseline']=d
out=base/'6y3o-dot-router-20261022-a01/final-status-20261023-a01.json'
out.write_text(json.dumps(result,indent=2)+'\n')
print('ROUTER_FINAL_STATUS '+json.dumps(result),flush=True)
print('STATUS_FILE_SHA256 '+hashlib.sha256(out.read_bytes()).hexdigest(),flush=True)
REMOTE
