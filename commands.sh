#1 +30+a
#th2-6y3o-dot-router-confirm-preflight-20261023-a01
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
/mnt/local/conda-py311/envs/train/bin/python3.11 - <<'REMOTE'
import json,hashlib,shutil
from pathlib import Path
from scripts.gpu_process import inspect
from scripts.native_baseline_pipeline import burn_gate
from scripts.train_then_burn import burn_progress,digest,BURN_HASH,GUARD_HASH
base=Path('/mnt/local/_outputs/deep-llms_th2')
prior=base/'6y3o-dot-router-20261022-a01';v3=base/'6y3o-dot-v3-20261021-a01'
root=base/'6y3o-dot-router-confirm-20261023-a01';source=prior/'source'
read=lambda p:json.loads(Path(p).read_text())
assert not root.exists()
burn_gate(prior)
assert read(prior/'supervised/supervisor.json')['training_status']=='ok'
assert read(prior/'supervised/run/checks/summary.json')['status']=='passed'
assert burn_progress((prior/'supervised/burn.log').read_text())
manifest=read(source/'SOURCE_MANIFEST.json')
assert digest(source/'SOURCE_MANIFEST.json')=='65afe4d86c3e154638d512f2e63c38303d9659b57ef1c4ebf96ff7e857c096c9'
assert all(digest(source/k)==v for k,v in manifest['files'].items())
assert digest(source/'resources/llm_pretrain_burn.py')==BURN_HASH
assert digest('/mnt/local/_gpu_guard/gpu_guard.sh')==GUARD_HASH
assert shutil.disk_usage(base).free>300*1024**3
baseline=v3/'prepared/runs/D-s1';bt=read(v3/'prepared/D-s1.train.json')
assert bt['seed']==1 and bt['data_seed']==0 and bt['max_steps']==4768 and bt['warmup_steps']==96
cp=baseline/'checkpoint-4768'
assert read(cp/'trainer_state.json')['global_step']==4768
assert all((cp/f).is_file() and (cp/f).stat().st_size>0 for f in ['optimizer.pt','scheduler.pt','model.safetensors',*[f'rng_state_{i}.pth' for i in range(8)]])
recipes={}
for arm in ['Partial-25-cosine','Partial-50-cosine-detach']:
 t=read(prior/'prepared'/(arm+'-s0.train.json'))
 diff={k:[bt.get(k),t.get(k)] for k in bt.keys()|t.keys() if bt.get(k)!=t.get(k)}
 assert set(diff)<= {'config_name','output_dir','seed'},diff
 assert read(prior/'prepared/runs'/(arm+'-s0')/'data_recipe.json')==read(baseline/'data_recipe.json')
 recipes[arm]={'train':t,'model':read(t['config_name']),'differences_from_D1':diff}
live=inspect();assert not live['guard_disabled']
root.mkdir()
result={'status':'ok','live':live,'source':str(source),'source_manifest_sha256':digest(source/'SOURCE_MANIFEST.json'),'baseline_root':str(baseline),'baseline_train':bt,'baseline_model':read(bt['config_name']),'baseline_eval':read(baseline/'final_eval.json'),'recipes':recipes,'free_gib':shutil.disk_usage(base).free/1024**3}
(root/'preflight-initial.json').write_text(json.dumps(result,indent=2)+'\n')
print('ROUTER_CONFIRM_PREFLIGHT '+json.dumps(result),flush=True)
REMOTE
