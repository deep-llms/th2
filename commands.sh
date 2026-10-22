#1 +30+a
#th2-6y3o-dot-long-pair-preflight-20261023-a01
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
/mnt/local/conda-py311/envs/train/bin/python3.11 - <<'REMOTE'
import json,hashlib,shutil
from pathlib import Path
from scripts.gpu_process import inspect
from scripts.native_baseline_pipeline import burn_gate
from scripts.train_then_burn import burn_progress,digest,BURN_HASH,GUARD_HASH
base=Path('/mnt/local/_outputs/deep-llms_th2')
prior=base/'6y3o-dot-router-confirm-20261023-a01'
router=base/'6y3o-dot-router-20261022-a01';source=router/'source'
pilot=base/'6y3o-dot-pilot-20261020-a01'
root=base/'6y3o-dot-long-pair-20261023-a01'
read=lambda p:json.loads(Path(p).read_text())
assert not root.exists()
burn_gate(prior)
assert read(prior/'supervised/run/run.json')['status']=='ok'
assert read(prior/'supervised/run/confirmation-results.json')['status']=='ok'
assert read(prior/'supervised/supervisor.json')['training_status']=='ok'
assert read(router/'supervised/run/checks/summary.json')['status']=='passed'
assert burn_progress((prior/'supervised/burn.log').read_text())
assert digest(source/'SOURCE_MANIFEST.json')=='65afe4d86c3e154638d512f2e63c38303d9659b57ef1c4ebf96ff7e857c096c9'
assert all(digest(source/k)==v for k,v in read(source/'SOURCE_MANIFEST.json')['files'].items())
assert digest(source/'resources/llm_pretrain_burn.py')==BURN_HASH
assert digest('/mnt/local/_gpu_guard/gpu_guard.sh')==GUARD_HASH
assert shutil.disk_usage(base).free>700*1024**3
recipes={}
for name,p in [('D',pilot/'prepared/D.train.json'),('Partial-25-cosine',router/'prepared/Partial-25-cosine-s0.train.json')]:
 t=read(p);run=Path(t['output_dir']);cp=run/'checkpoint-4768'
 assert t['seed']==t['data_seed']==0 and t['max_steps']==4768 and t['warmup_steps']==96
 assert read(cp/'trainer_state.json')['global_step']==4768
 assert all((cp/f).is_file() and (cp/f).stat().st_size>0 for f in ['optimizer.pt','scheduler.pt','model.safetensors',*[f'rng_state_{i}.pth' for i in range(8)]])
 recipes[name]={'train_path':str(p),'train':t,'model':read(t['config_name']),'final_eval':read(run/'final_eval.json'),'data_recipe_sha256':digest(run/'data_recipe.json')}
a=recipes['D']['train'];b=recipes['Partial-25-cosine']['train']
diff={k:[a.get(k),b.get(k)] for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
assert set(diff)<= {'config_name','output_dir','keep_checkpoint_steps'},diff
assert recipes['D']['data_recipe_sha256']==recipes['Partial-25-cosine']['data_recipe_sha256']
live=inspect();assert not live['guard_disabled']
root.mkdir()
r={'status':'ok','live':live,'source':str(source),'source_manifest_sha256':digest(source/'SOURCE_MANIFEST.json'),'recipes':recipes,'parent_recipe_differences':diff,'free_gib':shutil.disk_usage(base).free/1024**3}
with (root/'preflight-initial.json').open('x') as f:f.write(json.dumps(r,indent=2)+'\n')
print('LONG_PAIR_PREFLIGHT '+json.dumps(r),flush=True)
REMOTE
