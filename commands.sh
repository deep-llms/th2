#1 +120+a
#th2-wc69-depth-resume-preflight-20261106-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'CHECK'
from pathlib import Path
import json
from datetime import datetime,timezone
from scripts.a2_quant_io import lock,write
from scripts.a2_remaining import read_locked
from scripts.train_then_burn import digest
from scripts.gpu_process import inspect,process
from scripts.a2_depth_continue import validate_existing
from scripts.launch_a2_depth_partb import predecessor
from scripts.a2_prepare_corpora import verify_packed
from efficient_llm.a2.data import PackedRows
from efficient_llm.a2_depth.heal_long import run_identity
from efficient_llm.a2_depth.resume import select_checkpoint
base=Path('/mnt/local/_outputs/deep-llms_th2');depth=base/'wc69-a2-depth-partc-20261103-a01';why=base/'wc69-a2-why-rest-20261105-a01'
target=base/'wc69-a2-depth-resume-preflight-20261106-a01.json';assert not target.exists()
launch=json.loads((depth/'launch.json').read_text())
record,cfg,frozen=validate_existing(depth,Path(launch['input_plan']['path']),'e2af735a59127f1116dbf8d671fe5aa95f1bb12f')
live=inspect();ids={}
for g in live['gpus']:
 for pid in g['pids']:
  w=process(pid);ids[str(pid)]=w;ids[str(w['ppid'])]=process(w['ppid'])
b=dict(root=str(why),burn_script=str(why/'source/resources/llm_pretrain_burn.py'),supervisor=lock(why/'supervised/supervisor.json'),burn=lock(why/'supervised/burn-verified.json'),source_manifest=lock(why/'source-manifest.json'),prior_status='ok',known_burn_identities=ids)
predecessor({'predecessor':b},depth)
pause=read_locked(dict(path=str(base/'wc69-a2-why-pause-20261104-a01/pause/saved_state.json'),sha256='bc8a633435bab3f1d410a9bec7c3cc341cc25a3b65fa6a47bad1d17d9d431faa'))
assert pause['global_step']==24000 and pause['identity']==run_identity(cfg,'P300') and len(pause['hashes'])==22
for name,expected in pause['hashes'].items():assert digest(Path(pause['checkpoint'])/name)==expected,name
print('PAUSE_22_ACTUAL_SHA_PASS',flush=True)
out=Path(cfg['results_root'])/'heals/ST1-dclm-P300-S1-12.8B-lr1e-3'
selected,rejected=select_checkpoint(out/'trainer',pause['identity'],8)
original_hashes={name:digest(selected/name) for name in pause['hashes']} if selected else None
matching=bool(selected and selected.name=='checkpoint-24000' and original_hashes==pause['hashes'])
stream=cfg['heal_streams']['dclm'];manifest=read_locked(dict(path=stream['path']+'/manifest.json',sha256=stream['sha256']))
verify_packed(Path(stream['path']),manifest)
for split in ['tuning','final']:PackedRows(cfg['data_root'],split,verify=True)
print('DEPTH_ACTUAL_DATA_SHA_PASS',flush=True)
physical=predecessor({'predecessor':b},depth)
result=dict(utc=datetime.now(timezone.utc).isoformat(),pause_22_sha=True,selected=str(selected) if selected else None,selected_matches_pause=matching,rejected=rejected,final_exists=(out/'final').exists(),arm_complete_exists=(out/'arm-complete.json').exists(),data_actual_hash_pass=True,depth_launch=record,depth_config=cfg,predecessor=b,physical=physical,why_decision=lock(why/'results/a2_why_20261104/decision_why.json'),pause_receipt=lock(base/'wc69-a2-why-pause-20261104-a01/pause/saved_state.json'))
write(target,result)
print('DEPTH_RESUME_PREFLIGHT',json.dumps(result),flush=True)
CHECK
