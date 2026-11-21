#1 +60+a
#th2-ohmv-layer-adaptation-preflight-20261121-a01
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=2 PYTHONPATH="$PWD"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 - <<'PY'
import hashlib,json,socket,sys
from pathlib import Path
from datetime import datetime,timezone
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
source={'scripts/gpu_process.py': '0bd5b1e6d6fd51c887b14ffdc35493075ce45e2d6421af367a4d2d6d06b6c446', 'scripts/gpu_status.py': '63506e500bd24c4b105234116cd601543e1cdf5bd50c599b22550730b31c38eb', 'scripts/train_then_burn.py': 'e5c6d4e3a6cae85aaa4d34663860d3f2c2bf77b34e86cbe45f81f37209c871fc'}
pins={'/mnt/local/_data/deep-llms_th2/a2_dot_20261118/dot/e2/train.bin': '90d971d9203c9ab6c1498165dd35a2c6887ee8d5f886cd1242cc07ddc08368d6', '/mnt/local/_data/deep-llms_th2/a2_dot_20261118/dot/e2/train.starts': '8302b82ca68fa93c62ca378b60e7f57bd20090ad3490bdb2ce5eaa7e32b898ed', '/mnt/local/_outputs/deep-llms_th2/horizon_20261119/runs/D-s0/train/fullstate.pt': 'd3355e72e4373eff6d2b5a7060b4133a56fdce4ce3d82b2fd37ff99caff44115', '/mnt/local/_outputs/deep-llms_th2/horizon_20261119/runs/D-s0/train/weights.pt': '63e1ea6dd3926bc9d5bd837add7f4f9eff68dea24d7a0ddb7f8f83eb17ca25e5', '/mnt/local/_outputs/deep-llms_th2/horizon_20261119/runs/D-s0/train/recipe.json': 'b792b87152320c80a8bf68c9f30f4d8938a43d93cd0c116796e16eb1d5f97881', '/mnt/local/_outputs/deep-llms_th2/horizon_20261119/runs/D-s0/train/complete.json': '78dc0833010db6551e18eaa0fa9d1bf9753cc842bcc6b6ac1ac6c457611b29a8', '/mnt/local/_outputs/deep-llms_th2/horizon_20261119/runs/D-s0/train/checkpoint.json': '3034011a33c7a3c9191d98f0473cec4ec6129b4fa9dcb7bf4d81962669d32414', '/mnt/local/_outputs/deep-llms_th2/horizon_20261119/runs/D-s1/train/fullstate.pt': 'a509951aab397db803b9446595b8d35c535b75da0728553688b3e086e5ed8c79', '/mnt/local/_outputs/deep-llms_th2/horizon_20261119/runs/D-s1/train/weights.pt': 'c90d56447ae857083cc59228ae2bda9784d3155af63986ee258cfaa21fc1b3de', '/mnt/local/_outputs/deep-llms_th2/horizon_20261119/runs/D-s1/train/recipe.json': 'ebafe6b72c4494b8dbed1d2e49dd6960f5afc262a0760eaf7665e3941146a072', '/mnt/local/_outputs/deep-llms_th2/horizon_20261119/runs/D-s1/train/complete.json': '98fe22c3c0bdbd00d1141bd3910ada7b33405a60d3896a3c1a1281f290f99661', '/mnt/local/_outputs/deep-llms_th2/horizon_20261119/runs/D-s1/train/checkpoint.json': '7d6b737d43ee48a31b671fd445a7dc0b1d8b428e93c431c67550a4afe73404f2'}
assert all(sha(p)==h for p,h in source.items()),'Inspection source mismatch'
from scripts.gpu_process import inspect,process
from scripts.train_then_burn import approved_launcher,GUARD_HASH
import torch,numpy
state=inspect()
r=dict(utc=datetime.now(timezone.utc).isoformat(),ownership=state,runtime=dict(python=sys.version,torch=torch.__version__,numpy=numpy.__version__),source=source)
print('ADAPTATION_PREFLIGHT_START',json.dumps(r),flush=True)
assert socket.gethostname()=='thiennh-p6-ohmv-worker-0','Unexpected host'
assert torch.__version__=='2.14.1+cu130' and numpy.__version__=='2.4.6','Runtime mismatch'
native='/tmp/llm_pretrain_burn.py';digest='3cdcc857bd01b096e20a02640fa85f0b8be7607e3c2b22a89a704bbac3650857'
assert len(state['workers'])==8 and not state['guard_disabled'],'Ownership requires review'
assert sha(native)==digest and sha('/mnt/local/_gpu_guard/gpu_guard.sh')==GUARD_HASH
assert all(approved_launcher(process(p)['ppid'],{native:digest}) for p in state['workers'])
r['actual_files']={}
for p,want in pins.items():
 f=Path(p);got=sha(f);r['actual_files'][p]=dict(sha256=got,bytes=f.stat().st_size,match=got==want)
 assert got==want,p
r.update(complete=True,native_ownership_verified=True,train_jobs_launched=0,model_loaded=False,utc=datetime.now(timezone.utc).isoformat())
print('ADAPTATION_PREFLIGHT_COMPLETE',json.dumps(r),flush=True)
PY
