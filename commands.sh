#1 +60+a
#th2-wc69-family-f6-prepare-20261108-a02
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'PY'
from pathlib import Path
import json,subprocess,os,sys
from scripts.launch_a2_downstream import freeze_source
from scripts.a2_family_v14_gate import sha,read_lock
from scripts.a2_family_v14_install_gate import verify_owner
root=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f6-preparation-20261108-a02')
source=Path.cwd();commit='0d7d0bbb7e7698022f7d4c71d88ea2644a598ca2'
plan=json.loads((source/'resources/a2_family_v14_gate_install_20261108.json').read_text())
family=Path(plan['family_root']);transition=Path(plan['transition_root']);installed=json.loads((transition/'installed.json').read_text())
manifest=read_lock(installed['original_manifest'])
for name,digest in manifest['files'].items():
 expected=installed['patched_sha256'] if name=='scripts/a2_family_depth_bridge.py' else digest
 if sha(family/'source'/name)!=expected:raise ValueError('Unauthorized changed Family source: '+name)
if sha(transition/'gate.py')!=installed['gate_sha256']:raise ValueError('Changed F6 gate')
owner=verify_owner(plan)
print('FAMILY_F6_PREPARATION_OWNERSHIP',json.dumps(owner),flush=True)
print('FAMILY_CURRENT_PROGRESS',(family/'results/a2_family_20261107/progress.json').read_text(),flush=True)
proof=json.loads((source/'resources/a2_family_f6_cpu_20261108.json').read_text())
for rel,digest in proof['files'].items():
 if sha(source/rel)!=digest:raise ValueError('Changed tested F6 code: '+rel)
root.mkdir(parents=True,exist_ok=False)
extra=['docs/A2_family_spec_v1.4.md','resources/a2_family_f6_cpu_20261108.json','resources/a2_family_f6_plan_20261108.json',*[f for f in proof['files'] if f.startswith(('tests/','resources/'))]]
frozen=freeze_source(source,root,commit,extra_files=extra)
print('FAMILY_F6_CPU_PREPARATION_STARTED',json.dumps(dict(root=str(root),source=str(frozen),commit=commit,source_manifest_sha256=sha(root/'source-manifest.json'),GPU_run_started=False)),flush=True)
os.chdir(frozen);os.environ['PYTHONPATH']=str(frozen)
os.execv(sys.executable,[sys.executable,'-u','-m','scripts.a2_family_f6_prepare','--plan',str(frozen/'resources/a2_family_f6_plan_20261108.json'),'--output','/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f6-20261108-a01','--code-commit',commit])
PY
