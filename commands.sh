#1 +20+a
#th2-wc69-family-f10-launch-20261111-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'PY'
from pathlib import Path
import json,subprocess,shlex
from scripts.launch_a2_downstream import freeze_source
from eval.a2_family_f10 import write
from efficient_llm.diagnostic_data import sha256
source=Path.cwd();out=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f10-admission-20261111-a01')
prep=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f10-preparation-20261111-a01/prepared')
for p in [prep.parent/'prepare_recovery_a03.log',prep/'prepare.log',prep/'ready.json']:
 print(p.name,p.read_text()[-3500:] if p.exists() else 'ABSENT',flush=True)
assert not out.exists() and not Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f10-20261111-a01').exists()
proofname='resources/a2_family_f10_launch_cpu_20261111.json';proof=json.load(open(proofname));assert proof['launch_ready']
for name,h in proof['files'].items():assert sha256(source/name)==h,name
session='a2_family_f10_admission_20261111_a01'
assert subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode!=0
out.mkdir()
frozen=freeze_source(source,out,'834e2cf1a883761b291d8d1d6687d45c27403931',extra_files=['docs/A2_family_spec_v1.11.md','resources/a2_family_f10_inputs_20261111.json','resources/a2_family_f10_plan_20261111.json',proofname,*proof['files']])
write(out/'source-manifest.json',json.load(open(out/'source-manifest.json')))
cmd=['env','PYTHONPATH='+str(frozen),'CUDA_VISIBLE_DEVICES=','HF_HUB_OFFLINE=1','HF_DATASETS_OFFLINE=1','TRANSFORMERS_OFFLINE=1','HF_HUB_DISABLE_TELEMETRY=1','WANDB_MODE=offline','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','PYTHONDONTWRITEBYTECODE=1',
'/mnt/local/conda-py311/envs/a2_env/bin/python3.11','-u','-m','scripts.launch_a2_family_f10','--preparation',str(prep),'--wait-preparation-session','a2_family_f10_prepare_20261111_a03',
'--output','/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f10-20261111-a01','--session','a2_family_f10_20261111_a01','--code-commit','834e2cf1a883761b291d8d1d6687d45c27403931','--cpu-proof',str(frozen/proofname)]
shell='exec '+shlex.join(cmd)+' >'+shlex.quote(str(out/'admission.log'))+' 2>&1'
subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(frozen),shell],check=True)
subprocess.run(['tmux','set-option','-w','-t',session,'remain-on-exit','on'],check=True)
print('F10_IMMUTABLE_ADMISSION_STARTED_CPU_READY_THEN_NATIVE_GPU_QUEUE',flush=True)
PY
