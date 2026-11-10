#1 +20+a
#th2-wc69-family-f10-prepare-20261111-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'PYCODE'
from pathlib import Path
import json,subprocess,shlex
from scripts.launch_a2_downstream import freeze_source
from eval.a2_family_f10 import write
root=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f10-preparation-20261111-a01')
assert not root.exists()
old=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f10-cpu-20261111-a01')
print('F10A_LOG', (old/'cpu.log').read_text()[-4000:],flush=True)
root.mkdir()
frozen=freeze_source(Path.cwd(),root,'1f115f211e9ec28b8e48119445206db9076982f0',extra_files=['docs/A2_family_spec_v1.11.md','resources/a2_family_f10_inputs_20261111.json','resources/a2_family_f10_plan_20261111.json','resources/a2_family_f10_cpu_20261111.json'])
write(root/'source-manifest.json',json.loads((root/'source-manifest.json').read_text()))
script=root/'wait_prepare.py'
script.write_text("""import time,json,subprocess,os
from pathlib import Path
from efficient_llm.diagnostic_data import sha256
cfg=json.load(open('resources/a2_family_f10_inputs_20261111.json'))
files=[f for r in cfg['repositories'] for f in r['files']]
for attempt in range(360):
 pending=[]
 for f in files:
  p=Path(f['local_dir'])/Path(f['path']).name
  if not p.is_file() or p.stat().st_size!=f['bytes'] or sha256(p)!=f['sha256']:pending.append(str(p))
 marker=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f10-cpu-20261111-a01/f10a/complete.json')
 if not pending and marker.exists():break
 if attempt%6==0:print('WAIT_INPUTS',len(pending),'F10a_complete',marker.exists(),flush=True)
 time.sleep(10)
else:raise TimeoutError('Inputs or F10a not ready; no GPU mutation')
subprocess.run(['/mnt/local/conda-py311/envs/a2_env/bin/python3.11','-u','-m','scripts.a2_family_f10_prepare','--plan','resources/a2_family_f10_plan_20261111.json','--output','/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f10-preparation-20261111-a01/prepared','--code-commit','1f115f211e9ec28b8e48119445206db9076982f0'],check=True)
""")
shell='cd '+shlex.quote(str(frozen))+' && exec /mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u '+shlex.quote(str(script))+' >'+shlex.quote(str(root/'prepare.log'))+' 2>&1'
subprocess.run(['tmux','new-session','-d','-s','a2_family_f10_prepare_20261111_a01',shell],check=True)
subprocess.run(['tmux','set-option','-w','-t','a2_family_f10_prepare_20261111_a01','remain-on-exit','on'],check=True)
print('F10_PREPARATION_SUBMITTED',flush=True)
PYCODE
