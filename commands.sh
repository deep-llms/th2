#1 +20+a
#th2-wc69-family-f10-prepare-recovery-20261111-a03
set -euo pipefail
cd /mnt/local/deep-llms_th2
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'PY'
from pathlib import Path
import subprocess,json,shlex
from efficient_llm.diagnostic_data import sha256
root=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f10-preparation-20261111-a01');frozen=root/'source'
pane=subprocess.check_output(['tmux','list-panes','-t','a2_family_f10_prepare_20261111_a02','-F','#{pane_dead} #{pane_dead_status}'],text=True).strip()
assert pane=='1 1' and not (root/'prepared').exists(),pane
assert "ModuleNotFoundError: No module named 'efficient_llm'" in (root/'prepare_recovery_a02.log').read_text()
for name,h in json.load(open(root/'source-manifest.json'))['files'].items():assert sha256(frozen/name)==h,name
manifest=json.load(open(frozen/'resources/a2_family_f10_inputs_20261111.json'))
for f in [f for r in manifest['repositories'] for f in r['files']]:
 p=Path(f['local_dir'])/Path(f['path']).name
 assert p.stat().st_size==f['bytes'] and sha256(p)==f['sha256'],str(p)
session='a2_family_f10_prepare_20261111_a03'
assert subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode!=0
cmd=['env','PYTHONPATH='+str(frozen),'CUDA_VISIBLE_DEVICES=','HF_HUB_OFFLINE=1','HF_DATASETS_OFFLINE=1','TRANSFORMERS_OFFLINE=1','HF_HUB_DISABLE_TELEMETRY=1','WANDB_MODE=offline','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','PYTHONDONTWRITEBYTECODE=1',
'/mnt/local/conda-py311/envs/a2_env/bin/python3.11','-u','-m','scripts.a2_family_f10_prepare','--plan','resources/a2_family_f10_plan_20261111.json','--output',str(root/'prepared'),'--code-commit','1f115f211e9ec28b8e48119445206db9076982f0']
shell='exec '+shlex.join(cmd)+' >'+shlex.quote(str(root/'prepare_recovery_a03.log'))+' 2>&1'
subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(frozen),shell],check=True)
subprocess.run(['tmux','set-option','-w','-t',session,'remain-on-exit','on'],check=True)
print('F10_PREP_A03_59HASH_PASS_STARTED_NATIVE_MODULE_ENV_IN_PANE')
PY
