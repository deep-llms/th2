#1 +20+a
#th2-wc69-family-admission-20261107-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'PYCPU'
from pathlib import Path
import subprocess,socket,sys,shlex,os
from efficient_llm.a2_family.common import read,write,sha,SPEC_SHA
from scripts.launch_a2_downstream import freeze_source
from scripts.launch_a2_family import verify_proof
if socket.gethostname()!='thiennh-p6-wc69-worker-0':raise ValueError('Wrong node')
source=Path.cwd();root=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-admission-20261107-a01')
session='a2_family_admission_20261107_a01';commit='6d1068f155071a3f400f48b1f18f294bbfe59912'
family='/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-20261107-a01';family_session='a2_family_20261107_a01'
if root.exists() or Path(family).exists():raise ValueError('Existing admission or Family output')
for name in (session,family_session,family_session+'_burn'):
 if subprocess.run(['tmux','has-session','-t',name],capture_output=True).returncode==0:raise ValueError('Existing session: '+name)
proof=verify_proof(source)
extras=[str(p.relative_to(source)) for p in (source/'resources').glob('a2_family*.json')]
extras+=['docs/A2_family_spec_v1.3.md','resources/a2_data_rebuild_wc69_20261103.json',*proof['files']]
root.mkdir(parents=True,exist_ok=False);frozen=freeze_source(source,root,commit,extra_files=extras);verify_proof(frozen)
if sha(frozen/'docs/A2_family_spec_v1.3.md')!=SPEC_SHA:raise ValueError('Wrong frozen spec')
args=[sys.executable,'-u','-m','scripts.a2_family_admit',
 '--common','/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-preparation-20261107-a02',
 '--mixture','/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-mixture-20261107-a02/mixture',
 '--depth','/mnt/local/_outputs/deep-llms_th2/wc69-a2-depth-partc-20261103-a01',
 '--output',str(root/'admission'),'--family-output',family,'--session',family_session,'--code-commit',commit]
cmd='exec '+shlex.join(args)+' >'+shlex.quote(str(root/'supervisor.log'))+' 2>&1'
subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(frozen),cmd],env={**os.environ,'PYTHONPATH':str(frozen)},check=True)
subprocess.run(['tmux','set-option','-w','-t',session,'remain-on-exit','on'],check=True)
write(root/'launch.json',dict(root=str(root),session=session,code_commit=commit,source_manifest_sha256=sha(root/'source-manifest.json'),wait_inputs_before_pause=True,family_root=family,automatic_Family_then_Depth=True))
print('FAMILY_ADMISSION_SUBMITTED',root,session,commit,flush=True)
PYCPU
