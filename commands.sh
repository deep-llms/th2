#1 +20+a
#th2-wc69-family-f12-health-20261111-a02
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'CODE'
from pathlib import Path
from scripts.a2_family_f12_health import health
from scripts.a2_family_f10_health import health as f10health
from scripts.a2_family_f11_health import health as f11health
from eval.a2_family_f12 import read
from efficient_llm.diagnostic_data import sha256
health('resources/a2_family_f12_plan_20261111.json')
f11health('resources/a2_family_f11_plan_20261111.json')
f10health('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f10-20261111-a01')
root=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f12-queue-20261111-a01')
for name in ('prepared_config.json','actual_inputs.json'):
 p=root/name
 if p.is_file():
  value=read(p)
  print('F12_CPU_PROOF',name,sha256(p),'verified',value.get('verified'),'files',len(value.get('files',{})),'heal_rows',value.get('heal_rows'),'f11_source_verified',value.get('f11_source_verified'),flush=True)
p=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f10-20261111-a01/results/a2_family_20261107/f10/verification.json')
print('F10_VERIFICATION_SHA',sha256(p),flush=True)
CODE
