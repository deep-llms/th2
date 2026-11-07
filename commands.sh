#1 +60+a
#th2-wc69-family-v14-gate-install-20261108-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u -m scripts.a2_family_v14_install_gate --plan resources/a2_family_v14_gate_install_20261108.json
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'PY'
from pathlib import Path
import json
from scripts.a2_family_v14_gate import sha,write,SPEC_SHA
plan=json.loads(Path('resources/a2_inputs_20261028.json').read_text());rows=[]
for ds in ['dclm','fwe']:
 for arm in ['S72','S188']:
  a=plan['assets'][ds+'/'+arm]
  for directory,key in [('weights_dir','files'),('code_dir','code_files')]:
   for name,h in a[key].items():
    p=Path(a[directory])/name;r=dict(dataset=ds,arm=arm,path=str(p),expected_sha256=h,exists=p.is_file())
    if p.is_file():r.update(bytes=p.stat().st_size,sha256=sha(p));r['verified']=r.get('sha256')==h
    rows.append(r)
root=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-v14-transition-20261108-a01');result=dict(spec_sha256=SPEC_SHA,models=rows,full_training_data_verified=False)
write(root/'input-inventory.json',result)
print('FAMILY_F6_INPUT_INVENTORY',json.dumps(result),flush=True)
PY
