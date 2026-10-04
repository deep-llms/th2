#1 +60+a
#th2-tjx3-baseline-data-layout-20261005-a01
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-tjx3-worker-0
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
/mnt/local/conda-py311/envs/train_env/bin/python -u - <<'PY'
import json,shutil
from pathlib import Path
root=Path('/mnt/local/_data/deep-llms_th2')
print('DATA_ROOT',[(p.name,p.is_dir()) for p in root.iterdir()],flush=True)
recipe=json.loads(Path('proxy_heads.b200.json').read_text())
for key in ['data_dir','eval_data_dir']:
    p=Path(recipe[key]);print('CONFIGURED_PATH',key,str(p),p.exists(),flush=True)
    for folder in [p.parent.parent,p.parent,p]:
        if folder.is_dir():print('FOLDER',str(folder),[(x.name,x.is_dir()) for x in sorted(folder.iterdir())][:25],flush=True)
for suffix in ('*.parquet','dataset_info.json','state.json'):
    files=list(root.rglob(suffix));print('LAYOUT_FILES',suffix,len(files),[str(p) for p in files[:25]],flush=True)
for name in ['config.json','tokenizer.json','tokenizer_config.json']:
    p=Path(recipe['tokenizer_name'])/name;print('MODEL_ASSET',str(p),p.exists(),p.stat().st_size if p.exists() else None,flush=True)
print('OUTPUT_DISK',shutil.disk_usage('/mnt/local/_outputs'),flush=True)
PY
