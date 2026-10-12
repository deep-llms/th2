#1 +60+a
#th2-q359-position-extension-source-check-20261012-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
/mnt/local/conda-py311/envs/eval_fa4/bin/python3.11 - <<'PY'
import hashlib,json
from pathlib import Path
from transformers import AutoTokenizer
from eval.paired.run import token_hash
root=Path('/mnt/local/_data/deep-llms_th2')
p=root/'cx_sampled_old/dataset_manifest.json';print('OLD_MANIFEST_SHA256',hashlib.sha256(p.read_bytes()).hexdigest())
for split in ('validation','train'):
 p=root/f'position-extension-20261012/subsets/qwen3_0.6b_base_en_30B/{split}/shard_000000/data-00000-of-00001.arrow'
 print('EXTENSION_SOURCE',split,p.stat().st_size,hashlib.sha256(p.read_bytes()).hexdigest())
cfg=json.loads(Path('/mnt/local/_outputs/deep-llms_th2/q359-proxy-resume20k-seed1042-20261011-a01/supervised/run/seed-1042/A/train_config.json').read_text())
t=AutoTokenizer.from_pretrained(cfg['model']['tokenizer_name'],local_files_only=True)
if t.pad_token is None:t.pad_token=t.eos_token
print('REMOTE_TOKENIZER',token_hash(t))
print('VALIDATION_SOURCE',cfg['data']['eval_data_dir'])
PY
