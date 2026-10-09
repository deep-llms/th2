#1 +60+a
#th2-q359-verify-qwen-assets-20261009-a01
set -euo pipefail
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
/mnt/local/conda-py311/envs/attention_bench/bin/python3.11 -u - <<'PYREMOTE'
import hashlib,json
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer
manifest=json.loads(Path('resources/qwen3_base_assets.json').read_text())
recipe=json.loads(Path('proxy_heads.b200.json').read_text())
root=Path(recipe['tokenizer_name'])
for name,expected in manifest['files'].items():
    path=root/name
    assert path.is_file(),str(path)
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual==expected,(name,actual,expected)
    print('HASH_OK',name,actual,flush=True)
config=AutoConfig.from_pretrained(recipe['config_name'],local_files_only=True)
tokenizer=AutoTokenizer.from_pretrained(str(root),local_files_only=True)
assert config.model_type=='qwen3' and config.num_hidden_layers==28
assert tokenizer.eos_token_id is not None and tokenizer.encode('A short offline tokenizer check.')
assert tokenizer.eos_token_id < config.vocab_size
print('QWEN_ASSETS_VERIFIED',json.dumps(dict(revision=manifest['revision'],model_type=config.model_type,layers=config.num_hidden_layers,vocab_size=config.vocab_size,eos_token_id=tokenizer.eos_token_id)),flush=True)
PYREMOTE
