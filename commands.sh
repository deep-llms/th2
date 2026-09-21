#1 +60+a
#th2-tpbw-verify-qwen-assets-20260921-a01
set -euo pipefail
date -u
hostname
test "$(hostname)" = thiennh-p6-tpbw-worker-0
/mnt/local/conda-py311/envs/train_env/bin/python -B - <<'PY'
import hashlib, json
from pathlib import Path
spec = json.loads(Path('resources/qwen3_base_assets.json').read_text())
assert spec['repo_id'] == 'Qwen/Qwen3-0.6B-Base'
rev = spec['revision']
assert len(rev) == 40 and all(c in '0123456789abcdef' for c in rev)
root = Path(f'/mnt/local/_models/deep-llms_th2/Qwen3-0.6B-Base-{rev}')
for name, wanted in spec['files'].items():
    p = root/name
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest == wanted, ('HASH_MISMATCH', name, digest)
    print('ASSET_VERIFIED', name, p.stat().st_size, 'bytes')
from transformers import AutoTokenizer, AutoConfig
config = AutoConfig.from_pretrained(root, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
assert config.num_hidden_layers == 28 and config.vocab_size == 151936
assert tokenizer.eos_token_id is not None
sample = tokenizer('hello world')['input_ids']
print('QWEN_ASSETS_VERIFIED', json.dumps(dict(success=True, revision=rev,
    layers=config.num_hidden_layers, vocab=config.vocab_size,
    eos=tokenizer.eos_token_id, sample_ids=sample)), flush=True)
PY
echo TPBW_QWEN_ASSET_CHECK_COMPLETE
