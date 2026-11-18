#1 +60+a
#th2-ohmv-token-order-terminal-proof-20261118-a01
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=2 PYTHONPATH="$PWD"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 - <<'PY'
import json,time
from pathlib import Path
from efficient_llm.warm_order.data import sha
from scripts.gpu_process import inspect,process,ownership
from scripts.train_then_burn import approved_launcher,GUARD_HASH
base=Path('/mnt/local/_outputs/deep-llms_th2/token_order_20261118');state=inspect()
terminal=json.loads((base/'terminal.json').read_text());assert terminal['success'] and terminal['native_verified']
assert not Path('/mnt/local/_gpu_guard/DISABLED').exists()
assert sha('/mnt/local/_gpu_guard/gpu_guard.sh')==GUARD_HASH
native=json.loads((base/'native-verified.json').read_text());prior=native['samples'][-1]
assert ownership(state['gpus'])==ownership(prior['gpus'])
assert all(process(int(pid))==p for pid,p in prior['processes'].items())
path='/tmp/llm_pretrain_burn.py';assert sha(path)==native['script_sha256']
assert len(state['workers'])==8 and all(approved_launcher(process(p)['ppid'],{path:native['script_sha256']}) for p in state['workers'])
source=json.loads((base/'execution/source.json').read_text());assert all(sha(base/'execution/source'/p)==v for p,v in source.items())
backup=json.loads((base/'backup.json').read_text());assert sha(base/'results.tar.gz')==backup['archive_sha256'];assert (base/'results.tar.gz').stat().st_size==backup['bytes']
assert all(sha(p)==v for p,v in backup['node_copies'].items())
assert all(sha(base/p)==v for p,v in backup['files'].items())
heads={}
for name in ['R-s0','R-s1','W-s0','W-s1']:
 p=base/'runs'/name/'head';c=json.loads((p/'complete.json').read_text());assert sha(p/'fullstate.pt')==c['fullstate_sha256'] and sha(p/'router.pt')==c['router_sha256'];heads[name]=c
print('TOKEN_ORDER_TERMINAL_PASS',json.dumps(dict(utc=time.time(),terminal=terminal,ownership=state,native=native,source_files=len(source),backup_sha256=sha(base/'backup.json'),archive_sha256=sha(base/'results.tar.gz'),bytes=backup['bytes'],head_checks=heads,guard_released=True,all_member_hashes=True,all_node_backup_hashes=True)),flush=True)
PY
