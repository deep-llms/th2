#1 +60+a
#th2-78gg-deep-kv-refactor-smoke-20260927-a01
set -euo pipefail
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate train_env
test "$(hostname)" = thiennh-p6-78gg-worker-0
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
python -u <<'PY'
import hashlib, json, os, signal, socket, subprocess, time
from pathlib import Path
from scripts.gpu_status import require_free
from scripts.verified_gpu_reclaim import inspect, ownership
from accelerate.commands.config.config_args import default_yaml_config_file

preflight = Path('/mnt/local/_outputs/deep-llms_th2/deep-kv-refactor-preflight-20260927-a01')
expected = json.loads((preflight / 'gpu_inspection.json').read_text())
assert socket.gethostname() == expected['host'] == 'thiennh-p6-78gg-worker-0'
assert expected['workers'] == list(range(501, 509)) and expected['stop_roots'] == [434]
assert all(expected['processes'][str(pid)]['ppid'] == 434 for pid in expected['workers'])
assert expected['processes']['434']['command_prefix'] == ['/usr/bin/python3', '/tmp/llm_pretrain_burn.py']
assert hashlib.sha256(Path('/tmp/llm_pretrain_burn.py').read_bytes()).hexdigest() == '3cdcc857bd01b096e20a02640fa85f0b8be7607e3c2b22a89a704bbac3650857'
guard = Path('/mnt/local/_gpu_guard')
assert hashlib.sha256((guard / 'gpu_guard.sh').read_bytes()).hexdigest() == '3657a891b9e75e1bf57e9cf13d6db24f89ca2b625bbbfb80a19a141d64a1e3b5'
assert Path(default_yaml_config_file).read_bytes() == Path('resources/accelerate_config.yaml').read_bytes()
marker = guard / 'DISABLED'
payload = json.dumps({'owner': 'deep-kv-refactor-smoke-20260927-a01', 'pid': os.getpid()}) + '\n'
owned = None
handles = {}
try:
    try:
        with marker.open('x') as handle:
            handle.write(payload)
            handle.flush()
            owned = os.fstat(handle.fileno())
    except FileExistsError:
        print('PRESERVING_PREEXISTING_GUARD_DISABLE', flush=True)
    # Allow a guard pass already in progress to finish before final inspection.
    time.sleep(30)
    for pid in expected['workers']:
        handles[pid] = os.pidfd_open(pid)
    current = inspect()
    assert current['host'] == expected['host']
    assert ownership(current['gpus']) == ownership(expected['gpus'])
    assert current['workers'] == expected['workers'] and current['stop_roots'] == expected['stop_roots']
    assert current['processes'] == expected['processes']
    assert marker.is_file()
    print('VERIFIED_BURN_WORKER_STOP', json.dumps(current), flush=True)
    for pid, handle in handles.items():
        try:
            signal.pidfd_send_signal(handle, signal.SIGKILL)
        except ProcessLookupError:
            pass
    time.sleep(30)
    print('ALL_EIGHT_GPUS_FREE', json.dumps(require_free(list(range(8)))), flush=True)
    subprocess.run(['bash', 'scripts/smoke_deep_kv_b200.sh'], check=True)
finally:
    for handle in handles.values():
        os.close(handle)
    if owned is not None:
        try:
            require_free(list(range(8)))
        except Exception as error:
            print('GUARD_LEFT_DISABLED_GPU_OWNERSHIP_NEEDS_REVIEW', type(error).__name__, flush=True)
        else:
            if (marker.exists() and marker.stat().st_ino == owned.st_ino
                    and marker.stat().st_dev == owned.st_dev and marker.read_text() == payload):
                marker.unlink()
                print('GUARD_POLICY_RESTORED', flush=True)
PY
