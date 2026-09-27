#1 +60+a
#th2-78gg-deep-kv-smoke-postflight-20260927-a01
set -euo pipefail
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate train_env
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
python -u <<'PY'
from datetime import datetime, timezone
import hashlib, json, math, socket
from pathlib import Path
from scripts.verified_gpu_reclaim import inspect
root = Path('/mnt/local/_outputs/deep-llms_th2/deep-kv-refactor-smoke-20260927-a01')
assert socket.gethostname() == 'thiennh-p6-78gg-worker-0'
receipt = json.loads((root / 'smoke_complete.json').read_text())
assert receipt['status'] == 'ok' and receipt['resumed_step'] == 3
assert not Path('/mnt/local/_gpu_guard/DISABLED').exists(), 'Smoke guard marker remains'
hashes = {}
for arm in 'ABCD':
    path = root / 'run' / arm
    config = json.loads((path / 'train_config.json').read_text())
    result = json.loads((path / 'result.json').read_text())
    state = json.loads((path / 'checkpoint-3/trainer_state.json').read_text())
    assert config['world_size'] == 8 and config['tokens_per_update'] == 1048576
    assert config['model_config']['num_hidden_layers'] == 28
    assert config['data']['block_size'] == 2048
    assert config['training']['bf16'] and config['training']['per_device_train_batch_size'] == 16
    assert config['training']['gradient_accumulation_steps'] == 4
    assert state['global_step'] == result['global_step'] == 3 and state['max_steps'] == 4
    assert result['input_tokens'] == 3145728 and result['evaluation']['eval_rows'] == 129
    assert result['evaluation']['eval_input_tokens'] == 129 * 2048
    assert result['evaluation']['eval_target_tokens'] == 129 * 2047
    assert all(math.isfinite(result['evaluation'][k]) for k in ('eval_lm_loss', 'eval_loss_k', 'eval_loss_v'))
    assert len(list((path / 'checkpoint-3').glob('rng_state_*.pth'))) == 8
    for name in ('model.safetensors', 'optimizer.pt', 'scheduler.pt'):
        assert (path / 'checkpoint-3' / name).stat().st_size > 0
    for name in ('result.json', 'train_config.json', 'trainer_state.json', 'eval_results.json'):
        hashes[f'{arm}/{name}'] = hashlib.sha256((path / name).read_bytes()).hexdigest()
for name in ('smoke_complete.json', 'run/comparison.json'):
    hashes[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
status = inspect()
verification = {'status': 'ok', 'time': datetime.now(timezone.utc).isoformat(),
                'host': socket.gethostname(), 'guard_restored': not status['guard_disabled'],
                'smoke': receipt, 'sha256': hashes, 'gpu_inspection': status}
with (root / 'postflight.json').open('x') as handle:
    json.dump(verification, handle, indent=2)
print(json.dumps(verification, indent=2), flush=True)
print('DEEP_KV_SMOKE_POSTFLIGHT_OK', flush=True)
PY
