#1 +60+a
#th2-swt-verify-live-burn-and-package-final5k-results-20260909-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-oish-worker-0
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate swt_eval
test "$(command -v python)" = /mnt/local/conda-py311/envs/swt_eval/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
TASK_BURN=/tmp/llm_pretrain_burn.py
TASK_BURN_ROOT=/mnt/local/_outputs/deep-llms_th2/swt/burn_after_final5k_20260909_a01
TASK_RUN=/mnt/local/_outputs/deep-llms_th2/swt/final5k_diagnostics_finetune_20260909_a01
TASK_EXPORT=/mnt/local/_outputs/deep-llms_th2/swt/exports/final5k_diagnostics_finetune_20260909_a01
date -u
python - "$TASK_BURN" "$TASK_BURN_ROOT" <<'PY'
import json, re, sys, time
from pathlib import Path
from capacity_allocation.data import write_json
from scripts.reclaim_verified_burn import verified_workers
from scripts.gpu_status import snapshot
burn, root = sys.argv[1], Path(sys.argv[2])
log = root/'burn.log'
workers, launcher = verified_workers(burn, list(range(8)))
assert len(workers) == 8
first_text = log.read_text(errors='replace')
first_progress = [line for line in first_text.splitlines() if line.startswith('gpu_burn_progress rank=0 ')]
assert first_progress
first = first_progress[-1]
time.sleep(15)
second_text = log.read_text(errors='replace')
second_progress = [line for line in second_text.splitlines() if line.startswith('gpu_burn_progress rank=0 ')]
assert second_progress
second = second_progress[-1]
def values(line):
    return {key: float(re.search(r'\b'+key+r'=([\d.]+)', line).group(1))
            for key in ('completed_cycles','completed_collective_payload_gib','average_cycle_seconds')}
a, b = values(first), values(second)
assert b['completed_cycles'] > a['completed_cycles'] > 0
assert b['completed_collective_payload_gib'] > a['completed_collective_payload_gib'] > 0
assert 0.5 < b['average_cycle_seconds'] < 1.5
status = snapshot()
assert [g['index'] for g in status] == list(range(8))
assert all(len(g['pids']) == 1 for g in status)
assert {g['pids'][0] for g in status} == {w['pid'] for w in workers}
assert all(0.83 < g['memory_used_mib']/g['memory_total_mib'] < 0.87 for g in status)
assert 'GPU burn: 8 visible GPU(s)' in second_text
assert 'world_size=8' in second_text and 'collective_probe_sum=36' in second_text
target = root/'burn_verified_live.json'
assert not target.exists()
write_json(target, dict(success=True, verification='live_worker_memory_and_advancing_collectives',
    launcher=launcher, workers=workers, gpus=status, first_progress=first,
    second_progress=second))
print('ALL_EIGHT_LIVE_BURN_WORKERS_MEMORY_AND_ADVANCING_COLLECTIVES_VERIFIED')
print(json.dumps(status))
print(first)
print(second)
PY
CUDA_VISIBLE_DEVICES='' python - "$TASK_RUN" "$TASK_EXPORT" <<'PY'
import hashlib, json, sys, zipfile
from pathlib import Path
source, output = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
assert source.is_dir() and not output.exists() and not output.is_symlink()
pipeline = json.loads((source/'complete.json').read_text())
diagnostics = json.loads((source/'diagnostics/complete.json').read_text())
finetune = json.loads((source/'finetune/complete.json').read_text())
assert pipeline == dict(success=True, checkpoints=6, step=5000, diagnostic_jobs=18,
                        finetune_jobs=54, completed_utc=pipeline['completed_utc'])
assert diagnostics['success'] is True and len(diagnostics['completed']) == 18
assert finetune['success'] is True and len(finetune['completed']) == 54
def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()
def checked_result_paths(stage, complete, expected):
    paths = []
    for record in complete['completed']:
        assert record['exit_code'] == 0 and record['error'] is None
        path = Path(record['result']).resolve(strict=True)
        assert path.is_relative_to(source/stage) and path.name == 'result.json' or (
            stage == 'diagnostics' and path.is_relative_to(source/stage) and path.name.startswith('diagnostic_'))
        item = json.loads(path.read_text())
        assert item['success'] is True
        paths.append(path)
    assert len(paths) == expected and len(set(paths)) == expected
    return paths
diag_files = checked_result_paths('diagnostics', diagnostics, 18)
fine_files = checked_result_paths('finetune', finetune, 54)
diag_files += [source/name for name in ('complete.json','inputs_verified.json','diagnostic_plan.json',
                                        'frequency_comparison.json','diagnostics/plan.json',
                                        'diagnostics/complete.json')]
fine_files += [source/name for name in ('complete.json','inputs_verified.json','finetune_plan.json',
                                        'finetune/plan.json','finetune/complete.json')]
for path in diag_files+fine_files:
    assert path.is_file() and not path.is_symlink()
output.mkdir(parents=True)
archives = {}
sources = {}
for name, paths in (('diagnostics_results.zip',diag_files),('finetune_results.zip',fine_files)):
    archive = output/name
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for path in sorted(set(paths)):
            relative = path.relative_to(source).as_posix()
            z.write(path, relative)
            sources[relative] = dict(bytes=path.stat().st_size, sha256=sha(path))
    assert archive.stat().st_size < 25_000_000
    archives[name] = dict(bytes=archive.stat().st_size, sha256=sha(archive))
manifest = dict(success=True, source_root=str(source), diagnostics=18, finetune=54,
                excludes_model_weights=True, sources=sources, archives=archives)
with (output/'export_manifest.json').open('x') as handle:
    json.dump(manifest, handle, indent=2, sort_keys=True)
    handle.write('\n')
print('FINAL5K_RESULTS_PACKAGED', json.dumps(archives))
print('EXPORT_MANIFEST_SHA256', sha(output/'export_manifest.json'))
PY
echo BURN_VERIFIED_AND_FINAL5K_RESULTS_PACKAGED
