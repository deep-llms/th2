#1 +60+a
#th2-swt-package-completed-eval-results-only-20260909-a01
set -euo pipefail
date -u
hostname
cd /mnt/local/deep-llms_th2
python3 - <<'PY'
import hashlib, json, math, zipfile
from datetime import datetime, timezone
from pathlib import Path
root = Path('/mnt/local/_outputs/deep-llms_th2/swt/full_eval_finetune_42ckpt_20260909_a01')
destination = Path('/mnt/local/_outputs/deep-llms_th2/swt/exports/eval_42ckpt_20260909_a01')
assert not destination.exists()
inputs = json.loads((root/'inputs_verified.json').read_text())
complete = json.loads((root/'eval/complete.json').read_text())
plan = json.loads((root/'eval/plan.json').read_text())['jobs']
assert inputs['success'] is True and len(inputs['checkpoints']) == 42
assert complete['success'] is True and len(complete['completed']) == 84 and len(plan) == 84
expected = {f'{arm}_step{step}' for arm in ('B0','A128','A256','A512','C','D') for step in (250,500,1000,2000,3000,4000,5000)}
assert set(inputs['checkpoints']) == expected
paths = ['inputs_verified.json', 'eval_plan.json', 'eval/plan.json', 'eval/complete.json']
for name in sorted(expected):
    for stage in ('ppl', 'benchmarks'):
        relative = f'eval/{name}/{stage}.json'
        item = json.loads((root/relative).read_text())
        assert item['success'] is True and item['languages'] == ['en']
        assert item['checkpoint'] == inputs['checkpoints'][name]
        assert item['settings']['precision'] == 'bf16'
        if stage == 'ppl':
            row = item['ppl']['by_language']['en']
            assert row['scored_targets'] == 9829694 and row['data_fingerprint'] == inputs['ppl_fingerprint']
            assert math.isfinite(row['nll']) and math.isfinite(row['ppl']) and row['ppl'] > 0
        else:
            assert set(item['benchmarks']) == set(inputs['benchmark_coverage'])
            assert len(item['benchmarks']) == 78
            for task, result in item['benchmarks'].items():
                assert result['samples']['effective'] == inputs['benchmark_coverage'][task]['eval_examples']
                assert all(math.isfinite(value) for value in result['metrics'].values())
        paths.append(relative)
assert len(paths) == len(set(paths)) == 88
expected_results = {str(root/path) for path in paths if path.endswith(('/ppl.json','/benchmarks.json'))}
assert {r['result'] for r in complete['completed']} == expected_results
assert all(r['exit_code'] == 0 and r['error'] is None for r in complete['completed'])
payload = {path: (root/path).read_bytes() for path in paths}
destination.mkdir(parents=True, exist_ok=False)
archive = destination/'eval_results.zip'
with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as handle:
    for name, data in payload.items():
        handle.writestr(name, data)
manifest = dict(success=True, source_run=str(root), checkpoints=42, result_files=84,
    benchmark_results=42*78, exported_utc=datetime.now(timezone.utc).isoformat(),
    files=[dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()) for name,data in payload.items()],
    archive=dict(name=archive.name, bytes=archive.stat().st_size, sha256=hashlib.sha256(archive.read_bytes()).hexdigest()))
assert archive.stat().st_size < 25_000_000
manifest_path = destination/'export_manifest.json'
with manifest_path.open('x') as handle:
    json.dump(manifest, handle, indent=2, allow_nan=False)
    handle.write('\n')
print('VERIFIED_EXPORT', json.dumps({k:v for k,v in manifest.items() if k != 'files'}), flush=True)
print('MANIFEST_SHA256', hashlib.sha256(manifest_path.read_bytes()).hexdigest(), flush=True)
print('EVAL_RESULTS_EXPORT_READY_FINETUNING_UNTOUCHED', flush=True)
PY
