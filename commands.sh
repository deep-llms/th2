#1 +30+a
#th2-swt-readonly-eval98-health-20260910-1642
set -euo pipefail
date -u
hostname
cd /mnt/local/deep-llms_th2
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
/mnt/local/conda-py311/envs/swt_eval/bin/python -B - <<'PY'
import ast, collections, json, os, re
from pathlib import Path
root = Path('/mnt/local/_outputs/deep-llms_th2/swt/next_capacity_eval98_diag294_ft126_20260910_a02')
lines = (root/'pipeline.log').read_text(errors='replace').splitlines()
running, completed, failures = {}, [], []
for line in lines:
    match = re.fullmatch(r'START (\S+) physical_gpu=(\d+) pid=(\d+)', line)
    if match:
        name, gpu, pid = match.groups()
        if '_step' in name:
            running[name] = (int(gpu), int(pid))
    if line.startswith('DONE '):
        row = ast.literal_eval(line[5:])
        if '_step' in row['name']:
            running.pop(row['name'], None)
            (failures if row['error'] or row['exit_code'] else completed).append(row)
print('COMPLETED_WORKERS', len(completed), 'FAILED_WORKERS', failures)
for plan_name in ('eval', 'finetune'):
    plan = json.loads((root/f'{plan_name}_plan.json').read_text())
    valid, invalid = collections.Counter(), []
    for job in plan:
        path = Path(job['result'])
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
            assert data['success'] is True
            assert data['checkpoint']['path'] == job['checkpoint']
            assert all(data.get(k) == v for k,v in job['expected'].items())
            if job['stage'] == 'diagnostics':
                assert set(data['diagnostics']) == set(job['tasks'])
            elif job['stage'] != 'ppl':
                assert set(data['benchmarks']) == set(job['tasks'])
            valid[job['stage']] += 1
        except Exception as exc:
            invalid.append((job['name'], str(exc)))
    print('RESULT_CHECK', plan_name, 'planned', len(plan), 'valid', dict(valid), 'invalid', invalid)
for name, (gpu,pid) in running.items():
    process = Path('/proc')/str(pid)
    alive = process.exists()
    print('ACTIVE_JOB', name, 'gpu', gpu, 'pid', pid, 'exists', alive)
    if alive:
        command = (process/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        print('COMMAND', command)
    folder = 'finetune' if '_seed' in name else 'eval'
    log = root/folder/f'{name}.log'
    if log.exists():
        with log.open('rb') as handle:
            handle.seek(max(0,log.stat().st_size-4000))
            tail = handle.read().decode(errors='replace').replace('\r','\n').splitlines()
        print('WORKER_LOG_TAIL', name, '\n'.join(tail[-8:]))
for relative in ('eval/failed.json','finetune/failed.json','eval_verified.json','finetune_verified.json','complete.json','burn_verified.json'):
    path = root/relative
    print('MARKER', relative, path.read_text()[:3000] if path.exists() else 'absent')
print('PIPELINE_TAIL', '\n'.join(lines[-20:]))
PY
date -u
echo SWT_READONLY_EVAL_HEALTH_CHECK_COMPLETE
