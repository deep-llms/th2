#1 +180+a
#th2-readonly-audit-current-burn-ancestry-before-eval-20260904-a01
set -euo pipefail

echo '=== current physical GPU ownership ==='
date -u
hostname
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader,nounits

echo '=== process ancestry for every GPU PID ==='
python3 - <<'PY'
import csv
from pathlib import Path
import subprocess

def output(*args):
    return subprocess.check_output(args, text=True).strip()

def rows(value):
    return [[field.strip() for field in row] for row in csv.reader(value.splitlines()) if row]

def cmdline(pid):
    path = Path(f'/proc/{pid}/cmdline')
    if not path.is_file():
        return '<exited>'
    return path.read_bytes().replace(b'\0', b' ').decode(errors='replace')

def parent(pid):
    path = Path(f'/proc/{pid}/status')
    if not path.is_file():
        return 0
    for line in path.read_text().splitlines():
        if line.startswith('PPid:'):
            return int(line.split()[1])
    return 0

gpu_rows = rows(output(
    'nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'
))
uuid_to_index = {uuid: int(index) for index, uuid in gpu_rows}
app_rows = rows(output(
    'nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
    '--format=csv,noheader,nounits'
))
assert len(app_rows) == 8, app_rows
seen = set()
common_candidates = None
for uuid, pid_text in app_rows:
    index = uuid_to_index[uuid]
    pid = int(pid_text)
    assert index not in seen and pid != 1
    seen.add(index)
    chain = []
    candidates = set()
    cursor = pid
    visited = set()
    while cursor > 1 and cursor not in visited:
        visited.add(cursor)
        command = cmdline(cursor)
        chain.append((cursor, command))
        if '/tmp/llm_pretrain_burn.py' in command:
            candidates.add(cursor)
        cursor = parent(cursor)
    assert candidates, (index, pid, chain)
    common_candidates = candidates if common_candidates is None else common_candidates & candidates
    print(f'GPU={index} PID={pid}')
    for ancestor_pid, command in chain:
        print(f'  ancestor_pid={ancestor_pid} cmd={command}')

assert seen == set(range(8)), seen
assert common_candidates and len(common_candidates) == 1, common_candidates
launcher = next(iter(common_candidates))
assert launcher != 1
print(f'CURRENT_BURN_ANCESTRY_OK launcher={launcher} workers=8 gpus=8')
PY
echo 'TH2 CURRENT BURN READ-ONLY AUDIT COMPLETE; NO PROCESS SIGNALED'
