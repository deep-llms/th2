#1 +15+a
#th2-q359-p7-inspect-stop-wrapper-20261015-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
/mnt/local/conda-py311/envs/attention_bench/bin/python3.11 -u - <<'REMOTE'
from pathlib import Path
import json,subprocess
from scripts.verified_gpu_reclaim import process
from scripts.gpu_status import snapshot
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-p7-exploration8-20261015-a03')
pane=subprocess.check_output(['tmux','display-message','-p','-t',root.name,'#{pane_pid} #{pane_dead}'],text=True)
print('PANE',pane)
guard=json.loads(Path('/mnt/local/_gpu_guard/DISABLED').read_text());print('GUARD',guard)
for p in [int(pane.split()[0]),guard['owner_pid']]:
 print('IDENTITY',process(p));print('ARGV',p,(Path('/proc')/str(p)/'cmdline').read_bytes().replace(b'\0',b' '))
print('GPU_SNAPSHOT',json.dumps(snapshot(list(range(8)))))
print('RUN', (root/'supervised/run/run.json').read_text()[-6000:])
REMOTE
