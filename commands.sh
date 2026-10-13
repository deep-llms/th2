#1 +30+a
#th2-q359-proxy-followup11-20261014-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
/mnt/local/conda-py311/envs/attention_bench/bin/python3.11 -u - <<'PYREMOTE'
import json,shlex,subprocess
from pathlib import Path
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-proxy-followup11-20261014-a01')
previous=Path('/mnt/local/_outputs/deep-llms_th2/q359-proxy-12layer-screen22-20261013-a01')
state=json.loads((previous/'supervised/run/run.json').read_text())
print('PREDECESSOR_STATUS',state['status'],flush=True)
print('PREDECESSOR_RUNNING',[j['name'] for j in state['jobs'] if j['status']=='running'],flush=True)
assert state['status'] in ('running','ok'), 'Previous queue failed; no follow-up launched'
assert not root.exists(), 'Refuse existing follow-up output'
assert subprocess.run(['tmux','has-session','-t',root.name],capture_output=True).returncode!=0
root.mkdir()
command=shlex.join(['bash','scripts/launch_proxy_12layer.sh',str(root),'thiennh-p6-q359-worker-0',str(previous)])
command='exec '+command+' >'+shlex.quote(str(root/'supervisor.log'))+' 2>&1'
subprocess.run(['tmux','new-session','-d','-s',root.name,command],check=True)
subprocess.run(['tmux','set-option','-w','-t',root.name,'remain-on-exit','on'],check=True)
print('FOLLOWUP_WAITER_STARTED',root.name,flush=True)
PYREMOTE
sleep 10
tmux display-message -p -t q359-proxy-followup11-20261014-a01 '#{pane_dead} #{pane_pid}'
tail -n 15 /mnt/local/_outputs/deep-llms_th2/q359-proxy-followup11-20261014-a01/supervisor.log
