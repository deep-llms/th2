#1 +20+a
#th2-wc69-family-f10-status-20261111-a02
set -euo pipefail
cd /mnt/local/deep-llms_th2
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'PY'
from pathlib import Path
import json,subprocess
from efficient_llm.diagnostic_data import sha256
root=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f10-preparation-20261111-a01')
for name in ['prepare_recovery_a02.log','prepared/prepare.log','prepared/ready.json']:
 p=root/name
 print(name,p.read_text()[-5000:] if p.exists() else 'ABSENT',flush=True)
print(subprocess.run(['tmux','list-panes','-t','a2_family_f10_prepare_20261111_a02','-F','#{pane_pid} #{pane_dead} #{pane_dead_status}'],capture_output=True,text=True).stdout)
manifest=json.load(open('resources/a2_family_f10_inputs_20261111.json'))
files=[f for r in manifest['repositories'] for f in r['files']]
good=[];bad=[]
for f in files:
 p=Path(f['local_dir'])/Path(f['path']).name
 (good if p.is_file() and p.stat().st_size==f['bytes'] and sha256(p)==f['sha256'] else bad).append(f['path'])
print('ACTUAL_INPUT_HASHES',len(good),'/',len(files),'BAD',bad,flush=True)
p=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f10-cpu-20261111-a01/f10a/complete.json');print('F10A_COMPLETE',p.read_text())
PY
