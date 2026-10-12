#1 +60+a
#th2-q359-expanded-download-layout-repair-20261012-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
python3 - <<'PY'
import hashlib,json,os,subprocess
from pathlib import Path
root=Path('/mnt/local/_data/deep-llms_th2/expanded-20261012')
manifest=json.loads(Path('resources/expanded_inputs_20261012.json').read_text())
files={f['path']:f for r in manifest['repositories'] for f in r['files']}
fixed=[];missing=[]
for rel,f in files.items():
    p=root/rel
    if p.is_dir():
        child=p/p.name
        print('NESTED_LAYOUT',rel,sorted(x.name for x in p.iterdir()),flush=True)
        assert child.is_file() and child.stat().st_size==f['bytes'],rel
        assert hashlib.sha256(child.read_bytes()).hexdigest()==f['sha256'],rel
        backup=p.with_name(p.name+'.download-layout-backup')
        assert not backup.exists(),str(backup)
        p.rename(backup)
        os.link(backup/p.name,p)
        fixed.append(rel)
    if p.is_file():
        assert p.stat().st_size==f['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256'],rel
    else:missing.append(rel)
print('VERIFIED_INPUTS',len(files)-len(missing),'/',len(files),'REPAIRED',len(fixed),flush=True)
print('MISSING_PATHS',json.dumps(missing),flush=True)
subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv'],check=True)
subprocess.run(['tmux','list-panes','-t','q359-expanded-paired-20k-20261012-a01','-F','#{pane_pid} #{pane_dead} #{pane_current_command}'],check=True)
PY
tail -5 /mnt/local/_outputs/deep-llms_th2/q359-expanded-paired-20k-20261012-a01/supervisor.log
