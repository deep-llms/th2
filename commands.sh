#1 +60+a
#th2-q359-training-smoke50-status-20261009-a02
set -euo pipefail
/mnt/local/conda-py311/envs/attention_bench/bin/python3.11 -u - <<'PYREMOTE'
import json
from pathlib import Path
from scripts.gpu_status import snapshot
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-proxy-training-smoke50-20261009-a01')
for path in (root/'supervised/supervisor.json',root/'supervised/run/run.json',root/'supervised/run/complete.json'):
    if path.is_file():print(path.name,path.read_text(),flush=True)
print('SUPERVISOR_LOG', '\n'.join((root/'supervisor.log').read_text(errors='replace').splitlines()[-12:]),flush=True)
for path in sorted((root/'supervised/run').glob('*.log')):
    print('JOB_LOG',path.name,'\n'.join(path.read_text(errors='replace').splitlines()[-12:]),flush=True)
for arm in ('A','P6-iso'):
    folder=root/'supervised/run/seed-1042'/arm
    result=folder/'result.json'
    if result.is_file():print('RESULT',arm,result.read_text(),flush=True)
    validation=root/'supervised/run'/('validate-'+arm+'.json')
    if validation.is_file():print('VALIDATION',arm,validation.read_text(),flush=True)
print('GPUS',json.dumps(snapshot(list(range(8)))),flush=True)
final=root/'supervised/supervisor.json'
if final.is_file():
    receipt=json.loads(final.read_text())
    if 'finished_at' in receipt:
        assert receipt['training_status']=='ok' and receipt['training_returncode']==0,receipt
        assert receipt['burn']['collective_progress_verified'] and 'handoff_error' not in receipt
        complete=json.loads((root/'supervised/run/complete.json').read_text())
        assert len(complete['jobs'])==4 and all(j['status']=='ok' for j in complete['jobs'])
        import re,time
        from scripts.verified_gpu_reclaim import inspect,process
        from scripts.train_then_burn import approved_launcher,BURN_HASH
        burn=root/'supervised/burn.log'
        def cycles():
            return int(re.findall(r'gpu_burn_progress rank=0 completed_cycles=(\d+)',burn.read_text())[-1])
        before=cycles();time.sleep(15);after=cycles()
        assert after>before,(before,after)
        live=inspect()
        approved={str(Path.cwd()/'resources/llm_pretrain_burn.py'):BURN_HASH}
        assert live['host']=='thiennh-p6-q359-worker-0' and not live['guard_disabled']
        assert len(live['workers'])==8 and all(len(g['pids'])==1 for g in live['gpus'])
        assert all(approved_launcher(process(pid)['ppid'],approved) for pid in live['workers'])
        print('BOTH_50_STEP_SMOKES_AND_LIVE_BURN_VERIFIED',before,after,flush=True)
PYREMOTE
