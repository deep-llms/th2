"""Adaptive sequential phase coordinator, run UNDER the existing GPU supervisor.

Does not acquire GPUs, stop burns, copy Accelerate config or start remote jobs.
Those remain the existing supervisor/launcher's responsibility. Each child uses
all configured ranks. Failed numerical/infra jobs stop; only specified divergence
gets a half-LR retry. Never fills in an undefined LR with a guessed default.
"""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import subprocess
import time
from .data import write_json
from .protocol import matrix,choose_lr,CAPS


def execute_phase(*,phase,root,launcher,common,pre,estimated,spent_seconds):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    cap=CAPS[phase][1]*3600
    started=time.monotonic(); rows=[]; chosen={}; failed_grids=set(); skipped=[]
    eligibility=None
    if '--preflight' in common:
        gate=json.loads(Path(common[common.index('--preflight')+1]).read_text())
        eligibility=gate['parents']
    if not math.isfinite(spent_seconds) or spent_seconds<0 or spent_seconds>=cap:raise ValueError('Phase budget already exhausted')
    if not launcher or not all(isinstance(a,str) and a for a in launcher):raise ValueError('Explicit launcher argv required')
    def launch(run,rerun_of=None):
        remaining=cap-spent_seconds-(time.monotonic()-started)
        estimate=estimated[run.budget]
        if not math.isfinite(estimate) or estimate<=0 or remaining<estimate:raise TimeoutError('Insufficient remaining phase budget')
        folder=root/run.run_id
        runfile=root/(run.run_id+'.json'); prefile=root/(run.run_id+'-pre.json')
        write_json(runfile,dict(run.record(),rerun_of=rerun_of))
        baseline=pre[f'{run.dataset}/{run.parent}']
        if baseline.get('provenance')!=pre[f'{run.dataset}/{run.teacher}'].get('provenance'):
            raise ValueError('Parent and teacher baselines have different provenance')
        values=dict(baseline['sets'][run.set],parent_final=baseline['parent']['final']['loss'],
                    teacher_final=pre[f'{run.dataset}/{run.teacher}']['parent']['final']['loss'],
                    parent_secondary=baseline['parent'].get('secondary',{}).get('loss'),
                    teacher_secondary=pre[f'{run.dataset}/{run.teacher}']['parent'].get('secondary',{}).get('loss'),
                    provenance=baseline.get('provenance'),sets_sha256=baseline.get('sets_sha256'),
                    parent=run.parent,teacher=run.teacher)
        write_json(prefile,values)
        argv=[*launcher,'heal',*common,'--dataset',run.dataset,'--run',str(runfile),
              '--pre-values',str(prefile),'--output',str(folder),'--estimated-seconds',str(estimate)]
        # Parent run_jobs/supervisor owns this process and descendants. A timeout
        # exits this coordinator; the supervisor cleans its entire owned tree.
        with (root/(run.run_id+'.log')).open('x') as log:
            child=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,timeout=min(1.5*estimate,remaining),check=False)
        result_path=folder/'result.json'
        if not result_path.is_file():raise RuntimeError(f'No terminal artifact: {run.run_id}, exit={child.returncode}')
        result=json.loads(result_path.read_text())
        if any(result.get(k)!=v for k,v in run.record().items()) or result.get('rerun_of')!=rerun_of:
            raise ValueError('Wrong run receipt or retry identity')
        if result.get('provenance')!=baseline.get('provenance'):
            raise ValueError('Run receipt provenance differs from its baseline')
        if child.returncode and not (result.get('diverged') and result.get('status')=='failed'):
            raise RuntimeError('Training failed outside the registered divergence rule')
        if result['status'] not in ('complete','failed'):raise ValueError('Nonterminal result')
        if result['status']=='failed' and not result.get('diverged'):
            raise RuntimeError('Non-divergence failure cannot be accepted as completed')
        rows.append(result)
        with (root/'runs.jsonl').open('a') as records:records.write(json.dumps(result,allow_nan=False)+'\n')
        return result
    for prototype in matrix():
        if prototype.part!=phase:continue
        key=prototype.dataset,prototype.arm
        parents={f'{prototype.dataset}/{p}' for p in (prototype.parent,prototype.teacher)}
        if any(p not in pre or eligibility is not None and not eligibility.get(p,{}).get('passed') for p in parents):
            skipped.append(prototype.run_id);continue
        if prototype.budget!='0.03B':
            if key in failed_grids:continue
            if key not in chosen:raise ValueError('LR grid must precede dependent runs')
            run=replace(prototype,lr=chosen[key])
        else:run=prototype
        result=launch(run)
        if result.get('diverged'):
            retry=replace(run,lr=run.lr/2,suffix=run.suffix+'-rerun')
            launch(retry,run.run_id)
        if run.budget=='0.03B':
            grid=[r for r in rows if (r['dataset'],r['arm'],r['budget'])==(*key,'0.03B') and not r.get('rerun_of')]
            if len(grid)==3:
                lr=choose_lr(grid)
                if lr is None:failed_grids.add(key)
                else:chosen[key]=lr
    summary=dict(status='complete' if not failed_grids and not skipped and all(
        (not r.get('diverged')) or any(x.get('rerun_of')==r['run_id'] and not x.get('diverged') for x in rows)
        for r in rows if r['budget']!='0.03B') else 'incomplete',
        phase=phase,skipped_missing_or_ineligible_parent=skipped,wallclock_h=(spent_seconds+time.monotonic()-started)/3600,
        failed_grids=[list(k) for k in sorted(failed_grids)],runs=len(rows),
        note='Complete means coordinator finished; decision labels are generated separately. No deferred phases.')
    write_json(root/'complete.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    execute_phase(**config)

if __name__=='__main__':main()
