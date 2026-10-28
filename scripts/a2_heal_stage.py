"""Remaining registered Phase0A orchestration under the native GPU supervisor.

Reuse completed training and full-pattern numerical receipts with pinned bytes;
score the six current all-pattern baselines, then execute only pending fits.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from efficient_llm.a2.data import write_json
from efficient_llm.a2.protocol import DATASETS,Run,validate_shared_check
from efficient_llm.a2.pair import verify_grid
from efficient_llm.a2.early_grid import verify_reuse
from efficient_llm.diagnostic_data import sha256
from scripts.a2_remaining import (read_locked,load_history,remaining_runs,freeze_sets,
    same_science,verify_current_core)
from scripts.a2_remaining_heals import execute_remaining


def read(path):return json.loads(Path(path).read_text())


def current_provenance(source,inputs,commit):
    from efficient_llm.a2.__main__ import provenance
    result=provenance(argparse.Namespace(code_commit=commit,spec=Path(source)/'docs/A2_pilot_spec.md',
        assets=Path(inputs)/'assets.json',data=Path(inputs)/'data'))
    result['secondary_map_sha256']=sha256(Path(inputs)/'secondary_map.json')
    return result


def reviewed_prerequisites(lock,current,assets,identity,history_sha):
    """Verify exact successful predecessor artifacts and unchanged numerical code."""
    if lock.get('schema')!=1:raise ValueError('Reviewed prerequisite lock required')
    records={key:read_locked(lock[key]) for key in ('complete','supervisor','burn','sets','check')}
    if (records['complete'].get('status')!='ok'
        or records['supervisor'].get('training_status')!='ok'
        or records['supervisor'].get('training_returncode')!=0
        or not records['burn'].get('collective_progress_verified')):
        raise ValueError('Prerequisite stage/handoff incomplete')
    frozen=records['sets'];check=records['check'];old=check['provenance']
    same_science(old,current)
    if old['code_files']!=current['code_files']:raise ValueError('Numerical source changed after check')
    if frozen['history_sha256']!=history_sha:raise ValueError('Prerequisites used different completed history')
    identity_sha=lock.get('identity_sha256')
    if frozen['identity_sha256']!=identity_sha:raise ValueError('Prerequisite identity lock differs')
    expected=json.loads(json.dumps(freeze_sets(frozen['survey_evidence'],identity,current,assets)))
    if any(frozen[k]!=v for k,v in expected.items()):raise ValueError('Frozen BI selection/eligibility differs')
    sets_sha=lock['sets']['sha256']
    validate_shared_check(check,frozen['sets'],assets,old,sets_sha)
    numerical=dict(check,provenance=current,historical_receipt=check,
        reuse_note='Pinned full-pattern receipt; identical numerical code_files/runtime/data, coordinator-only commit change')
    validate_shared_check(numerical,frozen['sets'],assets,current,sets_sha)
    return dict(records=records,numerical=numerical,sets_sha256=sets_sha)


def assemble_gate(old_gate,reviewed,current,identity,identity_sha,history_plan):
    frozen=reviewed['records']['sets']
    if frozen['identity_sha256']!=identity_sha:raise ValueError('Current identity metadata differs')
    required=('T4','T5','T6','T7','T8','T10_formula','T11','T12','T13')
    if old_gate.get('status')!='passed' or any(old_gate.get('tests',{}).get(k,{}).get('passed') is not True for k in required):
        raise ValueError('Historical recipe checks incomplete')
    parents={}
    for ds in DATASETS:
        if identity['tests'][ds]['T2'] is not True:raise ValueError('Annealed pair identity failed')
        for p in ('P300','P1T'):
            e=frozen['eligibility'][ds+'/'+p]
            if e['T1'] is not True or e['T3'] is not True:raise ValueError('Affected annealed parent ineligible')
            parents[ds+'/'+p]=dict(passed=True,T1=True,T2=True)
    return dict(old_gate,status='passed',launch_scope='all',provenance=current,
        sets_sha256=reviewed['sets_sha256'],shared_numerical=reviewed['numerical'],parents=parents,
        chosen_lrs=history_plan['chosen_lrs'],
        reuse_note='Actual pinned full-pattern check and historical recipe evidence; no new numerical claim')


def survey_diagnostics(frozen):
    reports={}
    for ds in DATASETS:
        evidence=frozen['survey_evidence']
        if any(ds+'/'+p not in evidence for p in ('P300','S72','S188')):
            reports[ds]={'available':False,'reason':'Missing excluded branch survey'}
            continue
        loss={p:evidence[ds+'/'+p]['final']['loss'] for p in ('P300','S72','S188')}
        anneal=loss['S72']-loss['P300'];stable=loss['S72']-loss['S188']
        reports[ds]=dict(available=True,annealing_gain=anneal,stable_length_gain=stable,
            annealing_gain_below_002=anneal<.02,stable_length_nonpositive=stable<=0,
            lr_history='Documented by WSD schedule/iteration, not verified from logs',
            changes_eligibility=False)
    return reports


def prepare_heals(source,inputs,history_path,prerequisites_path,prior,grid,commit,spent_seconds):
    source,inputs=Path(source),Path(inputs)
    verify_current_core(source)
    current=current_provenance(source,inputs,commit)
    history_lock=read(history_path);history=load_history(history_lock,current);plan=remaining_runs(history['rows'])
    identity_path=source/'docs/results/a2_review_20261028/identity_metadata.json';identity=read(identity_path)
    prereq_lock=read(prerequisites_path);assets=read(inputs/'assets.json')
    reviewed=reviewed_prerequisites(prereq_lock,current,assets,identity,sha256(history_path))
    initial=verify_reuse(source,inputs,prior,read(source/'resources/a2_early_grid_reuse.json'))
    old_grid=verify_grid(source,grid)
    gate=assemble_gate(read(Path(grid)/'supervised/run/grid/gate.json'),reviewed,current,identity,sha256(identity_path),plan)
    gate['reuse']=dict(initial=initial,grid=old_grid,prerequisites=prereq_lock)
    step=initial['step_seconds']
    if not math.isfinite(step) or step<=0:raise ValueError('Invalid actual T8 measurement')
    estimates={b:Run('nem','P300','S1',b,3e-4).steps*step+180 for b in ('0.05B','0.2B','0.8B')}
    expected=6*240+sum(estimates[r['budget']] for r in plan['pending'])
    worst=6*900+sum(1.5*estimates[r['budget']] for r in plan['pending'])+600
    if not math.isfinite(spent_seconds) or spent_seconds<0 or spent_seconds+worst>18*3600:
        raise ValueError('Insufficient Phase0A time for pending fits and handoff')
    from scripts.a2_preflight_inputs import _parent_asset_plan
    asset_plan=read(source/'resources/a2_inputs_20261028.json');downloads={}
    for ds in DATASETS:
        for parent in ('P300','P1T'):
            for item in _parent_asset_plan(asset_plan,parent,ds)['downloads']:downloads[item['path']]=item
    return dict(reference=current,history_lock=history_lock,history=history,plan=plan,gate=gate,
        prerequisites_lock=prereq_lock,reviewed=reviewed,asset_plan=dict(asset_plan,downloads=list(downloads.values())),
        budget=dict(phase='0A',spent_seconds=spent_seconds,estimated_seconds=expected,worst_seconds=worst,
            hard_cap_seconds=18*3600,report_threshold_seconds=15*3600),estimates=estimates)


def heal_jobs(inputs,history,prerequisites,prior,grid,port,spent_seconds,timeout):
    return [dict(name='remaining-phase0a',gpus=list(range(8)),timeout_seconds=timeout,
        argv=['{python}','-m','scripts.a2_heal_stage','--inputs',str(inputs),'--history',str(history),
            '--prerequisites',str(prerequisites),'--prior',str(prior),'--grid',str(grid),
            '--port',str(port),'--spent-seconds',str(spent_seconds),'--output','{run_dir}/remaining'],
        required_outputs=[dict(path='remaining/heals/complete.json',json_equals=dict(status='complete',scientific_complete=True))])]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('inputs','history','prerequisites','prior','grid','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--port',type=int,required=True);parser.add_argument('--spent-seconds',type=float,required=True)
    a=parser.parse_args();source=Path(__file__).resolve().parents[1];started=time.monotonic()
    state=prepare_heals(source,a.inputs,a.history,a.prerequisites,a.prior,a.grid,os.environ.get('A2_CODE_COMMIT'),a.spent_seconds)
    a.output.mkdir(parents=True,exist_ok=False)
    write_json(a.output/'prelaunch-report.json',dict(state['budget'],
        chosen_lrs=state['plan']['chosen_lrs'],pending_ids=[r['run_id'] for r in state['plan']['pending']],
        diagnostics=survey_diagnostics(state['reviewed']['records']['sets'])))
    write_json(a.output/'gate.json',state['gate']);write_json(a.output/'reuse.json',state['prerequisites_lock'])
    sets_path=Path(state['prerequisites_lock']['sets']['path'])
    launcher=[sys.executable,'-m','accelerate.commands.launch','--config_file',str(source/'resources/accelerate_b200.yaml'),
        '--main_process_port',str(a.port),'-m','efficient_llm.a2']
    common=['--assets',str(a.inputs/'assets.json'),'--data',str(a.inputs/'data'),'--secondary-map',str(a.inputs/'secondary_map.json'),
        '--sets',str(sets_path)]
    pre={}
    for ds in DATASETS:
        for parent in ('P300','P1T'):
            name=ds+'-'+parent;path=a.output/(name+'-pre.json')
            if a.spent_seconds+time.monotonic()-started+900+600>18*3600:raise TimeoutError('Phase cap before baseline')
            print('A2_BASELINE_START',name,flush=True)
            with (a.output/(name+'-pre.log')).open('x') as log:
                subprocess.run(launcher+['pre',*common,'--dataset',ds,'--parent',parent,'--output',str(path)],
                    stdout=log,stderr=subprocess.STDOUT,timeout=900,check=True)
            base=read(path)
            if base['provenance']!=state['reference'] or base['sets_sha256']!=state['reviewed']['sets_sha256'] or not base['parent_sanity']['passed']:
                raise ValueError('Baseline identity/sanity differs')
            old=next(r for r in state['history']['rows'] if r['dataset']==ds and r['parent']==parent)
            if abs(base['parent']['final']['loss']-old['val_teacher'])>1e-6:raise ValueError('Original parent baseline changed')
            pre[ds+'/'+parent]=base;print('A2_BASELINE_END',name,flush=True)
    write_json(a.output/'baselines.json',pre)
    execute_remaining(root=a.output/'heals',launcher=launcher,common=common+['--preflight',str(a.output/'gate.json')],
        history=state['history'],pre=pre,gate=state['gate'],sets=state['reviewed']['records']['sets']['sets'],
        assets=read(a.inputs/'assets.json'),sets_sha256=state['reviewed']['sets_sha256'],estimated=state['estimates'],
        spent_seconds=a.spent_seconds+time.monotonic()-started)


if __name__=='__main__':main()
