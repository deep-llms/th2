"""Continue the registered DCLM S1 pair, reusing the completed P300 grid."""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import os
import subprocess
import sys
import time

from efficient_llm.diagnostic_data import sha256
from .data import write_json
from .early_grid import read, verify_reuse, ast_hash
from .protocol import FIXED_SETS, LRS, Run, choose_lr, validate_shared_check

SCOPE = 'dclm_S1_pair'


def verify_grid(source, grid):
    source, grid = Path(source), Path(grid)
    lock=read(source/'resources/a2_pair_reuse.json')
    for name,digest in lock['artifacts'].items():
        if sha256(grid/name)!=digest:raise ValueError('Completed grid receipt changed: '+name)
    for name,digest in lock['unchanged_files'].items():
        if sha256(source/name)!=digest:raise ValueError('Numerical code changed: '+name)
    for name,definitions in lock['unchanged_definitions'].items():
        for function,digest in definitions.items():
            if ast_hash(source/name,function)!=digest:raise ValueError('Scientific definition changed: '+function)
    base=grid/'supervised/run/grid'
    done=read(base/'complete.json'); rows=read(base/'results.json')
    chosen=choose_lr([r for r in rows if not r.get('rerun_of')])
    if done['status']!='complete' or chosen is None or done['chosen_lr']!=chosen:
        raise ValueError('P300 grid incomplete or LR invalid')
    for r in rows:
        if ((r['dataset'],r['arm'],r['set'],r['budget'])!=('dclm','P300','S1','0.03B')
            or r['provenance']!=read(base/'check.json')['provenance']
            or r['status']!='complete' or not r['coverage']['exact_order'] or r['roundtrip_difference']>1e-6):
            raise ValueError('P300 terminal run failed')
    return dict(status='passed',grid=str(grid),lock=lock,chosen_lr=chosen)


def reuse_record(record, provenance, sets_sha):
    """Current envelope with the unchanged original receipt kept explicitly inside.

    verify_grid checks exact historical bytes and scientific definitions first.
    Only protocol execution-scope code may differ in historical provenance;
    the new coordinator file has no numerical implementation.
    """
    old=record['provenance']
    for key,value in old.items():
        if key not in ('code_commit','code_files') and provenance.get(key)!=value:
            raise ValueError('Reused receipt runtime/data differs: '+key)
    if set(provenance['code_files'])-set(old['code_files']) != {'a2/pair.py'}:
        raise ValueError('Unexpected added numerical code')
    for key,value in old['code_files'].items():
        if key!='a2/protocol.py' and provenance['code_files'].get(key)!=value:
            raise ValueError('Reused numerical code differs: '+key)
    return dict(record,provenance=provenance,sets_sha256=sets_sha,historical_receipt=record,
                reuse_note='Exact receipt and unchanged science verified by resources/a2_pair_reuse.json')


def validate_result(result,run,provenance,rerun_of,exitcode):
    if (any(result.get(k)!=v for k,v in run.record().items()) or result.get('provenance')!=provenance
        or result.get('rerun_of')!=rerun_of or result.get('status') not in ('complete','failed')
        or exitcode and not (result['status']=='failed' and result.get('diverged'))
        or result['status']=='failed' and not result.get('diverged')):
        raise ValueError('Failed/wrong run receipt outside divergence policy')
    if result['status']=='complete' and (not result['coverage']['exact_order'] or result['roundtrip_difference']>1e-6):
        raise ValueError('Coverage/roundtrip failed')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('inputs','prior','grid','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--port',type=int,required=True);a=p.parse_args()
    source=Path(__file__).resolve().parents[2];a.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();durations={};results=[]
    reuse=verify_reuse(source,a.inputs,a.prior,read(source/'resources/a2_early_grid_reuse.json'))
    grid_reuse=verify_grid(source,a.grid);write_json(a.output/'reuse.json',dict(initial=reuse,grid=grid_reuse))
    with (a.output/'protocol-tests.log').open('x') as log:
        subprocess.run([sys.executable,'-m','unittest','discover','-s','tests/a2','-p','test_protocol.py','-v'],
            cwd=source,env={**os.environ,'CUDA_VISIBLE_DEVICES':''},stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
    sets=a.output/'sets.json';write_json(sets,dict(scope=SCOPE,sets={'dclm':{'S1':FIXED_SETS['S1']}}))
    launcher=[sys.executable,'-m','accelerate.commands.launch','--config_file',str(source/'resources/accelerate_b200.yaml'),
              '--main_process_port',str(a.port),'-m','efficient_llm.a2']
    common=['--assets',str(a.inputs/'assets.json'),'--data',str(a.inputs/'data'),
            '--secondary-map',str(a.inputs/'secondary_map.json'),'--dataset','dclm']
    def stage(name,action,extra,timeout=900,allow_divergence=False):
        begin=time.monotonic();remaining=10800-(begin-started)
        if remaining<timeout:raise TimeoutError('Insufficient bounded pair wall time')
        print('A2_STAGE_START',name,datetime.now(timezone.utc).isoformat(),flush=True)
        with (a.output/(name+'.log')).open('x') as log:
            job=subprocess.run(launcher+[action]+common+extra,cwd=source,stdout=log,stderr=subprocess.STDOUT,
                               timeout=timeout,check=False)
        durations[name]=time.monotonic()-begin;write_json(a.output/'timings.json',durations)
        print('A2_STAGE_END',name,'exit',job.returncode,'seconds',durations[name],flush=True)
        if job.returncode and not allow_divergence:raise RuntimeError('Stage failed: '+name)
        return job.returncode
    survey=a.output/'P1T-survey.json';pre=a.output/'P1T-pre.json'
    stage('P1T-survey','survey',['--parent','P1T','--output',str(survey)])
    measured=read(survey)
    if not measured['T1_precision_passed'] or not measured['parent_sanity']['passed']:
        raise ValueError('P1T precision/sanity failed; preserve evidence, stop this pair')
    prov=measured['provenance'];sh=sha256(sets);old=a.grid/'supervised/run/grid'
    numerical=reuse_record(read(old/'check.json'),prov,sh)
    baseline={'P300':reuse_record(read(old/'pre.json'),prov,sh)}
    validate_shared_check(numerical,read(sets)['sets'],read(a.inputs/'assets.json'),prov,sh,scope=SCOPE)
    write_json(a.output/'shared-check-reuse.json',numerical)
    write_json(a.output/'P300-pre-reuse.json',baseline['P300'])
    stage('P1T-pre','pre',['--parent','P1T','--sets',str(sets),'--output',str(pre)])
    baseline['P1T']=read(pre)
    if baseline['P1T']['provenance']!=prov or not baseline['P1T']['parent_sanity']['passed']:
        raise ValueError('P1T baseline provenance/sanity failed')
    chosen={'P300':grid_reuse['chosen_lr']}
    oldgate=read(old/'gate.json')
    gate=dict(oldgate,launch_scope=SCOPE,provenance=prov,sets_sha256=sh,shared_numerical=numerical,
        parents={f'dclm/{parent}':dict(passed=True) for parent in ('P300','P1T')},chosen_lrs=chosen,
        reuse=dict(initial=reuse,grid=grid_reuse),note='Only registered DCLM S1 pair, no full pilot claim')
    gatepath=a.output/'gate.json'
    # Observed 1.2s/update + ~95s scoring/load/save; conservative +180s allowance.
    estimate=lambda run:run.steps*reuse['step_seconds']+180
    write_json(a.output/'prelaunch-report.json',dict(scope=SCOPE,estimated_seconds=4800,
        hard_cap_seconds=10800,historical_phase0A_seconds=322+read(old/'complete.json')['elapsed_seconds'],
        charge_all_elapsed_to_phase0A=True,fresh_initialization_per_budget=True,
        reused_grid=str(old/'results.json'),new_fits=7,full_pilot_complete=False))
    def fit(run,rerun_of=None):
        write_json(gatepath,gate)
        base=baseline[run.parent]
        values=dict(base['sets']['S1'],parent_final=base['parent']['final']['loss'],teacher_final=base['parent']['final']['loss'],
            parent_secondary=base['parent']['secondary']['loss'],teacher_secondary=base['parent']['secondary']['loss'],
            provenance=prov,sets_sha256=sh,parent=run.parent,teacher=run.teacher)
        prefile=a.output/(run.run_id+'-pre.json');write_json(prefile,values)
        runfile=a.output/(run.run_id+'.json');write_json(runfile,dict(run.record(),rerun_of=rerun_of))
        out=a.output/run.run_id
        code=stage(run.run_id,'heal',['--sets',str(sets),'--run',str(runfile),'--pre-values',str(prefile),
            '--preflight',str(gatepath),'--output',str(out),'--estimated-seconds',str(estimate(run))],
            timeout=1.5*estimate(run),allow_divergence=True)
        result=read(out/'result.json');validate_result(result,run,prov,rerun_of,code)
        results.append(result);write_json(a.output/'results.json',results)
        return result
    def fit_with_retry(run):
        result=fit(run)
        if result.get('diverged'):fit(replace(run,lr=run.lr/2,suffix='-rerun'),run.run_id)
    for lr in LRS:fit_with_retry(Run('dclm','P1T','S1','0.03B',lr))
    chosen['P1T']=choose_lr([r for r in results if not r.get('rerun_of')])
    if chosen['P1T'] is None:raise ValueError('No eligible P1T LR; matched pair cannot proceed')
    write_json(a.output/'chosen-lrs.json',chosen)
    for budget in ('0.05B','0.2B'):
        for parent in ('P300','P1T'):fit_with_retry(Run('dclm',parent,'S1',budget,chosen[parent]))
    scientific_complete=all(not r.get('diverged') or any(
        x.get('rerun_of')==r['run_id'] and not x.get('diverged') for x in results)
        for r in results if r['budget']!='0.03B' and not r.get('rerun_of'))
    write_json(a.output/'complete.json',dict(status='complete',scientific_complete=scientific_complete,
        scope=SCOPE,runs=len(results),chosen_lrs=chosen,phase='0A',elapsed_seconds=time.monotonic()-started,
        full_pilot_complete=False,reused_grid=str(old/'results.json')))


if __name__=='__main__':main()
