"""Bounded DCLM/P300 S1 grid under the existing GPU supervisor; no full pilot."""
import argparse
import ast
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from efficient_llm.diagnostic_data import sha256
from .data import write_json
from .protocol import LRS, FIXED_SETS, Run, choose_lr, validate_shared_check

SCOPE = 'dclm_P300_S1_grid'


def read(path):
    return json.loads(Path(path).read_text())


def ast_hash(path, name):
    source = Path(path).read_text()
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    start = min([node.lineno]+[d.lineno for d in node.decorator_list])
    # Source bytes, not ast.dump: the latter changes between Python versions.
    return hashlib.sha256(''.join(source.splitlines(keepends=True)[start-1:node.end_lineno]).encode()).hexdigest()


def verify_reuse(source, inputs, prior, lock):
    """Verify original receipts and unchanged numerical paths, never relabel them."""
    source, inputs, prior = map(Path, (source, inputs, prior))
    for rel, digest in lock['artifacts'].items():
        if sha256(prior/rel) != digest:
            raise ValueError('Historical receipt changed: '+rel)
    for rel, digest in lock['unchanged_files'].items():
        if sha256(source/rel) != digest:
            raise ValueError('Historical numerical code changed: '+rel)
    for rel, names in lock['unchanged_definitions'].items():
        for name, digest in names.items():
            if ast_hash(source/rel, name) != digest:
                raise ValueError('Historical numerical function changed: '+name)
    old = read(prior/'supervised/run/survey.json')
    speed = read(prior/'supervised/run/throughput.json')
    if (old.get('T1_precision_passed') is not True or old['dataset'] != 'dclm' or old['parent'] != 'P300'
        or speed['status'] != 'passed' or not speed['coverage']['exact_order']
        or read(prior/'supervised/run/complete.json')['status'] != 'ok'):
        raise ValueError('Historical parent/T8 evidence failed')
    for record in (old, speed):
        prov = record['provenance']
        if (prov['data_sha256'] != sha256(inputs/'data/manifest.json')
            or prov['assets_sha256'] != sha256(inputs/'assets.json')
            or prov['secondary_map_sha256'] != sha256(inputs/'secondary_map.json')):
            raise ValueError('Historical inputs differ')
        from importlib.metadata import version
        import platform
        if prov['python'] != platform.python_version() or any(version(k) != v for k,v in prov['software'].items()):
            raise ValueError('Historical runtime differs')
    return dict(status='passed', prior_root=str(prior), lock=lock,
        original_provenance=old['provenance'], step_seconds=speed['step_seconds'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--prior',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--port',type=int,required=True)
    a=p.parse_args(); source=Path(__file__).resolve().parents[2]
    a.output.mkdir(parents=True,exist_ok=False); started=time.monotonic()
    lock=read(source/'resources/a2_early_grid_reuse.json')
    reuse=verify_reuse(source,a.inputs,a.prior,lock)
    write_json(a.output/'reuse.json',reuse)
    # Execute the pure schedule/decision tests on this exact frozen source.
    with (a.output/'protocol-tests.log').open('x') as log:
        subprocess.run([sys.executable,'-m','unittest','discover','-s','tests/a2','-p','test_protocol.py','-v'],
            cwd=source,env={**os.environ,'CUDA_VISIBLE_DEVICES':''},stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
    sets=a.output/'sets.json'
    write_json(sets,dict(scope=SCOPE,sets={'dclm':{'S1':FIXED_SETS['S1']}}))
    launcher=[sys.executable,'-m','accelerate.commands.launch','--config_file',
        str(source/'resources/accelerate_b200.yaml'),'--main_process_port',str(a.port),'-m','efficient_llm.a2']
    common=['--assets',str(a.inputs/'assets.json'),'--data',str(a.inputs/'data'),
        '--secondary-map',str(a.inputs/'secondary_map.json'),'--dataset','dclm','--sets',str(sets)]
    durations={}
    def stage(name, action, extra, timeout=900, allow_divergence=False):
        begin=time.monotonic()
        remaining=3600-(begin-started)
        if remaining<=0: raise TimeoutError('Bounded grid exceeded one node-hour')
        print('A2_STAGE_START',name,datetime.now(timezone.utc).isoformat(),flush=True)
        with (a.output/(name+'.log')).open('x') as log:
            job=subprocess.run(launcher+[action]+common+extra,cwd=source,stdout=log,stderr=subprocess.STDOUT,
                timeout=min(timeout,remaining),check=False)
        durations[name]=time.monotonic()-begin
        write_json(a.output/'timings.json',durations)
        print('A2_STAGE_END',name,'exit',job.returncode,'seconds',durations[name],flush=True)
        if job.returncode and not allow_divergence: raise RuntimeError('Stage failed: '+name)
        return job.returncode
    check=a.output/'check.json'; pre=a.output/'pre.json'
    stage('check','check',['--parent','P300','--output',str(check)])
    stage('pre','pre',['--parent','P300','--output',str(pre)])
    numerical=read(check); baseline=read(pre); prov=numerical['provenance']
    if baseline['provenance']!=prov or not baseline['parent_sanity']['passed']:
        raise ValueError('Unhealed baseline check/provenance failed')
    validate_shared_check(numerical,read(sets)['sets'],read(a.inputs/'assets.json'),prov,sha256(sets),scope=SCOPE)
    identity=read(source/'docs/results/a2_review_20261028/identity_metadata.json')
    ddp=read(source/'docs/results/a2_review_20261028/validation.json')
    if not identity['tests']['dclm']['T2'] or not ddp['gpu_ddp']['teacher_no_gradient']:
        raise ValueError('Identity or teacher isolation evidence missing')
    if not read(a.inputs/'data/manifest.json')['disjoint_by_id_and_content']:
        raise ValueError('Data splits overlap')
    # Previously measured survey contains many more scoring passes than a fit.
    # Use its entire elapsed time as a conservative scoring/load/save allowance.
    old_complete=read(a.prior/'supervised/run/complete.json')
    allowance=next(j['elapsed_seconds'] for j in old_complete['jobs'] if j['name']=='survey')
    estimate=114*reuse['step_seconds']+allowance
    gate=dict(status='passed',launch_scope=SCOPE,provenance=prov,sets_sha256=sha256(sets),
        shared_numerical=numerical,parents={'dclm/P300':dict(passed=True)},
        tests={name:dict(passed=True,evidence=evidence) for name,evidence in (
            ('T4','check.json'),('T5','check.json'),('T11','check.json'),('T12','check.json + pinned GPU DDP receipt'),
            ('T10_formula','pre.json'),('T6','protocol-tests.log + pinned GPU LR receipt'),
            ('T13','protocol-tests.log'),('T7','original data manifest'),('T8','reuse.json'))},
        reuse=reuse,estimated_fit_seconds=estimate,note='Only S1 P300 DCLM grid; not full pilot approval')
    gatepath=a.output/'gate.json'; write_json(gatepath,gate)
    write_json(a.output/'prelaunch-report.json',dict(status='passed',scope=SCOPE,
        estimated_three_fits_seconds=3*estimate,extra_seconds_per_retry=estimate,
        completed_check_and_baseline_seconds=sum(durations.values()),hard_cap_seconds=3600,
        historical_phase0A_seconds=322,charge_all_elapsed_to_phase0A=True,gate_sha256=sha256(gatepath)))
    print('A2_PREFLIGHT_PASSED',json.dumps(dict(scope=SCOPE,estimated_three_fits_seconds=3*estimate)),flush=True)
    values=dict(baseline['sets']['S1'],parent_final=baseline['parent']['final']['loss'],
        teacher_final=baseline['parent']['final']['loss'],parent_secondary=baseline['parent']['secondary']['loss'],
        teacher_secondary=baseline['parent']['secondary']['loss'],provenance=prov,
        sets_sha256=sha256(sets),parent='P300',teacher='P300')
    prevalues=a.output/'pre-values.json'; write_json(prevalues,values)
    results=[]
    def fit(run,rerun_of=None):
        runfile=a.output/(run.run_id+'.json'); out=a.output/run.run_id
        write_json(runfile,dict(run.record(),rerun_of=rerun_of))
        exitcode=stage(run.run_id,'heal',['--run',str(runfile),'--pre-values',str(prevalues),
            '--preflight',str(gatepath),'--output',str(out),'--estimated-seconds',str(estimate)],
            timeout=1.5*estimate,allow_divergence=True)
        result=read(out/'result.json')
        if (any(result.get(k)!=v for k,v in run.record().items()) or result.get('provenance')!=prov
            or result.get('rerun_of')!=rerun_of or result.get('status') not in ('complete','failed')
            or exitcode and not (result['status']=='failed' and result.get('diverged'))
            or result['status']=='failed' and not result.get('diverged')):
            raise ValueError('Failed/wrong run receipt outside divergence policy')
        if result['status']=='complete' and (not result['coverage']['exact_order'] or result['roundtrip_difference']>1e-6):
            raise ValueError('Coverage/roundtrip failed')
        results.append(result); write_json(a.output/'results.json',results)
        return result
    for lr in LRS:
        run=Run('dclm','P300','S1','0.03B',lr)
        result=fit(run)
        if result.get('diverged'): fit(replace(run,lr=lr/2,suffix='-rerun'),run.run_id)
    selected=choose_lr([r for r in results if not r.get('rerun_of')])
    write_json(a.output/'complete.json',dict(status='complete',scope=SCOPE,runs=len(results),chosen_lr=selected,
        elapsed_seconds=time.monotonic()-started,phase='0A',full_pilot_complete=False))


if __name__=='__main__':main()
