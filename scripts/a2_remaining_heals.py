"""Exact pending Phase0A fits under the existing supervisor, no GPU acquisition.

Consumes reviewed historical receipts and a current full-pattern preflight.
No automatic survey, LR sweep, source conversion, or checkpoint continuation.
"""
from dataclasses import replace
import json
import math
from pathlib import Path
import subprocess
import time

from efficient_llm.a2.data import write_json
from efficient_llm.a2.protocol import Run,lr_for_update,validate_shared_check
from efficient_llm.a2.pair import validate_result
from scripts.a2_remaining import remaining_runs


def verify_result(row,run,provenance,rerun_of,exitcode):
    try:
        validate_result(row,run,provenance,rerun_of,exitcode)
    except KeyError as error:
        raise ValueError('Incomplete native result receipt') from error
    if row['status']=='complete':
        expected=dict(examples=run.steps*64,microbatches=run.steps*8,world_size=8,exact_order=True)
        if (row.get('diverged') is not False or row.get('coverage')!=expected
                or row.get('roundtrip_difference')!=0):
            raise ValueError('Pending fit coverage/roundtrip differs')
        lrs=row.get('lr_updates',[])
        if len(lrs)!=run.steps or any(not math.isfinite(v) or abs(v-lr_for_update(i,run.steps,run.lr))>1e-12
                for i,v in enumerate(lrs,1)):
            raise ValueError('Pending fit full schedule differs')
        if any(not math.isfinite(row[k]) for k in ('val_teacher','val_post','val_tune_post','delta_post')):
            raise ValueError('Nonfinite completed fit')
        if abs(row['delta_post']-(row['val_post']-row['val_teacher']))>1e-12:
            raise ValueError('Teacher-normalized loss differs')


def execute_remaining(*,root,launcher,common,history,pre,gate,sets,assets,
                      sets_sha256,estimated,spent_seconds,subset=None):
    """Caller must use load_history to validate pinned historical inputs first.

    subset is an explicit bounded list from the registered pending matrix; omitted
    means all pending fits. This never invokes the old full-matrix coordinator.
    """
    plan=remaining_runs(history['rows'])
    pending={r['run_id']:r for r in plan['pending']}
    ids=list(pending) if subset is None else list(subset)
    if not ids or len(set(ids))!=len(ids) or not set(ids)<=pending.keys():
        raise ValueError('Explicit subset must contain only unique pending fits')
    prov=gate['provenance']
    if (gate.get('status')!='passed' or gate.get('launch_scope','all')!='all'
            or gate.get('sets_sha256')!=sets_sha256):
        raise ValueError('Current full-pattern preflight required')
    required=('T4','T5','T6','T7','T8','T10_formula','T11','T12','T13')
    if any(gate.get('tests',{}).get(t,{}).get('passed') is not True for t in required):
        raise ValueError('Incomplete preflight evidence')
    validate_shared_check(gate['shared_numerical'],sets,assets,prov,sets_sha256)
    if not launcher or not all(isinstance(v,str) and v for v in launcher):
        raise ValueError('Explicit native distributed launcher required')
    for flag in ('--preflight','--sets','--assets','--data','--secondary-map'):
        if common.count(flag)!=1 or common.index(flag)+1>=len(common):
            raise ValueError('Missing or repeated native argument: '+flag)
    if '--dataset' in common:raise ValueError('Dataset is chosen per pending fit')
    # Recheck the actual files used by children, not just in-memory gate objects.
    if json.loads(Path(common[common.index('--preflight')+1]).read_text())!=gate:
        raise ValueError('Child preflight differs')
    from efficient_llm.diagnostic_data import sha256
    if sha256(common[common.index('--sets')+1])!=sets_sha256:
        raise ValueError('Child frozen sets differ')
    for key,field in (('--assets','assets_sha256'),('--secondary-map','secondary_map_sha256')):
        if sha256(common[common.index(key)+1])!=prov[field]:raise ValueError('Child input identity differs')
    if sha256(Path(common[common.index('--data')+1])/'manifest.json')!=prov['data_sha256']:
        raise ValueError('Child data manifest differs')
    if not math.isfinite(spent_seconds) or not 0<=spent_seconds<18*3600-600:
        raise ValueError('Phase0A cap exhausted')
    for record in (pending[k] for k in ids):
        run=Run(**{k:record[k] for k in ('dataset','arm','set','budget','lr','suffix')})
        for parent in {run.parent,run.teacher}:
            key=run.dataset+'/'+parent
            if not gate.get('parents',{}).get(key,{}).get('passed'):
                raise ValueError('Ineligible affected parent: '+key)
            baseline=pre[key]
            if (baseline.get('provenance')!=prov or baseline.get('sets_sha256')!=sets_sha256
                    or not baseline['parent_sanity']['passed'] or run.set not in baseline['sets']):
                raise ValueError('Current baseline differs: '+key)
        estimate=estimated[run.budget]
        if not math.isfinite(estimate) or estimate<=0:raise ValueError('Invalid estimate')
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    write_json(root/'reuse-plan.json',plan)
    started=time.monotonic();results=[];finished=[]
    def fit(run,rerun_of=None):
        remaining=18*3600-spent_seconds-(time.monotonic()-started)-600
        timeout=1.5*estimated[run.budget]
        if remaining<timeout:raise TimeoutError('Insufficient Phase0A time including handoff reserve')
        base=pre[run.dataset+'/'+run.parent];teacher=pre[run.dataset+'/'+run.teacher]
        values=dict(base['sets'][run.set],parent_final=base['parent']['final']['loss'],
            teacher_final=teacher['parent']['final']['loss'],
            parent_secondary=base['parent'].get('secondary',{}).get('loss'),
            teacher_secondary=teacher['parent'].get('secondary',{}).get('loss'),
            provenance=prov,sets_sha256=sets_sha256,parent=run.parent,teacher=run.teacher)
        rf=root/(run.run_id+'.json');pf=root/(run.run_id+'-pre.json');out=root/run.run_id
        write_json(rf,dict(run.record(),rerun_of=rerun_of));write_json(pf,values)
        argv=[*launcher,'heal',*common,'--dataset',run.dataset,'--run',str(rf),
              '--pre-values',str(pf),'--output',str(out),'--estimated-seconds',str(estimated[run.budget])]
        with (root/(run.run_id+'.log')).open('x') as log:
            child=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,timeout=timeout,check=False)
        row=json.loads((out/'result.json').read_text())
        verify_result(row,run,prov,rerun_of,child.returncode)
        if row['status']=='complete' and row['val_teacher']!=values['teacher_final']:
            raise ValueError('Result teacher baseline differs')
        results.append(row);write_json(root/'results.json',results)
        return row
    for key in ids:
        record=pending[key];run=Run(**{k:record[k] for k in ('dataset','arm','set','budget','lr','suffix')})
        row=fit(run)
        if row.get('diverged'):
            row=fit(replace(run,lr=run.lr/2,suffix=run.suffix+'-rerun'),run.run_id)
        if row['status']=='complete':finished.append(key)
    result=dict(status='complete' if len(finished)==len(ids) else 'incomplete',phase='0A',
        scientific_complete=len(finished)==len(ids),requested_ids=ids,completed_original_ids=finished,
        reused_ids=plan['reused_ids'],runs=len(results),elapsed_seconds=time.monotonic()-started,
        historical_phase0A_seconds=spent_seconds,
        full_phase_complete=len(finished)==len(pending),
        unexecuted_ids=[k for k in pending if k not in ids])
    write_json(root/'complete.json',result)
    return result
