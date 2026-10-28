"""CPU-only, pinned-history planning for the remaining registered Phase 0A.

No GPU acquisition, subprocess training or implicit checkpoint resume. Historical
results stay untouched; the output names fresh fits and required branch surveys.
The execution coordinator must still validate its current preflight and handoff.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from efficient_llm.a2.protocol import (
    DATASETS, PARENTS, FIXED_SETS, LRS, Run, choose_lr, choose_s2,
    lr_for_update, matrix, shared_check_sets,
)

SCIENCE_KEYS = ('spec_sha256', 'assets_sha256', 'data_sha256', 'data_spec_sha256',
                'secondary_map_sha256', 'software', 'python')


def read_locked(item):
    """Read only the explicitly pinned artifact, without accepting self hashes."""
    path = Path(item['path'])
    digest = item['sha256']
    if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('Invalid artifact SHA256')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('Historical artifact changed: '+str(path))
    return json.loads(raw)


def same_science(old, current):
    # Code revisions are independently pinned in each source's exact provenance.
    # The caller must additionally verify the reviewed numerical code lock.
    for key in SCIENCE_KEYS:
        if key not in old or key not in current or old[key] != current[key]:
            raise ValueError('Historical scientific identity differs: '+key)


def same_survey_science(survey,current):
    """Recognize only the exact original survey predating the scope amendment.

    Keep its original provenance intact; validate the already-approved change in
    launch dependencies and the later-added data-spec metadata explicitly.
    """
    old=survey['provenance']
    try:
        same_science(old,current)
        return
    except ValueError:
        project=Path(__file__).resolve().parents[1]
        lock=json.loads((project/'resources/a2_early_grid_reuse.json').read_text())
        digest=hashlib.sha256((json.dumps(survey,indent=2,allow_nan=False)+'\n').encode()).hexdigest()
        from efficient_llm.a2.protocol import compatible_data_spec
        if (digest!=lock['artifacts']['supervised/run/survey.json']
                or old['code_commit']!=lock['source_commit']
                or not compatible_data_spec(old['spec_sha256'],current['spec_sha256'])
                or 'data_spec_sha256' in old or current['data_spec_sha256']!=old['spec_sha256']):
            raise ValueError('Unreviewed historical survey identity')
        verify_current_core(project)
        # Validation view only; never replace the stored original provenance.
        same_science(dict(old,spec_sha256=current['spec_sha256'],
                          data_spec_sha256=old['spec_sha256']),current)


def validate_completed(row, provenance, pre):
    run = Run(**{k:row[k] for k in ('dataset','arm','set','budget','lr','suffix')})
    if (run.part != '0A' or run.set != 'S1' or run.suffix or row.get('rerun_of')
            or row.get('status') != 'complete' or row.get('diverged') is not False
            or any(row.get(k) != v for k,v in run.record().items())
            or row.get('provenance') != provenance):
        raise ValueError('Only successful original Phase0A S1 fits may be reused')
    expected = dict(examples=run.steps*64, microbatches=run.steps*8,
                    world_size=8, exact_order=True)
    if row.get('coverage') != expected or row.get('roundtrip_difference') != 0:
        raise ValueError('Historical data coverage or roundtrip failed')
    lrs = row.get('lr_updates', [])
    if len(lrs) != run.steps or any(
        not math.isfinite(lr) or abs(lr-lr_for_update(i,run.steps,run.lr))>1e-12
        for i,lr in enumerate(lrs,1)):
        raise ValueError('Historical full LR schedule differs')
    for key in ('val_teacher','val_post','val_tune_post','delta_post'):
        if not math.isfinite(row[key]): raise ValueError('Nonfinite historical metric')
    same_science(pre['provenance'],provenance)
    if (not pre['parent_sanity']['passed']
            or row['val_teacher'] != pre['parent']['final']['loss']
            or abs(row['delta_post']-(row['val_post']-row['val_teacher']))>1e-12):
        raise ValueError('Historical parent baseline differs')
    return run


def load_history(lock, current):
    """Fail closed on partial queues, duplicates or changed parent/data/receipts.

    Source descriptors are operator-reviewed and pinned *before* execution.
    Each contains provenance, result/pre paths, and terminal supervisor receipts.
    Failed historical fits need an explicit reviewed retry-import extension;
    this helper never silently launches them again.
    """
    rows, surveys, receipts = {}, {}, []
    if lock.get('schema') != 1 or not lock.get('sources'):
        raise ValueError('Reviewed nonempty history lock required')
    for source in lock['sources']:
        prov = source['provenance']; same_science(prov,current)
        completed = read_locked(source['complete'])
        supervisor = read_locked(source['supervisor'])
        run_receipt = read_locked(source['run_complete'])
        burn = read_locked(source['burn'])
        if (completed.get('status') != 'complete'
                or completed.get('scientific_complete',True) is not True
                or supervisor.get('training_status') != 'ok'
                or supervisor.get('training_returncode') != 0
                or run_receipt.get('status') != 'ok'
                or not burn.get('collective_progress_verified')):
            raise ValueError('Historical queue/handoff incomplete')
        fits = read_locked(source['results'])
        if completed.get('runs',len(fits)) != len(fits):
            raise ValueError('Historical run count differs')
        pre = {arm:read_locked(item) for arm,item in source['pre'].items()}
        for row in fits:
            run = validate_completed(row,prov,pre[row['arm']])
            if run.run_id in rows: raise ValueError('Duplicate historical fit')
            rows[run.run_id] = row
        for key,item in source.get('surveys',{}).items():
            survey = read_locked(item)
            if (key != survey['dataset']+'/'+survey['parent']
                    or survey['provenance'] != prov or key in surveys):
                raise ValueError('Wrong or duplicate historical survey')
            surveys[key] = survey
        receipts.append(source)
    for key,item in lock.get('surveys',{}).items():
        survey=read_locked(item['artifact'])
        if key in surveys or key != survey['dataset']+'/'+survey['parent'] or survey['provenance'] != item['provenance']:
            raise ValueError('Wrong or duplicate standalone survey')
        same_survey_science(survey,current)
        surveys[key]=survey
    return dict(rows=list(rows.values()),surveys=surveys,historical_receipts=receipts)


def remaining_runs(rows):
    """Require all six complete grids, then subtract exact completed run IDs."""
    completed = {r['run_id']:r for r in rows}
    if len(completed) != len(rows): raise ValueError('Duplicate completed ID')
    chosen = {}
    for ds in DATASETS:
        for arm in ('P300','P1T'):
            grid = [r for r in rows if (r['dataset'],r['arm'],r['budget'])==(ds,arm,'0.03B')]
            if len(grid)!=3 or {r['lr'] for r in grid}!=set(LRS):
                raise ValueError('Complete three-point parent grid required')
            lr = choose_lr(grid)
            if lr is None: raise ValueError('No eligible parent LR')
            chosen[ds,arm] = lr
    registered = [r for r in matrix(chosen) if r.part=='0A']
    by_id = {r.run_id:r for r in registered}
    if not set(completed)<=set(by_id):
        raise ValueError('History includes a fit outside selected-LR matrix')
    for key,row in completed.items():
        if any(row.get(k)!=v for k,v in by_id[key].record().items()):
            raise ValueError('Completed run recipe differs')
    return dict(chosen_lrs={ds+'/'+arm:lr for (ds,arm),lr in chosen.items()},
                reused_ids=list(completed),
                pending=[r.record() for r in registered if r.run_id not in completed])


def branch_sequence(surveys,identity):
    """Never drop an unmeasured branch based on loss or guessed eligibility."""
    todo=[]
    for ds in DATASETS:
        if identity['tests'][ds]['T2'] is not True:
            raise ValueError('Dataset pair identity failed: '+ds)
        for parent in PARENTS:
            key=ds+'/'+parent
            if key in surveys: continue
            if parent in ('P300','P1T'):
                raise ValueError('Missing completed annealed survey: '+key)
            if identity['tests'][ds]['T3'][parent]['passed']:
                todo.append(dict(dataset=ds,parent=parent,action='survey'))
    return todo


def freeze_sets(surveys,identity,current,assets):
    if branch_sequence(surveys,identity):
        raise ValueError('Unmeasured eligible branches; S2 cannot be frozen')
    decisions={};sets={};eligibility={}
    for ds in DATASETS:
        usable=[]
        for parent in PARENTS:
            key=ds+'/'+parent
            t3=parent in ('P300','P1T') or identity['tests'][ds]['T3'][parent]['passed']
            if not t3:
                eligibility[key]=dict(T1=False,T3=False,reason='Pinned branch identity failed')
                continue
            s=surveys[key]
            if (s['dataset'],s['parent'])!=(ds,parent): raise ValueError('Wrong survey parent')
            same_survey_science(s,current)
            passed=s['T1_precision_passed'] is True and s['parent_sanity']['passed'] is True
            eligibility[key]=dict(T1=passed,T3=True)
            if passed:usable.append(parent)
        decisions[ds]=choose_s2({p:surveys[ds+'/'+p]['group_bi'] for p in usable},usable)
        sets[ds]=dict(FIXED_SETS,S2=decisions[ds]['groups'])
    shared_check_sets(sets,assets)
    return dict(sets=sets,selection=decisions,eligibility=eligibility)


def verify_current_core(project):
    """Use reviewed historical byte/AST locks; no new numerical waiver."""
    from efficient_llm.a2.early_grid import ast_hash
    project=Path(project)
    for filename in ('a2_pair_reuse.json','a2_early_grid_reuse.json'):
        lock=json.loads((project/'resources'/filename).read_text())
        for name,digest in lock['unchanged_files'].items():
            if hashlib.sha256((project/name).read_bytes()).hexdigest()!=digest:
                raise ValueError('Numerical source changed: '+name)
        for name,functions in lock['unchanged_definitions'].items():
            for fn,digest in functions.items():
                if ast_hash(project/name,fn)!=digest:raise ValueError('Scientific definition changed: '+fn)


def prerequisite_jobs(*, source, inputs, port, history, reference, identity, sequence):
    """Build an ordinary supervised job manifest, no execution or GPU reclaim."""
    source,inputs=Path(source),Path(inputs)
    if len({(s['dataset'],s['parent']) for s in sequence}) != len(sequence):
        raise ValueError('Duplicate branch survey')
    launch=['{python}','-m','accelerate.commands.launch','--config_file',str(source/'resources/accelerate_b200.yaml'),
            '--main_process_port',str(port),'-m','efficient_llm.a2']
    common=['--assets',str(inputs/'assets.json'),'--data',str(inputs/'data'),
            '--secondary-map',str(inputs/'secondary_map.json')]
    jobs=[]
    for item in sequence:
        ds,parent=item['dataset'],item['parent']
        if ds not in DATASETS or parent not in ('S72','S188') or item['action']!='survey':
            raise ValueError('Only missing branch surveys belong in this stage')
        name=ds+'-'+parent
        jobs.append(dict(name=name,gpus=list(range(8)),timeout_seconds=900,
            argv=launch+['survey']+common+['--dataset',ds,'--parent',parent,
                '--output','{run_dir}/surveys/'+name+'.json'],
            required_outputs=[dict(path='surveys/'+name+'.json',json_equals=dict(dataset=ds,parent=parent))]))
    jobs.append(dict(name='freeze-sets',timeout_seconds=120,
        argv=['{python}','-m','scripts.a2_remaining','--history',str(history),'--reference',str(reference),
              '--identity',str(identity),'--survey-dir','{run_dir}/surveys','--assets',str(inputs/'assets.json'),
              '--output','{run_dir}/sets.json'],
        required_outputs=[dict(path='sets.json',json_equals=dict(status='frozen'))]))
    jobs.append(dict(name='shared-numerical',gpus=list(range(8)),timeout_seconds=3600,
        argv=launch+['check']+common+['--dataset','dclm','--parent','P300',
            '--sets','{run_dir}/sets.json','--output','{run_dir}/check.json'],
        required_outputs=[dict(path='check.json',json_equals=dict(passed=True,scope='shared_mechanism'))]))
    return jobs


def prepare_prerequisites(source, inputs, history_path, code_commit, spent_seconds):
    """CPU launch gate: actual history, native provenance, exact input set/budget.

    Does not acquire GPUs. The existing launcher owns source freezing, Accelerate
    configuration and native supervisor handoff after this function succeeds.
    """
    from argparse import Namespace
    from efficient_llm.a2.__main__ import provenance
    from scripts.a2_preflight_inputs import _parent_asset_plan
    source,inputs=Path(source),Path(inputs)
    verify_current_core(source)
    reference=provenance(Namespace(code_commit=code_commit,spec=source/'docs/A2_pilot_spec.md',
        assets=inputs/'assets.json',data=inputs/'data'))
    reference['secondary_map_sha256']=hashlib.sha256((inputs/'secondary_map.json').read_bytes()).hexdigest()
    lock=json.loads(Path(history_path).read_text())
    history=load_history(lock,reference)
    plan=remaining_runs(history['rows'])
    identity=json.loads((source/'docs/results/a2_review_20261028/identity_metadata.json').read_text())
    sequence=branch_sequence(history['surveys'],identity)
    # Sum worst-case child timeouts plus cleanup reserve, not only expected cost.
    worst_seconds=len(sequence)*900+120+3600+600
    if not math.isfinite(spent_seconds) or spent_seconds<0 or spent_seconds+worst_seconds>18*3600:
        raise ValueError('Insufficient Phase0A budget for prerequisites and handoff')
    assets=json.loads((source/'resources/a2_inputs_20261028.json').read_text())
    required={('dclm','P300')} | {(s['dataset'],s['parent']) for s in sequence}
    downloads={}
    for ds,parent in sorted(required):
        for item in _parent_asset_plan(assets,parent,ds)['downloads']:
            if item['path'] in downloads and downloads[item['path']]!=item:
                raise ValueError('Conflicting pinned asset')
            downloads[item['path']]=item
    return dict(history_lock=lock,reference=reference,plan=plan,sequence=sequence,
        asset_plan=dict(assets,downloads=list(downloads.values())),
        budget=dict(phase='0A',spent_seconds=spent_seconds,worst_seconds=worst_seconds,
                    hard_cap_seconds=18*3600,report_threshold_seconds=15*3600))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--history',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True,help='Current provenance JSON')
    p.add_argument('--identity',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--survey-dir',type=Path)
    p.add_argument('--assets',type=Path)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh plan output required')
    verify_current_core(Path(__file__).resolve().parents[1])
    current=json.loads(a.reference.read_text());lock=json.loads(a.history.read_text())
    history=load_history(lock,current)
    identity=json.loads(a.identity.read_text())
    if a.survey_dir is not None:
        if a.assets is None:raise ValueError('Assets required for frozen sets')
        for item in branch_sequence(history['surveys'],identity):
            ds,parent=item['dataset'],item['parent']
            survey=json.loads((a.survey_dir/(ds+'-'+parent+'.json')).read_text())
            history['surveys'][ds+'/'+parent]=survey
        result=freeze_sets(history['surveys'],identity,current,json.loads(a.assets.read_text()))
        result.update(status='frozen',history_sha256=hashlib.sha256(a.history.read_bytes()).hexdigest(),
                      identity_sha256=hashlib.sha256(a.identity.read_bytes()).hexdigest(),
                      survey_evidence=history['surveys'])
        # Complete survey evidence is retained: no relabelling old source commits.
    else:
        result=dict(remaining_runs(history['rows']),branch_surveys=branch_sequence(history['surveys'],identity),
                launch=False,history_sha256=hashlib.sha256(a.history.read_bytes()).hexdigest(),
                note='S2 and actual numerical coverage must precede remaining heals')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
