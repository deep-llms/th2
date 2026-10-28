"""Build downstream checkpoint locks from reviewed, completed A2 receipts.

CPU only. Does not launch jobs or infer a checkpoint from a live process. The
outer launcher must independently check live communicating burns before reclaim.
"""
import argparse
import json
from pathlib import Path

from efficient_llm.a2.protocol import Run
from eval.a2_checkpoint import checkpoint_identity
from eval.a2_run import lock, write
from eval.a2_suite import select_initial_screen
from scripts.a2_remaining import load_history, read_locked, remaining_runs
from scripts.a2_remaining_heals import verify_result


def prepare_identities(stage_lock, history_lock, assets_lock, sets_lock, output):
    stage=read_locked(stage_lock);history_spec=read_locked(history_lock)
    reference=read_locked(stage['reference'])
    if reference['assets_sha256']!=assets_lock['sha256']:raise ValueError('Wrong parent assets')
    for field in ('complete','supervisor','run_complete','burn'):
        value=read_locked(stage[field])
        ok={'complete':value.get('scientific_complete') is True and value.get('status')=='complete',
            'supervisor':value.get('training_status')=='ok' and value.get('training_returncode')==0,
            'run_complete':value.get('status')=='ok',
            'burn':value.get('collective_progress_verified') is True}[field]
        if not ok:raise ValueError('Training stage or terminal handoff incomplete: '+field)
    history=load_history(history_spec,reference);plan=remaining_runs(history['rows'])
    if read_locked(stage['plan'])!=plan:raise ValueError('Current stage differs from exact pending plan')
    current=read_locked(stage['results']);expected={r['run_id'] for r in plan['pending']}
    original={r['run_id'] for r in current if not r.get('rerun_of')}
    if original!=expected:raise ValueError('Missing original registered fits')
    for row in current:
        run=Run(**{k:row[k] for k in ('dataset','arm','set','budget','lr','suffix')})
        if (row.get('rerun_of') or run.run_id) not in expected:raise ValueError('Unexpected current fit')
        verify_result(row,run,reference,row.get('rerun_of'),0 if row['status']=='complete' else 1)
    rows=history['rows']+current
    if len({r['run_id'] for r in rows})!=len(rows):raise ValueError('Duplicate original receipts')
    selection=select_initial_screen(rows)
    # Resolve the exact adjacent checkpoint result and compare it with its
    # reviewed aggregate. Preserve each original source's provenance verbatim.
    sources=[(read_locked(s['results']),s['results'],s['provenance']) for s in history_spec['sources']]
    sources.append((current,stage['results'],reference))
    by_id={r['run_id']:(r,aggregate,prov) for fits,aggregate,prov in sources for r in fits}
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    identities={}
    for name,item in selection.items():
        args=dict(assets_lock=assets_lock,dataset=item['dataset'],parent=item['parent'])
        if item['kind']=='healed':
            row,aggregate,provenance=by_id[item['run_id']]
            path=Path(aggregate['path']).parent/item['run_id']/'result.json'
            result_lock=lock(path)
            if read_locked(result_lock)!=row or row['provenance']!=provenance:
                raise ValueError('Export-adjacent result differs from reviewed original receipt')
            reference_lock=write(out/(name+'-reference.json'),provenance)
            args.update(checkpoint=str(path.parent/'final'),result_lock=result_lock,
                        reference_lock=reference_lock,sets_lock=sets_lock)
        identity=checkpoint_identity(**args)
        identities[name]=write(out/(name+'-identity.json'),identity)
    result=dict(schema=1,identities=identities,selection=selection,source_locks=dict(
        stage=stage_lock,history=history_lock,assets=assets_lock,sets=sets_lock))
    write(out/'identities.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('stage','history','assets','sets'):
        p.add_argument('--'+name,required=True);p.add_argument('--'+name+'-sha256',required=True)
    p.add_argument('--output',required=True);a=p.parse_args()
    locks=[dict(path=getattr(a,k),sha256=getattr(a,k+'_sha256')) for k in ('stage','history','assets','sets')]
    prepare_identities(*locks,a.output)


if __name__=='__main__':main()
