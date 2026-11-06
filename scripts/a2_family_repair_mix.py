"""Repair observed text-column failures, reuse verified sources, then interleave."""
import argparse
import copy
import concurrent.futures
import time
from pathlib import Path
from efficient_llm.a2_family.common import read, write, sha, SPEC_SHA
from scripts.a2_remaining import read_locked
from scripts.a2_family_prepare_mix import prepare_source, primary_exclusions, quotas, interleave


def corrected_plan(old,fix):
    plan=copy.deepcopy(old);entries={e['name']:e for e in plan['sources']}
    for name,override in fix['source_overrides'].items():
        if set(override)-{'text_columns','path_suffixes','download_group','repo','revision','source_pool','substitution','historical_dclm_prefix'}:
            raise ValueError('Correction attempts to change numerical recipe or weights')
        change=dict(override);group=change.pop('download_group',None);historical=change.pop('historical_dclm_prefix',None)
        if historical is not None:
            recipe=read_locked(historical)
            indices=recipe['selected_dclm_indices']
            if name!='dclm' or indices!=list(range(216)):raise ValueError('Not the verified historical DCLM pool')
            change['files']=entries[name]['files'][:216]
            if [r['index'] for r in change['files']]!=indices:raise ValueError('DCLM raw prefix indices differ')
        if group is not None:change['files']=fix['download_groups'][group]
        entries[name].update(change)
    return plan


def verify_source(receipt, target):
    value=read_locked(receipt)
    if value['status']!='complete' or value['tokens']!=target or value['seed']!=2030:
        raise ValueError('Source coverage/order differs')
    for path,key in [('token_file','token_sha256'),('documents_path','documents_sha256')]:
        if sha(value[path])!=value[key]:raise ValueError('Prepared source bytes changed')
    directory=Path(receipt['path']).parent
    if sha(directory/'document_order.npy')!=value['document_order_sha256']:
        raise ValueError('Source document order changed')
    if Path(value['token_file']).stat().st_size!=target*4:raise ValueError('Source token count differs')
    return value


def repair(plan_path, old_root, output, cx_root, corrections):
    old_root=Path(old_root);output=Path(output)
    old=read(plan_path);fix=read(corrections);plan=corrected_plan(old,fix)
    if old['spec_sha256']!=SPEC_SHA or fix['original_plan_sha256']!=sha(plan_path):
        raise ValueError('Unregistered repair source')
    progress=read(old_root/'progress.json');failed=progress.get('failed',{})
    entries={e['name']:e for e in plan['sources']}
    for name in fix['source_overrides']:
        historical=fix['source_overrides'][name].get('historical_dclm_prefix')
        if not historical and (name not in failed or 'No registered text column' not in failed[name]):
            raise ValueError('Repair is only for an observed text-column failure')
        columns=entries[name]['text_columns']
        if not columns or any(not isinstance(c,str) for c in columns):raise ValueError('Invalid column override')
    output.mkdir(parents=True,exist_ok=False)
    effective=write(output/'effective_plan.json',plan)
    excluded,exclusion_locks=primary_exclusions(plan);plan['heldout_hashes']={}
    for group in ('tuning','final'):
        folder=Path(cx_root)/group;m=read(folder/'manifest.json')
        if sha(folder/'documents.json')!=m['documents_sha256']:raise ValueError('CX metadata changed')
        for i,d in enumerate(read(folder/'documents.json')['documents']):
            plan['heldout_hashes'].setdefault(d['text_sha256'],[]).append(dict(set=group,index=i))
    weights={e['name']:e['effective_weight'] for e in plan['sources']};targets=quotas(weights,plan['prepared_tokens']);repaired={}
    failures={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        tasks={pool.submit(prepare_source,plan,entries[name],output/'sources'/name,targets[name],excluded):name for name in fix['source_overrides']}
        for future in concurrent.futures.as_completed(tasks):
            name=tasks[future]
            try:repaired[name]=future.result()
            except Exception as error:failures[name]=repr(error)
            write(output/'progress.json',dict(stage='repairing_sources',repaired=repaired,failed=failures))
    while True:
        terminal=read(old_root/'complete.json') if (old_root/'complete.json').is_file() else read(old_root/'progress.json')
        available=set(terminal.get('ready',terminal.get('sources',{})))
        observed=available|set(terminal.get('failed',{}))|set(repaired)|set(failures)
        if set(weights)<=observed:break
        time.sleep(30)
    unresolved=set(terminal.get('failed',{}))-set(repaired)-set(failures)
    if unresolved:
        failures.update({n:terminal['failed'][n] for n in unresolved})
    if failures:
        write(output/'complete.json',dict(status='incomplete',repaired=repaired,failed=failures))
        return
    ready={**terminal.get('ready',terminal.get('sources',{})),**repaired}
    if set(ready)!=set(weights):raise ValueError('Missing prepared source')
    sources={name:verify_source(receipt,targets[name]) for name,receipt in ready.items()}
    stream=interleave(sources,weights,output/'stream',plan['prepared_tokens'])
    write(output/'complete.json',dict(status='complete',stream=stream,sources=ready,
        primary_exclusions=exclusion_locks,plan_sha256=effective['sha256'],effective_plan=effective,
        original_plan_sha256=sha(plan_path),original_preparation=str(old_root),
        schema_corrections=fix,prior_failure=failed,flags=plan['flags'],missing_sources=plan['missing_sources']))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('plan','old-root','output','cx-root','corrections'):p.add_argument('--'+k,required=True)
    a=p.parse_args();repair(a.plan,a.old_root,a.output,a.cx_root,a.corrections)
