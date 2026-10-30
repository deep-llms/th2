"""Combine the reviewed four and two identity bundles for one final FWE screen."""
from pathlib import Path
from eval.a2_run import lock
from eval.a2_stage1b_plan import prepare_identities,validate_terminal_bundle
from eval.a2_stage1b_run import validate_plan as validate_part_plan
from eval.a2_stage1b_stats import PART_MODELS
from eval.a2_stage1b_regression import admit_regression
from eval.a2_request_bridge import digest
from eval.downstream_suite import COUNTS
from scripts.a2_remaining import read_locked
from efficient_llm.diagnostic_data import sha256

# Only new, explicitly scoped adapters may be added to the passing regression's
# evaluation/model tree. Every original file remains byte-identical.
ADDITIONS=frozenset('eval/'+n+'.py' for n in (
    'a2_stage1b_check','a2_stage1b_parent_score','a2_stage1b_batch_compare',
    'a2_stage1b_final_plan','a2_stage1b_final_run','a2_stage1b_batch_report'))
POLICY_SHA='8797ab8557de5ef6076c2d1aa8cedb906dd89217566009e2f93d8077262efa4c'
TIME_POLICY_SHA='e9cf0904bf92c8365b209ce940150cb33ce83c5987512242d3dcb2ef4d79a3ac'
PREPARED_SHA='85cf7f2119e35e81a79aea48f58080bdc9ba5371b27accbda0eac52aa6770bbf'


def validate_policy(item):
    if item['sha256']!=POLICY_SHA:raise ValueError('Explicit user continuation policy changed')
    return read_locked(item)


def validate_time_policy(item):
    if item['sha256']!=TIME_POLICY_SHA:raise ValueError('Explicit time-policy override changed')
    return read_locked(item)


def admit_final_regression(bundle_lock,source,policy_lock):
    source=Path(source)
    validate_policy(policy_lock)
    bundle=read_locked(bundle_lock)
    old_source=Path(bundle['source_manifest']['path']).parent/'source'
    # Validate all original terminal/artifact/accuracy evidence against its own
    # frozen source, then independently compare current scoring/model files.
    _,values,report=admit_regression(bundle_lock,old_source)
    old={k:h for k,h in values['source_manifest']['files'].items() if k.startswith(('eval/','efficient_llm/'))}
    actual={str(p.relative_to(source)) for folder in ('eval','efficient_llm') for p in (source/folder).rglob('*.py')}
    if actual!=set(old)|ADDITIONS or any(sha256(source/k)!=h for k,h in old.items()):
        raise ValueError('Original regression scoring/model core changed or unregistered additions')
    return dict(original_files=len(old),unchanged=True,additions={k:sha256(source/k) for k in sorted(ADDITIONS)},
                regression=bundle_lock,regression_passed=report['passed'])


def prepare_plan(part1,part2,assets,output,*,common,policy,time_policy,regression,failed_gate,prepared):
    out=Path(output);bundles={}
    for phase,stage in [('part1',part1),('part2',part2)]:
        prepare_identities(stage,assets,out/phase)
        bundles[phase]=lock(out/phase/'identities.json')
    identities={}
    for item in bundles.values():identities.update(read_locked(item)['identities'])
    identities={name:identities[name] for name in PART_MODELS['final']}
    plan=dict(common,schema=3,phase='final',identities=identities,identity_bundles=bundles,
        policy=policy,time_policy=time_policy,regression=regression,failed_gate=failed_gate,prepared=prepared)
    validate_plan(plan)
    return plan


def validate_plan(plan):
    validate_policy(plan['policy']);validate_time_policy(plan['time_policy']);read_locked(plan['regression'])
    if (plan.get('schema')!=3 or plan.get('phase')!='final'
            or tuple(plan['identities'])!=PART_MODELS['final']
            or set(plan['identity_bundles'])!={'part1','part2'} or plan['batch_size']!=2):
        raise ValueError('Require registered six-model final screen')
    identities={};references=[];rows=[];assets=[];all_locks={}
    for phase in ('part1','part2'):
        item=plan['identity_bundles'][phase];bundle=read_locked(item)
        all_locks.update(bundle['identities'])
        sub=dict(plan,schema=2,phase=phase,identity_bundle=item,identities=bundle['identities'])
        identities.update(validate_part_plan(sub))
        locks=bundle['source_locks'];assets.append(locks['assets'])
        _,ref,by_id,_=validate_terminal_bundle(locks['stage'],locks['assets'])
        references.append(ref);rows.append(by_id)
    if (assets[0]!=assets[1] or references[0]!=references[1]
            or any(rows[1].get(k)!=v for k,v in rows[0].items())
            or plan['identities']!={k:all_locks[k] for k in PART_MODELS['final']}):
        raise ValueError('Part2 changed inherited Part1 fits/identity locks')
    prepared=read_locked(plan['prepared']);protocol=prepared['protocol']
    if (plan['prepared']['sha256']!=PREPARED_SHA or len(prepared['requests'])!=78259
            or protocol['tasks']!=list(COUNTS) or protocol['max_length']!=4096
            or protocol['limit'] is not None or protocol['seed']!=42 or protocol['num_fewshot']!=0
            or protocol['manifest_sha256']!=plan['manifest']['sha256']
            or protocol['tokenizer_files']!=read_locked(plan['input_spec'])['tokenizer_provenance']['gpt_neox']['files']
            or prepared['request_sha256']!=digest(prepared['requests'])
            or prepared['protocol_sha256']!=digest(protocol)):
        raise ValueError('Original canonical requests changed')
    failed=read_locked(plan['failed_gate'])
    if failed['passed'] is not False or failed['batch_max_nats_per_token']!=.011164188385009766:
        raise ValueError('Original failed-gate receipt changed')
    return identities
