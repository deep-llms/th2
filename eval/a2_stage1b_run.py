"""Four/two-model Stage1b evaluation inside the native supervisor's GPU allocation.

No ownership changes here. The launcher must independently verify training and
communicating-burn handoff before entry. Canonical request/scoring code is shared
with the validated Phase0A screen; only selection and reporting differ.
"""
from pathlib import Path

from eval.a2_checkpoint import verify_identity
from eval.a2_run import _run_screen, write
from eval.a2_stage1b_plan import validate_terminal_bundle
from eval.a2_stage1b_stats import PART_MODELS, paired_report
from eval.downstream_suite import COUNTS
from scripts.a2_remaining import read_locked


def validate_plan(plan):
    phase=plan.get('phase')
    if phase not in ('part1','part2'):raise ValueError('Unknown FWE evaluation part')
    if (plan.get('schema') != 2 or set(plan['identities']) != set(PART_MODELS[phase])
            or plan.get('tasks') != list(COUNTS) or plan.get('max_length') != 4096
            or plan.get('gpus') != list(range(8)) or plan.get('batch_size') != 2):
        raise ValueError('Require the fixed FWE part selection at batch2')
    bundle = read_locked(plan['identity_bundle'])
    if (bundle.get('schema') != 2 or bundle['identities'] != plan['identities']
            or bundle.get('dataset') != 'fwe' or bundle.get('phase') != phase):
        raise ValueError('Plan differs from reviewed identity bundle')
    locks = bundle['source_locks']
    stage, reference, rows, selection = validate_terminal_bundle(locks['stage'], locks['assets'])
    if bundle['selection'] != selection or stage['phase']!=phase:
        raise ValueError('Stage1 checkpoint selection changed')
    identities = {}
    for name, item in plan['identities'].items():
        identity = read_locked(item)
        verify_identity(identity)
        selected = selection[name]
        if (identity['dataset'] != 'fwe' or identity['parent'] != selected['parent']
                or identity['kind'] != selected['kind'] or identity['assets'] != locks['assets']):
            raise ValueError('Checkpoint label or assets do not match Stage1')
        if selected['kind'] == 'healed':
            receipts = identity['training_receipts']
            if (identity['schema'] != 2 or receipts['reference'] != stage['reference']
                    or read_locked(receipts['result']) != rows[selected['run_id']]):
                raise ValueError('Student is not the reviewed main Stage1 fit')
        identities[name] = identity
    for name in ('native_python','eval_python','tokenizer','dataset_root'):
        if not Path(plan[name]).is_absolute():
            raise ValueError('Explicit absolute runtime/input paths required')
    for name in ('manifest','input_spec'):
        read_locked(plan[name])
    return identities


def finalize(plan, root, result, replay_locks):
    # Locks still have to match after scoring and canonical CPU replay.
    validate_plan(plan)
    replays = {name: read_locked(item) for name,item in replay_locks.items()}
    intervals = paired_report(replays,plan['phase'])
    return dict(paired_accuracy=write(root/'paired_accuracy.json',intervals))


def run_screen(plan, output, *, timeout_seconds):
    if plan.get('phase') not in ('part1','part2'):raise ValueError('Unknown FWE evaluation part')
    gates=('P300-parent','P300-student') if plan['phase']=='part1' else ('S72-parent','X72-student')
    return _run_screen(plan,output,timeout_seconds=timeout_seconds,
        validator=validate_plan,gate_names=gates,finalize=finalize)


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','plan-sha256','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--timeout-seconds',type=float,required=True)
    args=parser.parse_args()
    run_screen(read_locked(dict(path=args.plan,sha256=args.plan_sha256)),args.output,
               timeout_seconds=args.timeout_seconds)


if __name__ == '__main__':
    main()
