"""Fixed Stage1 downstream selection; no evaluation or GPU launch on import."""
from efficient_llm.a2_stage1.protocol import ARMS, Run, SPEC_SHA
from scripts.a2_stage1 import validate_fit


def select_stage1_screen(rows, chosen_lr, provenance):
    """Choose only three original parents and the three registered main fits.

    The coordinator must first verify terminal artifacts and lock the chosen LR
    from the completed grid. No benchmark scores are used in this selection.
    This validates original receipts without translating them into Phase0A runs.
    """
    if provenance.get('spec_sha256') != SPEC_SHA:
        raise ValueError('Wrong Stage1 preregistration')
    if len({row['run_id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate training receipt')
    selection = {name+'-parent':dict(kind='parent',dataset='dclm',parent=name)
                 for name in ('P300','P1T','S72')}
    for row in rows:
        run = Run(row['arm'], row['budget'], row['lr'])
        validate_fit(row, run, provenance)
    for arm in ARMS:
        expected = Run(arm, '3.2B', chosen_lr)
        candidates = [row for row in rows if row['run_id'] == expected.run_id]
        if len(candidates) != 1:
            raise ValueError('Missing registered main checkpoint: '+arm)
        row = candidates[0]
        if row['status'] != 'complete' or row.get('diverged') is not False:
            raise ValueError('Unsuccessful main checkpoint: '+arm)
        selection[arm+'-student'] = dict(kind='healed',dataset='dclm',
                parent=expected.parent,teacher=expected.teacher,arm=arm,
                run_id=expected.run_id,budget='3.2B',lr=chosen_lr)
    return selection


def validate_terminal_bundle(stage_lock, assets_lock):
    """Validate reviewed terminal locks and independently repeat LR selection.

    This consumes an archived successful handoff. The outer launcher must still
    verify current burn ownership before acquiring GPUs for evaluation.
    """
    from pathlib import Path
    from efficient_llm.a2_stage1.protocol import GRID, grid_choice, remaining
    from efficient_llm.diagnostic_data import sha256
    from scripts.a2_remaining import read_locked

    stage = read_locked(stage_lock)
    values = {name: read_locked(stage[name]) for name in
              ('reference','results','choice','tests','decision','complete',
               'supervisor','run_complete','burn')}
    reference = values['reference']
    if reference.get('assets_sha256') != assets_lock['sha256']:
        raise ValueError('Wrong Stage1 parent assets')
    read_locked(assets_lock)
    gate = values['tests']
    complete = values['complete']
    supervisor = values['supervisor']
    if (gate.get('status') != 'passed' or gate.get('eligible_x72') is not True
            or gate.get('provenance') != reference
            or complete.get('status') != 'complete'
            or complete.get('training_complete') is not True
            or complete.get('decision_sha256') != stage['decision']['sha256']
            or supervisor.get('training_status') != 'ok'
            or supervisor.get('training_returncode') != 0
            or values['burn'].get('collective_progress_verified') is not True):
        raise ValueError('Stage1 training or terminal handoff incomplete')
    terminal = values['run_complete']
    if terminal.get('status') != 'ok' or len(terminal.get('jobs',[])) != 1:
        raise ValueError('Stage1 supervisor job incomplete')
    job = terminal['jobs'][0]
    if job.get('name') != 'stage1-heals' or job.get('status') != 'ok' or job.get('returncode') != 0:
        raise ValueError('Wrong or unsuccessful Stage1 supervisor job')
    base = Path(stage['run_complete']['path']).resolve().parent
    if not job.get('artifacts'):
        raise ValueError('Missing terminal artifact receipts')
    complete_bound = False
    for artifact in job['artifacts']:
        path = (base/artifact['path']).resolve()
        if not path.is_relative_to(base) or sha256(path) != artifact['sha256']:
            raise ValueError('Terminal artifact changed')
        if path == Path(stage['complete']['path']).resolve() and artifact['sha256'] == stage['complete']['sha256']:
            complete_bound = True
    if not complete_bound:
        raise ValueError('Supervisor did not verify this Stage1 completion artifact')
    rows = values['results']
    by_id = {row['run_id']: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError('Duplicate Stage1 fit')
    grid_runs = [Run(arm,'0.2B',lr) for lr in GRID for arm in ARMS[:2]]
    if any(run.run_id not in by_id for run in grid_runs):
        raise ValueError('Incomplete base LR grid')
    grid = [by_id[run.run_id] for run in grid_runs]
    choice = grid_choice(grid)
    lrs = GRID
    if choice['edge_lr'] is not None:
        lrs = (*GRID,choice['edge_lr'])
        edge_runs = [Run(arm,'0.2B',lrs[-1]) for arm in ARMS[:2]]
        if any(run.run_id not in by_id for run in edge_runs):
            raise ValueError('Missing required edge LR pair')
        grid_runs += edge_runs
        grid += [by_id[run.run_id] for run in edge_runs]
        choice = grid_choice(grid,lrs)
    choice['own_best'] = {arm:grid_choice([r for r in grid if r['arm']==arm],lrs,(arm,))['chosen_lr']
                          for arm in ARMS[:2]}
    # JSON object keys are strings on disk; compare through canonical JSON.
    import json
    if json.loads(json.dumps(choice)) != values['choice']:
        raise ValueError('Stored common LR differs from the registered grid rule')
    expected = {run.run_id for run in grid_runs + remaining(choice['chosen_lr'])}
    if set(by_id) != expected or complete.get('runs') != len(rows):
        raise ValueError('Stage1 registered fit coverage incomplete or expanded')
    decision = values['decision']
    if (decision.get('chosen_lr') != choice['chosen_lr']
            or decision.get('stage1_spec_sha256') != SPEC_SHA):
        raise ValueError('Decision does not identify the completed Stage1')
    selection = select_stage1_screen(rows, choice['chosen_lr'], reference)
    for run in remaining(choice['chosen_lr']):
        row = by_id[run.run_id]
        if row['status'] != 'complete' or row.get('diverged') is not False:
            raise ValueError('Unsuccessful non-grid Stage1 fit')
    return stage, reference, by_id, selection


def prepare_identities(stage_lock, assets_lock, output):
    """Create six locked identities without GPU work or modifying fit receipts."""
    from pathlib import Path
    from eval.a2_checkpoint import checkpoint_identity as original_identity
    from eval.a2_stage1_checkpoint import checkpoint_identity as healed_identity
    from eval.a2_run import lock, write
    from scripts.a2_remaining import read_locked

    stage, reference, by_id, selection = validate_terminal_bundle(stage_lock, assets_lock)
    result_root = Path(stage['results']['path']).resolve().parent
    adjacent = {}
    for name, item in selection.items():
        if item['kind'] == 'healed':
            path = result_root/item['run_id']/'result.json'
            adjacent[name] = lock(path)
            if read_locked(adjacent[name]) != by_id[item['run_id']]:
                raise ValueError('Adjacent export receipt differs from reviewed aggregate')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    identities = {}
    for name, item in selection.items():
        if item['kind'] == 'parent':
            identity = original_identity(assets_lock=assets_lock,dataset='dclm',parent=item['parent'])
        else:
            identity = healed_identity(assets_lock=assets_lock,
                checkpoint=str(Path(adjacent[name]['path']).parent/'final'),
                result_lock=adjacent[name],reference_lock=stage['reference'])
        identities[name] = write(out/(name+'-identity.json'),identity)
    result = dict(schema=2,identities=identities,selection=selection,
                  source_locks=dict(stage=stage_lock,assets=assets_lock))
    write(out/'identities.json',result)
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('stage','assets'):
        parser.add_argument('--'+name,required=True)
        parser.add_argument('--'+name+'-sha256',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    prepare_identities(dict(path=args.stage,sha256=args.stage_sha256),
                       dict(path=args.assets,sha256=args.assets_sha256),args.output)


if __name__ == '__main__':
    main()
