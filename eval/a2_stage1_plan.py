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
