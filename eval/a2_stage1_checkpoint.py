"""Bind a Stage1 export to its original fit and reviewed provenance locks."""
import json
from pathlib import Path

from efficient_llm.a2_stage1.protocol import Run, SPEC_SHA
from efficient_llm.diagnostic_data import sha256
from scripts.a2_remaining import read_locked
from scripts.a2_stage1 import validate_fit


def checkpoint_identity(*, assets_lock, checkpoint, result_lock, reference_lock):
    from eval.a2_checkpoint import checkpoint_identity as parent_identity

    row = read_locked(result_lock)
    reference = read_locked(reference_lock)
    dataset = row.get('dataset')
    run_type = Run
    if dataset == 'fwe':
        from efficient_llm.a2_stage1b.protocol import Run as FWERun, REPLICATION_SPEC_SHA
        if (reference.get('stage1b_spec_sha256') != REPLICATION_SPEC_SHA
                or reference.get('dataset') != 'fwe'):
            raise ValueError('Wrong Stage1b replication provenance')
        run_type = FWERun
    elif dataset != 'dclm':
        raise ValueError('Unregistered Stage1 corpus')
    run = run_type(row['arm'], row['budget'], row['lr'])
    validate_fit(row, run, reference)
    if (reference.get('spec_sha256') != SPEC_SHA
            or reference.get('assets_sha256') != assets_lock['sha256']
            or row['status'] != 'complete' or row.get('diverged') is not False):
        raise ValueError('Unsuccessful or mismatched Stage1 receipt')
    original = parent_identity(assets_lock=assets_lock, dataset=dataset, parent=run.parent)
    root = Path(checkpoint).resolve()
    if root.name != 'final' or root.parent.name != run.run_id:
        raise ValueError('Checkpoint is not the declared Stage1 fit')
    if Path(result_lock['path']).resolve() != root.parent/'result.json':
        raise ValueError('Result must be adjacent to the Stage1 export')
    kept = list(range(16)) + [22, 23]
    config = json.loads((root/'config.json').read_text())
    if config.get('a2_kept_layers') != kept or config.get('num_hidden_layers') != 18:
        raise ValueError('Stage1 S1 exported layer identity changed')
    files = {name: sha256(root/name) for name in ('config.json', 'model.safetensors')}
    for name, value in original['code_files'].items():
        if sha256(root/name) != value:
            raise ValueError('Exported native code changed')
        files[name] = value
    return dict(schema=2, kind='healed', dataset=dataset, parent=run.parent,
                checkpoint=str(root), files=files, assets=assets_lock,
                code_dir=original['code_dir'], code_files=original['code_files'],
                kept_layers=kept, training_receipts=dict(result=result_lock,
                    reference=reference_lock, run_id=run.run_id, teacher=run.teacher))


def verify_identity(identity):
    receipts = identity['training_receipts']
    actual = checkpoint_identity(assets_lock=identity['assets'],
        checkpoint=identity['checkpoint'], result_lock=receipts['result'],
        reference_lock=receipts['reference'])
    if actual != identity:
        raise ValueError('Stage1 checkpoint identity changed')
    return actual
