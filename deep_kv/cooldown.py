"""Explicit linear cooldown branches; retain native Trainer checkpoint restore."""
import copy
import json
from pathlib import Path
import shutil


def validate(start, steps, maximum, warmup=0):
    if start is None and steps is None:
        return
    if (type(start) is not int or type(steps) is not int or start < warmup
            or start < 0 or steps <= 0 or start + steps > maximum):
        raise ValueError('Cooldown requires a start after warmup and positive duration within max_steps')


def schedule(original, start, steps):
    # A plain function keeps LambdaLR's checkpoint format unchanged (None for
    # function state). Its definition is recorded in the immutable run recipe.
    anchor = original(start)
    def multiplier(step):
        return original(step) if step <= start else anchor * max(0., (start + steps - step) / steps)
    return multiplier


def transition(previous, requested, checkpoint, output):
    """Authorize only a documented fresh fork, never editing an existing schedule."""
    keys = ('cooldown_start', 'cooldown_steps')
    old = tuple(previous['pilot'].get(k) for k in keys)
    new = tuple(requested['pilot'].get(k) for k in keys)
    if old == new:
        return requested, {}
    if old != (None, None) or None in new:
        raise ValueError('Cannot change or remove an existing cooldown schedule')
    receipt = json.loads((Path(output) / 'cooldown_source.json').read_text())
    state = json.loads((Path(checkpoint) / 'trainer_state.json').read_text())
    if (receipt['status'] != 'verified' or receipt['source_step'] != state['global_step']
            or Path(receipt['checkpoint']).resolve() != Path(checkpoint).resolve()
            or tuple(receipt[k] for k in keys) != new or state['global_step'] > new[0]
            or receipt['source_config'] != previous
            or Path(receipt['source_checkpoint']).resolve() == Path(checkpoint).resolve()):
        raise ValueError('Cooldown fork provenance or boundary differs')
    normalized = copy.deepcopy(requested)
    for key in keys:
        normalized['pilot'].pop(key)
    return normalized, {k: dict(before=None, after=v) for k, v in zip(keys, new)}


def stage(checkpoint, output, start, steps):
    """Copy and hash every state file; never hard-link mutable branch files."""
    from scripts.stage_deep_kv_resume import digest
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    if output.exists() or checkpoint.is_symlink():
        raise ValueError('Cooldown output must be fresh')
    config = json.loads((checkpoint.parent / 'train_config.json').read_text())
    state = json.loads((checkpoint / 'trainer_state.json').read_text())
    validate(start, steps, config['training']['max_steps'], config['training']['warmup_steps'])
    if (state['global_step'] > start or state['max_steps'] != config['training']['max_steps']
            or config['training'].get('ignore_data_skip', False)
            or config['pilot'].get('cooldown_start') is not None):
        raise ValueError('Invalid source step/schedule or data skipping disabled')
    world = config['world_size']
    required = ['model.safetensors', 'optimizer.pt', 'scheduler.pt', 'trainer_state.json', 'training_args.bin']
    required += ['rng_state.pth'] if world == 1 else [f'rng_state_{i}.pth' for i in range(world)]
    if any(not (checkpoint / name).is_file() or not (checkpoint / name).stat().st_size for name in required):
        raise ValueError('Incomplete cooldown source checkpoint')
    if any(p.is_symlink() for p in checkpoint.rglob('*')):
        raise ValueError('Source checkpoint contains symbolic links')
    target = output / checkpoint.name
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(checkpoint, target)
    hashes = {}
    for source in sorted(p for p in checkpoint.rglob('*') if p.is_file()):
        relative = source.relative_to(checkpoint)
        hashes[str(relative)] = digest(source)
        if digest(target / relative) != hashes[str(relative)]:
            raise ValueError('Cooldown copy checksum mismatch')
    shutil.copy2(checkpoint.parent / 'train_config.json', output / 'train_config.json')
    receipt = dict(status='verified', source_checkpoint=str(checkpoint), checkpoint=str(target),
                   source_step=state['global_step'], source_config=config, files=hashes,
                   cooldown_start=start, cooldown_steps=steps)
    (output / 'cooldown_source.json').write_text(json.dumps(receipt, indent=2))
    return receipt


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--start', type=int, required=True)
    p.add_argument('--steps', type=int, required=True)
    args = p.parse_args()
    stage(args.checkpoint, args.output, args.start, args.steps)
