"""Copy complete checkpoints into a fresh queue, preserving earlier results."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deep_kv import ARMS


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def stage(sources, destination, step):
    if type(step) is not int or step <= 0:
        raise ValueError('Resume step must be positive')
    receipt = dict(status='ok', source_step=step, arms={})
    # Check every input before copying any checkpoint.
    for arm, source in sources.items():
        if arm not in ARMS:
            raise ValueError('Invalid arm')
        source = Path(source)
        checkpoint = source / f'checkpoint-{step}'
        config = json.loads((source / 'train_config.json').read_text())
        state = json.loads((checkpoint / 'trainer_state.json').read_text())
        result = json.loads((source / 'result.json').read_text())
        if (config['pilot']['arm'] != arm or result['arm'] != arm
                or state['global_step'] != step or result['global_step'] != step
                or state['max_steps'] != config['training']['max_steps']):
            raise ValueError('Checkpoint/config/result mismatch')
        world = config['world_size']
        rng = ['rng_state.pth'] if world == 1 else [f'rng_state_{rank}.pth' for rank in range(world)]
        for name in ['model.safetensors', 'optimizer.pt', 'scheduler.pt', 'training_args.bin', *rng]:
            path = checkpoint / name
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f'Incomplete checkpoint: {path}')
        if any(p.is_symlink() for p in checkpoint.rglob('*')):
            raise ValueError('Checkpoint contains symlinks')
        if (destination / arm).exists():
            raise ValueError('Continuation output already exists')
    for arm, source in sources.items():
        source = Path(source)
        checkpoint = source / f'checkpoint-{step}'
        target = destination / arm
        target.mkdir(parents=True, exist_ok=False)
        shutil.copytree(checkpoint, target / checkpoint.name)
        checksums = {}
        for old in sorted(p for p in checkpoint.rglob('*') if p.is_file()):
            relative = old.relative_to(checkpoint)
            copied = target / checkpoint.name / relative
            expected = digest(old)
            if old.stat().st_size != copied.stat().st_size or digest(copied) != expected:
                raise ValueError('Checkpoint copy checksum mismatch')
            checksums[str(relative)] = dict(bytes=old.stat().st_size, sha256=expected)
        for old_name, new_name in [('train_config.json', 'train_config.json'),
                                   ('result.json', 'previous-result.json')]:
            shutil.copy2(source / old_name, target / new_name)
        receipt['arms'][arm] = dict(source=str(source), checkpoint=str(target / checkpoint.name), files=checksums)
        print('CHECKPOINT_COPY_VERIFIED', arm, step, flush=True)
    with (destination / 'resume_inputs.json').open('x') as handle:
        json.dump(receipt, handle, indent=2)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    parser.add_argument('--step', required=True, type=int)
    args = parser.parse_args()
    stage(json.loads(args.sources.read_text()), args.destination, args.step)
