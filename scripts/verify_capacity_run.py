"""Validate a completed arm before allowing the next job or a burn handoff."""
import argparse
import json
import math
from pathlib import Path

from safetensors import safe_open
import torch

from capacity_allocation.data import write_json
from capacity_allocation.modeling import EXPECTED_COUNTS, SHARED_ARMS
from train import validate_resume_checkpoint


def verify(root, arm, step, world_size, cache_report):
    root = Path(root)
    result = json.loads((root/'result.json').read_text())
    spec = json.loads((root/'train_config.json').read_text())
    prepared = json.loads(Path(cache_report).read_text())
    if not result.get('success') or result['arm'] != arm or result['global_step'] != step:
        raise ValueError('Training result/arm/step mismatch')
    if result['schedule_steps'] <= step or result['status'] != 'stopped_at_step':
        raise ValueError('Expected a screening cutoff within the full epoch schedule')
    if result['parameters']['total'] != EXPECTED_COUNTS[arm]:
        raise ValueError('Unexpected production parameter count')
    if spec['model']['tiny_test'] or spec['model']['num_hidden_layers'] != 6:
        raise ValueError('Not the registered six-layer production run')
    if spec['execution'] != dict(world_size=world_size, effective_batch_size=512):
        raise ValueError('Unexpected distributed/effective batch configuration')
    for split in ('train', 'eval'):
        if spec[split+'_fingerprint'] != prepared[split]['fingerprint']:
            raise ValueError(f'{split} data differs from verified preprocessing')
    evaluation = result['eval_metrics']
    if evaluation['eval_scored_targets'] != prepared['eval']['scored_targets']:
        raise ValueError('Incomplete evaluation coverage')
    for metric in (result['train_metrics']['train_loss'], evaluation['eval_loss']):
        if not math.isfinite(metric):
            raise ValueError('Non-finite training/evaluation metric')
    checkpoint = root/f'checkpoint-{step}'
    state = validate_resume_checkpoint(checkpoint, world_size)
    if state['global_step'] != step:
        raise ValueError('Saved checkpoint step mismatch')
    for name in ('final', checkpoint.name):
        path = root/name
        config = json.loads((path/'config.json').read_text())
        if config.get('experiment_arm') != arm or config['num_hidden_layers'] != 6:
            raise ValueError('Saved model configuration mismatch')
        if config['tie_word_embeddings'] != (arm == 'B0' or arm in SHARED_ARMS):
            raise ValueError('Saved tying configuration mismatch')
        for filename in ('tokenizer.json', 'tokenizer_config.json'):
            if not (path/filename).is_file() or not (path/filename).stat().st_size:
                raise ValueError('Saved tokenizer is missing')
        # Check all saved tensor data, not merely presence of a weight filename.
        count, keys = 0, set()
        files = sorted(path.glob('*.safetensors'))
        if not files:
            raise ValueError('Missing safetensors weights')
        for file in files:
            with safe_open(file, framework='pt', device='cpu') as handle:
                for key in handle.keys():
                    if key in keys:
                        raise ValueError('Duplicate saved tensor')
                    keys.add(key)
                    tensor = handle.get_tensor(key)
                    count += tensor.numel()
                    if not torch.isfinite(tensor).all():
                        raise ValueError(f'Non-finite model tensor: {key}')
        if count != EXPECTED_COUNTS[arm]:
            raise ValueError('Saved model tensor count differs from actual parameter budget')
    return dict(success=True, arm=arm, step=step, parameters=result['parameters']['total'],
                eval_loss=evaluation['eval_loss'], scored_targets=evaluation['eval_scored_targets'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--arm', required=True, choices=list(EXPECTED_COUNTS))
    parser.add_argument('--step', type=int, default=10000)
    parser.add_argument('--world-size', type=int, default=8)
    parser.add_argument('--cache-report', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    result = verify(args.run_dir, args.arm, args.step, args.world_size, args.cache_report)
    write_json(args.output, result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
