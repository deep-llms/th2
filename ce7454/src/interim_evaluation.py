"""Snapshot-only dev/test inference, with no training state or optimizer access."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import time

import torch

from models import build_model, count_trainable
from predict_external import predict_external
from train import set_norm


def stable_checkpoint(path, attempts=6):
    """The existing trainer writes last.pt in place: reject concurrent writes."""
    for attempt in range(attempts):
        try:
            before = path.stat()
            raw = path.read_bytes()
            after = path.stat()
            signature = lambda s: (s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
            if signature(before) != signature(after) or len(raw) != after.st_size:
                raise ValueError('Checkpoint changed during read')
            ck = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
            return ck, raw
        except (OSError, ValueError, RuntimeError, EOFError) as error:
            if attempt + 1 == attempts:
                raise RuntimeError('No stable complete checkpoint available') from error
            time.sleep(.5)


def validate(ck, expected_model):
    expected = {'model': expected_model, 'train_split': 'train', 'val_frac': .1,
                'split_seed': 42, 'seed': 0, 'data_seed': 42, 'norm': 'imagenet'}
    if any(ck['args'].get(k) != v for k, v in expected.items()):
        raise ValueError('Checkpoint recipe/identity mismatch')
    if not 1 <= ck['epoch'] <= ck['args']['epochs']:
        raise ValueError('Invalid checkpoint epoch')
    if not all(torch.isfinite(t).all().item() for t in ck['model'].values()):
        raise ValueError('Nonfinite checkpoint state')


def state_digest(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=8)
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    ck, raw = stable_checkpoint(args.checkpoint)
    validate(ck, args.model)
    sha = hashlib.sha256(raw).hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    # Keep a real immutable snapshot for reproducibility; never overwrite trainer files.
    (args.output/'checkpoint.pt').write_bytes(raw)
    del raw
    model = build_model(args.model)
    model.load_state_dict(ck['model'], strict=True)
    if count_trainable(model) != ck['n_params_training'] or count_trainable(model) > 12_300_000:
        raise ValueError('Parameter count mismatch')
    torch.cuda.set_per_process_memory_fraction(0.10)
    model = model.cuda().eval().to(memory_format=torch.channels_last)
    set_norm(ck['args']['norm'])
    before = state_digest(model)
    start = time.monotonic()
    receipt = predict_external(model, args.cache, args.output/'external', sha, args.batch_size)
    if before != state_digest(model) or any(t.grad is not None for t in model.parameters()):
        raise ValueError('Inference mutated model state or created gradients')
    result = {'status': 'passed', 'model': args.model, 'epoch': ck['epoch'],
              'checkpoint_sha256': sha, 'source': str(args.checkpoint),
              'training_crop': ck['args']['crop'], 'batch_size': args.batch_size,
              'inference_seconds': time.monotonic()-start,
              'state_unchanged': True, 'no_gradients': True,
              'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
              'receipt': receipt, 'predictions': {}}
    for split in ('dev', 'test_official'):
        data = json.loads((args.output/'external'/f'{split}_predictions.json').read_text())
        result['predictions'][split] = data['predictions']
    (args.output/'result.json').write_text(json.dumps(result, allow_nan=False)+'\n')
    print('INTERIM_RESULT '+json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
