"""Fixed-budget distributed student optimization and adapter-only resume files."""
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .distill_model import StudentLoss
from .joint_config import JointSettings
from .joint_training import autocast, evaluate, require, restore_rng, rng_state, schedule
from .screen import write_json


def digest(module):
    value = hashlib.sha256()
    for name, parameter in module.named_parameters():
        value.update(name.encode())
        value.update(parameter.detach().cpu().contiguous().numpy())
    return value.hexdigest()


def optimizer(model):
    matrices, other = [], []
    for name, parameter in model.student.named_parameters():
        require(parameter.requires_grad and parameter.dtype == torch.float32, 'Student masters must be trainable fp32')
        (matrices if parameter.ndim >= 2 and name != 'gate.weight' else other).append(parameter)
    return torch.optim.AdamW([
        {'params': matrices, 'weight_decay': .01, 'peak_lr': 3e-4},
        {'params': other, 'weight_decay': 0., 'peak_lr': 3e-4},
    ], lr=3e-4, betas=(.9, .95), eps=1e-8, foreach=False)


def wrap(model):
    device = next(model.parameters()).device
    return DistributedDataParallel(StudentLoss(model),
        device_ids=[device.index] if device.type == 'cuda' else None,
        broadcast_buffers=False, find_unused_parameters=False)


def update(model, opt, data, step, settings, microbatch, *, parallel=None, mixed_precision=True):
    world, rank = (dist.get_world_size(), dist.get_rank()) if parallel is not None else (1, 0)
    count = settings.tokens_per_update // settings.context
    require(count % world == 0 and (count // world) % microbatch == 0, 'Invalid student batch allocation')
    start = (step - 1) * count
    global_context = data.batch(start, start + count, 'cpu')
    targets = int(global_context.targets().sum())
    require(int(global_context.valid.sum()) == settings.tokens_per_update and targets > 0, 'Incorrect student batch')
    model.train()
    opt.zero_grad(set_to_none=True)
    schedule(opt, step, settings)
    device = next(model.parameters()).device
    total_lm, total_corr = 0., 0.
    first, last = start + rank * (count // world), start + (rank + 1) * (count // world)
    began = time.monotonic()
    for offset in range(first, last, microbatch):
        context = data.batch(offset, offset + microbatch, device)
        sync = parallel.no_sync() if parallel is not None and offset + microbatch < last else nullcontext()
        with sync:
            with autocast(model, mixed_precision):
                sums, _, corr = parallel(context) if parallel is not None else model.training_losses(context)
                loss = world * (sums.sum() / targets + corr / (targets * model.model.config.hidden_size))
            require(bool(torch.isfinite(loss)), 'Nonfinite student objective')
            loss.backward()
        total_lm += float(sums.detach().sum())
        total_corr += float(corr.detach())
    norm = torch.nn.utils.clip_grad_norm_(model.student.parameters(), 1., error_if_nonfinite=True)
    opt.step()
    opt.zero_grad(set_to_none=True)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elapsed = time.monotonic() - began
    if parallel is not None:
        totals = torch.tensor([total_lm, total_corr], dtype=torch.float64, device=device)
        dist.all_reduce(totals)
        total_lm, total_corr = totals.tolist()
        seconds = torch.tensor(elapsed, dtype=torch.float64, device=device)
        dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
        elapsed = seconds.item()
    return {'update': step, 'input_tokens': step * settings.tokens_per_update,
            'target_tokens': targets, 'nll': total_lm / targets,
            'correction_loss': total_corr / (targets * model.model.config.hidden_size),
            'lambda_corr': 1. if model.arm == 'PCC' else 0., 'sigma_delta': model.scale,
            'grad_norm': float(norm), 'learning_rate': opt.param_groups[0]['lr'], 'seconds': elapsed}


def save(path, model, opt, identity, step, history, states):
    state = {'format': 'distill-v1', 'identity': identity, 'update': step,
             'student': model.student.state_dict(), 'optimizer': opt.state_dict(),
             'history': history, 'rank_rng': states}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.student-', suffix='.part', delete=False) as f:
            temporary = Path(f.name)
            torch.save(state, f)
            f.flush(); os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def restore(path, model, opt, identity, rank=0):
    state = torch.load(path, map_location='cpu', weights_only=True)
    require(state['format'] == 'distill-v1' and state['identity'] == identity, 'Student resume identity mismatch')
    step = state['update']
    require(type(step) is int and 0 <= step <= identity['settings']['updates'], 'Invalid resume update')
    require([r['update'] for r in state['history']['train']] == list(range(1, step + 1)), 'Incomplete resume history')
    require(all(r['input_tokens'] == r['update'] * identity['settings']['tokens_per_update']
                for r in state['history']['train']), 'Incorrect resume token history')
    require(len(state['rank_rng']) == identity['world_size'], 'Resume rank topology changed')
    require(all(t.dtype == torch.float32 and bool(torch.isfinite(t).all())
                for t in state['student'].values()), 'Invalid student checkpoint weights')
    for saved, expected in zip(state['optimizer']['param_groups'], opt.state_dict()['param_groups'], strict=True):
        require(all(saved[k] == expected[k] for k in ('params', 'peak_lr', 'betas', 'eps', 'weight_decay')),
                'Student optimizer recipe changed')
    for moments in state['optimizer']['state'].values():
        require(float(moments['step']) == step, 'Student optimizer update mismatch')
        require(all(t.dtype == torch.float32 and bool(torch.isfinite(t).all())
                    for t in (moments['exp_avg'], moments['exp_avg_sq'])), 'Invalid optimizer moments')
    model.student.load_state_dict(state['student'], strict=True)
    opt.load_state_dict(state['optimizer'])
    restore_rng(state['rank_rng'][rank])
    return step, state['history']


def train(model, data, dev, output, identity, *, microbatch=1, distributed=False,
          mixed_precision=True, stop_after=None, resume=None):
    world, rank = (dist.get_world_size(), dist.get_rank()) if distributed else (1, 0)
    settings = JointSettings(**identity['settings']).validate()
    require(identity['world_size'] == world and identity['arm'] == model.arm, 'Student identity mismatch')
    require((model.s, model.d) == (settings.s, settings.d), 'Student coordinates differ from protocol')
    require(int(data.valid.sum()) == settings.updates * settings.tokens_per_update, 'Wrong student pool size')
    require(int(dev.valid.sum()) == settings.dev_tokens, 'Wrong student evaluation budget')
    target = settings.updates if stop_after is None else stop_after
    require(type(target) is int and 0 < target <= settings.updates, 'Invalid student stopping update')
    output = Path(output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / 'identity.json', identity)
    if distributed:
        dist.barrier()
    try:
        frozen_hash = digest(model.teacher)
        require(not any(p.requires_grad or p.grad is not None for p in model.teacher.parameters()), 'Teacher is not frozen')
        initial_student_hash = digest(model.student)
        opt = optimizer(model)
        step, history = (0, {'train': [], 'validation': []})
        if resume:
            step, history = restore(resume, model, opt, identity, rank)
        require(step <= target, 'Resume exceeds stopping update')
        parallel = wrap(model) if distributed else None
        started = time.monotonic()
        device = next(model.parameters()).device
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)

        def monitor(i):
            sums, counts = evaluate(model, dev, microbatch, mixed_precision=mixed_precision,
                rows=settings.monitor_tokens // settings.context, distributed=distributed)
            history['validation'].append({'update': i, 'nll': float(sums.sum()/counts.sum()),
                                          'target_tokens': int(counts.sum())})

        def checkpoint(i):
            states = [rng_state(current_device_only=distributed)]
            if distributed:
                states = [None] * world
                dist.all_gather_object(states, rng_state(current_device_only=True))
            if rank == 0:
                save(output / 'latest.pt', model, opt, identity, i, history, states)
            if distributed:
                dist.barrier()

        if step == 0:
            monitor(0)
        with (output / 'train.jsonl' if rank == 0 else Path(os.devnull)).open('x' if rank == 0 else 'w') as log, \
             (output / 'validation.jsonl' if rank == 0 else Path(os.devnull)).open('x' if rank == 0 else 'w') as val:
            for record in history['train']:
                log.write(json.dumps(record, allow_nan=False) + '\n')
            for record in history['validation']:
                val.write(json.dumps(record, allow_nan=False) + '\n')
            log.flush(); val.flush()
            for i in range(step + 1, target + 1):
                record = update(model, opt, data, i, settings, microbatch,
                                parallel=parallel, mixed_precision=mixed_precision)
                history['train'].append(record)
                log.write(json.dumps(record, allow_nan=False) + '\n'); log.flush()
                if i % settings.eval_every == 0 or i == target:
                    monitor(i)
                    val.write(json.dumps(history['validation'][-1], allow_nan=False) + '\n'); val.flush()
                    checkpoint(i)
                if rank == 0:
                    print(f'student={model.arm} update={i}/{settings.updates} nll={record["nll"]:.6f}', flush=True)
        if not (output / 'latest.pt').exists():
            checkpoint(target)
        require(digest(model.teacher) == frozen_hash and not any(p.grad is not None for p in model.teacher.parameters()),
                'Frozen teacher/backbone changed')
        require(digest(model.student) != initial_student_hash, 'Student parameters did not update')
        replica_hash = digest(model.student)
        if distributed:
            from .distributed import identical_parameters
            replica_hash = identical_parameters(model.student)
        # Restore the saved adapter/optimizer into the same module and verify its
        # exact parameters before evaluating; the full teacher is referenced by hash.
        saved_rng = rng_state(current_device_only=distributed)
        restored_step, restored_history = restore(output / 'latest.pt', model, opt, identity, rank)
        require(restored_step == target and restored_history == history and digest(model.student) == replica_hash,
                'Student checkpoint roundtrip failed')
        restore_rng(saved_rng)
        if target != settings.updates:
            report = {'status': 'stopped_at_step', 'update': target, 'teacher_unchanged': True,
                      'student_changed': True, 'checkpoint_roundtrip': True, 'replicas_sha256': replica_hash}
            if rank == 0:
                write_json(output / 'stopped.json', report)
            return report
        if rank == 0:
            os.link(output / 'latest.pt', output / 'final.pt')
        eval_started = time.monotonic()
        sums, counts = evaluate(model, dev, microbatch, mixed_precision=mixed_precision, distributed=distributed)
        report = {'status': 'ok', 'arm': model.arm, 'updates': target,
                  'input_tokens': target * settings.tokens_per_update, 'nll': float(sums.sum()/counts.sum()),
                  'target_tokens': int(counts.sum()), 'world_size': world, 'teacher_unchanged': True,
                  'checkpoint_verified': True, 'replicas_sha256': replica_hash,
                  'elapsed_seconds': time.monotonic()-started,
                  'student_only_eval_seconds': time.monotonic()-eval_started,
                  'training_seconds': sum(r['seconds'] for r in history['train']),
                  'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None}
        if rank == 0:
            np.savez(output / 'eval.npz', loss_sums=sums, target_counts=counts, sequence_indices=np.arange(len(counts)))
            write_json(output / 'complete.json', report)
        if distributed:
            dist.barrier()
        return report
    except BaseException as error:
        if rank == 0:
            write_json(output / 'failure.json', {'status': 'failed', 'error': str(error)})
        raise
