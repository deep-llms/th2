"""Final bounded study: fixed best checkpoints, two staged IBN fits, two soups.

No external labels or adaptive selection. Runs under train_then_burn after the
previous queue completes its burn handoff. All new fits start at base epoch450.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from deadline_study import training_command, run_child, save, validate_checkpoint
from followup_study import ARMS, EXTERNAL, parent_path, verify_parents

SRC = Path(__file__).resolve().parent
DIRECT = Path('/mnt/local/_outputs/deep-llms_th2/q359-ce7454-followup-20261017-a02/supervised/run/study')
MODELS = ('grip_nodown_ibna', 'grip_nodown_ibnb')
FINISH_BY = '2026-10-17T06:30:00+00:00'  # 14:30SGT, 85min before submission deadline.


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def stage_command(model, cache, output, parent, crop, epochs):
    if model not in MODELS or crop not in (256, 512):
        raise ValueError('Unexpected staged recipe')
    cmd = training_command(model, cache, output, epochs, final=crop == 512,
                           parent=parent if crop == 512 else None)
    if crop == 256:
        cmd += ['--init-from', str(parent)]
    for flag, value in [('--lr', '0.0001'), ('--seed', '1'), ('--data-seed', '43')]:
        cmd[cmd.index(flag)+1] = value
    return cmd


def verify_inputs():
    import torch
    torch.set_num_threads(1)
    parents = verify_parents()
    complete = json.loads((DIRECT/'complete.json').read_text())
    if complete['status'] != 'passed' or len(complete['results']) != 5:
        raise ValueError('Previous five-arm study must complete first')
    recorded = {r['arm']: r['checkpoint'] for r in complete['results']}
    inputs = {}
    for arm in ARMS:
        folder = DIRECT/arm['name']
        if digest(folder/'last.pt') != recorded[arm['name']]['sha256']:
            raise ValueError('Direct final checkpoint changed')
        best = folder/'best.pt'
        ck = torch.load(best, map_location='cpu', weights_only=False)
        expected = {'model': arm['model'], 'seed': 1, 'data_seed': 43,
                    'split_seed': 42, 'val_frac': .1, 'train_split': 'train',
                    'init_from': str(parent_path(arm)), 'epochs': 40,
                    'crop': 512, 'norm': 'imagenet', 'kd_teacher': None}
        if any(ck['args'].get(k) != v for k, v in expected.items()):
            raise ValueError('Wrong best checkpoint identity')
        with (folder/'log.csv').open() as f:
            rows = list(csv.DictReader(f))
        if len(rows) != 40 or not 1 <= ck['epoch'] <= 40:
            raise ValueError('Incomplete direct run')
        if abs(ck['val_acc'] - max(float(r['val_acc']) for r in rows)) > 1e-6:
            raise ValueError('Best checkpoint not selected by internal validation')
        inputs[arm['name']] = {'best': str(best), 'best_sha256': digest(best),
                               'best_epoch': ck['epoch'], 'last_sha256': digest(folder/'last.pt')}
    return {'parents': parents, 'direct': inputs}


def validate_stage(model, dest, parent, crop, epochs):
    import torch
    torch.set_num_threads(1)
    r = validate_checkpoint(dest/'last.pt', model, epochs, parent, seed=1, data_seed=43)
    ck = torch.load(dest/'last.pt', map_location='cpu', weights_only=False)
    expected = {'crop': crop, 'rrc_size': crop, 'batch_size': 32 if crop == 256 else 16,
                'lr': 1e-4, 'optim': 'adam', 'sched': 'cosine', 'warmup_epochs': 0}
    if any(ck['args'].get(k) != v for k, v in expected.items()):
        raise ValueError('Wrong stage recipe')
    with (dest/'log.csv').open() as f:
        rows = list(csv.DictReader(f))
    if [int(row['epoch']) for row in rows] != list(range(1, epochs+1)):
        raise ValueError('Incomplete epoch history')
    if any(not math.isfinite(float(v)) for row in rows for v in row.values()):
        raise ValueError('Nonfinite training history')
    r['last_epoch_seconds'] = float(rows[-1]['sec'])
    return r


def average_states(left, right):
    """Equal floating weights/BN stats; max BN counters; reject other mismatches."""
    import torch
    if left.keys() != right.keys():
        raise ValueError('State keys differ')
    out = {}
    for key, a in left.items():
        b = right[key]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError('State layout differs: '+key)
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError('Nonfinite soup ingredient')
        if a.is_floating_point():
            out[key] = ((a.double()+b.double())*.5).to(a.dtype)
        elif key.endswith('num_batches_tracked'):
            out[key] = torch.maximum(a, b)
        elif torch.equal(a, b):
            out[key] = a.clone()
        else:
            raise ValueError('Nonfloating buffers differ: '+key)
    return out


def make_soup(model, direct, staged, dest):
    import torch
    from models import build_model, count_trainable
    torch.set_num_threads(1)
    a, b = [torch.load(p, map_location='cpu', weights_only=False) for p in (direct, staged)]
    for ck in (a, b):
        if ck['args']['model'] != model or ck['epoch'] != 40:
            raise ValueError('Soup needs same architecture, fixed final endpoints')
        if any(ck['args'].get(k) != v for k, v in
               {'norm': 'imagenet', 'split_seed': 42, 'val_frac': .1, 'train_split': 'train',
                'crop': 512, 'seed': 1, 'data_seed': 43}.items()):
            raise ValueError('Incompatible soup recipes')
    state = average_states(a['model'], b['model'])
    net = build_model(model).eval()
    net.load_state_dict(state, strict=True)
    n = count_trainable(net)
    if n > 12_300_000:
        raise ValueError('Parameter cap exceeded')
    info = {'weights': [.5, .5], 'sources': [str(direct), str(staged)],
            'sha256': [digest(direct), digest(staged)], 'bn_recalibrated': False,
            'note': 'Inference-only checkpoint; no inherited validation score. Single network, no TTA.'}
    ck = {'model': state, 'args': b['args'], 'epoch': 40, 'n_params': n,
          'n_params_training': n, 'weight_soup': info}
    if dest.exists():
        raise FileExistsError(dest)
    torch.save(ck, dest)
    restored = torch.load(dest, map_location='cpu', weights_only=False)
    net.load_state_dict(restored['model'], strict=True)
    for key in state:
        torch.testing.assert_close(restored['model'][key], state[key], rtol=0, atol=0)
    return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--cache', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--finish-by', default=FINISH_BY)
    args = p.parse_args()
    inputs = verify_inputs()
    if args.preflight_only:
        print('STAGED_INPUTS_VERIFIED', json.dumps(inputs), flush=True)
        return
    if not args.cache or not args.output or os.environ.get('CUDA_VISIBLE_DEVICES') != '0,1,2,3,4,5,6,7':
        raise ValueError('Expected fresh output, cache and eight-GPU supervisor')
    deadline = datetime.fromisoformat(args.finish_by).timestamp()
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output/'manifest.json', {'inputs': inputs, 'stages': [[256, 30], [512, 40]],
         'seed': 1, 'data_seed': 43, 'models': MODELS, 'finish_by': args.finish_by,
         'soup': 'Equal direct-last and staged-last, same architecture; no fitting/tuning on dev/test',
         'selection': 'Internal validation/dev only; external labels stay local'})

    def evaluate(checkpoint, name, gpu):
        dest = args.output/(name+'_evaluation')
        run_child([sys.executable, '-u', str(SRC/'evaluate_deadline.py'), '--checkpoint', str(checkpoint),
                   '--cache', str(args.cache), '--external-cache', str(EXTERNAL), '--output', str(dest)],
                  gpu, args.output/(name+'_evaluation.log'), deadline)
        r = json.loads((dest/'results.json').read_text())
        if r['status'] != 'passed' or r['checkpoint_sha256'] != digest(checkpoint):
            raise ValueError('Evaluation identity mismatch')
        ext = r['external_predictions']
        if ext['status'] != 'passed' or any(ext['splits'][s]['n'] != n for s,n in [('dev',900),('test_official',1500)]):
            raise ValueError('Incomplete external evaluation')
        return {'name': name, 'evaluation': r}

    def best_eval(item):
        gpu, arm = item
        return evaluate(Path(inputs['direct'][arm['name']]['best']), arm['name']+'_best', gpu)

    def staged(item):
        gpu, model = item
        base = Path(inputs['parents']['grip_nodown']['path'])
        smoke256 = args.output/(model+'_smoke256')
        run_child(stage_command(model,args.cache,smoke256,base,256,1),gpu,args.output/(model+'_smoke256.log'),deadline)
        s256 = validate_stage(model,smoke256,base,256,1)
        smoke512 = args.output/(model+'_smoke512')
        run_child(stage_command(model,args.cache,smoke512,smoke256/'last.pt',512,1),gpu,args.output/(model+'_smoke512.log'),deadline)
        s512 = validate_stage(model,smoke512,smoke256/'last.pt',512,1)
        estimate = 1.2*(s256['last_epoch_seconds']*30+s512['last_epoch_seconds']*40)+900
        if estimate >= deadline-time.time():
            raise RuntimeError('Measured staged run cannot fit reserved deadline')
        save(args.output/(model+'_smoke_passed.json'),{'status':'passed','stages':[s256,s512],'estimated_seconds_with_margin':estimate})
        dest256 = args.output/(model+'_stage256')
        run_child(stage_command(model,args.cache,dest256,base,256,30),gpu,args.output/(model+'_stage256.log'),deadline)
        c256 = validate_stage(model,dest256,base,256,30)
        dest512 = args.output/(model+'_stage512')
        run_child(stage_command(model,args.cache,dest512,dest256/'last.pt',512,40),gpu,args.output/(model+'_stage512.log'),deadline)
        c512 = validate_stage(model,dest512,dest256/'last.pt',512,40)
        single = evaluate(dest512/'last.pt',model+'_staged',gpu)
        soup = args.output/(model+'_soup.pt')
        soup_info = make_soup(model,DIRECT/(model+'_s1')/'last.pt',dest512/'last.pt',soup)
        averaged = evaluate(soup,model+'_soup',gpu)
        result = {'model':model,'stages':[c256,c512],'single':single,'soup':averaged,'soup_recipe':soup_info}
        save(args.output/(model+'_complete.json'),result)
        return result

    # Disjoint GPUs: staged chains0/1, five best-checkpoint evaluations2..6.
    with ThreadPoolExecutor(max_workers=7) as pool:
        fits = [pool.submit(staged,item) for item in enumerate(MODELS)]
        bests = [pool.submit(best_eval,item) for item in enumerate(ARMS,start=2)]
        best_results = [f.result() for f in bests]
        save(args.output/'best_evaluations_complete.json',{'status':'passed','results':best_results})
        fits = [f.result() for f in fits]
    if verify_inputs() != inputs:
        raise ValueError('Source checkpoints changed')
    save(args.output/'complete.json',{'status':'passed','best':best_results,'staged':fits,
         'finished_utc':datetime.now(timezone.utc).isoformat()})


if __name__ == '__main__':
    main()
