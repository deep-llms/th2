"""Gated distillation from completed joint-v2 teachers; offline distributed CLI."""
import argparse
from dataclasses import asdict, replace
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from .joint_config import SEEDS, load_config, settings_for


def settings(config):
    if config.get('experiment') not in ('joint-v2-ddp', 'joint-local-v3'):
        raise ValueError('Distillation requires completed joint-v2-ddp or joint-local-v3 inputs and teachers')
    return replace(settings_for(config), version='distill-local-v2' if config.get('experiment') == 'joint-local-v3' else 'distill-v1').validate()


def file_hash(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read_teacher(args, config, fingerprints):
    import torch
    import torch.distributed as dist
    from transformers import AutoConfig, AutoModelForCausalLM
    from .joint_model import JointQwen
    from .joint_training import audit_final_checkpoint, require
    from .protocol import validate_config
    root = args.joint_root / 'runs'
    queue = json.loads((root / 'complete.json').read_text())
    local = config.get('experiment') == 'joint-local-v3'
    if local:
        require(queue['status'] == 'ok' and len(queue['jobs']) == 1
                and queue['jobs'][0]['name'] == f'seed-{args.seed_index}-Deep'
                and queue['jobs'][0]['status'] == 'ok', 'Local teacher queue did not finish')
    else:
        result = json.loads((root / 'report/complete.json').read_text())
        require(queue['status'] == result['status'] == 'ok' and len(queue['jobs']) == 7
                and all(j['status'] == 'ok' for j in queue['jobs'])
                and all(s['passed'] for s in result['seeds']), 'Joint teacher stage did not pass')
    directory = root / f'seed-{args.seed_index}-Deep'
    identity = json.loads((directory / 'identity.json').read_text())
    complete = json.loads((directory / 'complete.json').read_text())
    job = next(j for j in queue['jobs'] if j['name'] == directory.name)
    require(file_hash(directory / 'complete.json') == job['artifacts'][0]['sha256'], 'Parent receipt changed')
    require(identity['arm'] == complete['arm'] == 'Deep' and identity['purpose'] == 'scientific_training'
            and identity['config'] == config and identity['inputs'] == fingerprints
            and identity['settings'] == asdict(settings_for(config))
            and identity['distributed']['world_size'] == (4 if local else 8)
            and identity['distributed']['physical_gpus'] == list(range(4 if local else 8))
            and complete['status'] == 'ok' and complete['world_size'] == (4 if local else 8)
            and (identity['adapter_seed'], identity['data_order_seed']) == SEEDS[args.seed_index]
            and complete['updates'] == settings_for(config).updates and complete['checkpoint_verified'],
            'Joint teacher identity mismatch')
    path = directory / 'final.pt'
    hashes = [None]
    if dist.get_rank() == 0:
        if args.mode == 'audit':
            audit_final_checkpoint(path, identity)
        hashes[0] = file_hash(path)
    dist.broadcast_object_list(hashes, src=0)
    state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    require(state['format'] == 'joint-v1' and state['identity'] == identity
            and state['update'] == complete['updates'], 'Parent checkpoint mismatch')
    cfg = AutoConfig.from_pretrained(config['model_path'], local_files_only=True, trust_remote_code=False)
    validate_config(cfg)
    cfg._attn_implementation = 'eager'
    model = JointQwen(AutoModelForCausalLM.from_config(cfg), 'Deep',
                      seed=identity['adapter_seed'], s=identity['settings']['s'], d=identity['settings']['d'])
    model.load_state_dict(state['model'], strict=True)
    del state
    model.eval().requires_grad_(False)
    model.to(torch.device('cuda', int(os.environ['LOCAL_RANK'])))
    return model, {'path': str(path.resolve()), 'sha256': hashes[0], 'identity': identity}, directory


def calibration(model, data, budget=1000000):
    import math
    import torch
    import torch.distributed as dist
    from .model import Context
    from .training import prefix_batches
    from .joint_training import autocast, require
    rank, world = dist.get_rank(), dist.get_world_size()
    device = next(model.parameters()).device
    totals = torch.zeros(4, dtype=torch.float64, device=device)
    for i, context in enumerate(prefix_batches(data, budget, 1, 'cpu')):
        if i % world != rank:
            continue
        context = Context(context.input_ids.to(device), context.valid.to(device), context.position_ids.to(device),
                          context.segments.to(device) if context.segments is not None else None)
        with torch.no_grad(), autocast(model, True):
            delta = model.teacher_correction(context)
        values = delta[:, :-1][context.targets()].double()
        totals += torch.stack([values.square().sum(), totals.new_tensor(values.numel()),
                               totals.new_tensor(int(context.valid.sum())), totals.new_tensor(int(context.targets().sum()))])
    dist.all_reduce(totals)
    square, elements, inputs, targets = totals.tolist()
    sigma = math.sqrt(square/elements) if elements else float('nan')
    require(inputs == budget and math.isfinite(sigma) and sigma > 1e-8, 'Unusable teacher correction scale')
    return {'sigma_delta': sigma, 'squared_sum_fp64': square, 'elements': int(elements),
            'input_tokens': int(inputs), 'target_tokens': int(targets)}


def audit(args, config, teacher, parent, parent_dir, data, dev, fingerprints):
    import numpy as np
    import torch
    import torch.distributed as dist
    from .distill_model import CorrectionStudent, FeedbackOff
    from .distill_training import digest
    from .joint_training import evaluate, require
    from .statistics import paired_bootstrap
    from .screen import write_json
    rank = dist.get_rank()
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    before = digest(teacher)
    sums, counts = evaluate(teacher, dev, config['microbatch'], distributed=True)
    with np.load(parent_dir / 'eval.npz', allow_pickle=False) as old:
        require(np.array_equal(counts, old['target_counts'])
                and np.array_equal(old['sequence_indices'], np.arange(len(counts)))
                and np.allclose(sums, old['loss_sums'], rtol=1e-6, atol=1e-4),
                'Loaded teacher does not reproduce original final evaluation')
    require(abs(float(sums.sum()/counts.sum()) - json.loads((parent_dir/'complete.json').read_text())['nll']) < 1e-6,
            'Teacher NLL reproduction failed')
    off, off_counts = evaluate(FeedbackOff(teacher), dev, config['microbatch'], distributed=True)
    require(np.array_equal(counts, off_counts), 'Feedback-off evaluation targets changed')
    student = CorrectionStudent(teacher, 'PCC', 1., seed=SEEDS[args.seed_index][0])
    context = dev.batch(0, 1, next(teacher.parameters()).device)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        torch.testing.assert_close(student(context), FeedbackOff(teacher)(context), atol=0, rtol=0)
    scale = calibration(student, data)
    require(digest(teacher) == before, 'Audit modified teacher weights')
    if rank == 0:
        stats = paired_bootstrap({'Deep': sums, 'FeedbackOff': off}, counts, resamples=2000, seed=20260922)
        ci = np.quantile(stats['samples']['Deep']-stats['samples']['FeedbackOff'], [.025,.975]).tolist()
        gain = stats['nll']['FeedbackOff']-stats['nll']['Deep']
        passed = gain >= .0005 and ci[1] < 0
        np.savez(args.output/'eval.npz', deep_loss_sums=sums, off_loss_sums=off,
                 target_counts=counts, sequence_indices=np.arange(len(counts)))
        write_json(args.output/'complete.json', {'status':'ok', 'seed_index':args.seed_index,
            'ready_for_students':passed, 'nll':stats['nll'], 'deep_minus_off_ci95':ci,
            'feedback_gain':gain, 'calibration':scale, 'parent':parent, 'inputs':fingerprints,
            'teacher_reproduced':True, 'teacher_unchanged':True, 'student_noop_exact':True,
            'decision':'train_matched_students' if passed else 'stop_no_feedback_benefit'})
    dist.barrier()


def run_worker(args, config):
    import random
    import numpy as np
    import torch
    import torch.distributed as dist
    from .joint import load_inputs
    from .joint_training import code_identity, require
    from .distill_model import CorrectionStudent, FeedbackOff
    from .distill_training import train
    from .screen import write_json
    torch.set_num_threads(4)
    seed = SEEDS[args.seed_index][0]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    data, dev, fingerprints = load_inputs(config, args.data_dir, args.seed_index)
    teacher, parent, parent_dir = read_teacher(args, config, fingerprints)
    if args.mode == 'audit':
        return audit(args, config, teacher, parent, parent_dir, data, dev, fingerprints)
    ready = json.loads((args.audit_dir/'complete.json').read_text())
    require(ready['status'] == 'ok' and ready['ready_for_students'] and ready['parent'] == parent
            and ready['inputs'] == fingerprints and ready['seed_index'] == args.seed_index,
            'Missing or mismatched teacher feedback audit')
    model = CorrectionStudent(teacher, args.arm, ready['calibration']['sigma_delta'], seed=seed)
    context = dev.batch(0, 1, next(model.parameters()).device)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        torch.testing.assert_close(model(context), FeedbackOff(teacher)(context), atol=0, rtol=0)
    identity = {'purpose':'student_capacity' if args.mode == 'capacity' else 'correction_distillation',
                'settings':asdict(settings(config)), 'arm':args.arm, 'seed_index':args.seed_index,
                'adapter_seed':seed, 'data_order_seed':SEEDS[args.seed_index][1], 'inputs':fingerprints,
                'config':config, 'parent':parent, 'audit_sha256':file_hash(args.audit_dir/'complete.json'),
                'sigma_delta':model.scale, 'code':code_identity(), 'world_size':dist.get_world_size(),
                'physical_gpus':args.physical_gpus, 'mixed_precision':True, 'torch':str(torch.__version__)}
    result = train(model, data, dev, args.output, identity, microbatch=config['microbatch'], distributed=True,
                   stop_after=2 if args.mode == 'capacity' else None, resume=getattr(args,'resume',None))
    if args.mode == 'capacity':
        # A fresh module and DDP reducer restore per-rank RNG/optimizer and advance.
        fresh = CorrectionStudent(teacher, args.arm, model.scale, seed=seed)
        resumed = train(fresh, data, dev, args.output/'resume-smoke', identity,
            microbatch=config['microbatch'], distributed=True, stop_after=3, resume=args.output/'latest.pt')
        require(resumed['update'] == 3 and resumed['checkpoint_roundtrip'], 'Distributed student resume failed')
        if dist.get_rank() == 0:
            write_json(args.output/'capacity.json', {**result, 'status':'ok', 'arm':args.arm, 'world_size':dist.get_world_size(),
                       'resume_next_update_verified':True, 'student_noop_exact':True,
                       'scientific_run':False})
    if dist.get_rank() == 0:
        print(json.dumps(result,indent=2),flush=True)


def manifest(args):
    base = ['--config',str(args.config.resolve()), '--joint-root',str(args.joint_root.resolve()),
            '--data-dir',str(args.data_dir.resolve())]
    jobs=[]
    for seed in range(2):
        for arm in ('LM','PCC'):
            name=f'seed-{seed}-{arm}'
            jobs.append({'name':name,'gpus':list(range(8)), 'argv':['{python}','-u','-m','pcc.distill','train',
                *base,'--seed-index',str(seed),'--arm',arm,'--physical-gpus',*map(str,range(8)),
                '--audit-dir',str(args.audit_root.resolve()/f'audit-seed-{seed}'), '--output','{run_dir}/'+name],
                'required_outputs':[{'path':name+'/complete.json','json_equals':{'status':'ok','updates':6144,
                    'input_tokens':201326592,'world_size':8,'teacher_unchanged':True,'checkpoint_verified':True}}]})
    jobs.append({'name':'report','argv':['{python}','-u','-m','pcc.distill','report',*base,
        '--audit-root',str(args.audit_root.resolve()),'--runs-dir','{run_dir}','--output','{run_dir}/report'],
        'required_outputs':[{'path':'report/complete.json','json_equals':{'status':'ok'}}]})
    with args.output.open('x') as f:
        json.dump({'jobs':jobs},f,indent=2)


def report(args, config):
    from .distill_report import report_results
    return report_results(args,config)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest='mode',required=True)
    for mode in ('audit','capacity','train','manifest','report'):
        p=subs.add_parser(mode)
        for name in ('config','joint-root','data-dir','output'):
            p.add_argument('--'+name,type=Path,required=True)
        if mode in ('audit','capacity','train'):
            p.add_argument('--physical-gpus',type=int,nargs='+',required=True)
            p.add_argument('--seed-index',type=int,choices=(0,1),required=True)
        if mode in ('capacity','train'):
            p.add_argument('--arm',choices=('LM','PCC'),required=True)
            p.add_argument('--audit-dir',type=Path,required=True)
        if mode == 'train':
            p.add_argument('--resume',type=Path)
        if mode in ('manifest','report'):
            p.add_argument('--audit-root',type=Path,required=True)
        if mode == 'report':
            p.add_argument('--runs-dir',type=Path,required=True)
            p.add_argument('--seed-count',type=int,choices=(1,2),default=2)
    args=parser.parse_args()
    config=load_config(args.config);settings(config)
    from .__main__ import configure_offline
    configure_offline()
    if args.mode == 'manifest':
        if config.get('experiment') == 'joint-local-v3':
            raise ValueError('Use the gated local restart pipeline')
        return manifest(args)
    if args.mode == 'report':
        return report(args,config)
    world = 4 if config.get('experiment') == 'joint-local-v3' else 8
    if args.physical_gpus != list(range(world)):
        raise ValueError(f'This stage requires all physical GPUs 0 through {world-1}')
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(map(str,args.physical_gpus))
    if 'LOCAL_RANK' not in os.environ:
        from scripts.gpu_status import require_free
        require_free(args.physical_gpus)
        if args.output.exists():
            raise ValueError('Use a fresh output directory')
        subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1',
            f'--nproc-per-node={world}','--max-restarts=0','-m','pcc.distill',*sys.argv[1:]],check=True)
        return
    import torch
    import torch.distributed as dist
    rank=int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE'])!=world or int(os.environ['RANK'])!=rank or not 0<=rank<world:
        raise ValueError('Expected the declared single-node rank allocation')
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl',timeout=timedelta(minutes=10),device_id=torch.device('cuda',rank))
    try:
        if rank==0 and args.output.exists():
            raise ValueError('Use a fresh output directory')
        dist.barrier()
        run_worker(args,config)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__=='__main__':
    main()
