"""Offline calibration, matched HF-Trainer recovery and evaluation for KV/MLP."""
import os
os.environ.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  HF_HUB_DISABLE_TELEMETRY='1', WANDB_MODE='offline', TOKENIZERS_PARALLELISM='false')
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, TrainerCallback, default_data_collator, set_seed

from efficient_llm.diagnostic import load_probe, write, paired_summary
from efficient_llm.diagnostic_data import sha256
from efficient_llm.kv_mlp import load_compressed, convert, select_mlp, allocation
from packed_evaluation import preserve_rng_and_mode


def precision(device):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda')


@torch.no_grad()
def score(model, tokens, *, distributed=True, batch=2):
    device = next(model.parameters()).device
    world = dist.get_world_size() if distributed and dist.is_initialized() else 1
    rank = dist.get_rank() if world > 1 else 0
    sums = torch.zeros(len(tokens), dtype=torch.float64, device=device)
    counts = torch.zeros_like(sums)
    with preserve_rng_and_mode(model):
        model.eval()
        indices = list(range(rank, len(tokens), world))
        for offset in range(0, len(indices), batch):
            ids = indices[offset:offset+batch]
            row = tokens[ids].to(device)
            with precision(device):
                hidden = model.model(row[:, :-1], use_cache=False).last_hidden_state
                for start in range(0, hidden.shape[1], 128):
                    logits = model.lm_head(hidden[:, start:start+128]).float()
                    nll = F.cross_entropy(logits.flatten(0, 1),
                        row[:, start+1:start+1+logits.shape[1]].reshape(-1), reduction='none')
                    sums[ids] += nll.reshape(len(ids), -1).double().sum(-1)
            counts[ids] = 1
    if world > 1:
        dist.all_reduce(sums); dist.all_reduce(counts)
    if not bool((counts == 1).all()) or not torch.isfinite(sums).all():
        raise ValueError('Invalid score coverage or nonfinite loss')
    losses = (sums / (tokens.shape[1]-1)).cpu().tolist()
    mean = float(np.mean(losses))
    return dict(loss=mean, ppl=math.exp(mean), documents=len(tokens),
        scored_tokens=len(tokens)*(tokens.shape[1]-1), per_document_nll=losses)


@torch.no_grad()
def rank_channels(model, tokens):
    device = next(model.parameters()).device
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    layers = model.model.layers
    width = model.config.intermediate_size
    sums = torch.zeros(len(layers), width, dtype=torch.float64, device=device)
    counts = torch.zeros(len(layers), dtype=torch.float64, device=device)
    handles = []
    for i, layer in enumerate(layers):
        if layer.mlp.down_proj.in_features != width:
            raise ValueError('Channel ranking needs unpruned MLPs')
        def hook(module, args, index=i):
            x = args[0][:, ::8].detach().float().flatten(0, 1)
            sums[index].add_(x.square().double().sum(0))
            counts[index].add_(len(x))
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(hook))
    try:
        with preserve_rng_and_mode(model):
            model.eval()
            for i in range(rank, len(tokens), world):
                with precision(device):
                    model.model(tokens[i:i+1, :-1].to(device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if world > 1:
        dist.all_reduce(sums); dist.all_reduce(counts)
    if not bool((counts > 0).all()):
        raise ValueError('No calibration tokens')
    scores = []
    for i, layer in enumerate(layers):
        scores.append(sums[i] / counts[i] * layer.mlp.down_proj.weight.float().square().sum(0).double())
    scores = torch.stack(scores)
    if not torch.isfinite(scores).all():
        raise ValueError('Nonfinite channel importance')
    return scores.argsort(dim=-1, descending=True, stable=True).cpu().tolist()


def channel_lists(rankings, removed, unit):
    return [sorted(order[:len(order)-k*unit]) for order, k in zip(rankings, removed)]


@torch.no_grad()
def sensitivity(model, tokens, rankings, *, levels=(0, 1, 2, 3, 4)):
    device = next(model.parameters()).device
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    width = model.config.intermediate_size
    if width % 8:
        raise ValueError('Eight equal channel groups required')
    base = score(model, tokens)['loss']
    result = torch.zeros(len(rankings), len(levels), dtype=torch.float64, device=device)
    covered = torch.zeros(len(rankings), dtype=torch.int64, device=device)
    for i in range(rank, len(rankings), world):
        old = model.model.layers[i].mlp
        try:
            for j, k in enumerate(levels):
                if not k:
                    result[i, j] = base
                    continue
                replacement = deepcopy(old)
                select_mlp(replacement, sorted(rankings[i][:width-k*(width//8)]))
                model.model.layers[i].mlp = replacement
                result[i, j] = score(model, tokens, distributed=False)['loss']
                model.model.layers[i].mlp = old
                del replacement
            covered[i] = 1
        finally:
            model.model.layers[i].mlp = old
    if world > 1:
        dist.all_reduce(result); dist.all_reduce(covered)
    if not bool((covered == 1).all()):
        raise ValueError('Missing layer sensitivity')
    return dict(base_loss=base, removal_eighths=list(levels), losses=result.cpu().tolist())


def calibrate(args, device, rank, world, tokens, manifest):
    a, b = tokens['calibration'][:args.calibration], tokens['calibration'][args.calibration:2*args.calibration]
    dense = load_compressed(args.model, device).eval().requires_grad_(False)
    n, width = dense.config.num_hidden_layers, dense.config.intermediate_size
    ranks_dense = rank_channels(dense, a)
    if rank == 0: print('KV_MLP_CALIBRATION dense sensitivity', flush=True)
    curves_dense = sensitivity(dense, b, ranks_dense)
    convert(dense, share=True)
    ranks_shared = rank_channels(dense, a)
    if rank == 0: print('KV_MLP_CALIBRATION shared, dense channels', flush=True)
    curves_same = sensitivity(dense, b, ranks_dense)
    if rank == 0: print('KV_MLP_CALIBRATION shared, recalibrated channels', flush=True)
    curves_shared = sensitivity(dense, b, ranks_shared)
    removed_dense = allocation(curves_dense['losses'], prune_units=n*2)
    removed_shared = allocation(curves_shared['losses'], prune_units=n*2)
    dense_channels = channel_lists(ranks_dense, removed_dense, width//8)
    plans = {
        'A': dict(share=False, channels=None),
        'B': dict(share=True, channels=None),
        'C': dict(share=False, channels=dense_channels),
        'D': dict(share=True, channels=dense_channels),
        'E': dict(share=True, channels=channel_lists(ranks_shared, removed_dense, width//8)),
        'F': dict(share=True, channels=channel_lists(ranks_shared, removed_shared, width//8)),
    }
    result = dict(status='passed', world_size=world, probe_sha256=sha256(args.probe/'manifest.json'),
        model_files=manifest['model_files'], ranking_documents=len(a), sensitivity_documents=len(b),
        ranks_dense=ranks_dense, ranks_shared=ranks_shared, dense=curves_dense,
        shared_same_channels=curves_same, shared_recalibrated_channels=curves_shared,
        dense_removal_eighths=removed_dense, shared_removal_eighths=removed_shared,
        arms=plans, retained_total_width=n*width*3//4)
    if rank == 0:
        write(args.output, result)
        print('KV_MLP_CALIBRATED', json.dumps(dict(dense=removed_dense, shared=removed_shared)), flush=True)


class Prefixes(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        return dict(input_ids=self.rows[i, :-1], targets=self.rows[i, 1:], sample_id=i)


class RecoveryTrainer(Trainer):
    def __init__(self, *args, teacher, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher.eval().requires_grad_(False)
        self.seen = []
        self.model_accepts_loss_kwargs = False
        if self.args.gradient_accumulation_steps != 1 or self.args.gradient_checkpointing:
            raise ValueError('Pilot requires accumulation=1 and no decoder checkpointing')

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        self.seen.extend(inputs.pop('sample_id').detach().cpu().tolist())
        if inputs['input_ids'].shape != inputs['targets'].shape:
            raise ValueError('Pilot uses equally sized next-token targets')
        with torch.no_grad(), precision(self.args.device):
            teacher_hidden = self.teacher.model(inputs['input_ids'], use_cache=False).last_hidden_state
        outputs = model(**inputs, teacher_hidden=teacher_hidden,
                        teacher_weight=self.teacher.lm_head.weight)
        return (outputs.loss, outputs) if return_outputs else outputs.loss


class Validation(TrainerCallback):
    def __init__(self, tokens, path, steps, rank):
        self.tokens, self.path, self.steps, self.rank = tokens, path, steps, rank
        self.history = {}

    def measure(self, model, step):
        result = score(model, self.tokens)
        self.history[str(step)] = result
        if self.rank == 0:
            write(self.path, self.history)
            print('KV_MLP_VALIDATION', step, json.dumps({k:v for k,v in result.items() if k != 'per_document_nll'}), flush=True)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step in self.steps:
            self.measure(model, state.global_step)
            control.should_save = True
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.rank == 0:
            print('KV_MLP_PROGRESS', datetime.now(timezone.utc).isoformat(),
                  state.global_step, json.dumps(logs or {}), flush=True)


def train(args, device, rank, world, tokens, manifest):
    parent = None
    parent_files = None
    if args.initial_model is not None:
        if args.parent_result is None or args.plan is not None or args.calibration != 0:
            raise ValueError('New recovery phase requires parent result, no calibration/plan')
        if args.initial_model.resolve() != (args.parent_result.parent/'final').resolve():
            raise ValueError('Initial checkpoint is not the parent final model')
        parent = json.loads(args.parent_result.read_text())
        if (parent['status'] != 'passed' or parent['smoke'] or parent['arm'] != args.arm
                or parent['probe_sha256'] != manifest.get('recovery_from_manifest_sha256')
                or not manifest.get('evaluation_reused')):
            raise ValueError('Recovery parent and reused evaluation do not match')
        parent_files = {p.name:sha256(p) for p in sorted(args.initial_model.iterdir())
                        if p.is_file() and p.suffix in {'.json','.safetensors'}} if rank == 0 else None
        plan = None
    else:
        if args.parent_result is not None or args.plan is None or args.calibration < 1:
            raise ValueError('Fresh conversion requires calibration and a plan')
        plan = json.loads(args.plan.read_text())
        if plan['probe_sha256'] != sha256(args.probe/'manifest.json') or plan['status'] != 'passed':
            raise ValueError('Calibration identity differs')
    rows = tokens['calibration'][2*args.calibration:]
    needed = args.steps * args.batch * world
    if args.smoke:
        if args.steps > 8: raise ValueError('Smoke is bounded to eight steps')
        rows = rows[:needed]
    if len(rows) != needed:
        raise ValueError(f'Require exactly {needed} recovery documents, found {len(rows)}')
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
    if world > 1: dist.barrier()
    teacher = load_compressed(args.model, device).eval().requires_grad_(False)
    model = load_compressed(args.initial_model or args.model, device)
    if parent is None:
        convert(model, **plan['arms'][args.arm])
    elif (model.config.kv_sources != parent['kv_sources'] or model.config.mlp_widths != parent['mlp_widths']
          or sum(p.numel() for p in model.parameters()) != parent['parameters']):
        raise ValueError('Parent architecture differs')
    model.requires_grad_(True)
    checkpoint_steps = sorted({max(1, args.steps//4), args.steps})
    if args.eval_every:
        checkpoint_steps = sorted(set(range(args.eval_every, args.steps+1, args.eval_every)) | {args.steps})
    callback = Validation(tokens['evaluation'], args.output/'validation.json', checkpoint_steps, rank)
    callback.measure(model, 0)
    initial_difference = None
    if parent is not None:
        old = parent['validation'][str(parent['steps'])]['per_document_nll']
        new = callback.history['0']['per_document_nll']
        torch.testing.assert_close(torch.tensor(new, dtype=torch.float64),
            torch.tensor(old, dtype=torch.float64), rtol=0, atol=1e-5)
        initial_difference = max(abs(a-b) for a,b in zip(new,old,strict=True))
    config = TrainingArguments(output_dir=str(args.output/'trainer'), max_steps=args.steps,
        per_device_train_batch_size=args.batch, gradient_accumulation_steps=1,
        learning_rate=5e-5, lr_scheduler_type='cosine', warmup_steps=max(1,round(.05*args.steps)),
        optim='adamw_torch', weight_decay=.01, max_grad_norm=1.,
        bf16=device.type=='cuda', use_cpu=device.type=='cpu', seed=42, data_seed=42,
        report_to='none', dataloader_drop_last=True, dataloader_num_workers=0,
        dataloader_pin_memory=device.type=='cuda', save_strategy='no', save_total_limit=None,
        logging_steps=max(1,args.steps//16), logging_first_step=True,
        remove_unused_columns=False, ddp_find_unused_parameters=False,
        ddp_broadcast_buffers=False, disable_tqdm=True, label_names=['targets'])
    set_seed(42)
    trainer = RecoveryTrainer(model=model, teacher=teacher, args=config,
        train_dataset=Prefixes(rows), data_collator=default_data_collator, callbacks=[callback])
    start = time.monotonic()
    result = trainer.train()
    elapsed = time.monotonic()-start
    if trainer.state.global_step != args.steps:
        raise ValueError('Recovery ended at wrong step')
    if any(p.requires_grad or p.grad is not None for p in teacher.parameters()):
        raise ValueError('Teacher received gradients')
    order = [None] * world
    if world > 1: dist.all_gather_object(order, trainer.seen)
    else: order[0] = trainer.seen
    flat = [i for part in order for i in part]
    if len(flat) != len(rows) or sorted(flat) != list(range(len(rows))):
        raise ValueError('Recovery data was duplicated, skipped or repeated')
    hashes = [hashlib.sha256(json.dumps(part).encode()).hexdigest() for part in order]
    if rank == 0:
        model.save_pretrained(args.output/'final')
    if world > 1: dist.barrier()
    reference = score(model, tokens['evaluation'][:max(2, world)])
    reloaded = load_compressed(args.output/'final', device).eval()
    actual = score(reloaded, tokens['evaluation'][:max(2, world)])
    torch.testing.assert_close(torch.tensor(actual['per_document_nll']),
        torch.tensor(reference['per_document_nll']), rtol=0, atol=1e-6)
    del reloaded, teacher
    gc.collect()
    if device.type == 'cuda': torch.cuda.empty_cache()
    after_sensitivity = None
    if args.arm in ('A','B') and args.calibration > 0:
        rankings = rank_channels(model, tokens['calibration'][:args.calibration])
        after_sensitivity = sensitivity(model, tokens['calibration'][args.calibration:2*args.calibration],
                                        rankings, levels=(0,2))
    inference = None
    if args.inference_eval:
        if device.type != 'cuda':raise ValueError('Inference precision check requires CUDA')
        deployed = load_compressed(args.output/'final', device, torch.bfloat16).eval()
        inference = score(deployed, tokens['evaluation'])
        del deployed
    report = dict(status='passed', arm=args.arm, world_size=world, steps=args.steps, smoke=args.smoke,
        input_tokens=needed*(rows.shape[1]-1), unique_documents=len(rows),
        parameters=sum(p.numel() for p in model.parameters()),
        kv_sources=model.config.kv_sources, mlp_widths=model.config.mlp_widths,
        sample_order_sha256_per_rank=hashes, probe_sha256=sha256(args.probe/'manifest.json'),
        plan_sha256=sha256(args.plan) if args.plan else None, training_seconds=elapsed, trainer_metrics=result.metrics,
        training_arguments=config.to_dict(), validation=callback.history,
        reloaded_max_nll_difference=max(abs(a-b) for a,b in zip(actual['per_document_nll'],reference['per_document_nll'])),
        post_recovery_sensitivity=after_sensitivity, inference_validation=inference,
        phase='new_recovery' if parent else 'initial_conversion',
        initial_model_files=parent_files, parent_result_sha256=sha256(args.parent_result) if parent else None,
        parent_steps=parent['steps'] if parent else 0, initial_max_nll_difference=initial_difference,
        optimizer_and_schedule_reset=parent is not None)
    if rank == 0:
        write(args.output/'result.json', report)
        print('KV_MLP_RECOVERY_COMPLETE', args.arm, flush=True)


def summarize(args):
    reports = {arm:json.loads((args.runs/arm/'result.json').read_text()) for arm in 'ABCDEF'}
    ref = reports['A']
    for arm, report in reports.items():
        for key in ['steps','world_size','input_tokens','sample_order_sha256_per_rank','probe_sha256','plan_sha256']:
            if report[key] != ref[key]: raise ValueError(f'Unmatched arm {arm}: {key}')
        if report['status'] != 'passed': raise ValueError('Incomplete arm')
    comparisons = {}
    for step in ref['validation']:
        values = {a:r['validation'][step]['per_document_nll'] for a,r in reports.items()}
        interaction = np.asarray(values['D']) - values['B'] - values['C'] + np.asarray(values['A'])
        comparisons[step] = dict(
            factorial_interaction=paired_summary(interaction.tolist(), [0.]*len(interaction)),
            recalibrated_channels=paired_summary(values['E'], values['D']),
            recalibrated_allocation=paired_summary(values['F'], values['E']),
            total_recalibration=paired_summary(values['F'], values['D']))
    write(args.output, dict(status='passed', reports={a:{k:r[k] for k in ['steps','parameters','mlp_widths',
        'training_seconds','validation','post_recovery_sensitivity']} for a,r in reports.items()},
        comparisons=comparisons, total_recovery_tokens=sum(r['input_tokens'] for r in reports.values()),
        limitations=['One model/seed; prefix LM probe, not full validation or downstream quality.',
                     'Document bootstrap does not quantify training-seed variation.',
                     'Calibration loss curves are single-layer perturbations, not a causal proof.']))


def inference_quality(args, device, rank, world, tokens):
    """Score the BF16-weight inference configuration used by the profiler."""
    if device.type != 'cuda':raise ValueError('Deployment precision check requires CUDA')
    reports={}
    for arm in 'ABCDEF':
        previous=json.loads((args.runs/arm/'result.json').read_text())
        if (previous['status']!='passed' or previous['steps']!=512 or previous['smoke']
                or previous['probe_sha256']!=sha256(args.probe/'manifest.json')):
            raise ValueError('Inference checkpoint/probe identity differs')
        model=load_compressed(args.runs/arm/'final',device,torch.bfloat16).eval()
        if (model.config.kv_sources!=previous['kv_sources'] or model.config.mlp_widths!=previous['mlp_widths']
                or sum(p.numel() for p in model.parameters())!=previous['parameters']):
            raise ValueError('Inference architecture differs')
        reports[arm]=score(model,tokens['evaluation'])
        reports[arm]['versus_training_precision']=paired_summary(reports[arm]['per_document_nll'],
            previous['validation']['512']['per_document_nll'])
        if rank==0:print('KV_MLP_INFERENCE_QUALITY',arm,reports[arm]['loss'],reports[arm]['ppl'],flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    if rank==0:write(args.output,dict(status='passed',world_size=world,weight_dtype='bfloat16',
        autocast_dtype='bfloat16',probe_sha256=sha256(args.probe/'manifest.json'),arms=reports))


def recovery_summary(args):
    reports = {a:json.loads((args.runs/a/'result.json').read_text()) for a in 'AB'}
    a,b = reports['A'], reports['B']
    for r in reports.values():
        if r['status'] != 'passed' or r['smoke'] or r['phase'] != 'new_recovery':
            raise ValueError('Recovery phase did not complete')
    for key in ['steps','world_size','input_tokens','sample_order_sha256_per_rank','probe_sha256','parent_steps']:
        if a[key] != b[key]:raise ValueError('Unmatched recovery pair: '+key)
    for key in ['learning_rate','lr_scheduler_type','warmup_steps','weight_decay','max_grad_norm',
                'per_device_train_batch_size','gradient_accumulation_steps','seed','data_seed','bf16']:
        if a['training_arguments'][key] != b['training_arguments'][key]:
            raise ValueError('Unmatched recovery recipe: '+key)
    if a['validation'].keys() != b['validation'].keys():
        raise ValueError('Unmatched validation steps')
    curve = {}
    for step in a['validation']:
        x,y = a['validation'][step],b['validation'][step]
        curve[step] = dict(dense_ppl=x['ppl'], shared_ppl=y['ppl'], ppl_ratio=y['ppl']/x['ppl'],
            paired=paired_summary(y['per_document_nll'],x['per_document_nll']))
    x,y = a['inference_validation'], b['inference_validation']
    if x is None or y is None:raise ValueError('Missing inference-precision validation')
    inference=dict(dense_ppl=x['ppl'],shared_ppl=y['ppl'],ppl_ratio=y['ppl']/x['ppl'],
                   paired=paired_summary(y['per_document_nll'],x['per_document_nll']))
    gate=inference['paired']['paired_document_bootstrap_95ci'][1] <= math.log(1.05)
    write(args.output,dict(status='passed',curve=curve,inference=inference,
        decision='prune_recovered' if gate else 'review_conversion',
        gate='Upper 95% document CI for shared-minus-dense NLL <= log(1.05)',
        phase_tokens=sum(r['input_tokens'] for r in reports.values()),
        limitations=['A diagnostic on reused 256-document validation; not a new independent test.',
                     'New optimizer/schedule phase, not an exact resume; one training seed.']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['calibrate','train','summarize','inference-quality','recovery-summary'])
    for name in ['model','probe','output','plan','runs']: p.add_argument('--'+name, type=Path)
    p.add_argument('--arm', choices=list('ABCDEF'))
    p.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    p.add_argument('--calibration', type=int, default=64)
    p.add_argument('--steps', type=int, default=512)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--initial-model', type=Path)
    p.add_argument('--parent-result', type=Path)
    p.add_argument('--eval-every', type=int, default=0)
    p.add_argument('--inference-eval', action='store_true')
    args = p.parse_args()
    if args.output is None or args.output.exists(): raise ValueError('Require fresh output')
    if args.mode == 'summarize':
        summarize(args); return
    if args.mode == 'recovery-summary':
        recovery_summary(args); return
    if min(args.steps,args.batch) < 1 or args.calibration < 0 or args.eval_every < 0:
        raise ValueError('Invalid recovery sizes')
    rank, world = int(os.environ.get('RANK',0)), int(os.environ.get('WORLD_SIZE',1))
    device = torch.device('cuda',int(os.environ.get('LOCAL_RANK',0))) if args.device == 'cuda' else torch.device('cpu')
    if device.type == 'cuda': torch.cuda.set_device(device)
    if world > 1: dist.init_process_group('nccl' if device.type=='cuda' else 'gloo')
    try:
        set_seed(42); torch.set_float32_matmul_precision('highest')
        manifest, tokens = load_probe(args.probe)
        if rank == 0:
            for name, expected in manifest['model_files'].items():
                if sha256(args.model/name) != expected: raise ValueError('Model identity mismatch: '+name)
        if world > 1: dist.barrier()
        if args.mode == 'calibrate': calibrate(args,device,rank,world,tokens,manifest)
        elif args.mode == 'inference-quality':inference_quality(args,device,rank,world,tokens)
        else: train(args,device,rank,world,tokens,manifest)
    finally:
        if dist.is_initialized(): dist.destroy_process_group()


if __name__ == '__main__': main()
