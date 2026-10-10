"""Disposable function-compilation trial; production model and recipe stay unchanged."""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
from pathlib import Path
import runpy
import statistics
import sys


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


@contextmanager
def implementation(mode, backend='inductor'):
    """Patch function symbols before model construction, never parameters/modules."""
    import torch
    import deep_kv.proxy_estimators as estimators
    if mode not in ('eager', 'cosine', 'pointwise'):
        raise ValueError(mode)
    names = [] if mode == 'eager' else ['cosine_loss']
    if mode == 'pointwise':
        names.append('gated_prediction')
    original = {name: getattr(estimators, name) for name in names}
    try:
        for name, function in original.items():
            setattr(estimators, name, torch.compile(function, backend=backend, fullgraph=True, dynamic=False))
        yield
    finally:
        for name, function in original.items():
            setattr(estimators, name, function)


def compare(reference, candidate, *, relative=0.01, peak=0.01, exact=False):
    import torch
    if reference.keys() != candidate.keys():
        raise ValueError('Tensor keys / gradient routing changed')
    differences, failures = [], []
    for key, a in reference.items():
        b = candidate[key]
        if a.shape != b.shape or a.dtype != b.dtype or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            failures.append(dict(key=key, reason='shape/dtype/nonfinite'))
            continue
        if torch.equal(a, b):
            continue
        delta = (a.double()-b.double())
        row = dict(key=key, relative_l2=float(delta.norm()/a.double().norm().clamp_min(1e-30)),
                   max_abs=float(delta.abs().max()), peak_scaled=float(delta.abs().max()/a.double().abs().max().clamp_min(1e-30)))
        differences.append(row)
        if exact or row['relative_l2'] > relative or row['peak_scaled'] > peak:
            failures.append(row)
    return dict(differences=differences, failures=failures)


def gate(args):
    import torch
    from transformers import AutoTokenizer
    from deep_kv.packing import preprocess_dataset, isolated_data_collator
    from deep_kv.proxy_training import ProxyTrainer
    from eval.models import load_checkpoint, file_hash
    from scripts.evaluate_recurrent_p6 import evaluation_runtime
    from scripts.proxy_speed_validation import deterministic_fa4
    from train import load_text
    saved = read(Path(args.checkpoint).parent/'train_config.json')
    training, accelerator = evaluation_runtime(saved['training'], args.output)
    if accelerator.num_processes != 8 or accelerator.device.type != 'cuda':
        raise ValueError('Eight CUDA ranks required')
    rank = accelerator.process_index
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(saved['model']['tokenizer_name'], local_files_only=True)
    dataset = preprocess_dataset(load_text(saved['data']['eval_data_dir']), tokenizer, 2048, training,
                                 num_proc=1, isolate_documents=True).select(range(saved['data']['eval_rows']))
    if dataset._fingerprint != saved['eval_fingerprint']:
        raise ValueError('Validation identity changed')
    candidate = ('cosine', 'pointwise')[rank % 2]
    checkpointing = bool((rank//2) % 2)
    length = (64, 2048)[rank//4]
    rows = [{k:v[:length] for k,v in dataset[i].items()} for i in (0, 1)]
    ctx = ProxyTrainer.context({k:v.to(accelerator.device) for k,v in isolated_data_collator(rows).items()})
    captures = []
    for mode in ('eager', candidate):
        torch.cuda.reset_peak_memory_stats()
        with implementation(mode), deterministic_fa4(True):
            adapter, _, metadata = load_checkpoint(args.checkpoint, accelerator.device, attention_backend='fa4')
        if metadata['arm'] != 'P6-iso' or metadata['step'] != 10000 or metadata['checkpoint_sha256'] != args.checkpoint_sha256:
            raise ValueError('Wrong source checkpoint')
        m = adapter.wrapped.train()
        m.checkpoint_layers = m.checkpoint_aux = m.checkpoint_lm = checkpointing
        outputs = {}
        handle = m.backbone.model.norm.register_forward_hook(lambda module, inputs, value: outputs.update(hidden=value.detach().cpu()))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = m(ctx, collect_target_statistics=True)
            loss = out['lm_sum']/out['lm_count'] + .1*out['aux_sum']/out['aux_count']
        loss.backward()
        handle.remove()
        outputs.update({k:v.detach().cpu().clone() for k,v in out.items()})
        gradients = {k:p.grad.detach().cpu().clone() for k,p in m.named_parameters() if p.grad is not None}
        # Actual per-step buffer update, with the same rule as the Trainer.
        m.update_statistics(out['center_sums'], out['center_squares'], out['center_counts'])
        buffers = {k:v.detach().cpu().clone() for k,v in m.named_buffers()}
        captures.append(dict(outputs=outputs, gradients=gradients, buffers=buffers, loss=float(loss.detach()),
                             peak_gib=torch.cuda.max_memory_allocated()/2**30))
        del m, adapter, out, loss
        gc.collect()
        torch.cuda.empty_cache()
    a, b = captures
    comparison = dict(outputs=compare(a['outputs'], b['outputs'], relative=1e-4, peak=1e-4),
                      gradients=compare(a['gradients'], b['gradients']),
                      buffers=compare(a['buffers'], b['buffers'], relative=1e-4, peak=1e-4))
    if candidate == 'cosine':
        # Compilation of isolated auxiliary loss must leave the LM path EXACT.
        keys = set(a['outputs'])-{'aux_sum','statistics'}
        # cosine diagnostics appear inside statistics; auxiliary tensor is the only other affected output.
        comparison['lm_outputs'] = compare({k:a['outputs'][k] for k in keys}, {k:b['outputs'][k] for k in keys}, exact=True)
        keys = [k for k in a['gradients'] if not (k.startswith('heads.') and '.w' in k)]
        comparison['lm_gradients'] = compare({k:a['gradients'][k] for k in keys}, {k:b['gradients'][k] for k in keys}, exact=True)
        comparison['exact_buffers'] = compare(a['buffers'], b['buffers'], exact=True)
    passed = not any(c['failures'] for c in comparison.values())
    write(Path(args.output)/f'rank-{rank}.json', dict(status='passed' if passed else 'failed', mode=candidate,
          checkpointing=checkpointing, sequence_length=length, comparisons=comparison,
          loss=[a['loss'],b['loss']], peak_gib=[a['peak_gib'],b['peak_gib']], checkpoint=metadata))
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        reports = [read(Path(args.output)/f'rank-{i}.json') for i in range(8)]
        accepted = [mode for mode in ('cosine','pointwise') if all(r['status']=='passed' for r in reports if r['mode']==mode)]
        # Prefer the narrowest successful optimization. Timing tests cosine first.
        selected = 'cosine' if 'cosine' in accepted else None
        if file_hash(Path(args.checkpoint)/'model.safetensors') != args.checkpoint_sha256:
            raise ValueError('Source weights changed')
        write(Path(args.output)/'summary.json',dict(status='passed' if selected else 'failed',selected=selected,
            accepted=accepted, reports=reports, checkpoint_unchanged=True,
            criterion='Outputs/buffers <=1e-4 relative L2 and peak-scaled max; gradients <=1%; cosine LM path and buffers exact. '
                      'Numerical equivalence, not bitwise predictor-gradient equivalence; no production/resume adoption.'))
    accelerator.wait_for_everyone()
    if read(Path(args.output)/'summary.json')['status'] != 'passed':
        raise ValueError('Cosine numerical gate failed; no training benchmark')


def train(args, rest):
    import torch
    from transformers import TrainerCallback
    import deep_kv.proxy_training as training
    mode = read(args.gate)['selected'] if args.implementation == 'selected' else args.implementation
    if args.gate and read(args.gate)['status'] != 'passed':
        raise ValueError('Numerical gate not passed')
    class Timing(TrainerCallback):
        def __init__(self, trainer):
            self.trainer, self.steps, self.hashes = trainer, [], []
            self.graph_counts = {}
        def on_step_end(self, arguments, state, control, **kwargs):
            callback = next(c for c in self.trainer.callback_handler.callbacks if isinstance(c, training.ProxyCallback))
            if arguments.process_index == 0:
                self.steps.append(dict(step=state.global_step, seconds=callback.seconds_per_update))
            if state.global_step in (20,90):
                from torch._dynamo.utils import counters
                self.graph_counts[str(state.global_step)] = dict(counters['stats'])
        def on_train_end(self, arguments, state, control, **kwargs):
            from torch._dynamo.utils import counters
            write(Path(arguments.output_dir)/f'compile-trial-rank{arguments.process_index}.json',dict(
                status='complete', implementation=mode, steps=self.steps, data_sha256=self.hashes,
                graph_counts=self.graph_counts,
                compile_counters={k:dict(v) for k,v in counters.items()},
                peak_gib=torch.cuda.max_memory_allocated()/2**30))
    init, step = training.ProxyTrainer.__init__, training.ProxyTrainer.training_step
    def initialize(self,*a,**kw):
        init(self,*a,**kw)
        self.compile_trial = Timing(self)
        self.add_callback(self.compile_trial)
    def training_step(self,model,inputs,*a,**kw):
        # Audit update 1 and final update only; excluded from measured window.
        if self.state.global_step in (0,99):
            h = hashlib.sha256()
            for key, value in sorted(inputs.items()):
                h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
            self.compile_trial.hashes.append(h.hexdigest())
        return step(self,model,inputs,*a,**kw)
    training.ProxyTrainer.__init__, training.ProxyTrainer.training_step = initialize, training_step
    sys.argv = ['train.py',*(rest[1:] if rest and rest[0]=='--' else rest)]
    with implementation(mode):
        runpy.run_module('train',run_name='__main__')


def summary(args):
    root = Path(args.run_dir)
    runs = []
    for name in ('A-first','P6-eager-first','P6-compiled-first','P6-compiled-second','P6-eager-second','A-second'):
        arm = 'A' if name.startswith('A-') else 'P6-iso'
        folder = root/name/'seed-1042'/arm
        reports = [read(folder/f'compile-trial-rank{i}.json') for i in range(8)]
        config, result = read(folder/'train_config.json'), read(folder/'result.json')
        if any(len(r['data_sha256']) != 8 for r in reports):
            raise ValueError('Expected first/final four-microbatch audits on every rank')
        if any(r['graph_counts']['20'] != r['graph_counts']['90'] for r in reports):
            raise ValueError('Compiler generated new graphs within measured window')
        if result['global_step'] != 100:
            raise ValueError('Incomplete timing run')
        times = [v['seconds'] for v in reports[0]['steps'] if 21<=v['step']<=90]
        if len(times)!=70 or not all(v>0 for v in times):
            raise ValueError('Missing timing steps')
        if runs:
            if config['train_fingerprint']!=runs[0]['train_fingerprint']:
                raise ValueError('Different training dataset/order')
            if [r['data_sha256'] for r in reports]!=runs[0]['data_sha256']:
                raise ValueError('Different batch order across arms/runs')
        runs.append(dict(name=name, median_seconds=statistics.median(times), mean_seconds=statistics.mean(times),
            times=times, peak_gib=max(r['peak_gib'] for r in reports), compile_counters=reports[0]['compile_counters'],
            train_fingerprint=config['train_fingerprint'],data_sha256=[r['data_sha256'] for r in reports],
            implementation=reports[0]['implementation'],
            final_eval={k:v for k,v in result['evaluation'].items() if k in ('eval_lm_loss','eval_aux_loss','eval_rows','eval_target_tokens')},
            training_cost=result['training_cost']))
    medians={key:statistics.mean(r['median_seconds'] for r in runs if r['name'].startswith(key))
             for key in ('A-','P6-eager-','P6-compiled-')}
    write(root/'summary.json',dict(status='passed',runs=runs,median_step_seconds=medians,
        p6_speedup=medians['P6-eager-']/medians['P6-compiled-'],
        optimized_overhead_vs_A=medians['P6-compiled-']/medians['A-']-1,
        production_defaults_changed=False, note='100 actual Trainer updates per run, eight GPUs, identical batches; '
        'steps 21–90 measured, reverse-order repeats. Compilation/startup/final evaluation excluded.'))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    g=sub.add_parser('gate')
    for key in ('checkpoint','checkpoint-sha256','output'):g.add_argument('--'+key,required=True)
    t=sub.add_parser('train');t.add_argument('--implementation',choices=('eager','selected'),required=True)
    t.add_argument('--gate',required=True)
    s=sub.add_parser('summary');s.add_argument('--run-dir',required=True)
    args,rest=p.parse_known_args()
    if args.command=='train':train(args,rest)
    else:
        if rest:p.error('Unexpected arguments: '+str(rest))
        globals()[args.command](args)


if __name__=='__main__':main()
