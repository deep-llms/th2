"""Offline fresh-process resume test through train.py, with per-rank traces.

The outer supervisor allocates GPUs. This probe never stops processes or burns.
Synthetic data and smaller width/vocabulary keep the 12-layer routing test cheap.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def digest(value):
    """Value digest independent of torch serialization/storage addresses."""
    import torch
    h = hashlib.sha256()
    def visit(x):
        if isinstance(x, torch.Tensor):
            y = x.detach().cpu().contiguous()
            h.update(str((str(y.dtype), tuple(y.shape))).encode())
            h.update(y.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x, dict):
            for k in sorted(x, key=lambda k: (type(k).__name__, str(k))):
                visit(k)
                visit(x[k])
        elif isinstance(x, (list, tuple)):
            h.update(type(x).__name__.encode())
            for item in x:
                visit(item)
        else:
            h.update(pickle.dumps(x, protocol=4))
    visit(value)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def worker(config, traces):
    import random
    import numpy as np
    import torch
    import train
    import bdot.trainer
    from transformers import TrainerCallback

    torch.set_num_threads(1)
    traces.mkdir(parents=True, exist_ok=True)

    class Trace(TrainerCallback):
        def __init__(self, trainer):
            self.trainer = trainer
            self.batches = []
            self.updates = []
            self.snapshots = {}
            self.current = None
            self.branches = None

        def state(self):
            t = self.trainer
            return dict(model=digest(t.model.state_dict()),
                        optimizer=digest(t.optimizer.state_dict()),
                        scheduler=digest(t.lr_scheduler.state_dict()),
                        completed_steps=int(t.model.completed_steps),
                        routing_counter=int(t.model.routing_counter))

        def on_train_begin(self, args, state, control, **kwargs):
            self.initial = self.state()

        def before(self, module, args, kwargs):
            if not module.training:
                return
            rng = dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state())
            if self.trainer.args.device.type == 'cuda':
                rng['cuda'] = torch.cuda.get_rng_state(self.trainer.args.device)
            self.current = dict(step=self.trainer.state.global_step,
                input_sha256=digest(kwargs['input_ids']), labels_sha256=digest(kwargs['labels']),
                rng_sha256=digest(rng), progress=module.progress,
                routing_counter=int(module.routing_counter))
            self.branches = []

        def after(self, module, args, kwargs, output):
            if not module.training:
                return
            self.current.update(loss=float(output.loss.detach()), routes=digest(self.branches))
            if hasattr(module, 'stop_controller'):
                self.current['stop_depths'] = digest(module.stop_controller.last_depths)
            self.batches.append(self.current)
            self.current = self.branches = None

        def on_step_end(self, args, state, control, model=None, optimizer=None, **kwargs):
            # Registered after RoutingUpdateCallback; includes its one-per-update state.
            self.updates.append(dict(step=state.global_step, lr=optimizer.param_groups[0]['lr'],
                completed_steps=int(model.completed_steps), routing_counter=int(model.routing_counter),
                beta=digest(model.balance_bias), progress=model.progress))

        def on_save(self, args, state, control, **kwargs):
            self.snapshots[str(state.global_step)] = self.state()

        def on_train_end(self, args, state, control, **kwargs):
            write(traces/f'rank{args.process_index}.json', dict(status='completed',
                rank=args.process_index, world_size=args.world_size,
                global_step=state.global_step, max_steps=state.max_steps,
                initial=self.initial, final=self.state(), snapshots=self.snapshots,
                batches=self.batches, updates=self.updates))

    original = bdot.trainer.BDOTTrainer
    class TracedTrainer(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            trace = Trace(self)
            self.add_callback(trace)
            self.model.register_forward_pre_hook(trace.before, with_kwargs=True)
            self.model.register_forward_hook(trace.after, with_kwargs=True)
            for step in self.model.routed_steps:
                select = step.select
                def wrapped(*args, _select=select, **kwargs):
                    selected, weights = _select(*args, **kwargs)
                    if trace.branches is not None:
                        trace.branches.append(selected.detach().cpu())
                    return selected, weights
                step.select = wrapped
    bdot.trainer.BDOTTrainer = TracedTrainer
    sys.argv = ['train.py', str(config)]
    train.main()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def prepare(out, backend, world, steps, split, soft_split, use_cpu, expert_backend='reference', curriculum='B',
            schedule_steps=None, variant_config=None):
    import torch
    from datasets import Dataset
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    schedule_steps = schedule_steps or steps
    out.mkdir(parents=True, exist_ok=False)
    vocab = {'[UNK]': 0, '[EOS]': 1, **{f'w{i}': i+2 for i in range(510)}}
    tok = Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
    tok.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token='[UNK]',
        eos_token='[EOS]', pad_token='[EOS]', model_max_length=128)
    tok.save_pretrained(out/'tokenizer')
    # Plenty of distinct documents for a single full schedule without repeating.
    gen = torch.Generator().manual_seed(0)
    n = (schedule_steps*world*2*128 + 16384)//127 + 1
    rows = torch.randint(510, (n,126), generator=gen).tolist()
    Dataset.from_dict({'id':[str(i) for i in range(n)],
        'text':[' '.join(f'w{x}' for x in row) for row in rows]}).save_to_disk(out/'data')
    cfg = json.loads((Path(variant_config) if variant_config else ROOT/'configs/bdot/P1-B.json').read_text())
    cfg.update(vocab_size=512, hidden_size=256, intermediate_size=768,
               num_attention_heads=4, num_key_value_heads=2, head_dim=128,
               eos_token_id=1)
    # Diagnostic: nontrivial routes, with exploration still active after step100.
    cfg['bdot'].update(default_prior=0. if curriculum=='B' else 1., explore_eps_max=.8,
                      explore_end_frac=1., seed=17, expert_backend=expert_backend, curriculum=curriculum)
    write(out/'model.json',cfg)
    base = dict(config_name=str(out/'model.json'), tokenizer_name=str(out/'tokenizer'),
        data_dir=str(out/'data'), data_order='document_stream', dataset_revision='synthetic-resume-v1',
        validation_manifest=str(out/'validation.json'), validation_tokens=1024,
        data_cache_dir=str(out/'cache'), map_batch_size=31, block_size=128,
        attention_backend=backend, use_cpu=use_cpu, bf16=not use_cpu,
        max_steps=schedule_steps, save_steps=soft_split, save_total_limit=None,
        logging_steps=soft_split, report_to='none', learning_rate=3e-4,
        warmup_steps=min(math.ceil(.02*schedule_steps),
                        math.ceil((.1 if curriculum=='A' else .15)*schedule_steps)-1),
        lr_scheduler_type='cosine', adam_beta2=.95,
        per_device_train_batch_size=1, gradient_accumulation_steps=2,
        dataloader_num_workers=0, disable_tqdm=True, gradient_checkpointing=False,
        seed=42, data_seed=0, train_sampling_strategy='sequential',
        eval_strategy='steps', eval_steps=soft_split, per_device_eval_batch_size=1)
    if variant_config:
        base['keep_checkpoint_steps'] = sorted({soft_split, split, min(2*soft_split,steps), steps})
    jobs = [('full',None,steps),('resume_hard',split,steps),
            ('resume_soft',soft_split,min(2*soft_split,steps))]
    for name,start,end in jobs:
        args = dict(base,output_dir=str(out/name),stop_after=end)
        if start is not None:
            args['resume_from_checkpoint']=str(out/'full'/f'checkpoint-{start}')
        write(out/(name+'.json'),args)
    return jobs


def compare(out, world, split, soft_split, steps, schedule_steps=None):
    schedule_steps = schedule_steps or steps
    results=[]
    for name,start,end in [('resume_hard',split,steps),('resume_soft',soft_split,min(2*soft_split,steps))]:
        for rank in range(world):
            full=json.loads((out/'full_trace'/f'rank{rank}.json').read_text())
            resumed=json.loads((out/(name+'_trace')/f'rank{rank}.json').read_text())
            expected=[x for x in full['batches'] if start<=x['step']<end]
            actual=resumed['batches']
            errors=[]
            if len(expected)!=len(actual) or len(actual)!=(end-start)*2:
                errors.append('batch_count')
            fields=['step','input_sha256','labels_sha256','rng_sha256','progress','routing_counter','routes']
            if expected and 'stop_depths' in expected[0]:
                fields.append('stop_depths')
            mismatches={field:sum(a[field]!=b[field] for a,b in zip(expected,actual)) for field in fields}
            errors.extend(k for k,v in mismatches.items() if v)
            loss_max=max((abs(a['loss']-b['loss']) for a,b in zip(expected,actual)),default=0.)
            if not all(math.isfinite(x['loss']) for x in expected+actual) or loss_max>1e-6:
                errors.append('loss')
            initial_ok=full['snapshots'][str(start)]==resumed['initial']
            final_ok=full['snapshots'][str(end)]==resumed['final']
            if not initial_ok:errors.append('restored_state')
            if not final_ok:errors.append('final_state')
            updates=[x for x in full['updates'] if start<x['step']<=end]
            if updates!=resumed['updates']:errors.append('update_state')
            if resumed['global_step']!=end or resumed['max_steps']!=schedule_steps:errors.append('schedule')
            results.append(dict(case=name,rank=rank,status='failed' if errors else 'passed',
                errors=errors,batches=len(actual),mismatches=mismatches,max_loss_abs=loss_max,
                restored_state_exact=initial_ok,final_state_exact=final_ok))
    report=dict(status='passed' if all(r['status']=='passed' for r in results) else 'failed',
        world_size=world,steps=steps,schedule_steps=schedule_steps,split=split,soft_split=soft_split,results=results,
        scope='Real train.py and strict data/training recipe checks, fresh processes, 12-layer reduced-width synthetic fixture.')
    write(out/'summary.json',report)
    print('RESUME_SUMMARY '+json.dumps(report),flush=True)
    return 0 if report['status']=='passed' else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['suite','worker','compare'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path)
    p.add_argument('--accelerate-config',type=Path)
    p.add_argument('--cpu',action='store_true')
    p.add_argument('--steps',type=int,default=200)
    p.add_argument('--schedule-steps',type=int,
                   help='Full horizon; use 1000 with --steps 200 for spec v1.4 T12')
    p.add_argument('--split',type=int,default=100)
    p.add_argument('--soft-split',type=int,default=20)
    p.add_argument('--expert-backend',choices=['reference','grouped'],default='reference')
    p.add_argument('--curriculum',choices=['A','B','none'],default='B')
    p.add_argument('--variant-config',type=Path,help='New routing options in the same reduced-width resume fixture')
    a=p.parse_args();out=a.output.resolve()
    if a.mode=='worker':
        worker(a.config,out)
        return 0
    world=1 if a.cpu else 8
    if not 0<a.soft_split<a.split<a.steps or a.split%a.soft_split:
        p.error('Require 0 < soft-split < split < steps, and split divisible by soft-split')
    if a.schedule_steps is not None and a.schedule_steps<a.steps:
        p.error('schedule-steps must be at least steps')
    if a.mode=='compare':return compare(out,world,a.split,a.soft_split,a.steps,a.schedule_steps)
    jobs=prepare(out,'sdpa' if a.cpu else 'fa4',world,a.steps,a.split,a.soft_split,a.cpu,
                 a.expert_backend,a.curriculum,a.schedule_steps,a.variant_config)
    for name,_,_ in jobs:
        argv=[sys.executable]
        if not a.cpu:
            from scripts.gpu_status import require_free
            require_free(list(range(8)))
            if a.accelerate_config is None:raise ValueError('Explicit eight-GPU Accelerate config required')
            argv+=['-m','accelerate.commands.launch','--config_file',str(a.accelerate_config),
                   '--main_process_port','0']
        argv+=[str(Path(__file__).resolve()),'worker','--config',str(out/(name+'.json')),
               '--output',str(out/(name+'_trace'))]
        print('RESUME_START '+name,flush=True)
        env={**os.environ,'OMP_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false'}
        if a.cpu:env['CUDA_VISIBLE_DEVICES']=''
        with (out/(name+'.log')).open('x') as log:
            r=subprocess.run(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:
            write(out/'failure.json',dict(status='failed',case=name,returncode=r.returncode))
            raise RuntimeError(f'{name} failed; inspect its log')
    return compare(out,world,a.split,a.soft_split,a.steps,a.schedule_steps)


if __name__=='__main__':
    raise SystemExit(main())
