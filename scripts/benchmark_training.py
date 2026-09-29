"""Isolated performance experiments around the unchanged train.py entry point.

Run each worker in a fresh process. Experimental attention/model substitutions
are process-local, never applied to production training or saved checkpoints.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def configure_model(backend='original', native=False, checkpoint_lm=True, checkpoint_aux=True):
    import torch
    import deep_kv.model as module
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    class PackedContext(module.Context):
        def additive_mask(self, dtype):
            # This benchmark uses fully packed, unsegmented train.py contexts.
            if self.segments is not None or not bool(self.valid.all()):
                raise ValueError('Fast attention benchmark requires fully packed unsegmented inputs')
            return None

    def flash_attention(module, query, key, value, attention_mask, dropout=0., scaling=None, **kwargs):
        if attention_mask is not None or dropout != 0.:
            raise ValueError('FA4 benchmark supports only full causal attention without dropout')
        from flash_attn.cute import flash_attn_func
        # Match SDPA autocast: Q/K can be FP32 after QK norms/rotary.
        dtype = value.dtype
        # FA4 4.0.0b32 always returns (output, lse), even with return_lse=False.
        output, _ = flash_attn_func(query.to(dtype).transpose(1, 2), key.to(dtype).transpose(1, 2),
                                   value.transpose(1, 2), causal=True, softmax_scale=scaling)
        return output, None

    if backend == 'fa4':
        ALL_ATTENTION_FUNCTIONS.register('benchmark_fa4', flash_attention)

    class BenchmarkModel(module.DeepKV):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.checkpoint_lm, self.checkpoint_aux = checkpoint_lm, checkpoint_aux
            if backend == 'fa4':
                self.backbone.config._attn_implementation = 'benchmark_fa4'
            if native and self.arm != 'A':
                raise ValueError('Native model benchmark must be Arm A')

        def forward(self, context):
            if backend != 'original' or native:
                context = PackedContext(context.input_ids, context.valid, context.position_ids, context.segments)
            if not native:
                return super().forward(context)
            if context.segments is not None or not bool(context.valid.all()):
                raise ValueError('Native benchmark requires fully packed inputs')
            # Native Qwen forward and full-vocabulary head, no activation checkpointing.
            logits = self.backbone(input_ids=context.input_ids, position_ids=context.position_ids,
                                   use_cache=False).logits[:, :-1].float()
            labels = context.input_ids[:, 1:]
            rows = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                    labels.reshape(-1), reduction='none').view_as(labels).sum(1)
            counts, tokens = context.targets().sum(1), context.valid.sum(1)
            zeros = torch.zeros_like(rows)
            return dict(lm_sum=rows.sum(), lm_count=counts.sum(), k_sum=zeros.sum(), v_sum=zeros.sum(),
                        kv_count=tokens.sum(), statistics=torch.stack((rows.detach(), counts, zeros,
                                                                      zeros, tokens), dim=1).double())
    return BenchmarkModel


def objective(model, result):
    if model.functional_loss:
        return result['lm_sum']/result['lm_count'] + .3*(result['route_sum'] +
               (result['msg_sum'] if model.arm == 'G' else 0))/result['route_count'].clamp_min(1)
    return result['lm_sum']/result['lm_count'] + model.kv_loss_weight*(
        result['k_sum']+result['v_sum'])/(2*result['kv_count'])


def validate_variant(backend, native, arm, device='cpu', bf16=False, checkpoint_layers=False, lm_chunk=32,
                     checkpoint_lm=True, checkpoint_aux=True):
    import copy
    import torch
    from transformers import Qwen3Config
    from deep_kv.model import Context, DeepKV
    from contextlib import nullcontext
    cfg = Qwen3Config(vocab_size=256, hidden_size=256, intermediate_size=384,
                     num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
                     head_dim=128, max_position_embeddings=256, attention_dropout=0., tie_word_embeddings=True)
    cfg._attn_implementation = 'sdpa'
    options = dict(consumer=2, deep_target=4, checkpoint_layers=False, lm_chunk=32)
    reference = DeepKV.from_scratch(copy.deepcopy(cfg), arm, **options).to(device)
    candidate = configure_model(backend, native, checkpoint_lm, checkpoint_aux).from_scratch(copy.deepcopy(cfg), arm, **options).to(device)
    candidate.checkpoint_layers=checkpoint_layers
    candidate.lm_chunk=lm_chunk
    # Nonzero branch output tests attention/backbone gradients beyond zero-init.
    if reference.aux is not None:
        with torch.no_grad():
            reference.aux.out.weight.fill_(.001)
    candidate.load_state_dict(reference.state_dict())
    ids = (torch.arange(256, device=device).reshape(2, 128)*17+3)%256
    ctx = Context(ids, torch.ones_like(ids, dtype=torch.bool), torch.arange(128, device=device).expand_as(ids))
    values=[]
    for model in (reference, candidate):
        model.train()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16) if bf16 else nullcontext():
            loss=objective(model,model(ctx))
        loss.backward()
        values.append((float(loss.detach()), {k:p.grad.detach().float().cpu() for k,p in model.named_parameters() if p.grad is not None}))
    a,b=values
    if a[1].keys()!=b[1].keys():
        raise ValueError('Gradient parameter sets differ')
    numerator=sum((a[1][k]-b[1][k]).square().sum().item() for k in a[1])
    denominator=sum(a[1][k].square().sum().item() for k in a[1])
    relative=(numerator/max(denominator,1e-30))**.5
    if not abs(a[0]-b[0]) <= (0.01 if bf16 else 1e-5) or not relative <= (.03 if bf16 else 2e-5):
        raise ValueError(f'Loss/gradient mismatch: {a[0]}, {b[0]}, relative L2 {relative}')
    return dict(reference_loss=a[0], candidate_loss=b[0], gradient_relative_l2=relative,
                gradient_parameters=len(a[1]), bf16=bf16)


def worker(backend, native, variant, checkpoint_lm=True, checkpoint_aux=True, disposable=False):
    import torch
    import train
    from accelerate import PartialState
    state=PartialState()
    run_config=json.loads(Path(sys.argv[1]).read_text())
    validation=validate_variant(backend,native,run_config['arm'],state.device,True,
        checkpoint_layers=run_config.get('checkpoint_layers',True),lm_chunk=run_config.get('lm_chunk',128),
        checkpoint_lm=checkpoint_lm,checkpoint_aux=checkpoint_aux)
    torch.cuda.empty_cache()
    train.DeepKV=configure_model(backend,native,checkpoint_lm,checkpoint_aux)
    callbacks=[]
    original_trainer=train.DeepKVTrainer

    class BenchTrainer(original_trainer):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.input_digest=hashlib.sha256()
            self.first_batches=0
            self.input_rows=[]

        def _save_checkpoint(self, *args, **kwargs):
            if not disposable:
                return super()._save_checkpoint(*args, **kwargs)

        def save_model(self, *args, **kwargs):
            if not disposable:
                return super().save_model(*args, **kwargs)

        def compute_loss(self,model,inputs,*args,**kwargs):
            if model.training and self.is_in_train and self.state.global_step==0:
                rows=inputs['input_ids'].detach().cpu().numpy()
                self.input_digest.update(rows.tobytes())
                self.input_rows.extend(hashlib.sha256(row.tobytes()).hexdigest() for row in rows)
                self.first_batches+=1
            return super().compute_loss(model,inputs,*args,**kwargs)

    class Timing(train.PilotCallback):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.rows=[];self.profiler=None;self.profile=None;self.previous=None
            self.allocated=self.reserved=0
            callbacks.append(self)

        def on_step_begin(self,args,state,control,**kwargs):
            if state.global_step==5:
                torch.cuda.reset_peak_memory_stats()
            if state.global_step==2 and args.process_index==0 and not disposable:
                self.profiler=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],record_shapes=False)
                self.profiler.start()
            torch.cuda.synchronize()
            self.begin=time.perf_counter()

        def on_step_end(self,args,state,control,**kwargs):
            super().on_step_end(args,state,control,**kwargs)
            torch.cuda.synchronize()
            end=time.perf_counter()
            self.rows.append(dict(step=state.global_step,compute_seconds=end-self.begin,
                interval_seconds=None if self.previous is None else end-self.previous,**self.latest))
            self.previous=end
            if state.global_step>=6:
                self.allocated=torch.cuda.max_memory_allocated()
                self.reserved=torch.cuda.max_memory_reserved()
            if state.global_step==3 and self.profiler is not None:
                self.profiler.stop()
                averages=self.profiler.key_averages()
                ops=[dict(name=x.key,count=x.count,self_cpu_us=x.self_cpu_time_total,
                          self_device_us=x.self_device_time_total) for x in averages]
                kernels={}
                for event in self.profiler.events():
                    if event.device_type==torch.autograd.DeviceType.CUDA:
                        item=kernels.setdefault(event.name,dict(count=0,microseconds=0.))
                        item['count']+=1;item['microseconds']+=event.time_range.elapsed_us()
                self.profile=dict(top_ops=sorted(ops,key=lambda x:x['self_device_us'],reverse=True)[:40],
                    attention_ops=[x for x in ops if any(s in x['name'].lower() for s in ['attention','flash','cudnn','nccl','allreduce'])],
                    top_kernels=sorted([dict(name=k,**v) for k,v in kernels.items()],key=lambda x:x['microseconds'],reverse=True)[:35])
                self.profiler=None
            return control

    train.DeepKVTrainer=BenchTrainer
    train.PilotCallback=Timing
    try:
        train.main()
    except torch.cuda.OutOfMemoryError as error:
        # Structured capacity failure only after CUDA objective validation passed.
        write(Path(run_config['output_dir'])/f'capacity-oom-rank{state.process_index}.json',
              dict(status='cuda_oom',rank=state.process_index,variant=variant,error=str(error)))
        raise
    callback,=callbacks
    trainer=callback.trainer
    record=dict(status='ok',variant=variant,backend=backend,native=native,rank=trainer.args.process_index,
        world_size=trainer.args.world_size,steps=callback.rows,validation=validation,
        checkpoint_lm=checkpoint_lm,checkpoint_aux=checkpoint_aux,disposable=disposable,
        device_total_bytes=torch.cuda.get_device_properties(state.device).total_memory,
        allocated_peak_bytes=callback.allocated,reserved_peak_bytes=callback.reserved,
        profile=callback.profile,input_sha256=trainer.input_digest.hexdigest(),first_batches=trainer.first_batches,
        input_rows=trainer.input_rows,
        optimizer=type(trainer.optimizer).__name__,wrapped_model=type(trainer.model_wrapped).__name__)
    write(Path(trainer.args.output_dir)/f'benchmark-rank{record["rank"]}.json',record)


def summarize(root, names):
    results={};common_inputs=None
    for name in names:
        path=root/name
        ranks=[json.loads((path/f'benchmark-rank{i}.json').read_text()) for i in range(8)]
        config=json.loads((path/'train_config.json').read_text())
        result=json.loads((path/'result.json').read_text())
        assert result['global_step']==18 and config['tokens_per_update']==1048576
        assert all(r['world_size']==8 and r['first_batches']==config['training']['gradient_accumulation_steps'] for r in ranks)
        inputs=sorted(row for r in ranks for row in r['input_rows'])
        assert len(inputs)==512
        if common_inputs is None:common_inputs=inputs
        assert inputs==common_inputs,'Benchmark inputs/order changed'
        wall=[max(r['steps'][i]['interval_seconds'] for r in ranks) for i in range(5,18)]
        compute=[max(r['steps'][i]['compute_seconds'] for r in ranks) for i in range(5,18)]
        results[name]=dict(seconds_per_update=statistics.mean(wall),median_seconds=statistics.median(wall),
            stdev_seconds=statistics.stdev(wall),compute_seconds=statistics.mean(compute),
            tokens_per_second=1048576/statistics.mean(wall),allocated_peak_gib=max(r['allocated_peak_bytes'] for r in ranks)/2**30,
            reserved_peak_gib=max(r['reserved_peak_bytes'] for r in ranks)/2**30,
            train_fingerprint=config['train_fingerprint'],validation=ranks[0]['validation'],profile=ranks[0]['profile'],
            wrapped_model=ranks[0]['wrapped_model'])
    write(root/'benchmark-summary.json',dict(status='ok',measured_steps=[6,18],results=results))
    print('BENCHMARK_SUMMARY',json.dumps(results),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('worker');p.add_argument('--backend',choices=['original','causal','fa4'],default='original')
    p.add_argument('--native',action='store_true');p.add_argument('--variant',required=True);p.add_argument('config')
    p.add_argument('--no-checkpoint-lm',action='store_true');p.add_argument('--no-checkpoint-aux',action='store_true')
    p.add_argument('--disposable',action='store_true',help='No weight/optimizer saves or profiler; capacity probes only')
    p=sub.add_parser('summarize');p.add_argument('--root',type=Path,required=True);p.add_argument('--names',nargs='+',required=True)
    args=parser.parse_args()
    if args.command=='worker':
        sys.argv=['train.py',str(Path(args.config).resolve())]
        worker(args.backend,args.native,args.variant,not args.no_checkpoint_lm,not args.no_checkpoint_aux,args.disposable)
    else:summarize(args.root,args.names)
