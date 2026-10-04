"""Disposable full-model document-attention tests; production train.py is unchanged.

Uses the project's arm A, chunked LM loss and HF Trainer/Accelerate. The only
experimental model substitution is the attention interface and boundary metadata.
"""
import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

os.environ.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  HF_HUB_DISABLE_TELEMETRY='1', WANDB_MODE='offline')
os.environ.setdefault('NCCL_NVLS_ENABLE', '0')

import torch
from transformers import AutoConfig, AutoTokenizer, TrainerCallback, TrainingArguments, set_seed
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from deep_kv.model import Context, DeepKV
from deep_kv.packing import document_end_id, group_texts, tokenize_with_segments
from deep_kv.training import DeepKVTrainer, compute_metrics

MODES = ('sdpa_cross', 'sdpa_causal', 'fa4_cross', 'sdpa_isolated', 'fa4_isolated')


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


class Collator:
    def __init__(self, mode):
        if mode not in MODES:
            raise ValueError(mode)
        self.mode = mode

    def __call__(self, rows):
        ids = torch.tensor([r['input_ids'] for r in rows], dtype=torch.long)
        segments = torch.tensor([r['segments'] for r in rows], dtype=torch.long)
        b, n = ids.shape
        positions = torch.arange(n).expand(b, n).clone()
        isolated = self.mode.endswith('isolated')
        lengths = []
        if isolated:
            for i in range(b):
                starts = [0] + (segments[i, 1:] != segments[i, :-1]).nonzero().flatten().add(1).tolist() + [n]
                for a, z in zip(starts, starts[1:]):
                    positions[i, a:z] -= a
                    lengths.append(z - a)
        else:
            lengths = [n] * b
        result = dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids.clone(),
                      position_ids=positions)
        if isolated:
            result['segments'] = segments
        if self.mode.startswith('fa4'):
            result['cu_seqlens'] = torch.tensor([0] + lengths, dtype=torch.int32).cumsum(0, dtype=torch.int32)
            result['max_seqlen'] = max(lengths)
        return result


@dataclass
class BenchContext(Context):
    mode: str = 'sdpa_cross'
    cu_seqlens: torch.Tensor | None = None
    max_seqlen: int | None = None

    def allowed(self):
        # Arm A has no auxiliary attention/loss requiring a dense allowed mask.
        if self.mode.startswith('fa4') or self.mode == 'sdpa_causal':
            return None
        return super().allowed()

    def additive_mask(self, dtype):
        if self.mode.startswith('fa4') or self.mode == 'sdpa_causal':
            return None
        return super().additive_mask(dtype)


def context(inputs, mode):
    return BenchContext(inputs['input_ids'], inputs['attention_mask'].bool(), inputs['position_ids'],
                        inputs.get('segments'), inputs.get('labels'), mode,
                        inputs.get('cu_seqlens'), inputs.get('max_seqlen'))


def fa4_interface(module, query, key, value, attention_mask, dropout=0., scaling=None, **kwargs):
    if attention_mask is not None or dropout != 0.:
        raise ValueError('Benchmark FA4 requires unpadded zero-dropout packed inputs')
    from flash_attn.cute.interface import flash_attn_varlen_func
    b, h, n, d = query.shape
    def flatten(x):
        return x.to(value.dtype).transpose(1, 2).reshape(b*n, x.shape[1], d).contiguous()
    q, k, v = map(flatten, (query, key, value))
    cu, maximum = module._bench_layout
    output, _ = flash_attn_varlen_func(q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu,
                                     max_seqlen_q=maximum, max_seqlen_k=maximum,
                                     causal=True, softmax_scale=scaling)
    return output.reshape(b, n, h, d), None


class BenchmarkModel(DeepKV):
    def hidden_states(self, ctx):
        if self.arm != 'A' or not bool(ctx.valid.all()):
            raise ValueError('This benchmark supports only fully packed arm A')
        impl = 'benchmark_document_fa4' if ctx.mode.startswith('fa4') else 'sdpa'
        self.backbone.config._attn_implementation = impl
        if ctx.mode.startswith('fa4'):
            for layer in self.backbone.model.layers:
                layer.self_attn._bench_layout = (ctx.cu_seqlens, ctx.max_seqlen)
        return super().hidden_states(ctx)


ALL_ATTENTION_FUNCTIONS.register('benchmark_document_fa4', fa4_interface)


def new_model(config_path, device):
    cfg = AutoConfig.from_pretrained(config_path, local_files_only=True)
    cfg._attn_implementation = 'sdpa'
    cfg.use_cache = False
    return BenchmarkModel.from_scratch(cfg, 'A', seed=42, checkpoint_layers=False,
                                      checkpoint_lm=False, checkpoint_aux=False,
                                      lm_chunk=128).to(device)


def prepare(args):
    from train import load_text
    from datasets import load_from_disk
    root = Path(args.root)
    root.mkdir(exist_ok=False, parents=True)
    recipe = json.loads(Path(args.recipe).read_text())
    manifest = json.loads(Path('resources/qwen3_base_assets.json').read_text())
    for name, expected in manifest['files'].items():
        assert hashlib.sha256((Path(recipe['tokenizer_name']) / name).read_bytes()).hexdigest() == expected
    tokenizer = AutoTokenizer.from_pretrained(recipe['tokenizer_name'], local_files_only=True)
    raw = load_text(recipe['data_dir']).select(range(args.documents))
    # Same two batched maps as train.py, with one extra per-token document ID.
    tokenized = raw.map(tokenize_with_segments, with_indices=True, batched=True, num_proc=8,
                        fn_kwargs=dict(tokenizer=tokenizer, end_id=document_end_id(tokenizer)),
                        remove_columns=raw.column_names, cache_file_name=str(root/'tokenized.arrow'))
    packed = tokenized.map(group_texts, batched=True, num_proc=8, fn_kwargs=dict(block_size=2048),
                          cache_file_name=str(root/'packed.arrow'))
    if len(packed) < 4096:
        raise ValueError('Insufficient benchmark text')
    packed = packed.select(range(min(len(packed), 8192)//512*512))
    packed.save_to_disk(str(root/'data'))
    digest = hashlib.sha256()
    fragments = []
    for rows in packed.iter(batch_size=256):
        ids = torch.tensor(rows['input_ids'], dtype=torch.long)
        segments = torch.tensor(rows['segments'], dtype=torch.long)
        digest.update(ids.numpy().tobytes()); digest.update(segments.numpy().tobytes())
        fragments.extend((1 + (segments[:, 1:] != segments[:, :-1]).sum(1)).tolist())
    assert len(load_from_disk(str(root/'data'))) == len(packed)
    write(root/'data.json', dict(rows=len(packed), tokens=len(packed)*2048, sha256=digest.hexdigest(),
                               source=recipe['data_dir'], source_documents=args.documents,
                               tokenizer_revision=manifest['revision'],
                               mean_document_fragments=statistics.mean(fragments),
                               initialization='random seed 42, identical to train.py arm A',
                               recipe=recipe))
    if args.eval_rows:
        raw_eval = load_text(recipe['eval_data_dir'])
        raw_eval = raw_eval.select(range(min(4096, len(raw_eval))))
        tokenized_eval = raw_eval.map(tokenize_with_segments, with_indices=True, batched=True,
            fn_kwargs=dict(tokenizer=tokenizer, end_id=document_end_id(tokenizer)),
            remove_columns=raw_eval.column_names, cache_file_name=str(root/'eval-tokenized.arrow'))
        packed_eval = tokenized_eval.map(group_texts, batched=True, fn_kwargs=dict(block_size=2048),
                                        cache_file_name=str(root/'eval-packed.arrow'))
        if len(packed_eval) < args.eval_rows:
            raise ValueError('Insufficient held-out test data')
        packed_eval = packed_eval.select(range(args.eval_rows))
        packed_eval.save_to_disk(str(root/'eval'))
        eval_digest = hashlib.sha256()
        for batch in packed_eval.iter(batch_size=256):
            for key in ('input_ids', 'segments'):
                eval_digest.update(torch.tensor(batch[key], dtype=torch.long).numpy().tobytes())
        write(root/'eval-data.json', dict(rows=len(packed_eval), source=recipe['eval_data_dir'],
                                        sha256=eval_digest.hexdigest(), input_tokens=args.eval_rows*2048))
    print('BENCH_DATA', (root/'data.json').read_text(), flush=True)


def capture(model, rows, mode, bf16=True):
    inputs = {k: v.to('cuda') if torch.is_tensor(v) else v for k, v in Collator(mode)(rows).items()}
    ctx = context(inputs, mode)
    model.zero_grad(set_to_none=True); model.train()
    with torch.autocast('cuda', dtype=torch.bfloat16) if bf16 else nullcontext():
        # Retain real outputs without a second model forward.
        saved = {}
        def hook(module, inp, out):
            saved['hidden'] = out.detach().float().cpu()
            saved['logits'] = model.backbone.lm_head(out[:, ::128]).detach().float().cpu()
        handle = model.backbone.model.norm.register_forward_hook(hook)
        try:
            result = model(ctx)
        finally:
            handle.remove()
        loss = result['lm_sum']/result['lm_count']
    loss.backward()
    grads = {name: p.grad.detach().float().cpu().clone() for name, p in model.named_parameters()}
    assert all(torch.isfinite(g).all() for g in grads.values())
    return dict(loss=float(loss.detach()), targets=int(result['lm_count']), grads=grads, **saved)


def relative(a, b):
    # a is the reference; tensors live on CPU to limit device memory.
    delta = (a-b).double()
    return dict(relative_l2=float(delta.norm()/a.double().norm().clamp_min(1e-30)),
                relative_max=float(delta.abs().max()/a.double().abs().max().clamp_min(1e-30)))


def compare(a, b, enforce):
    assert a['grads'].keys() == b['grads'].keys()
    numerator = denominator = dot = other = 0.
    worst = []
    for name, x in a['grads'].items():
        y = b['grads'][name]
        # Streaming per parameter avoids a flattened 0.6B-element allocation.
        delta = (x-y).double()
        xx, yy = x.double(), y.double()
        n, d = delta.square().sum().item(), xx.square().sum().item()
        numerator += n; denominator += d
        dot += (xx*yy).sum().item(); other += yy.square().sum().item()
        worst.append(dict(name=name, relative_l2=(n/max(d, 1e-30))**.5))
    result = dict(reference_loss=a['loss'], candidate_loss=b['loss'],
                  loss_abs_diff=abs(a['loss']-b['loss']), targets=[a['targets'], b['targets']],
                  hidden=relative(a['hidden'], b['hidden']), logits=relative(a['logits'], b['logits']),
                  gradient_relative_l2=(numerator/max(denominator, 1e-30))**.5,
                  gradient_cosine=dot/max((denominator*other)**.5, 1e-30),
                  worst_parameters=sorted(worst, key=lambda r:r['relative_l2'], reverse=True)[:8])
    if enforce:
        assert a['targets'] == b['targets']
        assert result['loss_abs_diff'] < .01, result
        assert result['gradient_relative_l2'] < .03, result
        assert result['hidden']['relative_l2'] < .02 and result['logits']['relative_l2'] < .02, result
    return result


def correctness(args):
    from datasets import load_from_disk
    torch.set_num_threads(4); set_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    root = Path(args.root); meta = json.loads((root/'data.json').read_text())
    data = load_from_disk(str(root/'data'))
    rows = []
    for row in data:
        if len(set(row['segments'])) > 1:
            rows.append(row)
        if len(rows) == 2: break
    assert len(rows) == 2
    model = new_model(meta['recipe']['config_name'], 'cuda')
    results = {}
    cross = capture(model, rows, 'sdpa_cross')
    for mode in ('sdpa_causal', 'fa4_cross'):
        candidate = capture(model, rows, mode)
        results[mode+'_vs_sdpa_cross'] = compare(cross, candidate, True)
        del candidate
    isolated = capture(model, rows, 'sdpa_isolated')
    results['isolation_changes_vs_cross_expected'] = compare(cross, isolated, False)
    del cross
    candidate = capture(model, rows, 'fa4_isolated')
    results['fa4_isolated_vs_dense_isolated'] = compare(isolated, candidate, True)
    del candidate
    fp32 = capture(model, rows, 'sdpa_isolated', bf16=False)
    results['dense_bf16_vs_fp32_isolated'] = compare(fp32, isolated, True)
    del isolated
    candidate = capture(model, rows, 'fa4_isolated')
    results['fa4_bf16_vs_fp32_isolated'] = compare(fp32, candidate, True)
    del fp32, candidate
    # End-to-end hidden-state isolation and gradient isolation at embedding output.
    isolation = {}
    for mode in ('sdpa_cross', 'sdpa_isolated', 'fa4_isolated'):
        batch = Collator(mode)(rows[:1])
        segments = torch.tensor(rows[0]['segments'], device='cuda')
        first = segments == segments[0]
        batch = {k:v.cuda() if torch.is_tensor(v) else v for k,v in batch.items()}
        model.zero_grad(set_to_none=True)
        holder = {}
        def embedding_hook(module, inp, out):
            out.retain_grad(); holder['embedding'] = out
        handle = model.backbone.model.embed_tokens.register_forward_hook(embedding_hook)
        try:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                hidden = model.hidden_states(context(batch, mode))[0]
                probe = hidden[:, ~first, :16].float().sum()/int((~first).sum())
            probe.backward()
        finally:
            handle.remove()
        forbidden = holder['embedding'].grad[:, first].abs().max().item()
        changed = {**batch, 'input_ids':batch['input_ids'].clone()}
        changed['input_ids'][:, first] = (changed['input_ids'][:, first]+17) % model.backbone.config.vocab_size
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            after = model.hidden_states(context(changed, mode))[0]
        difference = (hidden[:, ~first]-after[:, ~first]).abs().max().item()
        if mode.endswith('isolated'):
            assert difference == 0 and forbidden == 0, (mode, difference, forbidden)
        else:
            assert difference > 0 and forbidden > 0
        isolation[mode] = dict(later_hidden_max_change=difference, previous_document_activation_grad_max=forbidden)
    results['isolation'] = isolation
    write(root/'correctness.json', dict(status='passed', results=results,
          tolerances=dict(loss_absolute=.01, global_gradient_relative_l2=.03, output_relative_l2=.02)))
    print('FULL_MODEL_CORRECTNESS', json.dumps(results), flush=True)


class BenchTrainer(DeepKVTrainer):
    def __init__(self, *args, mode, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = mode
        self.model_accepts_loss_kwargs = True
        self.input_digest = hashlib.sha256()
        self.full_input_digest = hashlib.sha256()
        self.observed_rows = 0

    def _get_num_items_in_batch(self, batch_samples, device):
        # Unlike equal-size cross-document targets, document-isolated counts
        # can vary across ranks/GAS. Normalize by the actual global target count.
        if self.is_in_train:
            for batch in batch_samples:
                # Audit the entire training stream, not just its first update.
                self.full_input_digest.update(batch['input_ids'].detach().cpu().numpy().tobytes())
                self.full_input_digest.update(batch['segments'].detach().cpu().numpy().tobytes()
                                              if 'segments' in batch else b'')
                self.observed_rows += len(batch['input_ids'])
        count = sum(context(x, self.mode).targets().sum() for x in batch_samples).to(device)
        return self.accelerator.reduce(count, reduction='sum')

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.state.global_step == 0:
            self.input_digest.update(inputs['input_ids'].detach().cpu().numpy().tobytes())
        result = model(context(inputs, self.mode))
        count = result['lm_count'] if num_items_in_batch is None else num_items_in_batch
        loss = result['lm_sum']/count
        if num_items_in_batch is not None:
            loss = loss*self.accelerator.num_processes
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite benchmark loss')
        return (loss, result) if return_outputs else loss


class Timing(TrainerCallback):
    def __init__(self, steps):
        self.stop = steps; self.rows = []; self.previous = None

    def on_step_begin(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        if state.global_step == 5: torch.cuda.reset_peak_memory_stats()
        self.begin = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize(); end = time.perf_counter()
        self.rows.append(dict(step=state.global_step, compute_seconds=end-self.begin,
                             interval_seconds=None if self.previous is None else end-self.previous))
        self.previous = end
        if state.global_step >= self.stop: control.should_training_stop = True
        return control


def train_worker(args):
    from datasets import load_from_disk
    root = Path(args.root); meta = json.loads((root/'data.json').read_text())
    assert json.loads((root/'correctness.json').read_text())['status'] == 'passed'
    recipe = meta['recipe']; output = root/args.mode
    set_seed(42)
    training = TrainingArguments(output_dir=str(output), per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4, max_steps=28600, warmup_steps=1430,
        learning_rate=3e-4, lr_scheduler_type='cosine_with_min_lr',
        lr_scheduler_kwargs={'min_lr_rate':.1}, weight_decay=.1, adam_beta1=.9, adam_beta2=.95,
        max_grad_norm=1., bf16=True, seed=42, data_seed=42, report_to=[],
        dataloader_num_workers=4, dataloader_drop_last=True, logging_steps=1,
        save_strategy='no', eval_strategy='no', remove_unused_columns=False,
        ddp_find_unused_parameters=False, ddp_timeout=1800, disable_tqdm=True)
    assert training.world_size == 8
    training.distributed_state.wait_for_everyone()
    if output.exists():
        raise ValueError('Use a fresh benchmark mode directory')
    # All ranks finish the freshness check before Trainer can create this path.
    training.distributed_state.wait_for_everyone()
    model = new_model(recipe['config_name'], 'cpu')
    timing = Timing(args.steps)
    trainer = BenchTrainer(model=model, mode=args.mode, args=training,
        train_dataset=load_from_disk(str(root/'data')), data_collator=Collator(args.mode), callbacks=[timing],
        compute_metrics=compute_metrics)
    before = time.perf_counter()
    result = trainer.train()
    elapsed = time.perf_counter()-before
    assert trainer.state.global_step == args.steps
    peak = torch.cuda.max_memory_allocated()/1024**3
    evaluation = None
    if args.eval_rows:
        eval_data = load_from_disk(str(root/'eval'))
        assert len(eval_data) == args.eval_rows and args.mode.endswith('isolated')
        # Score both trained weight sets through the same dense backend first.
        trainer.mode = 'sdpa_isolated'; trainer.data_collator = Collator(trainer.mode)
        common = trainer.evaluate(eval_dataset=eval_data, metric_key_prefix='heldout')
        trainer.mode = args.mode; trainer.data_collator = Collator(args.mode)
        native = (trainer.evaluate(eval_dataset=eval_data, metric_key_prefix='heldout')
                  if args.mode != 'sdpa_isolated' else common)
        evaluation = dict(common_dense=common, native=native,
                          data=json.loads((root/'eval-data.json').read_text()))
    # Save actual trained weights and logs, outside the measured intervals.
    trainer.save_model(str(output/'final_model')); trainer.save_state()
    record = dict(status='passed', mode=args.mode, rank=training.process_index,
        world_size=training.world_size, tokens_per_update=1048576, steps=timing.rows,
        elapsed_seconds=elapsed, peak_allocated_gib=peak, evaluation=evaluation,
        full_input_sha256=trainer.full_input_digest.hexdigest(), observed_rows=trainer.observed_rows,
        input_sha256=trainer.input_digest.hexdigest(), log_history=trainer.state.log_history,
        train_metrics=result.metrics, wrapped_model=type(trainer.model_wrapped).__name__)
    write(output/f'rank-{training.process_index}.json', record)
    trainer.accelerator.wait_for_everyone()
    if training.process_index == 0: print('FULL_TRAINING_MODE_FINISHED', args.mode, flush=True)


def summarize(args):
    root = Path(args.root); summary = {}
    digests = None
    streams = None
    for mode in args.modes:
        ranks = [json.loads((root/mode/f'rank-{i}.json').read_text()) for i in range(8)]
        assert all(r['status']=='passed' and len(r['steps'])==args.steps for r in ranks)
        current = [r['input_sha256'] for r in ranks]
        assert digests is None or current == digests, 'Input order differs between modes'
        digests = current
        if len({m.endswith('isolated') for m in args.modes}) == 1:
            full = [r['full_input_sha256'] for r in ranks]
            assert streams is None or full == streams, 'Full training streams differ'
            streams = full
        assert all(r['observed_rows'] == args.steps*64 for r in ranks)
        elapsed = [max(r['steps'][i]['interval_seconds'] for r in ranks) for i in range(5, args.steps)]
        compute = [max(r['steps'][i]['compute_seconds'] for r in ranks) for i in range(5, args.steps)]
        losses = [x['loss'] for x in ranks[0]['log_history'] if 'loss' in x]
        assert len(losses) == args.steps and all(torch.isfinite(torch.tensor(losses)))
        norms = [x['grad_norm'] for x in ranks[0]['log_history'] if 'grad_norm' in x]
        assert len(norms) == args.steps and all(torch.isfinite(torch.tensor(norms)))
        summary[mode] = dict(median_update_seconds=statistics.median(elapsed),
            mean_update_seconds=statistics.mean(elapsed), min_update_seconds=min(elapsed),
            max_update_seconds=max(elapsed), median_compute_seconds=statistics.median(compute),
            tokens_per_second=1048576/statistics.median(elapsed),
            peak_allocated_gib=max(r['peak_allocated_gib'] for r in ranks), losses=losses,
            gradient_norms=norms, evaluation=ranks[0]['evaluation'])
    reference_mode = 'sdpa_cross' if 'sdpa_cross' in summary else args.modes[0]
    base = summary[reference_mode]['median_update_seconds']
    for row in summary.values(): row['time_change_vs_reference_percent']=100*(row['median_update_seconds']/base-1)
    if args.eval_rows:
        assert set(args.modes) == {'sdpa_isolated', 'fa4_isolated'}
        a, b = (summary[m] for m in ('sdpa_isolated', 'fa4_isolated'))
        assert a['evaluation']['data'] == b['evaluation']['data']
        assert a['evaluation']['common_dense']['heldout_target_tokens'] == b['evaluation']['common_dense']['heldout_target_tokens']
        comparison = dict(max_training_loss_gap=max(abs(x-y) for x,y in zip(a['losses'],b['losses'])),
            max_relative_gradient_norm_gap=max(abs(x-y)/max(abs(x),1e-20) for x,y in zip(a['gradient_norms'],b['gradient_norms'])),
            heldout_loss_gap=abs(a['evaluation']['common_dense']['heldout_lm_loss']-b['evaluation']['common_dense']['heldout_lm_loss']),
            trained_fa4_backend_eval_gap=abs(b['evaluation']['common_dense']['heldout_lm_loss']-b['evaluation']['native']['heldout_lm_loss']))
        comparison['passed'] = (comparison['max_training_loss_gap'] < .01 and
            comparison['max_relative_gradient_norm_gap'] < .03 and comparison['heldout_loss_gap'] < .01 and
            comparison['trained_fa4_backend_eval_gap'] < .01)
        write(root/'comparison.json', comparison)
        assert comparison['passed'], comparison
    write(root/'summary.json', dict(status='passed', results=summary, measured_steps=[6,args.steps],
          reference_mode=reference_mode,
          includes='full model forward, LM loss, backward, DDP, clipping, optimizer, scheduler, data delivery',
          excludes='startup, first five updates, evaluation, final checkpoint writing'))
    print('FULL_TRAINING_SUMMARY', json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare','correctness','train','summarize'])
    parser.add_argument('--root', required=True)
    parser.add_argument('--recipe', default='deep_kv.b200.json')
    parser.add_argument('--documents', type=int, default=24000)
    parser.add_argument('--mode', choices=MODES)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--modes', nargs='+', choices=MODES, default=list(MODES))
    parser.add_argument('--eval-rows', type=int, default=0)
    args = parser.parse_args()
    if args.steps < 10: raise ValueError('At least 10 updates are required')
    {'prepare':prepare, 'correctness':correctness, 'train':train_worker, 'summarize':summarize}[args.command](args)


if __name__ == '__main__': main()
