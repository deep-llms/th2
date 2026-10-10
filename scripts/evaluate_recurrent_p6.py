"""Distributed, read-only PPL diagnostic for past-token real-target P6 memory."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from accelerate import Accelerator
from transformers import AutoTokenizer, TrainingArguments

from deep_kv.model import Context
from deep_kv.packing import isolated_data_collator, preprocess_dataset
from eval.models import load_checkpoint, file_hash
from eval.recurrent_p6 import recurrent_hidden
from scripts.evaluate_proxy_gates import state_hash
from train import load_text


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def evaluation_runtime(saved_training, output):
    # HF TrainingArguments initializes/resets distributed state while setting
    # up its device. Complete that setup BEFORE constructing Accelerator.
    training_args = TrainingArguments(**{**saved_training, 'output_dir':str(output), 'report_to':[]})
    _ = training_args.device
    accelerator = Accelerator(mixed_precision='bf16')
    return training_args, accelerator


def run(args):
    checkpoint = Path(args.checkpoint)
    output = Path(args.output)
    saved = json.loads((checkpoint.parent/'train_config.json').read_text())
    training_args, accelerator = evaluation_runtime(saved['training'], output)
    if accelerator.num_processes != 8 or accelerator.device.type != 'cuda':
        raise ValueError('Run the real checkpoint diagnostic on all eight B200 GPUs')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=False)
    accelerator.wait_for_everyone()
    model, _, metadata = load_checkpoint(checkpoint, accelerator.device, attention_backend='fa4')
    if metadata['arm'] != 'P6-iso' or metadata['step'] != 10000:
        raise ValueError('Expected P6-iso checkpoint-10000')
    if metadata['checkpoint_sha256'] != args.checkpoint_sha256:
        raise ValueError('Source checkpoint identity changed')
    m = model.wrapped
    before = state_hash(m)
    tokenizer = AutoTokenizer.from_pretrained(saved['model']['tokenizer_name'], local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Use the original train.py validation packing and its shared HF cache.
    dataset = preprocess_dataset(load_text(saved['data']['eval_data_dir']), tokenizer,
        saved['data']['block_size'], training_args, num_proc=1,
        isolate_documents=saved['data']['isolate_documents']).select(range(saved['data']['eval_rows']))
    if dataset._fingerprint != saved['eval_fingerprint']:
        raise ValueError('Validation fingerprint differs from the trained run')
    if not 8 <= args.rows <= len(dataset) or args.rows % 8 or args.batch_size < 1:
        raise ValueError('Use at least eight rows, divisible by eight, within validation')
    if not 2 <= args.tokens <= saved['data']['block_size']:
        raise ValueError('Invalid evaluation length')
    # Deterministic spread through the original validation rows, not a new split.
    selected = [i*len(dataset)//args.rows for i in range(args.rows)]
    signature = hashlib.sha256()
    for index in selected:
        signature.update(json.dumps([index,dataset[index]], sort_keys=True).encode())
    local = selected[accelerator.process_index::accelerator.num_processes]
    records, numeric = [], []
    modes = ('parallel_fa4', 'native_proxy', 'past_proxy', 'past_real')
    for begin in range(0, len(local), args.batch_size):
        indices = local[begin:begin+args.batch_size]
        rows = [{k:v[:args.tokens] for k,v in dataset[i].items()} for i in indices]
        inputs = {k:v.to(accelerator.device) for k,v in isolated_data_collator(rows).items()}
        ctx = Context(inputs['input_ids'], inputs['attention_mask'].bool(),
                      inputs['position_ids'], inputs['segments'], inputs['labels'])
        timings = {}
        statistics = {}
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            # Small same-weight SDPA control separates cache bugs from backend
            # differences before the full-length experimental recurrence.
            if begin == 0:
                n = min(64, args.tokens)
                short = Context(ctx.input_ids[:, :n],ctx.valid[:, :n],ctx.position_ids[:, :n],
                                ctx.segments[:, :n],ctx.labels[:, :n])
                m.attention_backend = 'sdpa'
                reference = m.hidden_states(short)[0]
                sequential = recurrent_hidden(m, short, 'native_proxy')
                relative = float((sequential.float()-reference.float()).norm()/reference.float().norm().clamp_min(1e-8))
                loss1, counts, _ = m.lm_statistics(short, reference)
                loss2, _, _ = m.lm_statistics(short, sequential)
                gap = abs(float((loss1.sum()-loss2.sum())/counts.sum()))
                if relative > .02 or gap > .01:
                    raise ValueError(f'Sequential SDPA control mismatch: {relative=} {gap=}')
                numeric.append(dict(rank=accelerator.process_index, hidden_relative_l2=relative, loss_gap=gap))
                m.attention_backend = 'fa4'
            for mode in modes:
                torch.cuda.synchronize(); started = time.perf_counter()
                progress = (lambda done,total: print('TOKEN_PROGRESS', mode, done, '/', total, flush=True)) if accelerator.is_main_process else None
                hidden = (m.hidden_states(ctx)[0] if mode == 'parallel_fa4' else
                          recurrent_hidden(m, ctx, mode, progress=progress))
                loss, counts, tokens = m.lm_statistics(ctx, hidden)
                if not bool(torch.isfinite(loss).all()):
                    raise ValueError(f'Nonfinite loss in {mode}')
                statistics[mode] = loss.double().cpu().tolist()
                torch.cuda.synchronize(); timings[mode] = time.perf_counter()-started
                if accelerator.is_main_process:
                    print('MODE_DONE', mode, 'batch', begin//args.batch_size,
                          'seconds', timings[mode], 'loss', float(loss.sum()/counts.sum()), flush=True)
            for j,index in enumerate(indices):
                records.append(dict(row=index, target_tokens=int(counts[j]), input_tokens=int(tokens[j]),
                                    nll_sum={mode:statistics[mode][j] for mode in modes}))
        write(output/f'rank-{accelerator.process_index}-batch-{begin//args.batch_size}.json',
              dict(timings=timings, rows=records[-len(indices):]))
    if before != state_hash(m):
        raise ValueError('Evaluation modified model weights or normalization buffers')
    write(output/f'rank-{accelerator.process_index}.json', dict(status='passed', rows=records,
        numerical_checks=numeric, selected_rows_sha256=signature.hexdigest(), model_state_unchanged=True))
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        reports = [json.loads((output/f'rank-{i}.json').read_text()) for i in range(8)]
        all_rows = sorted([row for report in reports for row in report['rows']], key=lambda row:row['row'])
        if [row['row'] for row in all_rows] != selected:
            raise ValueError('Missing, repeated or reordered evaluation rows')
        if any(r['selected_rows_sha256'] != signature.hexdigest() for r in reports):
            raise ValueError('Ranks used different validation selections')
        if file_hash(checkpoint/'model.safetensors') != metadata['checkpoint_sha256']:
            raise ValueError('Source weights changed')
        target_tokens = sum(r['target_tokens'] for r in all_rows)
        scores = {}
        for mode in modes:
            loss = sum(r['nll_sum'][mode] for r in all_rows)/target_tokens
            scores[mode] = dict(lm_loss=loss, ppl=math.exp(loss))
        gap = abs(scores['native_proxy']['lm_loss']-scores['parallel_fa4']['lm_loss'])
        if gap > .01:
            raise ValueError(f'Native sequential vs parallel FA4 control mismatch: {gap}')
        summary = dict(status='passed', checkpoint=metadata, eval_fingerprint=dataset._fingerprint,
            selected_rows=selected, selected_rows_sha256=signature.hexdigest(), rows=len(all_rows),
            sequence_length=args.tokens, target_tokens=target_tokens, scores=scores,
            current_token_proxy_in_experiment=False, real_targets='sum of current and next three MLP outputs of the SAME recurrent pass',
            memory='rebuild only completed token K/V from saved pre-injection residual and normalized target; no replay',
            attention='cached SDPA with explicit same-document mask; parallel control uses original FA4',
            model_state_unchanged=True, checkpoint_unchanged=True,
            numerical_checks=[x for r in reports for x in r['numerical_checks']],
            caveat='Inference-rule intervention; not an oracle bound or an isolated measure of predictor quality.')
        write(output/'summary.json', summary)
        print('RECURRENT_P6_SUMMARY', json.dumps(summary), flush=True)
    accelerator.wait_for_everyone()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--rows', type=int, default=128)
    parser.add_argument('--tokens', type=int, default=2048)
    parser.add_argument('--batch-size', type=int, default=16)
    run(parser.parse_args())
