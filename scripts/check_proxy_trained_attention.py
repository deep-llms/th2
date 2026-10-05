"""Read-only same-weight backend checks on completed production Arm A models.

Prepare on CPU before reclaiming GPUs. Eight independent cases cover two
checkpoints, two data splits, and microbatches 2/16; micro2 also checks FP32 math.
No optimizer is constructed and source checkpoints are never written.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import Qwen3Config, AutoTokenizer

from deep_kv.proxy import ProxyModel
from deep_kv.proxy_training import ProxyTrainer
from deep_kv.packing import (document_end_id, group_texts, isolated_data_collator,
                             tokenize_with_segments)
from scripts.benchmark_document_training import compare, write
from scripts.check_trained_attention import acceptable, fingerprint, restore

CASES = (('train', 2), ('validation', 2), ('train', 16), ('validation', 16))


def read(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def model_from_run(arm):
    saved = read(arm / 'train_config.json')
    config = Qwen3Config.from_dict(saved['model_config'])
    config._attn_implementation = 'sdpa'
    config.use_cache = False
    model = ProxyModel.from_scratch(config, 'A', seed=saved['training']['seed'],
        checkpoint_layers=False, checkpoint_lm=False, checkpoint_aux=False,
        lm_chunk=saved['pilot']['lm_chunk'])
    restore(model, arm / 'checkpoint-2500/model.safetensors')
    return model


def prepare(args):
    from train import load_text
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    configs = [read(Path(arm) / 'train_config.json') for arm in (args.dense, args.fa4)]
    assert configs[1]['pilot'].pop('attention_backend') == 'fa4'
    assert configs[0] == configs[1], 'Training recipes differ beyond backend'
    manifest = dict(checkpoints=[], batches={})
    for arm in map(Path, (args.dense, args.fa4)):
        assert read(arm / 'result.json')['global_step'] == 2500
        assert read(arm / 'checkpoint-2500/trainer_state.json')['global_step'] == 2500
        model = model_from_run(arm)  # Strictly validate weights/aliases before GPU stop.
        manifest['checkpoints'].append(dict(arm=str(arm), parameter_sha256=fingerprint(model),
            file_sha256=file_hash(arm / 'checkpoint-2500/model.safetensors')))
        del model
    tokenizer = AutoTokenizer.from_pretrained(configs[0]['model']['tokenizer_name'], local_files_only=True)
    for split, key in [('train', 'data_dir'), ('validation', 'eval_data_dir')]:
        raw = load_text(configs[0]['data'][key])
        raw = raw.select(range(min(512, len(raw))))
        tokenized = raw.map(tokenize_with_segments, with_indices=True, batched=True,
            fn_kwargs=dict(tokenizer=tokenizer, end_id=document_end_id(tokenizer)),
            remove_columns=raw.column_names, cache_file_name=str(root / f'{split}-tokenized.arrow'))
        packed = tokenized.map(group_texts, batched=True, fn_kwargs=dict(block_size=2048),
            cache_file_name=str(root / f'{split}-packed.arrow'))
        assert len(packed) >= 16
        rows = list(packed.select(range(16)))
        assert all(len(r['input_ids']) == 2048 for r in rows)
        assert any(len(set(r['segments'])) > 1 for r in rows[:2])
        path = root / f'{split}.json'
        write(path, rows)
        manifest['batches'][split] = dict(sha256=file_hash(path), source=configs[0]['data'][key],
            source_documents=len(raw), packed_rows=16)
    manifest['status'] = 'prepared'
    write(root / 'manifest.json', manifest)
    print('PROXY_TRAINED_CHECK_PREPARED', json.dumps(manifest), flush=True)


def capture(model, inputs, backend, fp32=False):
    model.attention_backend = backend
    model.zero_grad(set_to_none=True)
    model.train()
    saved = {}
    def hook(module, args, out):
        saved['hidden'] = out.detach().float().cpu()
        saved['logits'] = model.backbone.lm_head(out[:, ::128]).detach().float().cpu()
    handle = model.backbone.model.norm.register_forward_hook(hook)
    try:
        with sdpa_kernel(SDPBackend.MATH) if fp32 else nullcontext():
            with nullcontext() if fp32 else torch.autocast('cuda', dtype=torch.bfloat16):
                result = model(ProxyTrainer.context(inputs))
                loss = result['lm_sum'] / result['lm_count']
            loss.backward()
    finally:
        handle.remove()
    gradients = {name: p.grad.detach().float().cpu().clone() for name, p in model.named_parameters()}
    assert all(torch.isfinite(g).all() for g in gradients.values())
    assert torch.isfinite(loss) and all(torch.isfinite(x).all() for x in saved.values())
    return dict(loss=float(loss.detach()), targets=int(result['lm_count']), grads=gradients, **saved)


def worker(args):
    from deep_kv.fa4 import load_kernel
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source, output = Path(args.source), Path(args.output)
    manifest = read(source / 'manifest.json')
    checkpoint = manifest['checkpoints'][args.index // 4]
    split, batch = CASES[args.index % 4]
    assert file_hash(source / f'{split}.json') == manifest['batches'][split]['sha256']
    model = model_from_run(Path(checkpoint['arm']))
    initial = fingerprint(model)
    assert initial == checkpoint['parameter_sha256']
    model.cuda()
    model.fa4_kernel, kernel = load_kernel()
    assert model.backbone.config.attention_dropout == 0
    rows = read(source / f'{split}.json')[:batch]
    inputs = {k: v.cuda() for k, v in isolated_data_collator(rows).items()}
    digest = hashlib.sha256()
    for key in sorted(inputs):
        digest.update(key.encode()); digest.update(inputs[key].cpu().numpy().tobytes())
    result = dict(index=args.index, checkpoint=checkpoint, split=split, batch=batch,
                  input_sha256=digest.hexdigest(), kernel=kernel, comparisons={})
    def take(backend, fp32=False):
        print('CAPTURE', args.index, backend, 'fp32' if fp32 else 'bf16', flush=True)
        return capture(model, inputs, backend, fp32)
    dense, flash = take('sdpa'), take('fa4')
    result['comparisons']['fa4_vs_dense'] = compare(dense, flash, False)
    repeat = take('sdpa')
    result['comparisons']['dense_repeat'] = compare(dense, repeat, False)
    del repeat
    repeat = take('fa4')
    result['comparisons']['fa4_repeat'] = compare(flash, repeat, False)
    del repeat
    if batch == 2:
        reference = take('sdpa', True)
        result['comparisons']['dense_vs_fp32_math'] = compare(reference, dense, False)
        result['comparisons']['fa4_vs_fp32_math'] = compare(reference, flash, False)
        del reference
    del dense, flash
    result['parameters_unchanged'] = fingerprint(model) == initial
    result['peak_allocated_gib'] = torch.cuda.max_memory_allocated() / 1024**3
    result['backend_gate_passed'] = acceptable(result['comparisons']['fa4_vs_dense'])
    result['reference_gates_passed'] = (all(acceptable(result['comparisons'][key]) for key in
        ('dense_vs_fp32_math', 'fa4_vs_fp32_math')) if batch == 2 else None)
    result['status'] = 'completed'
    write(output, result)
    assert result['parameters_unchanged']
    print('SAME_WEIGHT_RESULT', json.dumps(result), flush=True)


def run(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    visible = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    assert len(visible) == 8
    children, logs = [], []
    try:
        for index, gpu in enumerate(visible):
            log = (root / f'case-{index}.log').open('x'); logs.append(log)
            command = [sys.executable, '-u', '-m', 'scripts.check_proxy_trained_attention', 'worker',
                '--source', args.source, '--output', str(root / f'case-{index}.json'), '--index', str(index)]
            children.append(subprocess.Popen(command, env={**os.environ, 'CUDA_VISIBLE_DEVICES': gpu},
                stdout=log, stderr=subprocess.STDOUT))
        codes = [child.wait() for child in children]
        assert codes == [0]*8, codes
    finally:
        # Only children created here; outer supervisor also owns orphan cleanup.
        for child in children:
            if child.poll() is None: child.terminate()
        for child in children:
            try: child.wait(timeout=20)
            except subprocess.TimeoutExpired: child.kill(); child.wait()
        for log in logs: log.close()
    rows = [read(root / f'case-{i}.json') for i in range(8)]
    manifest = read(Path(args.source) / 'manifest.json')
    for i, row in enumerate(rows):
        assert row['index'] == i and row['parameters_unchanged'] and row['status'] == 'completed'
        assert (row['split'], row['batch']) == CASES[i % 4]
        assert row['checkpoint'] == manifest['checkpoints'][i // 4]
        assert row['input_sha256'] == rows[(i+4) % 8]['input_sha256']
    for checkpoint in manifest['checkpoints']:
        assert file_hash(Path(checkpoint['arm']) / 'checkpoint-2500/model.safetensors') == checkpoint['file_sha256']
    passed = all(r['backend_gate_passed'] and r['reference_gates_passed'] is not False for r in rows)
    result = dict(status='passed' if passed else 'failed', checkpoint_files_unchanged=True,
        tolerances=dict(loss_absolute=.01, gradient_relative_l2=.03, output_relative_l2=.02), cases=rows)
    write(root / 'summary.json', result)
    print('FINAL_CHECK_SUMMARY', json.dumps(result), flush=True)
    assert passed, 'Numerical gate failed; complete evidence saved'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'worker', 'run'])
    parser.add_argument('--dense'); parser.add_argument('--fa4'); parser.add_argument('--source')
    parser.add_argument('--output', required=True)
    parser.add_argument('--index', type=int, choices=range(8))
    args = parser.parse_args()
    if args.command == 'prepare' and not (args.dense and args.fa4): parser.error('prepare needs both arms')
    if args.command != 'prepare' and not args.source: parser.error('source is required')
    if args.command == 'worker' and args.index is None: parser.error('worker needs index')
    globals()[args.command](args)


if __name__ == '__main__':
    main()
