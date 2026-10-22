"""Bounded real-CUDA workflow check. Requires explicitly allocated, free GPUs.

Run with --output NEW_DIR --backend sdpa|fa4 --world-size 1|2.
No downloads, installations, or process reclamation. Synthetic data only.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def run(argv, log):
    with log.open('x') as handle:
        subprocess.run(argv, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                       check=True, timeout=600, env=os.environ.copy())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--backend', choices=('sdpa', 'fa4', 'fa2'), required=True)
    p.add_argument('--world-size', type=int, choices=(1, 2, 8), required=True)
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      WANDB_MODE='offline', OMP_NUM_THREADS='1', NCCL_NVLS_ENABLE='0')
    import torch
    from datasets import Dataset
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen3Config
    from safetensors.torch import load_file
    assets = out / 'assets'
    assets.mkdir()
    vocab = {'[PAD]': 0, '[UNK]': 1, '[EOS]': 2, **{f't{i}': i for i in range(3, 128)}}
    tok = Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
    tok.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tok, unk_token='[UNK]', pad_token='[PAD]',
        eos_token='[EOS]', model_max_length=128).save_pretrained(assets)
    Qwen3Config(vocab_size=128, hidden_size=256, intermediate_size=512,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=64,
        max_position_embeddings=128, bos_token_id=None, eos_token_id=2, pad_token_id=0,
        attention_dropout=0.1 if args.backend == 'sdpa' else 0.).save_pretrained(assets)
    for split, shift in [('train', 0), ('validation', 19)]:
        texts = [' '.join(f't{3 + (i*11+j*7+shift)%125}' for j in range(23+i%71)) for i in range(80)]
        Dataset.from_dict({'text': texts}).to_parquet(out / f'{split}.parquet')
    accel = out / 'accelerate.yaml'
    accel.write_text(json.dumps(dict(compute_environment='LOCAL_MACHINE',
        distributed_type='NO' if args.world_size == 1 else 'MULTI_GPU',
        mixed_precision='bf16', num_processes=args.world_size, num_machines=1,
        machine_rank=0, main_training_function='main', use_cpu=False, rdzv_backend='static', same_network=True)))
    launch = ['{python}', '-m', 'accelerate.commands.launch', '--config_file', str(accel),
              '--main_process_port', '0']
    substitutions = {'/absolute/model/config.json': str(assets/'config.json'),
        '/absolute/tokenizer': str(assets), '/absolute/data/train': str(out/'train.parquet'),
        '/absolute/data/validation': str(out/'validation.parquet')}
    config = json.loads((ROOT/'jobs.training.example.json').read_text())
    for job in config['jobs']:
        a = [substitutions.get(v, v) for v in job['argv']]
        if job['name'] in ('train', 'resume', 'evaluate'):
            job['gpus'] = list(range(args.world_size))
            a[a.index('--use_cpu')+1] = 'false'
            a = launch + a[1:]
            if job['name'] == 'evaluate':
                a[a.index('--precision')+1] = 'bf16'
                a += ['--max_sequences', '7']
            else:
                a[a.index('--attention_backend')+1] = args.backend
                a[a.index('--block_size')+1] = '64'
                a += ['--bf16', 'true', '--tf32', 'false', '--gradient_checkpointing', 'true', '--eval_max_sequences', '7']
        job['argv'] = a
        job['timeout_seconds'] = 600
    manifest = out/'jobs.json'
    manifest.write_text(json.dumps(config, indent=2))
    queue = out/'queue'
    run([sys.executable, str(ROOT/'run_experiments.py'), '--config', str(manifest),
         '--project-dir', str(ROOT), '--run-dir', str(queue)], out/'workflow.log')
    # Same schedule, data, seed and rank count, but uninterrupted to step four.
    command = [v.replace('{python}', sys.executable).replace('{project_dir}', str(ROOT))
        .replace('{run_dir}', str(out/'reference')) for v in config['jobs'][0]['argv']]
    command[command.index('--stop_after')+1] = '4'
    run(command, out/'reference.log')
    resumed = queue/'training/checkpoint-4'
    reference = out/'reference/training/checkpoint-4'
    max_abs = {}
    def compare(a, b, name):
        if isinstance(a, torch.Tensor):
            if a.dtype.is_floating_point:
                if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                    raise AssertionError(f'Nonfinite {name}')
                max_abs[name.split('/')[0]] = max(max_abs.get(name.split('/')[0], 0.), float((a-b).abs().max()) if a.numel() else 0.)
                # Fixed smoke tolerance; record actual errors, including exact matches.
                torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6, msg=name)
            else:
                assert torch.equal(a, b), name
        elif isinstance(a, dict):
            assert a.keys() == b.keys(), name
            for k in a: compare(a[k], b[k], name+'/'+str(k))
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b), name
            for i, (x, y) in enumerate(zip(a, b)): compare(x, y, name+'/'+str(i))
        else:
            import numpy as np
            if isinstance(a, np.ndarray):
                assert np.array_equal(a, b), name
            else: assert a == b, name
    compare(load_file(resumed/'model.safetensors'), load_file(reference/'model.safetensors'), 'weights')
    for name in ('optimizer.pt', 'scheduler.pt'):
        compare(torch.load(resumed/name, map_location='cpu', weights_only=False),
                torch.load(reference/name, map_location='cpu', weights_only=False), name)
    rng_names = ['rng_state.pth'] if args.world_size == 1 else [f'rng_state_{i}.pth' for i in range(args.world_size)]
    for name in rng_names:
        compare(torch.load(resumed/name, map_location='cpu', weights_only=False),
                torch.load(reference/name, map_location='cpu', weights_only=False), name)
    state = json.loads((resumed/'trainer_state.json').read_text())
    grads = [v['grad_norm'] for v in state['log_history'] if 'grad_norm' in v]
    assert grads and all(math.isfinite(v) and v > 0 for v in grads), grads
    summary = json.loads((queue/'summary.json').read_text())
    evaluation = summary['evaluation']
    assert abs(evaluation['loss']-summary['stages'][-1]['validation']['loss']) < 1e-5, summary
    result = dict(status='passed', backend=args.backend, world_size=args.world_size,
        steps=4, schedule_steps=8, max_absolute_errors=max_abs, gradient_norms=grads,
        evaluation=evaluation, note='Tiny synthetic CUDA smoke; not long-run/throughput validation')
    (out/'complete.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
