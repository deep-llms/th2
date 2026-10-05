"""Evaluate saved proxy heads at 1x/2x/5x gates without training or saving weights."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3Config, TrainingArguments, TrainerState, set_seed

from deep_kv.packing import isolated_data_collator, preprocess_dataset
from deep_kv.proxy import ProxyModel, ProxySettings
from deep_kv.proxy_training import ProxyTrainer
from scripts.check_trained_attention import restore
from train import load_text


def read(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def scaled_gates(model, multiplier):
    if not math.isfinite(multiplier) or multiplier < 0:
        raise ValueError('Gate multiplier must be finite and nonnegative')
    if model.training or torch.is_grad_enabled():
        raise ValueError('Gate scaling is evaluation-only; use eval() and no_grad()')
    original = {name: head.alpha.detach().clone() for name, head in model.heads.items()}
    try:
        for name, head in model.heads.items():
            head.alpha.copy_(original[name] * multiplier)
        yield
    finally:
        for name, head in model.heads.items():
            head.alpha.copy_(original[name])


def run_sweep(arm_dir, output, step=2500):
    arm_dir, output = Path(arm_dir), Path(output)
    saved, result = read(arm_dir/'train_config.json'), read(arm_dir/'result.json')
    pilot = saved['pilot']
    if pilot['arm'] not in ('P1-block', 'P3-block') or result['global_step'] != step:
        raise ValueError('Expected a completed P1/P3-block checkpoint at the requested step')
    checkpoint = arm_dir/f'checkpoint-{step}'
    state = TrainerState.load_from_json(str(checkpoint/'trainer_state.json'))
    if state.global_step != step or state.max_steps != saved['training']['max_steps']:
        raise ValueError('Checkpoint step/schedule mismatch')
    if output.exists() or output.resolve().is_relative_to(arm_dir.resolve()):
        raise ValueError('Use a fresh output outside the source arm')
    args = TrainingArguments(**{**saved['training'], 'output_dir': str(output),
        'report_to': [], 'run_name': 'proxy-gate-evaluation', 'disable_tqdm': True})
    if args.world_size != saved['world_size']:
        raise ValueError('Evaluation must use the original world size')
    if args.should_save:
        output.mkdir(parents=True, exist_ok=False)
    args.distributed_state.wait_for_everyone()
    set_seed(args.seed)
    source = checkpoint/'model.safetensors'
    source_hash = file_hash(source) if args.should_save else None
    config = Qwen3Config.from_dict(saved['model_config'])
    config._attn_implementation = 'sdpa'
    config.use_cache = False
    settings = ProxySettings(**{key: (pilot.get('proxy_alpha_init', 0.) if key == 'alpha_init'
                                     else pilot['proxy_'+key]) for key in ProxySettings.__dataclass_fields__})
    model = ProxyModel.from_scratch(config, pilot['arm'], seed=args.seed,
        proxy_settings=settings, sequence_length=saved['data']['block_size'],
        attention_backend=pilot.get('attention_backend', 'sdpa'),
        **{key: pilot[key] for key in ('consumer','deep_target','lm_chunk','checkpoint_layers',
                                     'checkpoint_lm','checkpoint_aux','causal_attention')})
    restore(model, source)
    model.eval()
    initial = state_hash(model) if args.should_save else None
    tokenizer = AutoTokenizer.from_pretrained(saved['model']['tokenizer_name'], local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    data = preprocess_dataset(load_text(saved['data']['eval_data_dir']), tokenizer,
        saved['data']['block_size'], args, num_proc=1,
        isolate_documents=saved['data']['isolate_documents']).select(range(saved['data']['eval_rows']))
    if data._fingerprint != saved['eval_fingerprint']:
        raise ValueError('Validation data fingerprint differs from original run')
    trainer = ProxyTrainer(model=model, args=args, eval_dataset=data,
        processing_class=tokenizer, data_collator=isolated_data_collator)
    trainer.state = state
    measurements = []
    with torch.no_grad():
        for index, multiplier in enumerate((1, 2, 5, 1)):
            prefix = f'gate_{index}'
            with scaled_gates(model, multiplier):
                metrics = trainer.evaluate(metric_key_prefix=prefix)
            values = dict(multiplier=multiplier, lm_loss=metrics[prefix+'_lm_loss'],
                no_proxy_lm_loss=metrics[prefix+'_no_proxy_lm_loss'],
                target_tokens=metrics[prefix+'_target_tokens'], rows=metrics[prefix+'_rows'],
                auxiliary_loss=metrics[prefix+'_aux_loss'])
            if not all(math.isfinite(values[key]) for key in ('lm_loss','no_proxy_lm_loss','auxiliary_loss')):
                raise ValueError('Nonfinite evaluation metric')
            if values['rows'] != result['evaluation']['eval_rows'] or values['target_tokens'] != result['evaluation']['eval_target_tokens']:
                raise ValueError('Evaluation token/row counts differ')
            measurements.append(values)
            if args.should_save:
                print('GATE_RESULT', json.dumps(values), flush=True)
    # The repeated 1x and 0x passes separate scaling effects from execution noise.
    tolerance = 1e-5
    if abs(measurements[0]['lm_loss']-result['evaluation']['eval_lm_loss']) > tolerance:
        raise ValueError('1x control does not reproduce the saved evaluation')
    if abs(measurements[0]['lm_loss']-measurements[-1]['lm_loss']) > tolerance:
        raise ValueError('Restored 1x control changed')
    if any(abs(row['no_proxy_lm_loss']-result['evaluation']['eval_no_proxy_lm_loss']) > tolerance for row in measurements):
        raise ValueError('Proxy-disabled controls changed')
    if trainer.optimizer is not None or trainer.lr_scheduler is not None:
        raise ValueError('Unexpected optimizer/scheduler in evaluation-only diagnostic')
    if args.should_save:
        if initial != state_hash(model) or source_hash != file_hash(source):
            raise ValueError('Model state or source checkpoint changed')
        receipt = dict(status='passed', arm=pilot['arm'], checkpoint=str(checkpoint), step=step,
            checkpoint_sha256=source_hash, model_state_unchanged=True, checkpoint_unchanged=True,
            eval_fingerprint=data._fingerprint, world_size=args.world_size,
            attention_runtime=model.attention_runtime, original_lm_loss=result['evaluation']['eval_lm_loss'],
            repeat_tolerance=tolerance, measurements=measurements,
            fa4_receipts=getattr(trainer, '_fa4_receipts', []))
        with (output/'summary.json').open('x') as handle:
            json.dump(receipt, handle, indent=2, allow_nan=False)
    trainer.accelerator.wait_for_everyone()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--step', type=int, default=2500)
    cli = parser.parse_args()
    run_sweep(cli.arm_dir, cli.output, cli.step)
