"""Validate template training artifacts and summarize a sequential continuation.

Standard library only. Checks completeness/metadata, not tensor contents; native
HF resume and the real evaluator still load the weights and optimizer state.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.verify_manifest import resolve_file


def read_json(path):
    def invalid(value):
        raise ValueError(f"Nonfinite JSON value in {path}: {value}")
    value = json.loads(Path(path).read_text(), parse_constant=invalid)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def number(value, name, minimum=0, integer=False):
    if (type(value) not in ((int,) if integer else (int, float))
            or not math.isfinite(value) or value < minimum):
        raise ValueError(f"Invalid {name}: {value!r}")
    return value


def check_metric(data, prefix=''):
    loss = number(data[prefix + 'loss'], 'loss')
    ppl = number(data[prefix + 'perplexity'], 'perplexity', 1)
    if not math.isclose(math.log(ppl), loss, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError('Loss and perplexity disagree')
    tokens = number(data[prefix + 'num_tokens'], 'num_tokens', 1, integer=True)
    sequences = number(data[prefix + 'num_sequences'], 'num_sequences', 1, integer=True)
    nll = number(data[prefix + 'nll_sum'], 'nll_sum')
    if not math.isclose(nll / tokens, loss, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError('NLL/token count and loss disagree')
    return dict(loss=loss, perplexity=ppl, num_tokens=tokens, num_sequences=sequences)


def file_records(checkpoint, names):
    records = []
    for name in sorted(set(names)):
        path = resolve_file(checkpoint, name)
        stat = path.stat()
        if stat.st_size == 0:
            raise ValueError(f'Empty checkpoint file: {name}')
        item = dict(path=name, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
        if path.suffix == '.json':
            item['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append(item)
    return records


def weight_names(checkpoint):
    for single, index in [('model.safetensors', 'model.safetensors.index.json'),
                          ('pytorch_model.bin', 'pytorch_model.bin.index.json')]:
        if (checkpoint / index).exists():
            mapping = read_json(resolve_file(checkpoint, index))['weight_map']
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError('Empty/invalid checkpoint weight index')
            return [index, *sorted(set(mapping.values()))]
        if (checkpoint / single).exists():
            return [single]
    raise ValueError('Checkpoint has no model weights or shard index')


def check_training(output_dir, step, schedule, start_step=0):
    output_dir = Path(output_dir).resolve()
    number(step, 'expected step', 1, integer=True)
    number(schedule, 'schedule', step, integer=True)
    number(start_step, 'start step', 0, integer=True)
    if start_step >= step:
        raise ValueError('Start step must precede final step')
    checkpoint = output_dir / f'checkpoint-{step}'
    if checkpoint.resolve().parent != output_dir:
        raise ValueError('Checkpoint escapes output directory')
    state = read_json(resolve_file(checkpoint, 'trainer_state.json'))
    config = read_json(resolve_file(checkpoint, 'train_config.json'))
    recipe = read_json(resolve_file(checkpoint, 'training_recipe.json'))
    data = read_json(resolve_file(checkpoint, 'data_recipe.json'))
    metrics = read_json(output_dir / 'train_results.json')
    if (state['global_step'] != step or metrics['global_step'] != step
            or state['max_steps'] != schedule or metrics['schedule_steps'] != schedule
            or config['training']['max_steps'] != schedule
            or config['training'].get('stop_after') != step
            or recipe['training']['max_steps'] != schedule):
        raise ValueError('Training step/full schedule/cutoff does not match expected values')
    for name, saved in [('training_recipe.json', recipe), ('data_recipe.json', data),
                        ('train_config.json', config)]:
        if read_json(output_dir / name) != saved:
            raise ValueError(f'Run root and checkpoint disagree: {name}')
    if read_json(output_dir / 'trainer_state.json')['global_step'] != step:
        raise ValueError('Run root trainer state has a different step')
    resumed_from = config['training'].get('resume_from_checkpoint')
    if start_step:
        if not resumed_from:
            raise ValueError('Continuation requires an explicit resume_from_checkpoint')
        source = Path(resumed_from).resolve()
        if source != output_dir / f'checkpoint-{start_step}':
            raise ValueError('Resume source does not match expected preceding checkpoint')
        if read_json(source / 'trainer_state.json')['global_step'] != start_step:
            raise ValueError('Resume checkpoint has a different starting step')
    elif resumed_from:
        raise ValueError('Fresh stage unexpectedly resumed a checkpoint')
    execution = recipe['execution']
    world = number(execution['world_size'], 'world size', 1, integer=True)
    if state['train_batch_size'] != execution['train_batch_size_per_process']:
        raise ValueError('Saved effective batch size disagrees with training recipe')
    names = ['trainer_state.json', 'train_config.json', 'training_recipe.json',
             'data_recipe.json', 'config.json', 'tokenizer.json', 'tokenizer_config.json',
             'optimizer.pt', 'scheduler.pt', *weight_names(checkpoint)]
    names += ['rng_state.pth'] if world == 1 else [f'rng_state_{rank}.pth' for rank in range(world)]
    if execution['device_type'] == 'cuda' and execution['mixed_precision'] == 'fp16':
        names.append('scaler.pt')
    records = file_records(checkpoint, names)
    validation = None
    for entry in state['log_history']:
        for key in ('loss', 'train_loss', 'eval_loss'):
            if key in entry:
                number(entry[key], key)
        if 'eval_loss' in entry:
            validation = dict(step=number(entry['step'], 'validation step', integer=True),
                              **check_metric(entry, 'eval_'))
            if validation['step'] > step:
                raise ValueError('Validation is from a future step')
    return dict(status='ok', kind='training_stage', schema=1,
        checked_at=datetime.now(timezone.utc).isoformat(), checkpoint=str(checkpoint),
        output_dir=str(output_dir), start_step=start_step, global_step=step, schedule_steps=schedule,
        stage_train_seconds=number(metrics['train_runtime'], 'train_runtime'),
        train_loss=number(metrics['train_loss'], 'train_loss'), validation=validation,
        training_recipe_sha256=digest(recipe), data_recipe_sha256=digest(data),
        files=records, weights=weight_names(checkpoint),
        validation_scope='File completeness and metadata; tensors are loaded by HF resume/evaluation')


def summarize(reports, evaluation):
    stages = []
    previous = None
    for path in reports:
        stage = read_json(path)
        if (stage.get('status'), stage.get('kind'), stage.get('schema')) != ('ok', 'training_stage', 1):
            raise ValueError('Invalid training-stage report')
        checkpoint = Path(stage['checkpoint'])
        if file_records(checkpoint, [r['path'] for r in stage['files']]) != stage['files']:
            raise ValueError('Checkpoint changed after stage verification')
        if previous is None:
            if stage['start_step'] != 0:
                raise ValueError('Summary must start with the fresh stage')
        elif (stage['start_step'] != previous['global_step']
              or stage['output_dir'] != previous['output_dir']
              or stage['schedule_steps'] != previous['schedule_steps']
              or stage['training_recipe_sha256'] != previous['training_recipe_sha256']
              or stage['data_recipe_sha256'] != previous['data_recipe_sha256']):
            raise ValueError('Reports are not a contiguous continuation of the same recipe')
        stages.append(stage)
        previous = stage
    if not stages:
        raise ValueError('At least one training-stage report is required')
    result = read_json(evaluation)
    final = stages[-1]
    if (result.get('status') != 'ok' or result['global_step'] != final['global_step']
            or Path(result['checkpoint']).resolve() != Path(final['checkpoint']).resolve()
            or result['training_recipe_sha256'] != final['data_recipe_sha256']):
        # Historical evaluator field name refers to data_recipe.json, not training_recipe.json.
        raise ValueError('Evaluation does not belong to the verified final checkpoint/data recipe')
    checkpoint = Path(final['checkpoint'])
    validation = read_json(checkpoint / 'training_recipe.json').get('validation')
    if validation is not None:
        # Tokenizer file paths change when HF saves a checkpoint; compare actual
        # tokenizer/data/packing identities rather than those storage locations.
        keys = ('policy', 'sources', 'raw_fingerprint', 'tokenizer',
                'block_size', 'map_batch_size', 'requested_workers', 'effective_workers',
                'shuffle_seed', 'add_special_tokens', 'append_boundary', 'remainder',
                'attention', 'positions', 'labels')
        if any(result['data_recipe'][key] != validation['data_recipe'][key] for key in keys):
            raise ValueError('Evaluation data/packing differs from monitored validation')
        # Standalone scoring passes the saved EOS explicitly; training may have
        # selected the same token through tokenizer.eos_token.
        if any(result['data_recipe']['document_end'][key] != validation['data_recipe']['document_end'][key]
               for key in ('token', 'id')):
            raise ValueError('Evaluation document-end token differs from monitored validation')
        if any(result['selection'][key] != validation['selection'][key]
               for key in ('full_sequences', 'max_sequences', 'shuffle_seed')):
            raise ValueError('Evaluation selection differs from monitored validation')
    if result['config_sha256'] != hashlib.sha256((checkpoint / 'config.json').read_bytes()).hexdigest():
        raise ValueError('Evaluation model config differs from checkpoint')
    weights = {str((checkpoint / name).resolve()): (checkpoint / name).stat()
               for name in final['weights'] if not name.endswith('.json')}
    if {record['path'] for record in result['weights']} != set(weights):
        raise ValueError('Evaluation weight files differ from checkpoint')
    for record in result['weights']:
        stat = weights[record['path']]
        if (record['bytes'], record['mtime_ns']) != (stat.st_size, stat.st_mtime_ns):
            raise ValueError('Evaluation weights changed')
    metrics = check_metric(result)
    rows = [dict(checkpoint=s['checkpoint'], start_step=s['start_step'], step=s['global_step'],
                 schedule_steps=s['schedule_steps'], stage_train_seconds=s['stage_train_seconds'],
                 validation=s['validation']) for s in stages]
    return dict(status='ok', schema=1, stages=rows, evaluation=dict(
        file=str(Path(evaluation).resolve()), checkpoint=result['checkpoint'], step=result['global_step'],
        attention_backend=result['attention_backend'], precision=result['precision'],
        **metrics), total_stage_train_seconds=sum(s['stage_train_seconds'] for s in stages))


def markdown(summary):
    lines = ['# Training workflow results', '',
        '| Start → stop | Full schedule | Training seconds (stage) | Validation step | Loss | PPL |',
        '|---|---:|---:|---:|---:|---:|']
    for row in summary['stages']:
        valid = row['validation']
        fields = (f"{valid['step']} | {valid['loss']:.6f} | {valid['perplexity']:.6f}"
                  if valid else '— | — | —')
        lines.append(f"| {row['start_step']} → {row['step']} | {row['schedule_steps']} | "
                     f"{row['stage_train_seconds']:.3f} | {fields} |")
    result = summary['evaluation']
    lines += ['', f"Final standalone evaluation at step {result['step']}: "
        f"loss **{result['loss']:.6f}**, PPL **{result['perplexity']:.6f}** "
        f"({result['num_tokens']} targets, {result['num_sequences']} sequences).", '',
        'Times are HF train_runtime per invocation, including in-training validation; '
        'they exclude preprocessing and standalone evaluation. Resume-stage train_loss '
        'is not used for checkpoint quality comparisons.', '']
    return '\n'.join(lines)


def publish(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(payload)
            handle.flush()
            os.link(temporary, path)  # Atomic no-overwrite publication.
        finally:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    check = sub.add_parser('check')
    check.add_argument('--output-dir', required=True, type=Path)
    check.add_argument('--expected-step', required=True, type=int)
    check.add_argument('--schedule-steps', required=True, type=int)
    check.add_argument('--start-step', type=int, default=0)
    summary = sub.add_parser('summary')
    summary.add_argument('--reports', nargs='+', type=Path, required=True)
    summary.add_argument('--evaluation', type=Path, required=True)
    summary.add_argument('--markdown', type=Path)
    for command in (check, summary):
        command.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        outputs = [args.output] + ([args.markdown] if args.mode == 'summary' and args.markdown else [])
        if len({p.resolve() for p in outputs}) != len(outputs) or any(os.path.lexists(p) for p in outputs):
            raise ValueError('Choose distinct, fresh report paths')
        result = (check_training(args.output_dir, args.expected_step, args.schedule_steps, args.start_step)
                  if args.mode == 'check' else summarize(args.reports, args.evaluation))
        publish(args.output, json.dumps(result, indent=2, allow_nan=False) + '\n')
        if args.mode == 'summary' and args.markdown:
            publish(args.markdown, markdown(result))
        print(f'{args.mode}: verified; wrote {args.output}')
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f'Training artifact check FAILED: {error}\n')


if __name__ == '__main__':
    main()
