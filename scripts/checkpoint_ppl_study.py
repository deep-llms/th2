"""Read-only full packed-validation PPL using the original ProxyTrainer."""
import argparse
import json
import math
from pathlib import Path

from scripts.proxy_time_match import read, write

ARMS = ('A', 'P6-iso')
STEPS = (20000, 15000, 16000, 17000, 18000, 19000)


def prepare_output(arm_dir, output, *, is_main_process):
    """Reserve the shared output once; broadcast failures to every worker."""
    import torch.distributed as dist
    error = [None]
    if is_main_process:
        try:
            if output.exists() or output.is_relative_to(arm_dir):
                raise ValueError('Use a fresh evaluation output outside the source arm')
            output.mkdir(parents=True, exist_ok=False)
        except (OSError, ValueError) as exc:
            error[0] = str(exc)
    if dist.is_initialized():
        dist.broadcast_object_list(error, src=0)
    if error[0] is not None:
        raise ValueError(error[0])


def evaluate(arm_dir, output, step):
    import torch
    from transformers import AutoTokenizer, TrainingArguments, TrainerState, set_seed
    from deep_kv.packing import preprocess_dataset, isolated_data_collator
    from deep_kv.proxy_training import ProxyTrainer
    from eval.models import load_checkpoint, file_hash
    from scripts.evaluate_proxy_gates import state_hash
    from train import load_text
    arm_dir, output = Path(arm_dir).resolve(), Path(output).resolve()
    saved, reference = read(arm_dir/'train_config.json'), read(arm_dir/'result.json')
    checkpoint = arm_dir/f'checkpoint-{step}'
    state = TrainerState.load_from_json(str(checkpoint/'trainer_state.json'))
    if (saved['pilot']['arm'] not in ARMS or state.global_step != step
            or state.max_steps != saved['training']['max_steps']):
        raise ValueError('Unsupported arm or checkpoint step/schedule mismatch')
    args = TrainingArguments(**{**saved['training'], 'output_dir': str(output),
        'report_to': [], 'run_name': 'checkpoint-ppl', 'disable_tqdm': True})
    if args.world_size != saved['world_size']:
        raise ValueError('Use the original world size for comparable evaluation')
    prepare_output(arm_dir, output, is_main_process=args.process_index == 0)
    args.distributed_state.wait_for_everyone()
    set_seed(args.seed)
    adapter, _, metadata = load_checkpoint(checkpoint, args.device)
    model = adapter.wrapped
    initial = state_hash(model) if args.should_save else None
    tokenizer = AutoTokenizer.from_pretrained(saved['model']['tokenizer_name'], local_files_only=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    data = preprocess_dataset(load_text(saved['data']['eval_data_dir']), tokenizer,
        saved['data']['block_size'], args, num_proc=1,
        isolate_documents=saved['data']['isolate_documents']).select(range(saved['data']['eval_rows']))
    if data._fingerprint != saved['eval_fingerprint']:
        raise ValueError('Validation fingerprint differs from training')
    trainer = ProxyTrainer(model=model, args=args, eval_dataset=data,
        processing_class=tokenizer, data_collator=isolated_data_collator)
    trainer.state = state
    metrics = trainer.evaluate()
    for key in ('eval_rows', 'eval_target_tokens', 'eval_input_tokens'):
        if metrics[key] != reference['evaluation'][key]:
            raise ValueError(f'Validation count differs: {key}')
    loss = metrics['eval_lm_loss']
    if not math.isfinite(loss): raise ValueError('Nonfinite LM loss')
    gap = abs(loss-reference['evaluation']['eval_lm_loss']) if step == reference['global_step'] else None
    if gap is not None and gap > 1e-5:
        raise ValueError(f'Final-checkpoint control differs from saved PPL: {gap}')
    if trainer.optimizer is not None or trainer.lr_scheduler is not None:
        raise ValueError('Evaluation unexpectedly created optimizer or scheduler')
    if args.should_save:
        if initial != state_hash(model) or metadata['checkpoint_sha256'] != file_hash(checkpoint/'model.safetensors'):
            raise ValueError('Evaluation modified model state or source weights')
        result = dict(status='passed', arm=saved['pilot']['arm'], step=step, metadata=metadata,
            world_size=args.world_size, eval_fingerprint=data._fingerprint, metrics=metrics,
            lm_loss=loss, perplexity=math.exp(loss), final_control_gap=gap,
            model_state_unchanged=True, checkpoint_unchanged=True,
            fa4_receipts=getattr(trainer, '_fa4_receipts', []))
        write(output/'summary.json', result)
        print('CHECKPOINT_PPL', json.dumps(result), flush=True)
    trainer.accelerator.wait_for_everyone()


def make(source_run, output):
    from run_experiments import load_jobs
    source_run, output = Path(source_run).resolve(), Path(output).resolve()
    assert read(source_run/'continuation-summary.json')['status'] == 'passed'
    jobs = []
    for step in STEPS:  # Both final-checkpoint controls gate the earlier checkpoints.
        for arm in ARMS:
            folder = source_run/'seed-1042'/arm
            assert read(folder/f'checkpoint-{step}/trainer_state.json')['global_step'] == step
            name = f'{arm}-{step}'
            jobs.append(dict(name=name, gpus=list(range(8)), timeout_seconds=900,
                argv=['{python}', '-m', 'accelerate.commands.launch', '--config_file',
                    '{project_dir}/resources/accelerate_config.yaml', '--main_process_port', '29651',
                    '--module', 'scripts.checkpoint_ppl_study', 'evaluate', '--source-run', str(folder),
                    '--output', '{run_dir}/'+name, '--step', str(step)],
                required_outputs=[dict(path=name+'/summary.json',
                    json_equals=dict(status='passed', arm=arm, step=step, checkpoint_unchanged=True))]))
    jobs.append(dict(name='compare', argv=['{python}', '-m', 'scripts.checkpoint_ppl_study',
        'report', '--output', '{run_dir}'], required_outputs=[dict(path='summary.json', json_equals=dict(status='passed'))]))
    write(output/'jobs.json', dict(jobs=jobs)); load_jobs(output/'jobs.json')
    return jobs


def report(output):
    output = Path(output)
    results, protocol = [], None
    for step in sorted(STEPS):
        pair = {}
        for arm in ARMS:
            row = read(output/f'{arm}-{step}/summary.json')
            assert row['status'] == 'passed' and row['arm'] == arm and row['step'] == step
            assert row['model_state_unchanged'] and row['checkpoint_unchanged']
            signature = (row['eval_fingerprint'], row['world_size'],
                *(row['metrics'][k] for k in ('eval_rows','eval_input_tokens','eval_target_tokens')),
                row['metadata']['attention_backend'])
            assert protocol is None or protocol == signature
            protocol = signature
            if step == 20000: assert row['final_control_gap'] is not None and row['final_control_gap'] <= 1e-5
            pair[arm] = dict(lm_loss=row['lm_loss'], perplexity=row['perplexity'])
        results.append(dict(step=step, arms=pair,
            p6_ppl_reduction_percent=100*(1-pair['P6-iso']['perplexity']/pair['A']['perplexity'])))
    result = dict(status='passed', results=results, protocol=protocol)
    write(output/'summary.json', result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('evaluate', 'make', 'report'))
    p.add_argument('--source-run', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--step', type=int)
    a = p.parse_args()
    if a.mode == 'evaluate': evaluate(a.source_run, a.output, a.step)
    elif a.mode == 'make': make(a.source_run, a.output)
    else: report(a.output)
