"""D/F/G production-shape smoke using train.py; run under train_then_burn."""
import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def read(path):
    return json.loads(path.read_text())


def worker():
    """Add smoke-only timing/memory observations; keep the real training loop."""
    import torch
    import train
    from scripts.gpu_status import snapshot
    callbacks = []

    class TimingCallback(train.PilotCallback):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.steps, self.gpu_sample = [], None
            self.allocated = self.reserved = 0
            callbacks.append(self)

        def memory(self):
            self.allocated = max(self.allocated, torch.cuda.max_memory_allocated())
            self.reserved = max(self.reserved, torch.cuda.max_memory_reserved())

        def on_step_begin(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            self.begin = time.perf_counter()

        def on_step_end(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            seconds = time.perf_counter() - self.begin
            super().on_step_end(args, state, control, **kwargs)
            self.memory()
            self.steps.append(dict(step=state.global_step, seconds=seconds, **self.latest))
            if args.process_index == 0 and state.global_step == 3:
                self.gpu_sample = snapshot(list(range(8)))
            return control

        def on_evaluate(self, args, state, control, **kwargs):
            self.memory()

    train.PilotCallback = TimingCallback
    train.main()
    callback, = callbacks
    callback.memory()
    trainer = callback.trainer
    assert trainer.args.world_size == 8 and trainer.state.global_step in (10, 12)
    record = dict(arm=trainer.model.arm, rank=trainer.args.process_index,
                  step=trainer.state.global_step, steps=callback.steps,
                  allocated_peak_bytes=callback.allocated, reserved_peak_bytes=callback.reserved,
                  device_total_bytes=torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory,
                  gpu_sample=callback.gpu_sample)
    path = Path(trainer.args.output_dir) / f'smoke-step{record["step"]}-rank{record["rank"]}.json'
    with path.open('x') as handle:
        json.dump(record, handle, indent=2, allow_nan=False)


def validate_arm(path, arm, step):
    """Gate the long run on saved results, true resume and all-rank GPU telemetry."""
    result, config = read(path / 'result.json'), read(path / 'train_config.json')
    state = read(path / 'trainer_state.json')
    assert result['arm'] == arm and result['global_step'] == state['global_step'] == step
    assert result['schedule_steps'] == state['max_steps'] == 12
    assert result['status'] == ('stopped' if step == 10 else 'complete')
    assert result['input_tokens'] == step * 1048576
    assert config['world_size'] == 8 and config['tokens_per_update'] == 1048576
    assert config['model_config']['num_hidden_layers'] == 28 and config['data']['block_size'] == 2048
    assert config['training']['bf16'] and config['training']['per_device_train_batch_size'] == 16
    assert config['training']['gradient_accumulation_steps'] == 4
    checkpoint = path / f'checkpoint-{step}'
    assert (checkpoint / 'optimizer.pt').is_file() and (checkpoint / 'scheduler.pt').is_file()
    assert len(list(checkpoint.glob('rng_state_*.pth'))) == 8
    evaluation = result['evaluation']
    assert evaluation['eval_rows'] == 129
    assert all(math.isfinite(v) for v in evaluation.values() if isinstance(v, (float, int)))
    if arm in 'FG':
        assert evaluation['eval_route_queries'] == 129 * 2047
        assert evaluation['eval_loss_route'] >= -1e-6
        assert evaluation['eval_loss_msg'] >= 0
        assert arm != 'F' or evaluation['eval_loss_msg'] == 0
        expected = evaluation['eval_lm_loss'] + .3 * evaluation['eval_loss_route']
        if arm == 'G':
            expected += .3 * evaluation['eval_loss_msg']
        assert math.isclose(evaluation['eval_loss'], expected, abs_tol=1e-9)
    telemetry = [read(path / f'smoke-step{step}-rank{rank}.json') for rank in range(8)]
    for rank, record in enumerate(telemetry):
        assert record['arm'] == arm and record['rank'] == rank and record['step'] == step
        assert [s['step'] for s in record['steps']] == list(range(1 if step == 10 else 11, step + 1))
        assert 0 < record['allocated_peak_bytes'] <= record['reserved_peak_bytes']
        assert record['device_total_bytes'] - record['reserved_peak_bytes'] > 8 * 2**30
        for metrics in record['steps']:
            assert math.isfinite(metrics['seconds']) and metrics['seconds'] > 0
            if arm in 'FG':
                assert math.isfinite(metrics['loss_route']) and metrics['loss_route'] >= -1e-6
                assert math.isfinite(metrics['loss_msg']) and metrics['loss_msg'] >= 0
                assert arm != 'F' or metrics['loss_msg'] == 0
    if step == 10:
        sample = telemetry[0]['gpu_sample']
        assert len(sample) == 8 and [g['index'] for g in sample] == list(range(8))
        assert all(len(g['pids']) == 1 and g['memory_total_mib'] - g['memory_used_mib'] > 8192 for g in sample)
        seconds = sum(max(r['steps'][i]['seconds'] for r in telemetry) for i in range(2, 10)) / 8
    else:
        seconds = None
    return dict(step=step, seconds_per_update=seconds,
                allocated_peak_bytes=max(r['allocated_peak_bytes'] for r in telemetry),
                reserved_peak_bytes=max(r['reserved_peak_bytes'] for r in telemetry),
                evaluation=evaluation)


def run(root, recipe):
    from train import load_text
    from deep_kv.__main__ import jobs
    from deep_kv.report import report
    from run_experiments import load_jobs, run_jobs
    from scripts.gpu_status import require_free
    root.mkdir(parents=True, exist_ok=False)
    assert shutil.disk_usage(root).free > 180 * 2**30, 'Insufficient checkpoint disk space'
    config = read(recipe)
    for key, count, split in (('data_dir', 30000, 'train'), ('eval_data_dir', 2000, 'eval')):
        data = load_text(config[key])
        destination = root / 'text' / split
        data.select(range(min(count, len(data)))).save_to_disk(str(destination))
        config[key] = str(destination)
    config.update(max_steps=12, warmup_steps=1, stop_after=10, preprocessing_num_workers=8,
                  eval_rows=129, monitor_rows=16, logging_steps=1, save_steps=5, eval_steps=5,
                  dataloader_num_workers=2, skip_memory_metrics=False)
    recipe_path = root / 'recipe.json'
    recipe_path.write_text(json.dumps(config, indent=2))
    manifest = jobs(recipe_path, arms='DFG')
    # Replace only the entry point to attach observational callbacks for this smoke.
    for job in manifest['jobs'][:-1]:
        argv = job['argv']
        index = next(i for i, value in enumerate(argv) if value.endswith('/train.py'))
        argv[index:index + 1] = [str(Path(__file__).resolve()), 'worker']
    manifest_path = root / 'jobs.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))
    run_dir = root / 'run'
    run_jobs(load_jobs(manifest_path), Path.cwd(), run_dir)
    initial = {arm: validate_arm(run_dir / arm, arm, 10) for arm in 'DFG'}
    # Preserve step-10 receipts before the intentional in-place resume changes artifacts.
    for name in ('complete.json', 'run.json', 'comparison.json'):
        (run_dir / name).rename(root / (Path(name).stem + '-step10.json'))
    for arm in 'DFG':
        for name in ('result.json', 'trainer_state.json', 'eval_results.json'):
            shutil.copy2(run_dir / arm / name, run_dir / arm / (Path(name).stem + '-step10.json'))
    for job in manifest['jobs'][:-1]:
        require_free(list(range(8)))
        argv = [x.replace('{python}', sys.executable).replace('{run_dir}', str(run_dir)) for x in job['argv']]
        argv[argv.index('--stop_after') + 1] = '12'
        with (root / (job['name'] + '-resume.log')).open('x') as log:
            subprocess.run(argv, check=True, stdout=log, stderr=subprocess.STDOUT)
        require_free(list(range(8)))
    final = {arm: validate_arm(run_dir / arm, arm, 12) for arm in 'DFG'}
    assert report(run_dir, 'DFG')['compared_update'] == 12
    baseline = initial['D']['seconds_per_update']
    receipt = dict(status='ok', arms=list('DFG'), initial_cutoff=10, resumed_step=12,
                   schedule_steps=12, tokens_per_update=1048576, eval_rows=129,
                   initial=initial, final=final, scientific_result=False,
                   speed_ratios={arm: initial[arm]['seconds_per_update'] / baseline for arm in 'FG'})
    (root / 'smoke_complete.json').write_text(json.dumps(receipt, indent=2, allow_nan=False))
    print('FUNCTIONAL_SMOKE_PASSED', json.dumps(receipt), flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'worker':
        del sys.argv[1]
        worker()
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--root', required=True, type=Path)
        parser.add_argument('--recipe', required=True, type=Path)
        args = parser.parse_args()
        run(args.root.resolve(), args.recipe.resolve())
