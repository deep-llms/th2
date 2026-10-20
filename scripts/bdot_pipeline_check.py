"""T14: fresh native/fixed-B-DOT pair through unchanged train.py, before the pilot."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def initial_pair(native_config, routed_config, seed):
    """Verify that the two config-based, from-scratch initializations map exactly."""
    import torch
    from transformers import set_seed
    from train import build_model, ModelArguments
    torch.set_num_threads(8)
    set_seed(seed)
    routed = build_model(ModelArguments(config_name=str(routed_config)))
    mapped = routed.native_reference()
    set_seed(seed)
    native = build_model(ModelArguments(config_name=str(native_config)))
    left, right = native.state_dict(), mapped.state_dict()
    assert left.keys() == right.keys()
    digest = hashlib.sha256()
    for name, tensor in left.items():
        torch.testing.assert_close(tensor, right[name], atol=0, rtol=0, msg=name)
        digest.update(name.encode())
        digest.update(tensor.detach().contiguous().numpy().tobytes())
    return dict(status='passed', initialization='random from config; no pretrained weights',
                seed=seed, native_mapped_state_sha256=digest.hexdigest(),
                parameters=sum(p.numel() for p in native.parameters()),
                layers=native.config.num_hidden_layers)


def summarize_run(output, steps=500, window=100, logging_steps=10, batch_tokens=524288):
    output = Path(output)
    state = json.loads((output/'trainer_state.json').read_text())
    metrics = json.loads((output/'train_results.json').read_text())
    assert state['global_step'] == metrics['global_step'] == steps
    entries = [e for e in state['log_history'] if steps-window < e['step'] <= steps
               and ('lm_loss' in e or 'loss' in e)]
    if [e['step'] for e in entries] != list(range(steps-window+logging_steps, steps+1, logging_steps)):
        raise ValueError('Incomplete last-window training loss logs')
    losses = [e['lm_loss'] if 'lm_loss' in e else e['loss'] for e in entries]
    evaluations = [e for e in state['log_history'] if e['step'] == steps and 'eval_loss' in e]
    if not evaluations:
        raise ValueError('Missing periodic validation at the final update')
    ev = evaluations[0]  # B-DOT final/full validation is separate; compare periodic sets.
    values = [*losses, ev['eval_loss'], metrics['train_runtime']]
    if not all(math.isfinite(v) for v in values) or metrics['train_runtime'] <= 0:
        raise ValueError('Nonfinite pipeline-check metrics')
    return dict(last_window_lm_loss=sum(losses)/len(losses),
                periodic_validation_loss=ev['eval_loss'],
                validation_tokens=ev['eval_num_tokens'], validation_sequences=ev['eval_num_sequences'],
                trainer_seconds=metrics['train_runtime'],
                input_tokens_per_second_including_eval_save=steps*batch_tokens/metrics['train_runtime'])


def compare_runs(native, routed):
    for key in ('validation_tokens', 'validation_sequences'):
        if native[key] != routed[key]:
            raise ValueError('T14 validation sets differ')
    gaps = {key: abs(native[key]-routed[key]) for key in
            ('last_window_lm_loss', 'periodic_validation_loss')}
    return dict(status='passed' if all(v <= .01 for v in gaps.values()) else 'failed',
                absolute_gaps=gaps, tolerance_nats=.01, native=native, bdot=routed)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arguments', type=Path, required=True, help='Generated D.train.json')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--accelerate-config', type=Path, required=True)
    a = p.parse_args()
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        p.error('Explicit GPU allocation required')
    a.output.mkdir(parents=True, exist_ok=False)
    base = json.loads(a.arguments.read_text())
    routed = json.loads(Path(base['config_name']).read_text())
    routed['bdot'].update(routing='fixed', curriculum='none', r=0, step_film=False)
    native = copy.deepcopy(routed)
    del native['bdot']
    write(a.output/'native.model.json', native)
    write(a.output/'bdot.model.json', routed)
    write(a.output/'initial.json', initial_pair(a.output/'native.model.json',
                                              a.output/'bdot.model.json', base['seed']))
    configurations = {}
    def run(name, model):
        from scripts.gpu_status import require_free
        from scripts.training_results import check_training
        require_free(list(range(8)))
        options = dict(base, config_name=str(a.output/(model+'.model.json')),
            output_dir=str(a.output/name), max_steps=500, stop_after=500, warmup_steps=10,
            logging_steps=10, save_steps=500, eval_steps=500)
        configurations[name] = options
        file = a.output/(name+'.train.json')
        write(file, options)
        print('T14_START '+name, flush=True)
        with (a.output/(name+'.log')).open('x') as log:
            subprocess.run([sys.executable, '-m', 'accelerate.commands.launch', '--config_file',
                str(a.accelerate_config), '--main_process_port', '0', str(ROOT/'train.py'), str(file)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        write(a.output/(name+'.checked.json'), check_training(a.output/name, 500, 500))
        return summarize_run(a.output/name)
    n = run('native', 'native')
    b = run('bdot', 'bdot')
    for filename in ('data_recipe.json',):
        assert json.loads((a.output/'native'/filename).read_text()) == json.loads((a.output/'bdot'/filename).read_text())
    nr = json.loads((a.output/'native/training_recipe.json').read_text())
    br = json.loads((a.output/'bdot/training_recipe.json').read_text())
    assert nr['validation'] == br['validation'], 'Validation recipe changed between T14 runs'
    report = compare_runs(n, b)
    if report['status'] != 'passed':
        report['native_repeat'] = run('native_repeat', 'native')
        report['native_repeat_comparison'] = compare_runs(n, report['native_repeat'])
        report['decision'] = 'Review numerical control before authorizing the pilot'
    write(a.output/'summary.json', report)
    print('T14_RESULT '+json.dumps(report), flush=True)
    if report['status'] != 'passed':
        raise RuntimeError('T14 losses differ; pilot remains blocked pending review')


if __name__ == '__main__':
    main()
