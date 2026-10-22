"""Prepare/validate the checkpoint-only Step 1 study; never train or stop GPUs."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    from eval.packed_ppl import write_result
    write_result(path, value)


def prepare(output, pilot, no_reuse):
    from scripts.train_then_burn import digest
    from eval.pilot_log_analysis import analyze
    from run_experiments import load_jobs
    output.mkdir(parents=True, exist_ok=False)
    previous = read(pilot/'prepared/P1-A.train.json')
    reference = pilot/'prepared/runs/P1-A'
    for name, expected in read(reference/'training_recipe.json')['code_sha256'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('Training source differs from the pilot: '+name)
    for arm in ('P1-A', 'P1-B'):
        report = analyze(pilot/'prepared/runs'/arm)
        write(output/(arm+'.logs.json'), report)
        if not report['sign_replay']['passed']:
            raise ValueError('Balance update implementation failed replay')
    cases = [('P1-A',4768), ('P1-B',4768), ('N-A',4768),
             ('P1-A',1000), ('P1-A',2500), ('P1-B',1000), ('P1-B',2500)]
    jobs, manifest = [], []
    for arm, step in cases:
        run = (no_reuse if arm == 'N-A' else pilot)/'prepared/runs'/arm
        checkpoint = run/f'checkpoint-{step}'
        if read(checkpoint/'trainer_state.json')['global_step'] != step:
            raise ValueError('Checkpoint step mismatch')
        name = f'{arm}-{step}'
        result = output.parent/'supervised/run'/name/'result.json'
        config = dict(checkpoint=str(checkpoint), eval_data_dir=previous['eval_data_dir'],
            tokenizer_name=previous['tokenizer_name'], output_file=str(result),
            attention_backend='fa4', precision='bf16', batch_size=16,
            data_cache_dir=previous.get('data_cache_dir'), route_study=True,
            route_sequences=4096, route_artifact_sequences=256,
            pilot_diagnostics=True, router_gradient_check=step==4768 and arm!='N-A', routing_seed=0,
            max_sequences=None if step==4768 else 4096)
        config_path = output/(name+'.eval.json')
        write(config_path, config)
        jobs.append(dict(name=name, gpus=list(range(8)), timeout_seconds=1800,
            argv=['{python}','-m','accelerate.commands.launch','--config_file',
                  str(ROOT/'resources/accelerate_b200.yaml'), str(ROOT/'eval/packed_ppl.py'),str(config_path)],
            required_outputs=[dict(path=name+'/result.json',json_equals=dict(status='ok'))]))
        manifest.append(dict(name=name, arm=arm, step=step, result=str(result),
                             original_final=str(run/'final_eval.json')))
    write(output/'cases.json', manifest)
    jobs.append(dict(name='validate',timeout_seconds=120,
        argv=['{python}',str(ROOT/'scripts/bdot_diagnostic_queue.py'),'validate','--output',str(output)],
        required_outputs=[dict(path='summary.json',json_equals=dict(status='ok'))]))
    write(output/'jobs.json',dict(jobs=jobs))
    load_jobs(output/'jobs.json')
    print('DIAGNOSTICS_PREPARED '+str(output),flush=True)


def validate(output):
    results=[]
    for case in read(output/'cases.json'):
        result=read(case['result']);study=result['route_study']
        if result['global_step']!=case['step'] or result['world_size']!=8:
            raise ValueError('Incorrect checkpoint or GPU count')
        if study['router_health']['sequences']!=4096:
            raise ValueError('Incorrect diagnostic prefix')
        if case['step']==4768:
            original=read(case['original_final'])
            for key in ('num_sequences','num_tokens'):
                if result[key]!=original['eval_'+key]:raise ValueError('Validation population changed')
            delta=result['loss']-original['eval_loss']
            if abs(delta)>1e-5:raise ValueError('As-trained final loss did not reproduce: '+str(delta))
        else:delta=None
        health=study['router_health'];usage=study['route_diagnostics']
        # Criterion 2 is a diagnostic flag, never automatic permission to train.
        weak=(any(x<.5 or x>2 for x in usage['module_load_over_target'])
              and health['raw_gap']['median']>health['beta_range'])
        results.append(dict(name=case['name'],loss=result['loss'],ppl=result['perplexity'],
            replay_delta=delta,window_and_gate_pass=health['passed'],
            gradient_pass=study.get('gradient_check',{}).get('passed'),
            balancing_weak_rule=weak,result=case['result']))
    write(output.parent/'supervised/run/summary.json',dict(status='ok',results=results,
        scientific_checks_pass=all(r['window_and_gate_pass'] and r['gradient_pass'] is not False for r in results),
        scope='Checkpoint-only diagnostics. No new training; conditional arms require a separate decision.'))
    print('DIAGNOSTICS_COMPLETE '+json.dumps(results),flush=True)


def prepare_gradient(output, study):
    """Resolve saturated finite differences without redoing full validation."""
    from run_experiments import load_jobs
    output.mkdir(parents=True, exist_ok=False)
    jobs=[]
    for arm in ('P1-A','P1-B'):
        config=read(study/'prepared'/(arm+'-4768.eval.json'))
        config.update(max_sequences=1,route_sequences=1,route_artifact_sequences=1,
            pilot_diagnostics=False,router_gradient_check=True,router_gradient_center_logits=True,
            output_file=str(output.parent/'supervised/run'/arm/'result.json'))
        path=output/(arm+'.eval.json');write(path,config)
        jobs.append(dict(name=arm,gpus=list(range(8)),timeout_seconds=900,
            argv=['{python}','-m','accelerate.commands.launch','--config_file',
                  str(ROOT/'resources/accelerate_b200.yaml'),str(ROOT/'eval/packed_ppl.py'),str(path)],
            required_outputs=[dict(path=arm+'/result.json',json_equals=dict(status='ok'))]))
    jobs.append(dict(name='validate',timeout_seconds=120,
        argv=['{python}',str(ROOT/'scripts/bdot_diagnostic_queue.py'),'gradient-validate','--output',str(output)],
        required_outputs=[dict(path='summary.json',json_equals=dict(status='ok'))]))
    write(output/'jobs.json',dict(jobs=jobs));load_jobs(output/'jobs.json')


def validate_gradient(output):
    reports={}
    for arm in ('P1-A','P1-B'):
        result=read(output.parent/'supervised/run'/arm/'result.json')['route_study']
        raw,centered=result['gradient_check'],result['centered_gradient_check']
        if raw['loss']!=centered['loss']:raise ValueError('Centering changed the forward LM loss')
        if not centered['passed']:raise ValueError('Well-conditioned gradient check failed')
        if not all(r['local_pass'] for r in raw['steps']):raise ValueError('Stable gate VJP check failed')
        reports[arm]=dict(centered_passed=True,forward_loss_unchanged=True,
            stable_original_gate_vjp_passed=True,
            original_full_difference_passed=raw['passed'])
    write(output.parent/'supervised/run/summary.json',dict(status='ok',results=reports,
        scope='Centered logits test the unchanged ST machinery at a resolvable point; '
              'original saturated full-LM finite differences remain precision-limited.'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','validate','gradient-prepare','gradient-validate']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pilot',type=Path);p.add_argument('--no-reuse',type=Path)
    p.add_argument('--study',type=Path)
    a=p.parse_args()
    if a.action=='prepare':
        if not a.pilot or not a.no_reuse:p.error('prepare requires --pilot and --no-reuse')
        prepare(a.output,a.pilot,a.no_reuse)
    elif a.action=='validate':validate(a.output)
    elif a.action=='gradient-prepare':
        if not a.study:p.error('gradient-prepare requires --study')
        prepare_gradient(a.output,a.study)
    else:validate_gradient(a.output)


if __name__=='__main__':main()
