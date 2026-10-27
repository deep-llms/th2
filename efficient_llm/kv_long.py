"""Long matched dense/uniform25 recovery; reuse the validated trainer and supervisor."""
import json
from pathlib import Path

STEPS = 20752
DOCUMENTS = STEPS * 8 * 8
TOKENS = DOCUMENTS * 1024


def verify_probe(probe, short_probe, parent_probe, *, documents=DOCUMENTS):
    import torch
    from efficient_llm.diagnostic import load_probe
    from efficient_llm.diagnostic_data import sha256
    new, tokens = load_probe(probe)
    old, short = load_probe(short_probe)
    parent = json.loads((parent_probe/'manifest.json').read_text())
    if (new['calibration_documents'] != documents or new['length'] != old['length']
            or new['recovery_from_manifest_sha256'] != sha256(parent_probe/'manifest.json')
            or new['recovery_from_manifest_sha256'] != old['recovery_from_manifest_sha256']
            or new['seed'] != old['seed'] or not new['evaluation_reused']):
        raise ValueError('Long recovery probe provenance differs')
    n = old['calibration_documents']
    if (new['documents']['calibration'][:n] != old['documents']['calibration']
            or not torch.equal(tokens['calibration'][:n], short['calibration'])
            or new['documents']['evaluation'] != old['documents']['evaluation']
            or not torch.equal(tokens['evaluation'], short['evaluation'])):
        raise ValueError('Extended data changed the old prefix or fixed validation')
    for key in ('text_sha256', 'tokens_sha256'):
        ids = {r[key] for r in new['documents']['calibration']}
        excluded = {r[key] for records in parent['documents'].values() for r in records}
        if len(ids) != documents or ids & excluded:
            raise ValueError('Duplicate or overlapping training documents: '+key)
    return dict(status='passed', documents=documents, probe_sha256=sha256(probe/'manifest.json'),
        short_probe_sha256=sha256(short_probe/'manifest.json'),
        exact_short_training_prefix_documents=n, exact_reused_validation_documents=len(short['evaluation']),
        original_parent_overlap=0)


def verify_handoff(output_parent, burn_script, digest):
    root = burn_script.parents[2]
    if (root.parent.resolve() != output_parent.resolve()
            or burn_script.resolve() != (root/'source/resources/llm_pretrain_burn.py').resolve()):
        raise ValueError('Require a known completed native-supervisor burn owner')
    if (json.loads((root/'supervised/run/complete.json').read_text())['status'] != 'ok'
            or json.loads((root/'supervised/supervisor.json').read_text())['training_status'] != 'ok'
            or not json.loads((root/'supervised/burn-verified.json').read_text())['collective_progress_verified']):
        raise ValueError('Latest workload/burn handoff is incomplete')
    # The supervisor will independently reverify current live ownership and PID starts.
    return dict(root=str(root), complete_sha256=digest(root/'supervised/run/complete.json'))


def jobs_for(model, probe, config, port, dense_parent):
    parent = Path(dense_parent)/'supervised/run/A'
    launch = ['{python}', '-m', 'accelerate.commands.launch', '--config_file', str(config),
              '--main_process_port', str(port), '--module', 'efficient_llm.kv_mlp_study']
    jobs = [dict(name='long-numerical-gate', gpus=[0], timeout_seconds=900,
        argv=['{python}', '-m', 'efficient_llm.kv_mlp_check', '--model', str(model),
              '--probe', str(probe), '--sharing-layout', 'uniform25',
              '--output', '{run_dir}/gate.json'],
        required_outputs=[dict(path='gate.json', json_equals=dict(status='passed', sharing_layout='uniform25'))])]
    for arm, smoke in [('A', True), ('B', True), ('A', False), ('B', False)]:
        name = 'smoke-'+arm if smoke else arm
        steps = 8 if smoke else STEPS
        jobs.append(dict(name='long-'+name, gpus=list(range(8)), timeout_seconds=14400,
            argv=[*launch, 'train', '--model', str(model), '--probe', str(probe),
                  '--initial-model', str(parent/'final'), '--parent-result', str(parent/'result.json'),
                  '--calibration', '0', '--arm', arm, '--steps', str(steps), '--batch', '8',
                  '--eval-every', str(8 if smoke else 1024), '--output', '{run_dir}/'+name]
                  + (['--sharing-layout', 'uniform25'] if arm == 'B' else [])
                  + (['--smoke'] if smoke else ['--inference-eval', '--eval-steps', '5188', '10376']),
            required_outputs=[dict(path=name+'/result.json', json_equals=dict(status='passed',
                world_size=8, steps=steps, arm=arm, smoke=smoke, trainable_scope='all',
                input_tokens=steps*65536, parent_steps=512, optimizer_and_schedule_reset=True,
                phase='new_recovery' if arm == 'A' else 'placement_conversion'))]))
    jobs.append(dict(name='long-summary', timeout_seconds=300,
        argv=['{python}', '-m', 'efficient_llm.kv_mlp_study', 'long-summary',
              '--runs', '{run_dir}', '--output', '{run_dir}/summary.json'],
        required_outputs=[dict(path='summary.json', json_equals=dict(status='passed', phase_tokens=2*TOKENS))]))
    return dict(jobs=jobs)
