"""CPU readiness checks and a bounded initial preflight manifest (no execution)."""
import json
from pathlib import Path

from efficient_llm.diagnostic_data import sha256
from efficient_llm.a2.data import PackedRows
from efficient_llm.a2.protocol import compatible_data_spec


def initial_asset_plan(plan, parents=('P300',)):
    """Verify only actual launch dependencies; unrelated downloads may continue."""
    if not parents or not set(parents) <= {'P300','P1T'}:
        raise ValueError('Unsupported bounded launch parents')
    plans = [_parent_asset_plan(plan,parent) for parent in parents]
    downloads = {item['path']:item for part in plans for item in part['downloads']}
    return dict(plan, downloads=list(downloads.values()))


def _parent_asset_plan(plan, parent):
    asset = plan['assets']['dclm/'+parent]
    required = {str(Path(asset[directory])/name): digest
        for directory, files in (('weights_dir', 'files'), ('code_dir', 'code_files'))
        for name, digest in asset[files].items()}
    downloads = [item for item in plan['downloads'] if item['path'] in required]
    if ({item['path']: item['sha256'] for item in downloads} != required
        or not {'config.json','model.safetensors'} <= asset['files'].keys()
        or not {'modeling_opensci.py','configuration_opensci.py'} <= asset['code_files'].keys()):
        raise ValueError('Initial parent lock incomplete or inconsistent')
    return dict(plan, downloads=downloads)


def verify_prepared(root, spec, plan):
    root = Path(root)
    complete = json.loads((root/'complete.json').read_text())
    manifest = json.loads((root/'data/manifest.json').read_text())
    if ((root/'failure.json').exists() or complete.get('status') != 'passed'
        or complete['data_manifest_sha256'] != sha256(root/'data/manifest.json')
        or complete['verified_inputs_sha256'] != sha256(root/'verified_inputs.json')
        or complete['secondary_map_sha256'] != sha256(root/'secondary_map.json')):
        raise ValueError('Missing, failed or changed preparation receipt')
    if (not compatible_data_spec(manifest['provenance']['spec_sha256'], sha256(spec))
        or manifest['provenance']['revision'] != plan['pool_contract']['revision']
        or manifest['provenance']['manifest_sha256'] != plan['pool_contract']['manifest_sha256']
        or not manifest['disjoint_by_id_and_content']
        or manifest['streams']['heal']['tokens'] < 3051*64*4096
        or manifest['streams']['tuning']['rows'] != 1024
        or len(manifest['small_indices']) != 256):
        raise ValueError('Prepared data contract differs')
    if json.loads((root/'assets.json').read_text()) != plan['assets']:
        raise ValueError('Prepared model asset contract differs')
    # Actual packed bytes, including document boundaries, checked before GPU reclaim.
    for name in ('heal','tuning','final'):
        rows = PackedRows(root/'data', name)
        if rows.length != 4096:
            raise ValueError('Production context must be 4096')
    for name in ('documents.jsonl','indices.npz'):
        key = 'documents_sha256' if name.endswith('jsonl') else 'indices_sha256'
        if sha256(root/'data'/name) != manifest[key]:
            raise ValueError('Prepared document/order metadata changed')
    secondary = json.loads((root/'secondary_map.json').read_text())
    if set(secondary) != {'dclm','fwe','nem'}:
        raise ValueError('Secondary acquisition/skip records missing')
    for item in secondary.values():
        if item.get('path'):
            if sha256(Path(item['path'])/'manifest.json') != item['manifest_sha256']:
                raise ValueError('Secondary manifest changed')
            rows = PackedRows(item['path'], 'final')
            if rows.length != 4096 or len(rows) != 256:
                raise ValueError('Secondary shape differs')
        elif not item.get('skip_reason'):
            raise ValueError('Secondary requires acquisition or explicit skip')
    return complete


def initial_jobs(source, inputs, port):
    # Initial cost probe only; do not rerun when continuing after this stage.
    return [_model_job(source, inputs, port, action, limit, contract)
        for action, limit, contract in (
            ('throughput', 1800, {'status': 'passed'}),
            ('survey', 3600, {'dataset': 'dclm', 'parent': 'P300'}))]


def shared_numerical_jobs(source, inputs, port, sets_path, assets):
    """Post-S2 manifest: exactly one real-parent mechanism job, no T8 or heal."""
    from efficient_llm.a2.protocol import shared_check_sets
    shared_check_sets(json.loads(Path(sets_path).read_text())['sets'], assets)
    job = _model_job(source, inputs, port, 'check', 3600,
        {'passed': True, 'scope': 'shared_mechanism', 'representative': 'dclm/P300'})
    job['argv'].extend(['--sets', str(sets_path)])
    return [job]


def early_grid_jobs(inputs, prior, port):
    return [dict(name='early-grid',gpus=list(range(8)),timeout_seconds=3700,
        argv=['{python}','-u','-m','efficient_llm.a2.early_grid','--inputs',str(inputs),
              '--prior',str(prior),'--port',str(port),'--output','{run_dir}/grid'],
        required_outputs=[dict(path='grid/complete.json',json_equals={
            'status':'complete','scope':'dclm_P300_S1_grid','full_pilot_complete':False})])]


def _model_job(source, inputs, port, action, limit, contract):
    source, inputs = Path(source), Path(inputs)
    launch = ['{python}', '-m', 'accelerate.commands.launch', '--config_file',
        str(source/'resources/accelerate_b200.yaml'), '--main_process_port', str(port),
        '-m', 'efficient_llm.a2']
    common = ['--assets', str(inputs/'assets.json'), '--data', str(inputs/'data'),
        '--secondary-map', str(inputs/'secondary_map.json'), '--dataset', 'dclm', '--parent', 'P300']
    return dict(name=action, gpus=list(range(8)), timeout_seconds=limit,
        argv=launch+[action]+common+['--output', '{run_dir}/'+action+'.json'],
        required_outputs=[dict(path=action+'.json', json_equals=contract)])


def pair_jobs(inputs, prior, grid, port):
    return [dict(name='s1-pair',gpus=list(range(8)),timeout_seconds=10900,
        argv=['{python}','-u','-m','efficient_llm.a2.pair','--inputs',str(inputs),
              '--prior',str(prior),'--grid',str(grid),'--port',str(port),'--output','{run_dir}/pair'],
        required_outputs=[dict(path='pair/complete.json',json_equals={
            'status':'complete','scope':'dclm_S1_pair','full_pilot_complete':False})])]
