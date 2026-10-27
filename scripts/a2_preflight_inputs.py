"""CPU readiness checks and a bounded initial preflight manifest (no execution)."""
import json
from pathlib import Path

from efficient_llm.diagnostic_data import sha256
from efficient_llm.a2.data import PackedRows


def initial_asset_plan(plan):
    """Only DCLM/P300 is loaded by the initial jobs; other downloads may continue."""
    asset = plan['assets']['dclm/P300']
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
    if (manifest['provenance']['spec_sha256'] != sha256(spec)
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
    source, inputs = Path(source), Path(inputs)
    launch = ['{python}', '-m', 'accelerate.commands.launch', '--config_file',
        str(source/'resources/accelerate_b200.yaml'), '--main_process_port', str(port),
        '-m', 'efficient_llm.a2']
    common = ['--assets', str(inputs/'assets.json'), '--data', str(inputs/'data'),
        '--secondary-map', str(inputs/'secondary_map.json'), '--dataset', 'dclm', '--parent', 'P300']
    # Fresh discarded weights for T8. Never enqueue heal/queue from this launcher.
    return [dict(name=action, gpus=list(range(8)), timeout_seconds=limit,
        argv=launch+[action]+common+['--output', '{run_dir}/'+action+'.json'],
        required_outputs=[dict(path=action+'.json', **contract)])
        for action, limit, contract in (
            ('throughput', 1800, {'json_equals': {'status': 'passed'}}),
            ('survey', 3600, {'json_equals': {'dataset': 'dclm', 'parent': 'P300'}}))]
