"""Offline CPU input verification and frozen A2 data preparation. No GPU work.

Run only after the controller reports all pinned downloads complete. This job
refuses partial inputs/output reuse, verifies actual bytes and never downloads.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                  TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', WANDB_MODE='offline')

from efficient_llm.a2.data import write_json
from efficient_llm.diagnostic_data import sha256


def verify_inputs(plan):
    """Verify exactly the locked files; paths alone or controller success aren't enough."""
    verified = {}
    for item in plan['downloads']:
        p = Path(item['path'])
        if not p.is_file() or sha256(p) != item['sha256']:
            raise ValueError('A2 input missing or digest mismatch: '+str(p))
        verified[str(p)] = dict(bytes=p.stat().st_size, sha256=item['sha256'])
        print('A2_INPUT_VERIFIED', str(p), flush=True)
    for key, path in (('qwen', plan['qwen_path']), ('gpt_neox', plan['tokenizer_path'])):
        for name, expected in plan['tokenizer_provenance'][key]['files'].items():
            if sha256(Path(path)/name) != expected:
                raise ValueError('Tokenizer mismatch: '+key+'/'+name)
    manifest = Path(plan['pool_path'])/'subsets'/plan['pool_contract']['subset']/'manifest.json'
    if sha256(manifest) != plan['pool_contract']['manifest_sha256']:
        raise ValueError('Source pool manifest differs')
    return verified


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--spec', type=Path, default=Path('docs/A2_pilot_spec.md'))
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Fresh output required; incomplete outputs are preserved')
    a.output.mkdir(parents=True)
    started = time.monotonic()
    try:
        plan = json.loads(a.plan.read_text())
        verified = verify_inputs(plan)
        write_json(a.output/'verified_inputs.json', dict(status='passed', files=verified,
            plan_sha256=sha256(a.plan), elapsed_seconds=time.monotonic()-started))
        for name, value in [('assets', plan['assets']), ('pool_contract', plan['pool_contract']),
                            ('tokenizer_provenance', plan['tokenizer_provenance'])]:
            write_json(a.output/(name+'.json'), value)
        subprocess.run([sys.executable, '-u', '-m', 'efficient_llm.a2', 'prepare',
            '--pool', plan['pool_path'], '--qwen-tokenizer', plan['qwen_path'],
            '--tokenizer', plan['tokenizer_path'], '--pool-contract', str(a.output/'pool_contract.json'),
            '--dataset-revision', plan['pool_contract']['revision'], '--spec', str(a.spec),
            '--tokenizer-provenance', str(a.output/'tokenizer_provenance.json'),
            '--output', str(a.output/'data')], check=True)
        data = json.loads((a.output/'data/manifest.json').read_text())
        if (data['status'] != 'complete' or not data['disjoint_by_id_and_content']
            or data['streams']['heal']['tokens'] < 3051*64*4096
            or data['streams']['tuning']['rows'] != 1024 or len(data['small_indices']) != 256):
            raise ValueError('Prepared data does not satisfy the production budget/splits')
        write_json(a.output/'complete.json', dict(status='passed',
            elapsed_seconds=time.monotonic()-started, data_manifest_sha256=sha256(a.output/'data/manifest.json'),
            verified_inputs_sha256=sha256(a.output/'verified_inputs.json'),
            note='CPU assets/data only. No numerical/preflight or healing acceptance.'))
        print('A2_INPUTS_COMPLETE', flush=True)
    except BaseException as error:
        write_json(a.output/'failure.json', dict(status='failed', error_type=type(error).__name__,
            error=str(error), elapsed_seconds=time.monotonic()-started))
        raise


if __name__ == '__main__':
    main()
