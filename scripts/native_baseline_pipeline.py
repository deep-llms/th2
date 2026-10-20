"""Offline, gated preparation followed by the authorized native-baseline queue.

CPU preparation leaves the existing verified burns running. Only the established
GPU supervisor reclaims their workers, and only after every preparation gate.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def wait_for_download(data_root, model_root, contract, timeout_seconds):
    """Wait for complete, stable input sizes; the next stage verifies all hashes.

    The controller alone writes these inputs. This does not retry downloads,
    create cache files, signal processes or permit partially downloaded data.
    """
    from scripts.verify_cx_english import sha256
    manifest_path = data_root/'subsets'/contract['subset']/'manifest.json'
    marker_path = data_root/'UPLOAD_COMPLETE.json'
    # These small trusted metadata files must already have arrived before a
    # waiting pipeline is submitted. Never trust paths from unverified metadata.
    if (sha256(manifest_path) != contract['manifest_sha256']
            or sha256(marker_path) != contract['completion_sha256']):
        raise ValueError('Download metadata does not match the trusted contract')
    manifest = json.loads(manifest_path.read_text())
    files = {}
    for split in ('train', 'validation'):
        for shard in manifest['shards'][split]:
            files.update(shard['files'])
    if len(files) != contract['files']:
        raise ValueError('Unexpected download manifest file count')
    expected = [(data_root/name, meta['bytes']) for name, meta in files.items()]
    if any(not path.resolve().is_relative_to(data_root.resolve()) for path, _ in expected):
        raise ValueError('Unsafe download manifest path')
    expected += [(model_root/name, meta['bytes'])
                 for name, meta in contract['tokenizer_files'].items()]
    deadline = time.monotonic() + timeout_seconds
    previous = None
    while True:
        snapshot = []
        ready_bytes = 0
        for path, size in expected:
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            if path.is_file() and stat.st_size == size:
                snapshot.append((str(path), stat.st_size, stat.st_mtime_ns))
                ready_bytes += size
        complete = len(snapshot) == len(expected)
        print('DOWNLOAD_WAIT '+json.dumps(dict(complete_files=len(snapshot),
              expected_files=len(expected), complete_bytes=ready_bytes,
              stable=complete and snapshot == previous)), flush=True)
        if complete and snapshot == previous:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Input download did not finish within the bounded wait')
        previous = snapshot if complete else None
        time.sleep(min(30, remaining))


def burn_gate(comparison):
    from scripts.gpu_process import inspect, ownership
    state = json.loads((comparison/'supervised/supervisor.json').read_text())
    proof = json.loads((comparison/'supervised/burn-verified.json').read_text())
    live = inspect()
    if state.get('training_status') != 'ok' or state.get('handoff_error'):
        raise RuntimeError('Comparison or its final burn handoff did not pass')
    if (not proof['collective_progress_verified'] or live['guard_disabled']
            or ownership(proof['gpus']) != ownership(live['gpus'])):
        raise RuntimeError('Predecessor burn workers or ownership changed')
    print('PREDECESSOR_BURN_HANDOFF_VERIFIED', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comparison', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--model-root', type=Path, required=True)
    p.add_argument('--host', required=True)
    p.add_argument('--session', required=True)
    p.add_argument('--wait-for-download-seconds', type=int, default=0)
    a = p.parse_args()
    if a.wait_for_download_seconds < 0:
        p.error('--wait-for-download-seconds must be nonnegative')
    a.output.mkdir(parents=True, exist_ok=False)
    try:
        burn_gate(a.comparison)
        selection = a.comparison/'supervised/run/checks/selection.json'
        selected = json.loads(selection.read_text())
        if selected['status'] != 'passed':
            raise ValueError('Runtime comparison has not passed')
        if a.wait_for_download_seconds:
            contract = json.loads((ROOT/'resources/cx_english_20261020.json').read_text())
            wait_for_download(a.data_root, a.model_root, contract, a.wait_for_download_seconds)
        py = selected['python']
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'HF_HUB_OFFLINE': '1',
               'HF_DATASETS_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
               'HF_HUB_DISABLE_TELEMETRY': '1', 'WANDB_MODE': 'offline',
               'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
               'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '1'}

        def run(script, *argv, interpreter=py, environment=env):
            print('PIPELINE_STAGE '+script, flush=True)
            subprocess.run([interpreter, str(ROOT/script), *map(str, argv)],
                           cwd=ROOT, env=environment, check=True)

        assets = a.output/'assets-verified.json'
        run('scripts/verify_cx_english.py', '--data-root', a.data_root,
            '--model-root', a.model_root, '--output', assets)
        prepared = a.output/'prepared'
        run('scripts/prepare_dense_baseline.py', '--selection', selection,
            '--assets', assets, '--output', prepared)
        run('scripts/warm_training_cache.py', '--arguments', prepared/'baseline.json',
            '--output', a.output/'cache-ready.json')
        burn_gate(a.comparison)
        run('scripts/attention_env_inventory.py', '--host', a.host,
            '--compare', a.comparison/'preflight.json', '--output', a.output/'preflight.json',
            interpreter=sys.executable)
        launch_env = {**env, 'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7'}
        run('scripts/bdot_b200_launch.py', '--host', a.host, '--root', a.output,
            '--session', a.session, '--verify-python', py,
            '--jobs-config', prepared/'jobs.json', interpreter=sys.executable,
            environment=launch_env)
        print('NATIVE_BASELINE_QUEUE_SUBMITTED', flush=True)
    except BaseException as error:
        (a.output/'preparation-failed.json').write_text(json.dumps(
            dict(status='failed', error=repr(error)), indent=2))
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
