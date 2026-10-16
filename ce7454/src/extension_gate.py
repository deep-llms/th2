"""CPU-only future-evaluation gate for the existing four-arm queue.

Its original evaluator runs unchanged in a subprocess, releasing CUDA before
waiting. It waits for the companion's verified-free receipt, NOT its success:
an unrelated failed candidate must not invalidate the original science.
"""
import json
from pathlib import Path
import subprocess
import sys
import time


def wait_released(path):
    while True:
        if path.exists():
            data = json.loads(path.read_text())
            if data.get('gpus_free') is True and data.get('gpus') == [4, 5, 6, 7]:
                return
            raise RuntimeError('Invalid extension release receipt; fail closed')
        time.sleep(1)


def stop_interim(path, timeout=30):
    path = Path(path)
    (path/'STOP').touch()
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        released = path/'released.json'
        if released.exists():
            if json.loads(released.read_text()).get('own_evaluator_exited') is True:
                return
            raise RuntimeError('Invalid interim release receipt')
        time.sleep(.2)
    raise RuntimeError('Interim evaluator did not release; fail closed')


def main(config):
    cfg = json.loads(Path(config).read_text())
    if cfg.get('interim_watch'):
        stop_interim(cfg['interim_watch'])
    code = subprocess.call([sys.executable, cfg['original_evaluator'], *sys.argv[1:]])
    print('Original evaluation exited; waiting for companion GPU release', flush=True)
    wait_released(Path(cfg['release_receipt']))
    return code
