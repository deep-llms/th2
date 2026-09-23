"""Read-only ownership evidence for occupied GPUs; never signals a process."""
import hashlib
import json
from pathlib import Path

from scripts.gpu_status import snapshot


def main():
    status = snapshot()
    print('GPU_STATUS', json.dumps(status), flush=True)
    visited = set()
    for gpu in status:
        for worker in gpu['pids']:
            pid = worker
            while pid > 1 and pid not in visited:
                visited.add(pid)
                directory = Path('/proc') / str(pid)
                fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
                argv = (directory / 'cmdline').read_bytes().split(b'\0')
                record = {'pid': pid, 'ppid': int(fields[1]), 'start_ticks': int(fields[19]),
                          'argv': [s.decode(errors='replace') for s in argv if s]}
                selected = {}
                for item in (directory / 'environ').read_bytes().split(b'\0'):
                    key, _, value = item.partition(b'=')
                    if key == b'CUDA_VISIBLE_DEVICES' or key.startswith(b'GPU_BURN_'):
                        selected[key.decode()] = value.decode(errors='replace')
                record['gpu_environment'] = selected
                print('PROCESS', json.dumps(record), flush=True)
                pid = record['ppid']
    source = Path('/tmp/llm_pretrain_burn.py').read_bytes()
    if len(source) > 65536:
        raise ValueError('Unexpectedly large burn script; review separately')
    print('BURN_SHA256', hashlib.sha256(source).hexdigest(), flush=True)
    print('BURN_SOURCE_BEGIN', flush=True)
    print(source.decode(), flush=True)
    print('BURN_SOURCE_END', flush=True)
    print('OWNERSHIP_INSPECTION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
