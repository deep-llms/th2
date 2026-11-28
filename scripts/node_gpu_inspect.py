"""Read-only GPU-owner inspection receipt for a node handoff (same content as the reviewed c31dfe2a inline job).

Never signals a process. Records the GPU workers, each worker's ancestor chain (pid, ppid, start ticks, argv, exe),
the hashes of the known burn scripts and the guard script, and the guard directory listing.
  python scripts/node_gpu_inspect.py <receipt.json>   (refuses to overwrite)
"""
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.gpu_process import inspect  # noqa: E402

FILES = ['/tmp/llm_pretrain_burn.py', '/mnt/local/_gpu_guard/polite_burn.py', '/mnt/local/_gpu_guard/gpu_guard.sh',
         'resources/llm_pretrain_burn.py']


def cmd(pid):
    raw = (Path('/proc')/str(pid)/'cmdline').read_bytes()
    return [x.decode(errors='replace')[:160] for x in raw.split(b'\0') if x][:6]


def stat(pid):
    f = (Path('/proc')/str(pid)/'stat').read_text().rsplit(')', 1)[1].split()
    return int(f[1]), int(f[19])


def receipt():
    state = inspect()
    chains = {}
    for pid in state['workers']:
        chain, p = [], pid
        while p > 1 and len(chain) < 8:
            ppid, start = stat(p)
            chain.append(dict(pid=p, ppid=ppid, start_ticks=start, argv=cmd(p), exe=os.readlink(f'/proc/{p}/exe')))
            p = ppid
        chains[str(pid)] = chain
    state['ancestor_chains'] = chains
    state['files'] = {path: dict(exists=Path(path).is_file(),
                                 sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest() if Path(path).is_file() else None,
                                 bytes=Path(path).stat().st_size if Path(path).is_file() else None) for path in FILES}
    guard = Path('/mnt/local/_gpu_guard')
    state['guard_dir'] = sorted(x.name for x in guard.iterdir()) if guard.is_dir() else None
    return state


if __name__ == '__main__':
    out = Path(sys.argv[1])
    if out.exists():
        raise SystemExit(f'{out} exists')
    state = receipt()
    out.write_text(json.dumps(state, indent=1) + '\n')
    print(json.dumps(dict(host=state['host'], workers=state['workers'], stop_roots=state['stop_roots'],
                          guard_disabled=state['guard_disabled'], guard_dir=state['guard_dir'], files=state['files']),
                     indent=1))
    for pid, chain in state['ancestor_chains'].items():
        print('CHAIN', pid, ' <- '.join(f"{c['pid']}:{' '.join(c['argv'][:3])}" for c in chain))
