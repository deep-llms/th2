"""Dev-only generation/verification of one read-only OHMV batch inspection.

Used only for ENDED sessions lacking a terminal receipt, never for reported
controller failures. No download retries, remote network, or process signals.
"""
import json
from pathlib import Path


def payload(dev, command, job, download_commit):
    dev = Path(dev)
    cfg = json.loads((dev/'resources/a2_dot_data_ohmv_20261118.json').read_text())
    plan = json.loads((dev/cfg['plan']).read_text())
    known = {}
    for corpus, count in cfg['raw_counts'].items():
        for rec in plan['corpora'][corpus]['files'][:count]:
            path = f"{cfg['data_root']}/raw/{corpus}/{rec['index']:05d}.parquet"
            known[(rec['url'], path)] = dict(path=path, basename=Path(rec['path']).name,
                                           bytes=rec['bytes'], sha256=rec['sha256'])
    records = []
    for line in command.splitlines():
        if line.startswith('--'):
            fields = line.split()
            if len(fields) != 3 or fields[0] != '--url':
                raise ValueError('Unsupported reconciliation item')
            records.append(known[tuple(fields[1:])])
    if not 1 <= len(records) <= 10 or len({r['path'] for r in records}) != len(records):
        raise ValueError('Invalid batch coverage')
    source = json.loads((dev/'resources/a2_dot_rebuild_source_20261118.json').read_text())['files']
    return dict(job=job, download_commit=download_commit, host=cfg['host'],
                output=cfg['output'], session=cfg['session'], source=source, records=records)


def command_for(p):
    # JSON is embedded as a Python string literal in a quoted heredoc: never
    # interpolated into executable shell text.
    script = '''import hashlib,json,socket,subprocess
from pathlib import Path
p=json.loads(PAYLOAD_LITERAL)
assert socket.gethostname()==p['host'], 'Wrong node'
out=Path(p['output'])
source=json.loads((out/'source-manifest.json').read_text())['files']
assert source==p['source'], 'Source inventory changed'
assert all(hashlib.sha256((out/'source'/n).read_bytes()).hexdigest()==h for n,h in source.items()), 'Source bytes changed'
assert not (out/'failure.json').exists(), 'CPU preparation failure'
alive=subprocess.run(['tmux','has-session','-t',p['session']],capture_output=True).returncode==0
complete=out/'complete.json'
ready=complete.is_file() and json.loads(complete.read_text()).get('ready') is True
assert alive or ready, 'CPU worker absent without verified completion'
records=[]
for rec in p['records']:
 path=Path(rec['path']);path=path/rec['basename'] if path.is_dir() else path
 assert path.is_file() and path.stat().st_size==rec['bytes'], 'Missing or incomplete raw file'
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 assert h.hexdigest()==rec['sha256'], 'Raw hash mismatch'
 records.append(dict(path=rec['path'],bytes=path.stat().st_size,sha256=h.hexdigest()))
print('DOWNLOAD_RECONCILE '+json.dumps(dict(job=p['job'],download_commit=p['download_commit'],host=socket.gethostname(),source=p['source'],records=records,cpu_alive=alive,ready=ready)),flush=True)
'''.replace('PAYLOAD_LITERAL', repr(json.dumps(p, sort_keys=True)))
    compile(script, 'readonly_batch_reconcile', 'exec')
    return ('#1 +30+a\n#'+p['job']+'\nset -euo pipefail\n'
            'export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1\n'
            "/mnt/local/conda-py311/envs/a2_env/bin/python3.11 - <<'PYRECONCILE'\n"+
            script+'PYRECONCILE\n')


def verify(raw, p):
    lines = [s.removeprefix('DOWNLOAD_RECONCILE ') for s in raw.splitlines()
             if s.startswith('DOWNLOAD_RECONCILE ')]
    if len(lines) != 1:
        raise ValueError('Missing or ambiguous reconciliation proof')
    result = json.loads(lines[0])
    for key in ('job', 'download_commit', 'host', 'source'):
        if result[key] != p[key]:
            raise ValueError('Reconciliation identity differs: '+key)
    expected = [{k:r[k] for k in ('path','bytes','sha256')} for r in p['records']]
    if result['records'] != expected:
        raise ValueError('Actual raw hashes/coverage differ')
    if result.get('cpu_alive') is not True and result.get('ready') is not True:
        raise ValueError('CPU status unverified')
    return result
