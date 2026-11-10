"""Read-only filesystem/GPU census before Family v1.9 admission (no scoring)."""
import json
import os
from pathlib import Path
import socket
from datetime import datetime, timezone
from scripts.gpu_process import inspect


def census():
    roots = [Path('/mnt/local') / x / 'deep-llms_th2'
             for x in ('_models', '_outputs', '_backup')]
    rows, errors = [], []
    for root in roots:
        for directory, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [d for d in dirs if d not in ('source', '.git', '__pycache__')]
            weights = [f for f in files if f.endswith('.safetensors') or f == 'pytorch_model.bin']
            if not weights:
                continue
            p = Path(directory)
            record = dict(path=str(p), weights={f:(p/f).stat().st_size for f in weights},
                          config=None, provenance={})
            for name, file in [('config',p/'config.json')]:
                if file.is_file():
                    try: record[name]=json.loads(file.read_text())
                    except Exception as e: errors.append(dict(path=str(file),error=repr(e)))
            for parent in (p, p.parent, p.parent.parent):
                for name in ('recipe.json','result.json','summary.json'):
                    file=parent/name
                    if file.is_file() and file.stat().st_size<100_000_000:
                        try:
                            v=json.loads(file.read_text())
                            record['provenance'][str(file)]={k:x for k,x in v.items()
                                if k not in ('lr_updates','per_sequence_nll','per_sequence_tokens','checks','tests')}
                            if 'scores' in record['provenance'][str(file)]:
                                record['provenance'][str(file)]['scores']={k:({a:b for a,b in x.items()
                                    if not a.startswith('per_sequence')} if isinstance(x,dict) else x)
                                    for k,x in v['scores'].items()}
                        except Exception as e: errors.append(dict(path=str(file),error=repr(e)))
            rows.append(record)
    return dict(utc=datetime.now(timezone.utc).isoformat(),host=socket.gethostname(),
                inspection=inspect(),models=rows,errors=errors,
                stage='metadata census only; FT12 weight hashing/loading still required')


if __name__=='__main__':
    print('F9_CENSUS_JSON '+json.dumps(census()),flush=True)
