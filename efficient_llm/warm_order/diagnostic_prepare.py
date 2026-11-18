"""CPU-only validate pinned previous costs/selections and materialize immutable configs."""
import argparse,json,shutil
from pathlib import Path
from .data import sha
from .train import write_json

def prepare(base,plan):
    base=Path(base);out=base/'inputs'
    if out.exists():raise ValueError('Existing inputs; no overwrite')
    for p,want in plan['hashes'].items():
        if sha(p)!=want:raise ValueError('Input changed '+p)
    out.mkdir(parents=True);shutil.copyfile(plan['indices'],out/'indices.npz')
    for c in plan['jobs']:
        write_json(out/(c['name']+'.json'),dict(c,indices=str(out/'indices.npz'),hashes={**c['hashes'],str(out/'indices.npz'):sha(out/'indices.npz')}))
    write_json(out/'complete.json',dict(status='ready',hashes=plan['hashes'],files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print('DIAGNOSTIC_INPUTS_READY',sha(out/'complete.json'),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--plan',required=True);a=p.parse_args();prepare(a.base,json.loads(Path(a.plan).read_text()))
