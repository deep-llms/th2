"""Verify small frozen path-background results; no basis/weights download."""
import argparse,json,tarfile
from pathlib import Path
from efficient_llm.warm_order.data import sha
from efficient_llm.warm_order.path_background_study import replay

def verify(archive,expected_sha,out):
    archive,out=Path(archive),Path(out)
    if out.exists() or sha(archive)!=expected_sha:raise ValueError('Fresh output and trusted archive SHA required')
    if archive.stat().st_size>5_000_000:raise ValueError('Archive exceeds bound')
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers();names=[m.name for m in members]
        if len(names)!=len(set(names)) or sum(m.size for m in members)>20_000_000:raise ValueError('Duplicate/oversized members')
        for m in members:
            p=Path(m.name)
            if not m.isfile() or p.is_absolute() or '..' in p.parts or p.suffix=='.pt':raise ValueError('Unsafe/bulk member')
        out.mkdir(parents=True)
        for m in members:
            p=out/m.name;p.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(m) as f:p.write_bytes(f.read())
    inventory=json.loads((out/'small-files.json').read_text())
    if set(names)!=set(inventory)|{'small-files.json'}:raise ValueError('Member inventory mismatch')
    for name,h in inventory.items():
        if sha(out/name)!=h:raise ValueError('Member hash mismatch')
    top=json.loads((out/'REPORT.json').read_text());checks={}
    for seed in (0,1):
        root=out/'prepared'/f's{seed}';m=json.loads((root/'study/manifest.json').read_text());plan=json.loads((root/'plan.json').read_text())
        if sha(root/'plan.json')!=m['plan_sha256'] or m['model_pins']!=plan['parent_hashes'] or m['input_pins']!=plan['expected_inputs']:raise ValueError('Plan/provenance mismatch')
        if top['seeds'][str(seed)]!=json.loads((root/'study/REPORT.json').read_text()):raise ValueError('Report join mismatch')
        checks[str(seed)]=replay(root/'study')
    return dict(status='PASS',members=len(names),seeds=checks,archive_sha256=expected_sha,basis_and_weights_exported=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--sha256',required=True);p.add_argument('--out',required=True);a=p.parse_args();print(json.dumps(verify(a.archive,a.sha256,a.out)))
if __name__=='__main__':main()
