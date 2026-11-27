"""Small archive verification: loss replay locally, raw geometry moments node-only."""
import argparse,json,tarfile
from pathlib import Path
import numpy as np
from efficient_llm.warm_order.adaptation_eval import replay as replay_loss
from efficient_llm.warm_order.adaptation_timecourse import summarize
from efficient_llm.warm_order.data import sha


def replay_small_study(root):
    root=Path(root);read=lambda p:json.loads(Path(p).read_text())
    manifest=read(root/'manifest.json');config=read(root/'config.json');report=read(root/'REPORT.json')
    names=[f"step_{s['step']:06d}" for s in config['snapshots']]
    if manifest['snapshots']!=names or config['domain']!='dclm' or len(report['points'])!=len(names):raise ValueError('Study registration mismatch')
    for name,h in manifest['files'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Unsafe study member path')
        if sha(root/name)!=h:raise ValueError('Study member hash mismatch')
    trajectory=summarize([root/n/'loss' for n in names])
    for source,name in zip(trajectory['sources'],names):source['directory']=name+'/loss'
    for k,v in trajectory.items():
        if k not in ('points','interpretation') and report.get(k)!=v:raise ValueError('Loss trajectory metadata mismatch')
    checks=0
    for spec,name,point,saved in zip(config['snapshots'],names,trajectory['points'],report['points']):
        folder=root/name;replay_loss(folder/'loss');loss=read(folder/'loss/manifest.json');provenance=loss['provenance'];ck=provenance['checkpoints']
        if (ck['phase_files']!=spec['phase_files'] or ck['phase_step']!=spec['step'] or
            ck['parent_file_pins']!=config['parent_hashes'] or provenance['verified_input_file_hashes']!=config['expected_inputs']):
            raise ValueError('Loss checkpoint/input pairing changed')
        if {k:v for k,v in saved.items() if k!='geometry'}!=point:raise ValueError('Loss trajectory numeric mismatch')
        with np.load(folder/'loss/scores.npz',allow_pickle=False) as raw:ids=raw['row_ids'].copy()
        if set(saved['geometry'])!={'immediate','ordinary_drift','swapped_drift','remaining'}:raise ValueError('Geometry comparisons changed')
        for comparison in saved['geometry']:
            g=folder/'geometry'/comparison;m=read(g/'manifest.json')
            signature={k:m[k] for k in ('keys','positions','labels','batch_size','measurement_runtime')}
            if report['geometry_selection'][comparison]!=signature:raise ValueError('Geometry selection changed')
            if (m['provenance']!=ck or m['verified_input_file_hashes']!=config['expected_inputs'] or
                m['keys']!=config['geometry_keys'] or m['positions']!=config['positions'] or
                m['measurement_runtime']!=loss['measurement_runtime'] or m['comparison']!=comparison):
                raise ValueError('Geometry pairing changed')
            for member in ('REPORT.json','row_ids.npy'):
                if sha(g/member)!=m['files'][member]:raise ValueError('Geometry exported member hash mismatch')
            if not np.array_equal(ids,np.load(g/'row_ids.npy',allow_pickle=False)):raise ValueError('Geometry row pairing changed')
            if saved['geometry'][comparison]!={'metrics':read(g/'REPORT.json'),'manifest_sha256':sha(g/'manifest.json')}:raise ValueError('Geometry join changed')
            checks+=1
    return dict(loss_snapshots_replayed=len(names),geometry_summary_bindings_checked=checks,
                raw_geometry_moments_replayed_locally=False)


def verify(archive,expected_sha256,out):
    archive=Path(archive);out=Path(out)
    if sha(archive)!=expected_sha256:raise ValueError('Archive hash differs from node receipt')
    if archive.stat().st_size>25_000_000:raise ValueError('Archive exceeds small export limit')
    if out.exists():raise ValueError('Fresh local extraction directory required')
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers();names=[m.name for m in members]
        if len(names)!=len(set(names)) or sum(m.size for m in members)>100_000_000:raise ValueError('Duplicate/oversized archive')
        for m in members:
            p=Path(m.name)
            if not m.isfile() or p.is_absolute() or '..' in p.parts or p.suffix=='.pt' or p.name.startswith('moments_'):
                raise ValueError('Non-small/unsafe archive member')
        out.mkdir(parents=True)
        for m in members:
            p=out/m.name;p.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(m) as f:p.write_bytes(f.read())
    inventory=json.loads((out/'small-files.json').read_text())
    if set(names)!=set(inventory)|{'small-files.json'}:raise ValueError('Archive member inventory mismatch')
    for name,h in inventory.items():
        if sha(out/name)!=h:raise ValueError('Exported file changed')
    checks={str(s):replay_small_study(out/'study'/f's{s}') for s in (0,1)}
    report=json.loads((out/'REPORT.json').read_text())
    if report['seeds']!={str(s):json.loads((out/'study'/f's{s}'/'REPORT.json').read_text()) for s in (0,1)}:raise ValueError('Top-level study join mismatch')
    return dict(status='PASS',archive_sha256=expected_sha256,members=len(names),studies=checks,
                scope='Loss scores and joins replayed locally; geometry raw moments and model states stay on-node, not replayed here.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--sha256',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    print(json.dumps(verify(a.archive,a.sha256,a.out)),flush=True)
if __name__=='__main__':main()
