"""Verify exact small archive inventory, prepared/config joins and score replay."""
import argparse
import json
import tarfile
from pathlib import Path
import numpy as np

from efficient_llm.warm_order.data import sha
from efficient_llm.warm_order.adaptation_directions_pipeline import replay


def verify(archive,expected_sha,out,prior_study=None):
    archive,out=Path(archive),Path(out)
    if out.exists() or sha(archive)!=expected_sha:raise ValueError('Fresh output and trusted SHA required')
    if archive.stat().st_size>5_000_000:raise ValueError('Oversized archive')
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers();names=[m.name for m in members]
        if len(names)!=len(set(names)) or sum(m.size for m in members)>30_000_000:
            raise ValueError('Duplicate/oversized archive members')
        for m in members:
            p=Path(m.name)
            if not m.isfile() or p.is_absolute() or '..' in p.parts or p.suffix=='.pt' or 'moments' in p.name or 'basis' in p.name:
                raise ValueError('Unsafe/bulk member')
        out.mkdir(parents=True)
        for m in members:
            p=out/m.name;p.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(m) as f:p.write_bytes(f.read())
    read=lambda p:json.loads(p.read_text())
    inventory=read(out/'small-files.json')
    if set(names)!=set(inventory)|{'small-files.json'}:raise ValueError('Inventory mismatch')
    for name,h in inventory.items():
        if sha(out/name)!=h:raise ValueError('Member hash mismatch')
    prepared=read(out/'prepared/prepared.json');top=read(out/'REPORT.json');checks={}
    for seed in ('0','1'):
        root=out/'prepared'/f's{seed}'
        if sha(root/'config.json')!=prepared['seeds'][seed]['config_sha256']:
            raise ValueError('Prepared identity mismatch')
        config=read(root/'config.json');gate=read(root/'gate.json')
        if gate['pass_'] is not True or gate['config_sha256']!=sha(root/'config.json'):
            raise ValueError('Bound numerical gate missing')
        if config!=read(root/'study/config.json') or top['seeds'][seed]!=read(root/'study/REPORT.json'):
            raise ValueError('Joined config/report mismatch')
        checks[seed]=replay(root/'study')
        if prior_study is not None:
            old=Path(prior_study)/'prepared'/f's{seed}'/'study'
            oldmanifest=read(old/'manifest.json')
            if sha(old/'scores.npz')!=oldmanifest['files']['scores.npz']:
                raise ValueError('Prior score hash mismatch')
            newmanifest=read(root/'study/manifest.json')
            for field in ('checkpoints','input_pins','measurement_runtime'):
                if newmanifest[field]!=oldmanifest[field]:
                    raise ValueError('Prior checkpoint/row/runtime mismatch')
            with np.load(old/'scores.npz',allow_pickle=False) as a,np.load(root/'study/scores.npz',allow_pickle=False) as b:
                np.testing.assert_array_equal(a['row_ids'],b['row_ids'])
                np.testing.assert_array_equal(a['token_count'],b['token_count'])
                for branch in ('A','B'):
                    for new,previous in [('base','attn/000'),('full','attn/001')]:
                        np.testing.assert_allclose(a[branch+'/'+previous]/a['token_count'],
                                                   b[branch+'/'+new]/b['token_count'],atol=1e-4,rtol=1e-4)
            checks[seed]['prior_factor_endpoints_compared']=4
    return dict(status='PASS',members=len(names),archive_sha256=expected_sha,seeds=checks,
                model_or_reference_activations_replayed_locally=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--sha256',required=True)
    p.add_argument('--out',required=True);p.add_argument('--prior-study');a=p.parse_args()
    print(json.dumps(verify(a.archive,a.sha256,a.out,a.prior_study)))
