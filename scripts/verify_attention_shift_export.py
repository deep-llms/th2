"""Recompute small correction reports and authenticate row, fit, gate and prior joins."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np

from efficient_llm.warm_order.data import sha
from efficient_llm.warm_order.attention_shift_eval import replay


def verify(archive,expected_sha,out,prior):
    archive,out=Path(archive),Path(out)
    if out.exists() or sha(archive)!=expected_sha or archive.stat().st_size>5_000_000:
        raise ValueError('Fresh output and trusted small archive required')
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers();names=[m.name for m in members]
        if len(names)!=45 or len(set(names))!=len(names) or sum(m.size for m in members)>30_000_000:
            raise ValueError('Unexpected/duplicate/oversized archive')
        for m in members:
            p=Path(m.name)
            if not m.isfile() or p.is_absolute() or '..' in p.parts or p.suffix=='.pt':raise ValueError('Unsafe member')
        out.mkdir(parents=True)
        for m in members:
            p=out/m.name;p.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(m) as f:p.write_bytes(f.read())
    read=lambda p:json.loads(Path(p).read_text())
    inventory=read(out/'small-files.json')
    if set(names)!=set(inventory)|{'small-files.json'}:raise ValueError('Archive inventory mismatch')
    for n,h in inventory.items():
        if sha(out/n)!=h:raise ValueError('Member hash mismatch')
    prepared=read(out/'prepared/prepared.json');top=read(out/'REPORT.json');checks={}
    for seed in ('0','1'):
        root=out/'prepared'/f's{seed}';c=read(root/'config.json');study=read(root/'study/config.json')
        m=read(root/'study/manifest.json');done=read(root/'complete.json')
        if (sha(root/'config.json')!=prepared['seeds'][seed]['config_sha256'] or
                c['fits']!=study['fits'] or c['mean_reference']!=study['mean_reference'] or
                done['config_sha256']!=sha(root/'config.json') or done['runs']!=study['runs'] or
                top['seeds'][seed]!=read(root/'study/REPORT.json') or
                c['parent_fingerprint']!=m['parent_fingerprint'] or c['row_binding']!=m['row_binding'] or
                len(m['terminal_calibration_cpu_gpu_parity'])!=10):raise ValueError('Prepared/study/endpoint joins failed')
        checks[seed]=replay(root/'study');traces={}
        for branch in ('C','S'):
            d=root/'work'/branch;fit=d/'fit';cfg=c['fits'][branch];recipe=read(fit/'recipe.json');gate=read(d/'gate.json')
            numerical=read(d/'numerical.json');terminal=read(fit/'complete.json');cp=read(fit/'checkpoint.json')
            if (gate['pass_'] is not True or gate['step']!=40 or not gate['exact_resume'] or
                    gate['config']!=cfg or gate['source']!=prepared['source'] or
                    gate['numerical_sha256']!=sha(d/'numerical.json') or numerical['config']!=cfg or
                    numerical['prepared_config_sha256']!=sha(root/'config.json') or
                    recipe['config']!=cfg or recipe['source']!=prepared['source'] or
                    recipe['parent_fingerprint']!=c['parent_fingerprint'] or
                    recipe['donor_loaded'] or recipe['parent_optimizer_inherited'] or
                    terminal['step']!=500 or cfg['steps']!=500 or cfg['batch']!=32 or
                    terminal['trainable_parameters']!=768 or not terminal['frozen_body_unchanged'] or
                    cp['step']!=500 or cp['sha256']!=done['runs'][branch]['sha256'] or
                    terminal['fullstate_sha256']!=cp['sha256']):raise ValueError('Fit/gate/parent binding failed')
            t=[json.loads(line) for line in (fit/'trace.jsonl').read_text().splitlines()]
            if [x['step'] for x in t]!=list(range(1,501)) or any(len(x['rows'])!=32 or x['lr']!=.001 or x['branch']!=branch for x in t):
                raise ValueError('Training trajectory mismatch')
            ids=np.asarray([row for x in t for row in x['rows']],dtype='<i8')
            if hashlib.sha256(ids.tobytes()).hexdigest()!=c['row_binding']['selected_prefix_int64_sha256']:
                raise ValueError('Training prefix changed')
            traces[branch]=ids
        np.testing.assert_array_equal(traces['C'],traces['S'])
        old=Path(prior)/'prepared'/f's{seed}'/'study';om=read(old/'manifest.json')
        if sha(old/'scores.npz')!=om['files']['scores.npz']:raise ValueError('Prior scores changed')
        if c['fits']['C']['parent_hashes']!=om['checkpoints']['parent_file_pins']:raise ValueError('Prior parent mismatch')
        with np.load(old/'scores.npz',allow_pickle=False) as a,np.load(root/'study/scores.npz',allow_pickle=False) as b:
            np.testing.assert_array_equal(a['row_ids'],b['row_ids']);np.testing.assert_array_equal(a['token_count'],b['token_count'])
            for new,oldkey in [('S/zero','B/base'),('S/mu_A','A/mean'),('S/mu_B','B/mean')]:
                np.testing.assert_allclose(b[new]/b['token_count'],a[oldkey]/a['token_count'],atol=1e-4,rtol=1e-4)
        checks[seed]['prior_endpoints']=3
    return dict(status='PASS',members=45,seeds=checks,archive_sha256=expected_sha,
                actual_weights_replayed_locally=False,scope='Reused exploratory two-seed correction trial')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--sha256',required=True)
    p.add_argument('--out',required=True);p.add_argument('--prior',required=True);a=p.parse_args()
    print(json.dumps(verify(a.archive,a.sha256,a.out,a.prior)))
