"""Device scoring with original36-case report/replay and checkpoint joins."""
import json
from pathlib import Path
import random

import numpy as np
import torch

from .adaptation_runtime import measurement_runtime
from .attention_shift import fingerprint
from .data import sha
from .embedding_anchor_device import DeviceTrainer, device_sources
from .embedding_anchor_eval import CASES,LABELS,summarize,replay
from .embedding_anchor_runtime import anchor_runtime


def evaluate(runs,warmup,config,data,out,*,seed=0,device='cpu',prepared=None,
             step=1920,ids=None,batch=2):
    if (set(runs)!={'A','C'} or type(batch) is not int or batch<1
            or step not in (1152,1344,1536,1920)):
        raise ValueError('Invalid checkpoint evaluation')
    if not data.binding['fixture_only'] and (step!=1920 or ids is not None or prepared is None):
        raise ValueError('Production requires prepared binding/terminal/all512rows')
    ids=data.measurement_ids if ids is None else np.asarray(ids)
    if (ids.ndim!=1 or ids.dtype.kind not in 'iu' or not len(ids)
            or len(np.unique(ids))!=len(ids) or not np.isin(ids,data.measurement_ids).all()):
        raise ValueError('Only unique measurement IDs allowed')
    out=Path(out)
    if out.exists():raise ValueError('Fresh score directory required')
    python_rng,numpy_rng=random.getstate(),np.random.get_state()
    raw={k:[] for k in LABELS};arms={};fingerprints={}
    try:
        with torch.random.fork_rng(devices=[0] if device=='cuda' else []),measurement_runtime(device),anchor_runtime(device):
            ancestor=DeviceTrainer.restore(warmup['path'],warmup['sha256'],config,data,
                seed=seed,device=device,prepared=prepared)
            if ancestor.step!=768 or ancestor.arm!='shared':raise ValueError('Expected common768 ancestor')
            recipe=ancestor.recipe;trace=ancestor.trace;del ancestor
            for arm in ('A','C'):
                t=DeviceTrainer.restore(runs[arm]['path'],runs[arm]['sha256'],config,data,
                    seed=seed,device=device,prepared=prepared)
                if (t.arm!=arm or t.step!=step or t.fork_sha256!=warmup['sha256']
                        or t.recipe!=recipe or t.trace[:768]!=trace):
                    raise ValueError('Arm/step/ancestor/recipe mismatch')
                m=t.model;m.eval();before=fingerprint(m)
                with torch.no_grad():
                    for start in range(0,len(ids),batch):
                        x=data.batch(ids[start:start+batch]).to(device)
                        for name,case in CASES.items():
                            loss=m.token_losses(x,case['path'],skip=case['skip'],alpha=.5 if arm=='A' else 1.)
                            raw[arm+'/'+name].append(loss.double().sum(1).cpu().numpy())
                if fingerprint(m)!=before:raise ValueError('Scoring mutated weights')
                fingerprints[arm]=before
                arms[arm]=dict(checkpoint=dict(runs[arm]),step=step,arm=arm,
                    fork_sha256=t.fork_sha256,model_fingerprint=before)
                del t,m
    finally:
        random.setstate(python_rng);np.random.set_state(numpy_rng)
    proofs=dict(warmup=dict(warmup),warmup_step=768,step=step,arms=arms,
                data_binding=data.binding,model_config=recipe['model'],seed=seed,
                source=device_sources(),prepared=prepared)
    raw={k:np.concatenate(v) for k,v in raw.items()}
    raw.update(row_ids=ids.copy(),token_count=np.full(len(ids),config.length-config.prefix,dtype=np.int64))
    report=summarize(raw);out.mkdir(parents=True)
    np.savez_compressed(out/'scores.npz',**raw)
    (out/'REPORT.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    meta=dict(schema=2,cases=CASES,labels=LABELS,fixture_only=data.binding['fixture_only'],
        checkpoint_binding_verified=True,checkpoint_proofs=proofs,data_binding=data.binding,
        model_fingerprints=fingerprints,row_ids=ids.tolist(),tokens_per_row=config.length-config.prefix,
        files={n:sha(out/n) for n in ('scores.npz','REPORT.json')})
    (out/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n');replay(out)
    return report
