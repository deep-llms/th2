"""Ten frozen cross-path endpoints; small score replay, no donor-model fitting."""
from pathlib import Path

import numpy as np
import torch

from .attention_shift import PATHS, freeze, fingerprint, shifted
from .attention_shift_train import read, selections, validate_optimizer, source_hashes
from .adaptation_train import load_parent
from .adaptation_runtime import measurement_runtime, measurement_policy
from .layer_dependence import paired_scores
from .horizon_train import write_json
from .data import sha

VECTORS=('zero','mu_A','mu_B','learned_C','learned_S')
LABELS=[path+'/'+v for path in PATHS for v in VECTORS]


def load_bias(root, expected_config, expected_sha, width, body_digest):
    root=Path(root);checkpoint=read(root/'checkpoint.json');done=read(root/'complete.json')
    if sha(root/'fullstate.pt')!=expected_sha or checkpoint['sha256']!=expected_sha or done['fullstate_sha256']!=expected_sha:
        raise ValueError('Correction terminal hash mismatch')
    state=torch.load(root/'fullstate.pt',map_location='cpu',weights_only=False)
    if (read(root/'recipe.json')!=state['recipe'] or
            state['recipe']['config']!=expected_config or state['recipe']['source']!=source_hashes() or
            state['recipe']['parameter_names']!=['attention_shift'] or state['recipe']['donor_loaded'] or
            state['recipe']['parent_optimizer_inherited'] or state['recipe']['parent_fingerprint']!=body_digest or
            state['step']!=expected_config['steps'] or checkpoint['step']!=state['step'] or
            done['step']!=state['step'] or not done['frozen_body_unchanged']):
        raise ValueError('Correction recipe/parent/completion mismatch')
    validate_optimizer(state['optimizer'],state['step'],width)
    b=state['bias']
    if b.shape!=(width,) or b.dtype!=torch.float32 or not torch.isfinite(b).all():
        raise ValueError('Invalid learned vector')
    return b,dict(fullstate_sha256=expected_sha,recipe_sha256=sha(root/'recipe.json'),
                  parent_fingerprint=body_digest,step=state['step'],branch=expected_config['branch'])


def load_means(spec, c, width):
    root=Path(spec['directory'])
    if sha(root/'calibration.json')!=spec['metadata_sha256']:raise ValueError('Mean metadata hash mismatch')
    meta=read(root/'calibration.json');binding=meta['binding']
    if (binding['checkpoints']['parent_file_pins']!=c['parent_hashes'] or
            binding['data_pins']!={p:c['expected_inputs'][p] for p in (c['data']+'.bin',c['data']+'.starts')} or
            binding['calibration_sha256']!=c['expected_inputs'][c['calibration_indices']]):
        raise ValueError('Mean reference parent/data binding mismatch')
    means={}
    for branch in ('A','B'):
        p=root/(branch+'-basis.pt');digest=spec['basis_hashes'][branch]
        if sha(p)!=digest or meta['node_files'][p.name]!=digest:raise ValueError('Mean vector hash mismatch')
        fitted=torch.load(p,map_location='cpu',weights_only=True);mean=fitted['mean']
        if mean.shape!=(width,) or not torch.isfinite(mean).all():raise ValueError('Invalid donor mean')
        means['mu_'+branch]=mean.float()
    return means


def summarize(raw):
    if set(raw)!=set(LABELS)|{'row_ids','token_count'}:raise ValueError('Endpoint registration mismatch')
    ids=raw['row_ids'];counts=raw['token_count']
    if (ids.ndim!=1 or ids.dtype.kind not in 'iu' or not len(ids) or len(np.unique(ids))!=len(ids) or
            counts.shape!=ids.shape or counts.dtype.kind not in 'iu' or (counts<=0).any()):
        raise ValueError('Invalid row/count arrays')
    report={};den=int(counts.sum())
    for label in LABELS:
        v=raw[label]
        if v.shape!=ids.shape or not np.isfinite(v).all() or (v<0).any():raise ValueError('Invalid endpoint sums')
        difference=raw[label.split('/')[0]+'/zero']-v;perrow=difference/counts
        report[label]=dict(nll_nat=float(v.sum()/den),gain_vs_zero_nat=float(difference.sum()/den),
                           q10_median_q90=np.quantile(perrow,[.1,.5,.9]).tolist(),
                           positive_rows=int((difference>0).sum()),negative_rows=int((difference<0).sum()))
    return dict(cases=report,sequences=len(ids),tokens_per_case=den,
                scope='Frozen-body constant correction, exploratory/reused rows; no portability/generalization claim')


def evaluate(configs, runs, reference, out, *, device='cpu', batch=2):
    if set(configs)!=set(PATHS) or set(runs)!=set(PATHS):raise ValueError('Both matched fits required')
    if {k:v for k,v in configs['C'].items() if k!='branch'}!={k:v for k,v in configs['S'].items() if k!='branch'}:
        raise ValueError('Correction fit configurations are unmatched')
    c=configs['C'];out=Path(out)
    if out.exists() or type(batch) is not int or batch<1:raise ValueError('Fresh output and positive batch required')
    with torch.random.fork_rng(devices=[]):model,opt,state,_=load_parent(c['parent'],c['parent_hashes'])
    del opt,state
    freeze(model);body_digest=fingerprint(model);rows,_,cal,ids,row_binding=selections(c,model.c.length)
    vectors={'zero':torch.zeros(model.c.d)};proofs={}
    for b in PATHS:
        vectors['learned_'+b],proofs[b]=load_bias(runs[b]['path'],configs[b],runs[b]['sha256'],model.c.d,body_digest)
    vectors.update(load_means(reference,c,model.c.d))
    vector_norms={k:float(v.norm()) for k,v in vectors.items()}
    model.eval();endpoint_parity={}
    if device=='cuda':
        # Verify the ACTUAL terminal corrections, not just a small test vector.
        x=rows.batch(cal[:1],'cpu');cpu_reference={}
        with torch.no_grad(),measurement_runtime('cpu'):
            for path in PATHS:
                for name in VECTORS:
                    cpu_reference[path+'/'+name]=shifted(model,x,vectors[name],path,hidden=True).cpu()
        model.to(device);x=x.to(device)
        with torch.no_grad(),measurement_runtime(device) as policy:
            for path in PATHS:
                for name in VECTORS:
                    h=shifted(model,x,vectors[name].to(device),path,hidden=True).cpu()
                    expected=cpu_reference[path+'/'+name]
                    torch.testing.assert_close(h,expected,atol=policy['component_atol'],rtol=policy['component_rtol'])
                    endpoint_parity[path+'/'+name]=float((h-expected).abs().max())
        del cpu_reference,x,h,expected
    model.to(device);raw={k:[] for k in LABELS};counts=[];out.mkdir(parents=True)
    with torch.no_grad(),measurement_runtime(device):
        policy=measurement_policy()
        for start in range(0,len(ids),batch):
            x=rows.batch(ids[start:start+batch],device);denominator=None
            for path in PATHS:
                base=model.hidden(x,PATHS[path])
                for name in VECTORS:
                    h=shifted(model,x,vectors[name].to(device),path,hidden=True)
                    if name=='zero':torch.testing.assert_close(h,base,atol=0,rtol=0)
                    score=paired_scores(model,x,h,base)
                    current=score['token_count'].numpy()
                    if denominator is not None and not np.array_equal(current,denominator):raise ValueError('Case denominator mismatch')
                    denominator=current;raw[path+'/'+name].append(score['nll_sum'].numpy())
            counts.append(denominator);write_json(out/'progress.json',dict(rows=min(start+batch,len(ids)),total=len(ids)))
    if fingerprint(model)!=body_digest:raise ValueError('Evaluation mutated frozen parent')
    raw={k:np.concatenate(v) for k,v in raw.items()};raw.update(row_ids=ids,token_count=np.concatenate(counts))
    report=summarize(raw);np.savez_compressed(out/'scores.npz',**raw);write_json(out/'REPORT.json',report)
    config=dict(fits=configs,runs=runs,mean_reference=reference,measurement_device=device,batch_size=batch)
    write_json(out/'config.json',config)
    write_json(out/'manifest.json',dict(labels=LABELS,proofs=proofs,row_binding=row_binding,input_pins=c['expected_inputs'],
        row_ids=ids.tolist(),measurement_policy=policy,parent_fingerprint=body_digest,vector_norms=vector_norms,
        reference_metadata_sha256=config['mean_reference']['metadata_sha256'],weights_or_vectors_exported=False,
        terminal_calibration_cpu_gpu_parity=endpoint_parity,
        files={n:sha(out/n) for n in ('scores.npz','REPORT.json','config.json')}))
    replay(out);return report


def replay(out):
    out=Path(out);m=read(out/'manifest.json')
    if m['labels']!=LABELS or set(m['files'])!={'scores.npz','REPORT.json','config.json'}:raise ValueError('Manifest registration mismatch')
    for n,h in m['files'].items():
        if sha(out/n)!=h:raise ValueError('Output hash mismatch')
    c=read(out/'config.json')
    if (c['measurement_device']!=m['measurement_policy']['device'] or
            c['fits']['C']['expected_inputs']!=m['input_pins'] or
            c['mean_reference']['metadata_sha256']!=m['reference_metadata_sha256']):raise ValueError('Output binding mismatch')
    for branch in PATHS:
        if c['runs'][branch]['sha256']!=m['proofs'][branch]['fullstate_sha256'] or m['proofs'][branch]['parent_fingerprint']!=m['parent_fingerprint']:
            raise ValueError('Correction checkpoint binding mismatch')
    with np.load(out/'scores.npz',allow_pickle=False) as f:raw={k:f[k] for k in f.files}
    if raw['row_ids'].tolist()!=m['row_ids'] or summarize(raw)!=read(out/'REPORT.json'):raise ValueError('Score replay mismatch')
    return dict(status='PASS',cases=10,sequences=len(raw['row_ids']),model_or_vectors_reexecuted=False)
