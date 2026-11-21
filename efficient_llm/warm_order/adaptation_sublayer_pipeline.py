"""Row-pinned streaming and portable replay for fixed-sublayer crossover panels."""
import json
from itertools import product
from pathlib import Path

import numpy as np
import torch

from .adaptation_sublayer import sublayer_crossovers, score_sublayer, sublayer_effects
from .adaptation_runtime import measurement_device, measurement_policy, measurement_runtime
from .data import Rows, sha
from .horizon_train import write_json


CELLS = ['00','01','10','11','S']
LABELS = [f'{branch}/{cell}' for branch in ('A','B') for cell in CELLS]


def run(config, out):
    """External launcher must qualify runtime/ownership before any CUDA call."""
    from .adaptation_pipeline import load_snapshot_panel
    if (config.get('schema')!=1 or config.get('domain')!='dclm' or config.get('target')!=2 or
            config.get('path')!=[0,2,1,3] or config.get('measurement_device') not in ('cpu','cuda')):
        raise ValueError('Unregistered sublayer study configuration')
    if config.get('flags')!=['exploratory','reused_confirmation','historical_overlap_unknown','fixed_parent_input_reader']:
        raise ValueError('Required interpretation flags missing')
    if Path(out).exists():raise ValueError('Fresh output directory required')
    models,proof=load_snapshot_panel(config['parent'],config['parent_hashes'],config['phase_files'])
    if proof['phase_step']!=config['phase_step']:raise ValueError('Unexpected continuation snapshot')
    device=config['measurement_device']
    with measurement_runtime(device):
        for model in models.values():model.to(device)
        result=evaluate_stream(models,config['data'],config['indices'],config['expected_inputs'],
                               config['batch_size'],out,proof)
    bind_config(out,config)
    return result


def bind_config(out,config):
    write_json(Path(out)/'config.json',config)
    manifest=json.loads((Path(out)/'manifest.json').read_text())
    manifest['files']['config.json']=sha(Path(out)/'config.json')
    write_json(Path(out)/'manifest.json',manifest)
    replay(out)


def summary(raw):
    if set(raw) != {'row_ids','token_count'} | set(LABELS):
        raise ValueError('Unexpected score members')
    ids,counts=raw['row_ids'],raw['token_count']
    if (ids.ndim!=1 or ids.dtype.kind not in 'iu' or not len(ids) or len(np.unique(ids))!=len(ids)
            or counts.shape!=ids.shape or counts.dtype.kind not in 'iu' or (counts<=0).any()):
        raise ValueError('Invalid paired rows/counts')
    for label in LABELS:
        v=raw[label]
        if v.shape!=ids.shape or not np.isfinite(v).all() or (v<0).any():
            raise ValueError('Invalid loss sums')
    den=int(counts.sum())
    reports={}
    for branch in ('A','B'):
        effects=sublayer_effects({cell:torch.from_numpy(raw[f'{branch}/{cell}']).double() for cell in CELLS})
        report={}
        for name,v in effects.items():
            a=v.numpy();perrow=a/counts
            report[name]=dict(gain_nat=float(a.sum()/den),q10_median_q90=np.quantile(perrow,[.1,.5,.9]).tolist(),
                              positive_rows=int((a>0).sum()),negative_rows=int((a<0).sum()))
        reports[branch]=report
    return dict(cases={label:float(raw[label].sum()/den) for label in LABELS},effects=reports,
                sequences=len(ids),tokens=den,production_ready=False,
                scope='Conditional attention/FFN group effects at fixed parent input/reader; norms included; not causal shares',
                historical_document_disjointness_verified=False)


def evaluate_stream(models,data_stem,indices_path,expected_inputs,batch_size,out,checkpoint_provenance):
    if set(models)!={'parent','A','B'} or type(batch_size) is not int or batch_size<=0:
        raise ValueError('Paired panel and positive batch required')
    if not checkpoint_provenance.get('checkpoint_bytes_and_pairing_verified'):
        raise ValueError('Verified snapshot-panel provenance required')
    out=Path(out)
    if out.exists():raise ValueError('Fresh output directory required')
    pins={p:sha(p) for p in (data_stem+'.bin',data_stem+'.starts',str(indices_path))}
    if pins!=expected_inputs:raise ValueError('Input identity mismatch')
    ids=np.load(indices_path,allow_pickle=False)
    if ids.ndim!=1 or ids.dtype.kind not in 'iu' or not len(ids) or len(np.unique(ids))!=len(ids):
        raise ValueError('Unique nonempty row selection required')
    rows=Rows(data_stem,models['parent'].c.length)
    if ids.min()<0 or ids.max()>=len(rows.ids):raise ValueError('Row out of range')
    pieces={label:[] for label in LABELS};denominators=[];parity={branch:{} for branch in ('A','B')}
    for start in range(0,len(ids),batch_size):
        x=rows.batch(ids[start:start+batch_size],measurement_device());counts=None
        for branch in ('A','B'):
            result=sublayer_crossovers(models['parent'],models[branch],x)
            scored=score_sublayer(models['parent'],models[branch],x,result)
            for cell in CELLS:
                item=scored['cells'][cell];c=item['token_count'].numpy()
                if counts is None:counts=c
                elif not np.array_equal(counts,c):raise ValueError('Case denominators differ')
                pieces[f'{branch}/{cell}'].append(item['nll_sum'].numpy())
            for key,value in result['parity_max_abs'].items():
                parity[branch][key]=max(parity[branch].get(key,0.),value)
            del result,scored
        denominators.append(counts)
    raw={label:np.concatenate(v) for label,v in pieces.items()}
    raw.update(row_ids=ids,token_count=np.concatenate(denominators))
    report=summary(raw);out.mkdir(parents=True)
    np.savez_compressed(out/'scores.npz',**raw);write_json(out/'REPORT.json',report)
    write_json(out/'manifest.json',dict(schema=1,target=2,path=[0,2,1,3],bit_order=['attention','ffn'],
        labels=LABELS,input_pins=pins,checkpoints=checkpoint_provenance,batch_size=batch_size,
        measurement_runtime=measurement_policy(),parity_max_abs=parity,
        files={name:sha(out/name) for name in ('scores.npz','REPORT.json')},
        production_ready=False,weights_exported=False,activations_exported=False))
    replay(out)
    return report


def replay(out):
    out=Path(out);m=json.loads((out/'manifest.json').read_text())
    if m['labels']!=LABELS or m['target']!=2 or m['path']!=[0,2,1,3] or m['bit_order']!=['attention','ffn']:
        raise ValueError('Sublayer registration mismatch')
    if set(m['files']) not in ({'scores.npz','REPORT.json'},{'scores.npz','REPORT.json','config.json'}):
        raise ValueError('Unexpected manifest files')
    for name,h in m['files'].items():
        if sha(out/name)!=h:raise ValueError('Sublayer output hash mismatch')
    if 'config.json' in m['files']:
        c=json.loads((out/'config.json').read_text());p=m['checkpoints']
        if (c['phase_files']!=p['phase_files'] or c['parent_hashes']!=p['parent_file_pins'] or
                c['phase_step']!=p['phase_step'] or c['expected_inputs']!=m['input_pins'] or
                c['batch_size']!=m['batch_size'] or c['target']!=m['target'] or c['path']!=m['path'] or
                c['measurement_device']!=m['measurement_runtime']['device']):
            raise ValueError('Sublayer configuration binding mismatch')
    with np.load(out/'scores.npz',allow_pickle=False) as saved:raw={k:saved[k].copy() for k in saved.files}
    if summary(raw)!=json.loads((out/'REPORT.json').read_text()):raise ValueError('Sublayer replay mismatch')
    return dict(status='PASS',cases=len(LABELS),sequences=len(raw['row_ids']),model_reexecuted=False)
