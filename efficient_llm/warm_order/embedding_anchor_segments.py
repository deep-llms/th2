"""Bounded optimizer probes, explicitly NOT1920-step production lineages.

Each segment starts from the same authenticated initialization. Schedule
origins exercise single/donor/stack kernels without pretending an initial state
has already undergone768/1152 updates. Results cannot serve as trained models.
"""
import copy
from dataclasses import asdict
from pathlib import Path

import torch

from .adaptation_runtime import training_runtime
from .attention_shift import fingerprint
from .embedding_anchor import exposure
from .embedding_anchor_device import device_sources
from .embedding_anchor_runtime import anchor_runtime
from .embedding_anchor_train import tensor_sha
from .exposure_recovery import restore_rng
from .data import sha

SEGMENTS={'single':(0,'C'),'donor-A':(768,'A'),'donor-C':(768,'C'),
          'stack-A':(1152,'A'),'stack-C':(1152,'C')}


def validate(t,record):
    origin,arm=SEGMENTS[record['segment']]
    if (record['kind']!='qualification_segment_not_lineage' or record['source']!=device_sources()
            or record['recipe']!=t.recipe or type(record['completed']) is not int
            or not 0<=record['completed']<=40 or len(record['trace'])!=record['completed']):
        raise ValueError('Invalid qualification segment')
    counts=[0]*4
    for k,row in enumerate(record['trace']):
        e=exposure(origin+k,arm);ids=t.row_ids[e.row_start:e.row_stop]
        if row['exposure']!=asdict(e) or row['row_ids_sha256']!=tensor_sha(ids):
            raise ValueError('Segment schedule/row mismatch')
        for i in e.path:counts[i]+=1
    for name,p in t.model.named_parameters():
        n=counts[int(name.split('.')[1])] if name.startswith('blocks.') else record['completed']
        state=t.optimizer.state.get(p,{})
        if not torch.isfinite(p).all():raise ValueError('Nonfinite probe parameter')
        if n==0:
            if state:raise ValueError('Inactive segment parameter advanced')
        elif (set(state)!={'step','exp_avg','exp_avg_sq'} or state['step'].item()!=n
              or state['exp_avg'].shape!=p.shape or state['exp_avg_sq'].shape!=p.shape
              or state['exp_avg'].dtype!=p.dtype or state['exp_avg_sq'].dtype!=p.dtype
              or not torch.isfinite(state['exp_avg']).all() or not torch.isfinite(state['exp_avg_sq']).all()
              or (state['exp_avg_sq']<0).any()):
            raise ValueError('Segment Adam counter/moment mismatch')


def run(t,label,out,end=40,*,resume=None):
    if label not in SEGMENTS or type(end) is not int or not 1<=end<=40:
        raise ValueError('Registered bounded segment required')
    # Caller creates a fresh, authenticated initialization; never a fitted model.
    t.validate()
    if t.step!=0 or t.trace or t.optimizer.state:raise ValueError('Segment starts at initialization only')
    record=dict(kind='qualification_segment_not_lineage',segment=label,completed=0,trace=[],
                recipe=copy.deepcopy(t.recipe),source=device_sources(),initial_fingerprint=fingerprint(t.model))
    if resume is not None:
        if sha(resume['path'])!=resume['sha256']:raise ValueError('Segment checkpoint hash mismatch')
        saved=torch.load(resume['path'],map_location='cpu',weights_only=False)
        if (saved['record']['segment']!=label
                or saved['record']['initial_fingerprint']!=record['initial_fingerprint']
                or saved['state']['parameter_names']!=[n for n,_ in t.model.named_parameters()]):
            raise ValueError('Segment/initialization/optimizer identity mismatch')
        t.model.load_state_dict(saved['state']['model'],strict=True)
        t.optimizer.load_state_dict(saved['state']['optimizer'])
        record=saved['record'];validate(t,record);restore_rng(saved['state'],t.device)
    if not record['completed']<end:raise ValueError('Segment already reached endpoint')
    origin,arm=SEGMENTS[label]
    with training_runtime(t.device),anchor_runtime(t.device):
        for k in range(record['completed'],end):
            e=exposure(origin+k,arm)
            row=t.update(e,t.row_ids[e.row_start:e.row_stop]);record['trace'].append(row)
            record['completed']=k+1
    validate(t,record)
    # state intentionally contains no lineage step/arm/fork fields.
    state=t.state();state={k:state[k] for k in ('parameter_names','model','optimizer','torch_rng','cuda_rng','numpy_rng','python_rng')}
    with Path(out).open('xb') as f:torch.save(dict(record=record,state=state),f)
    return dict(path=str(out),sha256=sha(out))
