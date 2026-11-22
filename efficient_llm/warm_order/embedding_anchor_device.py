"""Device trainer and authenticated CPU-reference bridge.

Actual-corpus updates remain capped at40 warm-up qualification updates. This
module does not admit a full production fit or manage GPU ownership.
"""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import random

import numpy as np
import torch

from .adaptation_runtime import training_runtime, training_policy, measurement_runtime
from .attention_shift import fingerprint
from .data import sha
from .embedding_anchor import exposure
from .embedding_anchor_data import BoundRows
from .embedding_anchor_prepare import references, compare_references
from .embedding_anchor_runtime import anchor_runtime, verify_forward_bridge
from .embedding_anchor_train import CPUStreamTrainer, CPUFixtureTrainer, sources, tensor_sha
from .exposure_recovery import restore_rng
from .model import Config


def device_sources():
    root=Path(__file__).resolve().parent
    return dict(cpu=sources(), device={name:sha(root/name) for name in
                ('embedding_anchor_device.py','embedding_anchor_runtime.py',
                 'embedding_anchor_device_eval.py','embedding_anchor_segments.py','embedding_anchor_qualify.py')})


class DeviceTrainer(CPUStreamTrainer):
    def __init__(self, config, data, *, seed=0, device='cpu'):
        if device not in ('cpu','cuda'): raise ValueError('Explicit device required')
        if device=='cuda' and data.binding['fixture_only']:
            raise ValueError('Synthetic bindings cannot admit CUDA')
        if device=='cuda' and torch.cuda.device_count()!=1:
            raise ValueError('One visible GPU required')
        super().__init__(config,data,seed=seed)
        self.device=device
        self.recipe.update(device_source=device_sources(), training_policy=training_policy(device))
        self.model.to(device)
        if device=='cuda': torch.cuda.manual_seed_all(seed)

    def validate(self):
        super().validate()
        if (self.recipe['device_source']!=device_sources()
                or self.recipe['training_policy']!=training_policy(self.device)):
            raise ValueError('Device source/runtime changed')

    def update(self, e, ids):
        """Shared optimizer kernel for lineage and explicitly nonlineage probes."""
        x=self.data.batch(ids.numpy()).to(self.device)
        self.model.train()
        self.optimizer.param_groups[0]['lr']=e.lr
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.device=='cuda'):
            loss=self.model.nll(x,e.path,alpha=e.alpha,donor=e.donor).mean()
        if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.,error_if_nonfinite=True)
        self.optimizer.step()
        return dict(exposure=asdict(e),row_ids_sha256=tensor_sha(ids),
                    loss=float(loss.detach()),grad_norm=float(norm))

    def advance(self,end):
        self.validate()
        cap=768 if self.arm=='shared' else 1920
        if (type(end) is not int or not self.step<end<=cap
                or (not self.recipe['fixture_only'] and (self.arm!='shared' or end>40))):
            raise ValueError('Invalid endpoint or unadmitted production fit')
        with training_runtime(self.device),anchor_runtime(self.device):
            for s in range(self.step,end):
                e=exposure(s,'C' if self.arm=='shared' else self.arm)
                row=self.update(e,self.row_ids[e.row_start:e.row_stop])
                self.step=s+1;self.trace.append(dict(step=self.step,**row))
        self.validate()

    def state(self):
        return dict(recipe=self.recipe,step=self.step,arm=self.arm,
                    fork_sha256=self.fork_sha256,trace=self.trace,
                    parameter_names=[n for n,_ in self.model.named_parameters()],
                    model=self.model.state_dict(),optimizer=self.optimizer.state_dict(),
                    torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if self.device=='cuda' else [],
                    numpy_rng=np.random.get_state(),python_rng=random.getstate())

    def save(self,path):
        self.validate()
        with Path(path).open('xb') as f:torch.save(self.state(),f)
        return sha(path)

    @classmethod
    def restore(cls,path,digest,config,data,*,seed=0,device='cpu',fork_arm=None,prepared=None):
        if sha(path)!=digest:raise ValueError('Checkpoint hash mismatch')
        state=torch.load(path,map_location='cpu',weights_only=False)
        if prepared is None:
            t=cls(config,data,seed=seed,device=device)
        else:
            t,_=load_prepared(prepared['path'],prepared['sha256'],device=device)
            if t.model.c!=config or t.data.binding!=data.binding or t.recipe['seed']!=seed:
                raise ValueError('Prepared resume binding mismatch')
        t._restore_state(state,digest,fork_arm)
        restore_rng(state,device)
        return t


def load_prepared(root,expected_sha,*,device='cpu'):
    """Old CPU files remain immutable. No hashes or state lineage are rewritten."""
    root=Path(root)
    if sha(root/'config.json')!=expected_sha:raise ValueError('CPU config hash mismatch')
    meta=json.loads((root/'config.json').read_text())
    frozen=root.parents[1]/'source'
    for name,h in meta['source'].items():
        if sha(frozen/'efficient_llm'/name)!=h:raise ValueError('Frozen CPU source mismatch')
        if name!='warm_order/embedding_anchor.py' and sources().get(name)!=h:
            raise ValueError('CPU code changed outside device guard: '+name)
    verify_forward_bridge(frozen/'efficient_llm/warm_order/embedding_anchor.py')
    if (sha(root/'initial-fullstate.pt')!=meta['initial_sha256']
            or sha(root/'cpu-reference.pt')!=meta['reference_sha256']):
        raise ValueError('Initial/reference hash mismatch')
    data=BoundRows(meta['data_binding']['prior_recipe_pin'],seed=meta['seed'],
                   length=meta['model']['length'],fixture_only=meta['fixture_only'])
    if data.binding!=meta['data_binding'] or int(data.calibration_ids[0])!=meta['reference_row']:
        raise ValueError('Prepared data/row mismatch')
    t=DeviceTrainer(Config(**meta['model']),data,seed=meta['seed'],device=device)
    state=torch.load(root/'initial-fullstate.pt',map_location='cpu',weights_only=False)
    wanted={k:v for k,v in t.recipe.items() if k not in ('device_source','training_policy')}
    wanted['source']=meta['source']
    if (state['recipe']!=wanted or state['step']!=0 or state['arm']!='shared'
            or state['fork_sha256'] is not None or state['trace'] or state['optimizer']['state']
            or state['parameter_names']!=[n for n,_ in t.model.named_parameters()] or state['cuda_rng']):
        raise ValueError('Initial CPU fullstate identity mismatch')
    t.model.load_state_dict(state['model'],strict=True)
    t.optimizer.load_state_dict(state['optimizer']);t.validate()
    if fingerprint(t.model)!=meta['initial_fingerprint']:
        raise ValueError('Initial weight fingerprint mismatch')
    restore_rng(state,'cpu')
    if device=='cuda':torch.cuda.manual_seed_all(meta['seed'])
    t.recipe['prepared_initial']=dict(config_sha256=expected_sha,initial_sha256=meta['initial_sha256'])
    return t,meta


def check_references(trainer,meta,root):
    saved=torch.load(Path(root)/'cpu-reference.pt',map_location='cpu',weights_only=True)
    before=fingerprint(trainer.model)
    x=trainer.data.batch(np.array([meta['reference_row']])).to(trainer.device)
    with measurement_runtime(trainer.device) as policy,anchor_runtime(trainer.device):
        now=references(trainer.model,x)
        maximum=compare_references(saved,now,policy)
    if fingerprint(trainer.model)!=before:raise ValueError('Reference changed parameters')
    return dict(policy=policy,max_abs=maximum,gradient_scope=meta['gradient_scope'])
