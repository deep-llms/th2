"""Recover ten matched parents, audit exact phase entry, replay soft/hard NLL."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from .exposure_recovery import load_parent
from .exposure_recovery_train import train
from .soft_route import Model as SoftModel, ARMS
from .soft_exploration import Model as ExposureModel
from .soft_route_eval import Reference as SoftReference
from .soft_exploration_eval import Reference as ExposureReference, collect, audit, numerical
from .model import Config
from .data import Rows,sha
from .train import write_json
from .token_router import precision
from .strength_refinement import check
from .recursion_eval import check_train


def construct(c, reference=False):
    recipe=json.loads((Path(c['parent'])/'recipe.json').read_text())['config']
    exposed=c['name'].split('-')[0].endswith('F')
    cls=(ExposureReference if reference else ExposureModel) if exposed else (SoftReference if reference else SoftModel)
    m=cls(Config(**recipe['model']),ARMS[recipe['arm']],recipe['canonical_probability'])
    if exposed:m.disable_exploration()
    return m


@torch.no_grad()
def initial(c,device):
    m,opt,state=load_parent(c['parent'],c['parent_hashes'],c['name'],expected_steps=c['parent_steps'],device=device)
    direct=construct(c).to(device)
    w=torch.load(Path(c['parent'])/'weights.pt',map_location='cpu',weights_only=False)
    direct.load_state_dict(w['model']);m.eval();direct.eval()
    x=Rows(c['data'],c['model']['length']).batch(np.load(c['indices'])[:3],device)
    for hard in (False,True):
        if not torch.equal(m.hidden(x,hard),direct.hidden(x,hard)):raise ValueError('Parent output identity failed')
        if not torch.equal(m.nll(x,hard),direct.nll(x,hard)):raise ValueError('Parent NLL identity failed')
    proof=dict(parent_output_exact=True,fullstate_sha256=c['parent_hashes']['fullstate.pt'],
        parent_step=state['step'],inherited_lrs=[g['lr'] for g in opt.param_groups])
    proof.update(audit(m,x));return proof


def run(c,out,device='cuda'):
    out=Path(out)
    if out.exists():raise ValueError('Fresh run required')
    if str(device).startswith('cuda') and torch.cuda.device_count()!=1:raise ValueError('One GPU required')
    check(c['hashes']);precision();tc=c['train'];torch.set_num_threads(tc.get('threads',4))
    out.mkdir(parents=True);write_json(out/'config.json',c);start=time.monotonic()
    proof=numerical(device,ARMS[tc['arm']]);proof['initial']=initial(tc,device)
    write_json(out/'production-probe.json',proof)
    train(tc,out/'train',device,stop_after=min(40,tc['steps']-1))
    proof['first_checkpoint']=json.loads((out/'train/checkpoint.json').read_text())
    train(tc,out/'train',device,resume=True);done=check_train(out/'train')
    resume=json.loads((out/'train/resume.json').read_text())
    if resume['loaded_fullstate_sha256']!=proof['first_checkpoint']['sha256']:raise ValueError('Resume hash failed')
    proof['resume']=resume
    state=torch.load(out/'train/weights.pt',map_location='cpu',weights_only=False)
    m=construct(tc).to(device).eval();m.load_state_dict(state['model'])
    ref=construct(tc,True).to(device).eval();ref.load_state_dict(state['model'])
    for model in (m,ref):
        for p in model.parameters():p.requires_grad_(False)
    ev=c['eval'];ids=np.load(ev['indices']);rows={k:Rows(v,tc['model']['length']) for k,v in ev['stems'].items()}
    with torch.no_grad():
        proof.update(audit(m,rows['fit'].batch(ids['valid'][:3],device)))
        for domain in ('dclm','fwe'):
            dest=out/domain;dest.mkdir();a,parity=collect(m,ref,rows[domain],ids[domain],device)
            np.savez_compressed(dest/'summary.npz',**a);proof[domain]=dict(fullset_nll_parity=parity)
    check(c['hashes']);check_train(out/'train')
    write_json(out/'complete.json',dict(status='complete',arm=tc['arm'],seed=tc['seed'],train=done,proof=proof,
        parameters=sum(p.numel() for p in m.parameters()),candidate_blocks=6 if tc['arm']=='C' else 18,
        seconds=time.monotonic()-start,files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name not in ('weights.pt','fullstate.pt')}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cuda')
    a=p.parse_args();run(json.loads(Path(a.config).read_text()),a.out,a.device)
