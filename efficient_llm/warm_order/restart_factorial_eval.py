"""Dense restart fits with unchanged pair-control production gates/evaluation."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from .pair_order_eval import numerical,initial,construct,collect
from .restart_factorial_train import train
from .soft_exploration_eval import audit
from .soft_exploration_train import write_json
from .token_router import precision
from .recursion_eval import check_train
from .strength_refinement import check
from .data import Rows,sha


def run(c,out,device='cuda'):
    out=Path(out)
    if out.exists():raise ValueError('Fresh job required')
    check(c['hashes']);precision();tc=c['train'];torch.set_num_threads(tc.get('threads',4))
    out.mkdir(parents=True);write_json(out/'config.json',c);start=time.monotonic()
    proof=numerical(device,tc['arm']);proof['initial']=initial(tc,device)
    write_json(out/'production-probe.json',proof)
    train(tc,out/'train',device,stop_after=min(40,tc['steps']-1))
    proof['first_checkpoint']=json.loads((out/'train/checkpoint.json').read_text())
    train(tc,out/'train',device,resume=True);done=check_train(out/'train')
    proof['resume']=json.loads((out/'train/resume.json').read_text())
    if proof['resume']['loaded_fullstate_sha256']!=proof['first_checkpoint']['sha256']:raise ValueError('Resume identity failed')
    state=torch.load(out/'train/weights.pt',map_location='cpu',weights_only=False)
    m=construct(tc).to(device).eval();m.load_state_dict(state['model'])
    ref=construct(tc,True).to(device).eval();ref.load_state_dict(state['model'])
    for model in (m,ref):
        for p in model.parameters():p.requires_grad_(False)
    ev=c['eval'];ids=np.load(ev['indices']);rows={k:Rows(v,tc['model']['length']) for k,v in ev['stems'].items()}
    proof.update(audit(m,rows['fit'].batch(ids['valid'][:3],device)))
    for domain in ('dclm','fwe'):
        dest=out/domain;dest.mkdir();a,parity=collect(m,ref,rows[domain],ids[domain],device)
        np.savez_compressed(dest/'summary.npz',**a);proof[domain]=dict(fullset_nll_parity=parity)
    check(c['hashes']);check_train(out/'train')
    write_json(out/'complete.json',dict(status='complete',arm=tc['arm'],seed=tc['seed'],train=done,proof=proof,
        parameters=sum(p.numel() for p in m.parameters()),trainable_parameters=done['trainable_parameters'],candidate_blocks=10,
        seconds=time.monotonic()-start,files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name not in ('weights.pt','fullstate.pt')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cuda')
    a=p.parse_args();run(json.loads(Path(a.config).read_text()),a.out,a.device)
