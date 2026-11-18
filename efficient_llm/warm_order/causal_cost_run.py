"""Split frozen cache jobs, CPU head fits, and independent domain evaluations."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from .causal_cost import Policy, CostHead, decisions, cached_policy
from .causal_cost_fit import cache,fit,predict,select,constant
from .model import Config,Model as Dense
from .data import Rows,sha
from .evaluate import cluster_ids
from .train import write_json
from .token_router import precision
from .strength_refinement import check
from .window_cf import PATH


def parent(c,device):
    state=torch.load(c['weights'],map_location='cpu',weights_only=False)
    if state['arm']!=c['arm'] or state['seed']!=c['seed']:raise ValueError('Wrong parent')
    model=Policy(Config(**state['config']),'WG' if c['arm']=='GF' else 'WL',state['window'],state['stride'],c['seed']).to(device).eval()
    model.load_state_dict(state['model'])
    for p in model.parameters():p.requires_grad_(False)
    return model,state


def begin(c,out):
    out=Path(out)
    if out.exists():raise ValueError('Fresh stage required')
    check(c['hashes']);precision();torch.set_num_threads(4);out.mkdir(parents=True);write_json(out/'config.json',c)
    return out


def complete(c,out,extra):
    check(c['hashes']);write_json(out/'complete.json',dict(status='complete',**extra,files={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='complete.json'}))


@torch.no_grad()
def causal(model,x):
    before=model.hidden(x,PATH).clone()
    for cut in sorted(set((1,model.c.prefix,min(model.window,model.c.length-1),model.c.length-1))):
        changed=x.clone();changed[:,cut:]=(changed[:,cut:]+3)%model.c.vocab
        if not torch.equal(before[:,:cut],model.hidden(changed,PATH)[:,:cut]):raise ValueError('Noncausal selected policy')
    return dict(canonical_causal_max=0)


def cache_job(c,out,part,device='cuda'):
    if part not in (0,1):raise ValueError('Invalid cache partition')
    out=begin(c,out);model,state=parent(c,device);indices=np.load(c['indices']);fit_ids=indices['fit'][:c['fit_n']][part::2];valid_ids=indices['valid'][c['valid_start']:c['valid_stop']][part::2]
    rows=Rows(c['stems']['fit'],model.c.length)
    if set(cluster_ids(rows,fit_ids)) & set(cluster_ids(rows,indices['valid'])):raise ValueError('Fit/validation document overlap')
    proof=causal(model,rows.batch(valid_ids[:3],device));started=time.monotonic()
    for split,ids in (('fit',fit_ids),('valid',valid_ids)):
        a=cache(model,rows,ids,device);np.savez(out/(split+'.npz'),**a)
    complete(c,out,dict(part=part,seconds=time.monotonic()-started,proof=proof,weights_sha256=sha(c['weights'])))


def load_cache(c,root,split):
    arrays=[]
    for part in (0,1):
        p=Path(root)/f'cache-{part}';done=json.loads((p/'complete.json').read_text())
        if done['status']!='complete' or done['proof']['canonical_causal_max']!=0:raise ValueError('Cache incomplete')
        for rel,h in done['files'].items():
            if sha(p/rel)!=h:raise ValueError('Changed cache')
        arrays.append(dict(np.load(p/(split+'.npz'))))
    a={k:arrays[0][k] if k=='actions' else np.concatenate([v[k] for v in arrays]) for k in arrays[0]}
    np.testing.assert_array_equal(arrays[0]['actions'],arrays[1]['actions'])
    indices=np.load(c['indices']);expected=indices['fit'][:c['fit_n']] if split=='fit' else indices['valid'][c['valid_start']:c['valid_stop']]
    position={int(v):i for i,v in enumerate(a['ids'])}
    if len(position)!=len(expected) or set(position)!=set(map(int,expected)):raise ValueError('Cache reservation changed')
    order=[position[int(v)] for v in expected]
    return {k:v if k=='actions' else v[order] for k,v in a.items()}


def fit_job(c,root):
    out=begin(c,Path(root)/'head');a=load_cache(c,root,'fit');valid=load_cache(c,root,'valid');state=torch.load(c['weights'],map_location='cpu',weights_only=False)
    fit(a,state['config'],state['window'],state['stride'],c['seed']+771,out/'fit',epochs=1,hidden=c.get('hidden',128))
    first=sha(out/'fit/fullstate.pt')
    head,trace=fit(a,state['config'],state['window'],state['stride'],c['seed']+771,out/'fit',epochs=c.get('epochs',30),resume=True,hidden=c.get('hidden',128))
    pred=predict(head,valid,state['config']['length'],state['window'],state['stride']);const=constant(a);static=const[None].expand(len(valid['nll']),-1,-1)
    selection={kind:select(v,valid,state['config']['length'],state['window'],state['stride']) for kind,v in [('learned',pred),('constant',static)]}
    torch.save(dict(constant=const,head=head.state_dict(),hidden=c.get('hidden',128),selection=selection),out/'router.pt')
    np.savez(out/'validation.npz',**{k:v for k,v in valid.items() if k!='features'},learned_cost=pred.numpy(),constant_cost=static.numpy())
    write_json(out/'selection.json',selection);write_json(out/'fit-proof.json',dict(first_epoch_fullstate_sha256=first,epochs=c.get('epochs',30),resume_from_epoch=1,trace=trace))
    complete(c,out,dict(fullstate_sha256=sha(out/'fit/fullstate.pt'),router_sha256=sha(out/'router.pt')))


@torch.no_grad()
def evaluate_job(c,root,domain,device='cuda'):
    if domain not in ('dclm','fwe'):raise ValueError('Unknown final domain')
    root=Path(root);out=begin(c,root/'eval'/domain);model,state=parent(c,device);done=json.loads((root/'head/complete.json').read_text())
    if sha(root/'head/router.pt')!=done['router_sha256']:raise ValueError('Router changed')
    router=torch.load(root/'head/router.pt',map_location='cpu',weights_only=False);ids=np.load(c['indices'])[domain];rows=Rows(c['stems'][domain],model.c.length);result=dict(ids=ids,clusters=cluster_ids(rows,ids));proof={}
    for kind in ('factual','constant','learned'):
        model.cost_head=None;model.constant_cost=None;model.policy_depth=None
        if kind!='factual':
            policy=router['selection'][kind]['selected'];model.policy_depth=policy['depth'];model.threshold=policy['threshold']
            if kind=='constant':model.constant_cost=router['constant'].to(device)
            else:
                model.cost_head=CostHead(model.c.d,router['hidden']).to(device).eval();model.cost_head.load_state_dict(router['head'])
        proof[kind]=causal(model,rows.batch(ids[:3],device));values=[];edits=[]
        for i in range(0,len(ids),8):
            values.append(model.nll(rows.batch(ids[i:i+8],device),PATH).cpu().numpy());edits.append(torch.cat(model.edit_masks,1).sum(1).cpu().numpy())
        result[kind]=np.concatenate(values);result[kind+'_edits']=np.concatenate(edits)
        if np.max(result[kind+'_edits'])>1:raise ValueError('More than one edit')
        if kind!='factual' and model.policy_depth is None:np.testing.assert_array_equal(result[kind],result['factual'])
    del model;torch.cuda.empty_cache();dense=Dense(Config(**state['config'])).to(device).eval();ds=torch.load(c['dense_weights'],map_location='cpu',weights_only=False)
    if ds['arm']!='D6C' or ds['seed']!=c['seed'] or ds['config']!=state['config']:raise ValueError('Wrong dense control')
    dense.load_state_dict(ds['model']);result['dense']=np.concatenate([dense.nll(rows.batch(ids[i:i+8],device),PATH).cpu().numpy() for i in range(0,len(ids),8)])
    np.savez(out/'summary.npz',**result);complete(c,out,dict(proof=proof,sequences=len(ids),domain=domain))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--root',required=True);p.add_argument('--stage',choices=['cache','fit','eval'],required=True);p.add_argument('--part',type=int);p.add_argument('--domain');p.add_argument('--device',default='cuda');a=p.parse_args();c=json.loads(Path(a.config).read_text())
    if a.stage=='cache':cache_job(c,Path(a.root)/f'cache-{a.part}',a.part,a.device)
    elif a.stage=='fit':fit_job(c,a.root)
    else:evaluate_job(c,a.root,a.domain,a.device)
