"""One native decoder pass per sequence; temperature reductions share logits."""
import argparse
from pathlib import Path
import torch
from transformers import set_seed
from eval.a2_family_f9_score import load
from eval.a2_family_f11 import read,write,GUARD_GRID,choose_temperature,temperature_sums
from efficient_llm.a2.model import prune,autocast
from efficient_llm.a2.data import PackedRows

@torch.no_grad()
def evaluate(model,rows,chunk=128):
    device=next(model.parameters()).device;values={str(t):[] for t in GUARD_GRID};entropy=[];counts=[]
    model.eval()
    for i in range(len(rows)):
        ids=rows[i]['input_ids'][None].to(device)
        sums={str(t):torch.zeros((),device=device,dtype=torch.float64) for t in GUARD_GRID}
        ent=torch.zeros((),device=device,dtype=torch.float64)
        with autocast(device):
            hidden=model.model(ids,use_cache=False,return_dict=True).last_hidden_state
            for start in range(0,ids.shape[1]-1,chunk):
                end=min(start+chunk,ids.shape[1]-1)
                logits=model.lm_head(hidden[:,start:end]).float().flatten(0,1)
                totals,e=temperature_sums(logits,ids[:,start+1:end+1].reshape(-1))
                for t in sums:sums[t]+=totals[t]
                ent+=e
        n=ids.shape[1]-1;counts.append(n);entropy.append(float(ent/n))
        for t in sums:values[t].append(float(sums[t]/n))
        if i%50==0:print('F11_CALIBRATION',i+1,'/',len(rows),flush=True)
    total=sum(counts)
    return dict(per_sequence_nll=values,per_sequence_entropy=entropy,token_counts=counts,
                nll={t:sum(v*n for v,n in zip(xs,counts))/total for t,xs in values.items()},
                mean_entropy=sum(e*n for e,n in zip(entropy,counts))/total,
                sequences=len(rows),tokens=total,precision='fp32_parameters_bf16_cuda_autocast',
                decoder_forwards=len(rows),chunk=chunk)

def run(entry,output):
    set_seed(42);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    model,info=load(entry,'cuda:0')
    if entry.get('oneshot'):prune(model,entry['removed_layers'])
    results={}
    for split in ('tuning','final'):
        results[split]=evaluate(model,PackedRows(entry['cx_root'],split,verify=True))
        write(out/(split+'.json'),results[split])
    selection=choose_temperature(results['tuning']['nll'])
    expected=entry['oneshot_expected_nll'] if entry.get('oneshot') else entry['final_nll']
    difference=results['final']['nll']['100']-expected
    flags=[]
    if abs(difference)>1e-4:flags.append('FT17')
    if selection['still_at_edge']:flags.append('temperature_still_at_extended_edge')
    write(out/'result.json',dict(model=entry,selection=selection,FT17=dict(passed=abs(difference)<=1e-4,
        difference=difference,expected=expected,tolerance=1e-4,same_node=True),
        flags=flags,loading_info=info,complete=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--entry',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(read(a.entry),a.output)
