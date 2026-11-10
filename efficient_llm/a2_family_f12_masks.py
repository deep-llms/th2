"""Two own-parent activation masks, once per parent, before all six fits."""
import argparse
from pathlib import Path
import torch
from accelerate import PartialState
from eval.a2_family_f12 import read,write,SPEC_SHA
from efficient_llm.a2.model import load_parent,autocast
from efficient_llm.a2.data import PackedRows
from efficient_llm.a2_family_f12_width import importance,select_neurons,slice_width,calibration_indices
from scripts.a2_family_f12_checks import Checks
from scripts.a2_family_f9_run import locked

def run(cfg,arm):
    state=PartialState();rank=state.process_index;world=state.num_processes
    root=Path(cfg['results_root']);assets=locked(cfg['assets']);stream=cfg['heal_streams']['dclm']['2030']
    locked(dict(path=str(Path(stream['path'])/'manifest.json'),sha256=stream['sha256']))
    rows=PackedRows(stream['path'],'heal',verify=False);indices=calibration_indices(len(rows))
    checks=Checks(root/'masks'/f'{arm}-rank{rank}.json');model=load_parent(assets,'dclm',arm,state.device)
    checks.record('eight_ranks',world==8,world)
    ids=rows[indices[0]]['input_ids'][:32].unsqueeze(0).to(state.device)
    with torch.inference_mode(),autocast(state.device):before=model(ids).logits
    slice_width(model,[list(range(8192)) for _ in model.model.layers])
    with torch.inference_mode(),autocast(state.device):after=model(ids).logits
    checks.record('FT20_full_width_logits',torch.equal(before,after),dict(max_abs=float((before-after).abs().max())))
    means,counts=importance(model,rows,indices,rank,world);masks=select_neurons(means,5440)
    nonembedding=slice_width(model,masks)
    checks.record('FT20_5440_shape',len(model.model.layers)==24 and abs(nonembedding/cfg['reference_13']['nonembedding_parameters']-1)<=.01,dict(actual=nonembedding,expected=cfg['reference_13']['nonembedding_parameters']))
    allchecks=[None]*world;torch.distributed.all_gather_object(allchecks,checks.items)
    if rank==0:write(root/'masks'/(arm+'.json'),dict(parent=arm,parent_weight_sha256=assets['dclm/'+arm]['files']['model.safetensors'],masks=masks,mean_abs=means.tolist(),tokens_per_layer=counts,indices=indices,seed=2031,sampling='numpy.default_rng(2031).choice(total,1024,replace=False), same indices for both parents',stream_manifest=stream['sha256'],nonembedding_parameters=nonembedding,FT20=allchecks,flags=[dict(c,rank=i) for i,items in enumerate(allchecks) for c in items if not c['passed']]))
    state.wait_for_everyone()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--arm',choices=['P300','P1T'],required=True);a=p.parse_args();cfg=read(a.config)
    if cfg['spec_sha256']!=SPEC_SHA:raise ValueError('Wrong spec')
    run(cfg,a.arm)
