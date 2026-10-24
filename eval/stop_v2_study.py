"""One eligible exit at six: frozen head, independent calibration, PPL and decode."""
import argparse
from contextlib import nullcontext
from dataclasses import replace
import gc
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoTokenizer
from bdot.model import BDOTForCausalLM
from eval.stop_study import (read,write,fit_head,score,long_fragments,cached_policy_check,latency)
from train import DataArguments,prepare_validation_data
from training_data import prepare_training_data,require_resume_data


def select_policy(calibration,reference_loss,tolerance=.01):
    full=calibration[0]
    # A model cannot hide full-depth degradation by calibrating only to itself.
    ceiling=min(full['loss'],reference_loss)+tolerance
    feasible=[r for r in calibration if r['policy']['policy'] in ('none','learned') and r['loss']<=ceiling]
    chosen=min(feasible,key=lambda r:r['mean_middle_steps']) if feasible else full
    return chosen,dict(reference_loss=reference_loss,ceiling=ceiling,feasible=bool(feasible),
                      selection='Lowest mean depth within 0.01 nats of BOTH own full path and matched ordinary control')


def single_exit_model(checkpoint,backend):
    m=BDOTForCausalLM.from_pretrained(checkpoint,attn_implementation=backend).cuda().eval()
    # Old ordinary controls share the same six-only inference policy, head
    # initialization and fit data; their trunk weights remain untouched.
    m.stop_controller.options=replace(m.stop_controller.options,min_steps=6,exit_steps=(6,))
    with torch.no_grad():
        m.stop_controller.head.weight.zero_();m.stop_controller.bias.zero_();m.stop_controller.norm.weight.fill_(1)
    return m


def run(checkpoint,parent_recipe,previous,arm,output,progress500):
    from packed_fa4 import register_backend
    backend=register_backend();torch.set_num_threads(1)
    recipe=read(parent_recipe)
    tokenizer=AutoTokenizer.from_pretrained(recipe['tokenizer_name'],local_files_only=True)
    data_args=DataArguments(**{k:v for k,v in recipe.items() if k in DataArguments.__dataclass_fields__})
    settings=SimpleNamespace(data_seed=0,seed=0,main_process_first=lambda **kw:nullcontext())
    train,identity=prepare_training_data(data_args,settings,tokenizer,recipe['tokenizer_name'],1024,'fa4')
    require_resume_data(Path(recipe['output_dir'])/'checkpoint-10000',identity)
    start=5632000;fit=train.select(range(start,start+256));cal=train.select(range(start+256,start+512))
    reference=read(previous/'supervised/run/evaluation/summary.json')['results'][arm+'-continue']['calibration'][0]['loss']
    output.mkdir(parents=True,exist_ok=False)
    progress={}
    for step in ([500,1000] if progress500 else [1000]):
        cp=Path(checkpoint).parent/f'checkpoint-{step}'
        m=single_exit_model(cp,backend)
        if int(m.completed_steps)!=step:raise ValueError('Incomplete checkpoint')
        progress[str(step)]=[score(m,cal,'fa4',p) for p in
                            (dict(policy='none'),dict(policy='fixed',fixed_depth=6))]
        if step!=1000:del m;gc.collect();torch.cuda.empty_cache()
    write(output/'progress.json',progress)
    head=fit_head(m,fit,'fa4');torch.save(m.stop_controller.state_dict(),output/'stop_head.pt')
    write(output/'head.json',dict(status='ok',training_rows=[start,start+256],calibration_rows=[start+256,start+512],**head))
    policies=[dict(policy='none'),dict(policy='fixed',fixed_depth=6),
              *[dict(policy='learned',threshold=t) for t in (.1,.25,.4,.5,.6,.7,.8,.9,.95,.99)]]
    calibration=[score(m,cal,'fa4',p) for p in policies]
    if not progress500 and abs(calibration[0]['loss']-reference)>1e-4:
        raise ValueError('Ordinary-control calibration changed')
    chosen,selection=select_policy(calibration,reference)
    # Exactly one decision at depth 6: E[depth] = 8 - 2*p.
    matched=dict(policy='constant',constant_prob=max(0.,min(1.,(8-chosen['mean_middle_steps'])/2)))
    calibration.append(score(m,cal,'fa4',matched))
    write(output/'calibration.json',dict(status='ok',results=calibration,chosen=chosen,
                                        selection=selection,matched_random=matched))
    validation,_=prepare_validation_data(replace(data_args,eval_max_sequences=None),settings,
        tokenizer,recipe['tokenizer_name'],1024,'fa4',identity)
    def unique(items):
        result=[]
        for x in items:
            if x not in result:result.append(x)
        return result
    tested=unique([dict(policy='none'),dict(policy='fixed',fixed_depth=6),chosen['policy'],matched])
    final=[score(m,validation,'fa4',p) for p in tested]
    if any((r['num_tokens'],r['num_sequences'])!=(9915698,9704) for r in final):raise ValueError('Validation count changed')
    expected=read(Path(checkpoint).parent/'final_eval.json')['eval_loss']
    if abs(final[0]['loss']-expected)>1e-4:raise ValueError('Full-depth carry failed')
    m.set_attn_implementation('sdpa');tokens=long_fragments(cal)
    cached={label:cached_policy_check(m,p,tokens[0]) for label,p in
            [('full',dict(policy='none')),('fixed6',dict(policy='fixed',fixed_depth=6)),('chosen',chosen['policy'])]}
    times=[latency(m,p,tokens,b,context=c) for p in tested for b in (1,32) for c in (128,512)]
    write(output/'result.json',dict(status='ok',checkpoint=str(checkpoint),checkpoint_step=1000,parent_step=10000,
        chosen=chosen,selection=selection,head=head,calibration=calibration,final=final,latency=times,
        cached_checks=cached,progress=progress,
        scope='One seed, six-only exits, matched ordinary controls; equal tokens/updates, unequal training compute.'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','parent-recipe','previous','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--arm',required=True);p.add_argument('--progress500',action='store_true')
    a=p.parse_args();run(a.checkpoint,a.parent_recipe,a.previous,a.arm,a.output,a.progress500)


if __name__=='__main__':main()
