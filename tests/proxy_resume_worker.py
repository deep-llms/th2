"""Eight-rank CPU proof of proxy loss scaling, mean synchronization and native resume."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import torch
from datasets import Dataset
from safetensors.torch import load_file
from transformers import TrainingArguments

import train
from deep_kv.packing import isolated_data_collator
from deep_kv.proxy_training import ProxyTrainer
from tests.test_proxy_heads import model
from tests.test_proxy_training import proxy_fixture


def global_gradient_check(root,arm):
    rows=[]
    for i in range(16):
        ids=((torch.arange(8)+i)%30).tolist()
        labels=ids.copy()
        if i%3==0:labels[2]=-100
        rows.append(dict(input_ids=ids,labels=labels,attention_mask=[1]*8,
                         segments=(torch.arange(8)//(2+i%3)).tolist()))
    reference=model(arm,target_centering=False)
    with torch.no_grad():
        for head in reference.heads.values():head.alpha.fill_(.1)
    # Constant lambda tests normalization independently of the warmup schedule.
    out=reference(ProxyTrainer.context(isolated_data_collator(rows)))
    (out['lm_sum']/out['lm_count']+.1*out['aux_sum']/out['aux_count']).backward()
    torch.optim.SGD(reference.parameters(),lr=.01).step()
    actual=model(arm,target_centering=False)
    with torch.no_grad():
        for head in actual.heads.values():head.alpha.fill_(.1)
    actual.auxiliary_weight=lambda step:.1
    args=TrainingArguments(output_dir=str(root/(arm+'-gradient')),use_cpu=True,report_to=[],
        max_steps=1,per_device_train_batch_size=1,gradient_accumulation_steps=2,
        optim='sgd',learning_rate=.01,max_grad_norm=0,weight_decay=0,
        remove_unused_columns=False,ddp_find_unused_parameters=False,save_strategy='no',logging_strategy='no',disable_tqdm=True)
    trainer=ProxyTrainer(model=actual,args=args,train_dataset=Dataset.from_list(rows),data_collator=isolated_data_collator)
    trainer.train()
    delta=0.
    for name,value in actual.state_dict().items():
        expected=reference.state_dict()[name]
        torch.testing.assert_close(value,expected,atol=2e-7,rtol=2e-6,msg=name)
        if value.dtype != torch.bool and value.numel():delta=max(delta,(value-expected).abs().max().item())
    return delta


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1)
    state=TrainingArguments(output_dir=str(a.output),use_cpu=True,report_to=[]).distributed_state
    if state.num_processes!=8:raise ValueError('This check needs eight CPU ranks')
    root=a.output.resolve()
    if state.is_main_process:
        root.mkdir(exist_ok=False,parents=True)
        config=proxy_fixture(root,world=8);config['bf16']=True
        (root/'fixture.json').write_text(json.dumps(config))
    state.wait_for_everyone();config=json.loads((root/'fixture.json').read_text())
    result={}
    for arm in ('A','P1-flow','P3-block','P3-lambda0'):
        gradient_delta=global_gradient_check(root,arm) if arm in ('P1-flow','P3-block') else None
        for mode,stop in (('full',None),('resumed',2),('resumed',None)):
            args={**config,'arm':arm,'output_dir':str(root/arm/mode)}
            if stop is not None:args['stop_after']=stop
            invocation=root/'invocation.json'
            if state.is_main_process:invocation.write_text(json.dumps(args))
            state.wait_for_everyone()
            instances=[]
            from deep_kv.proxy_training import ProxyTrainer as RealTrainer
            class Observed(RealTrainer):
                def __init__(self,*a,**kw):super().__init__(*a,**kw);instances.append(self)
            with patch('deep_kv.proxy_training.ProxyTrainer',Observed),patch('sys.argv',['train.py',str(invocation)]):
                train.main()
            local=instances[-1].model.mu.detach().cpu().numpy().tobytes()
            checksum=hashlib.sha256(local).hexdigest()
            gathered=[None]*8;torch.distributed.all_gather_object(gathered,checksum)
            assert len(set(gathered))==1,('Rank mean disagreement',gathered)
            state.wait_for_everyone()
        if state.is_main_process:
            full=load_file(root/arm/'full/model.safetensors');resumed=load_file(root/arm/'resumed/model.safetensors')
            maximum=0.
            for name in full:
                torch.testing.assert_close(full[name],resumed[name],rtol=2e-5,atol=2e-6,msg=name)
                if full[name].dtype!=torch.bool and full[name].numel():maximum=max(maximum,(full[name]-resumed[name]).abs().max().item())
            result[arm]=dict(maximum_resume_difference=maximum,global_batch_gradient_difference=gradient_delta,
                             means_identical_across_ranks=True,mean_initialized=bool(resumed['mu_initialized']))
        state.wait_for_everyone()
    if state.is_main_process:
        (root/'verified.json').write_text(json.dumps(dict(status='passed',world_size=8,cpu_only=True,bf16=True,arms=result),indent=2))


if __name__=='__main__':main()
