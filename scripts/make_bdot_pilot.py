"""Generate explicit local configurations and a sequential queue; never launch it."""
import argparse
import json
import math
import re
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('tokenizer','data','revision','manifest','accelerate_config','output'):
        p.add_argument('--'+name.replace('_','-'),required=True)
    p.add_argument('--world-size',type=int,default=8)
    p.add_argument('--micro-batch',type=int,default=16)
    p.add_argument('--tokens',type=int,default=1_000_000_000)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--routing-seed',type=int,default=0)
    p.add_argument('--backend',choices=['sdpa','fa4'],default='sdpa')
    p.add_argument('--arms',nargs='+',choices=['D','P1-A','P1-B','R','N-A','N-B'],default=['D','P1-A','P1-B'])
    a=p.parse_args()
    if a.world_size < 1 or a.micro_batch < 1 or a.tokens < 524288 or a.seed < 0:
        p.error('Invalid batch, token budget or seed')
    if not re.fullmatch(r'[0-9a-fA-F]{40}',a.revision):p.error('revision must be a pinned dataset commit hash')
    per_micro=a.world_size*a.micro_batch
    if 512 % per_micro:p.error('world-size * micro-batch must divide 512 sequences/update')
    project=Path(__file__).resolve().parents[1]
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    steps=a.tokens//524288; jobs=[]
    for arm in a.arms:
        model=json.loads((project/'configs/bdot'/f'{arm}.json').read_text())
        model['bdot']['seed']=a.routing_seed
        config=out/f'{arm}.model.json';config.write_text(json.dumps(model,indent=2)+'\n')
        args=dict(config_name=str(config),tokenizer_name=str(Path(a.tokenizer).resolve()),
            data_dir=str(Path(a.data).resolve()),output_dir=str(out/'runs'/arm),
            data_order='document_stream',dataset_revision=a.revision,
            validation_manifest=str(Path(a.manifest).resolve()),document_end_token='<|endoftext|>',
            block_size=1024,attention_backend=a.backend,isolate_documents=True,reset_document_positions=True,
            seed=a.seed,data_seed=0,bf16=True,per_device_train_batch_size=a.micro_batch,
            gradient_accumulation_steps=512//per_micro,per_device_eval_batch_size=a.micro_batch,
            max_steps=steps,warmup_steps=math.ceil(.02*steps),learning_rate=3e-4,
            lr_scheduler_type='cosine_with_min_lr',lr_scheduler_kwargs={'min_lr_rate':.1},
            adam_beta1=.9,adam_beta2=.95,weight_decay=.1,max_grad_norm=1.,
            logging_steps=10,save_steps=250,eval_strategy='steps',eval_steps=100,
            eval_max_sequences=977,report_to='none',do_eval=True,
            dataloader_drop_last=True,train_sampling_strategy='sequential',
            ddp_find_unused_parameters=True,ddp_broadcast_buffers=False)
        recipe=out/f'{arm}.train.json';recipe.write_text(json.dumps(args,indent=2)+'\n')
        # Run directories are isolated by run_experiments. Use generated argument
        # JSON inside the existing native train entry point.
        jobs.append(dict(name=arm,argv=['{python}','-m','accelerate.commands.launch',
            '--config_file',str(Path(a.accelerate_config).resolve()),'--num_processes',str(a.world_size),
            str(project/'train.py'),str(recipe)],required_outputs=[
                dict(path=f'{arm}.verified.json',json_equals={'status':'ok'})]))
    # A completion wrapper places a small receipt under the runner's run_dir.
    for job in jobs:
        arm=job['name']
        job['argv']=['{python}',str(project/'scripts/bdot_job.py'),
                     '--receipt','{run_dir}/'+arm+'.verified.json',
                     '--output',str(out/'runs'/arm),'--steps',str(steps),'--',*job['argv']]
    (out/'jobs.json').write_text(json.dumps({'jobs':jobs},indent=2)+'\n')
    (out/'budget.json').write_text(json.dumps(dict(requested_tokens=a.tokens,steps=steps,
        actual_tokens=steps*524288,discarded_budget_tokens=a.tokens%524288,
        world_size=a.world_size,global_batch_sequences=512),indent=2)+'\n')
    print(out/'jobs.json')

if __name__=='__main__':main()
