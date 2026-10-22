"""Generate explicit local configurations and a sequential queue; never launch it."""
import argparse
import json
import math
import re
from pathlib import Path

V3_RUNS = [('P1-A-prime', 0), ('V2', 0), ('C2', 0), ('V1', 0),
           ('V3', 0), ('C3', 0), ('V4', 0), ('D', 1), ('V2', 1), ('C2', 1),
           ('V5', 0), ('V6', 0), ('V7', 0)]
MLP_RUNS = [(name, 0) for base in ('Partial-25', 'Partial-50', 'MLP-only')
            for name in (base, base+'-random')]
ROUTER_RUNS = [(base+'-'+variant, 0)
              for variant in ('detach', 'cosine', 'cosine-detach')
              for base in ('Partial-25', 'Partial-50')]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('tokenizer','data','eval_data','revision','accelerate_config','output'):
        p.add_argument('--'+name.replace('_','-'),required=True)
    p.add_argument('--manifest', help='Shared packed-validation manifest; created once by the first arm')
    p.add_argument('--world-size',type=int,default=8)
    p.add_argument('--micro-batch',type=int,default=16)
    p.add_argument('--tokens',type=int,default=2_500_000_000)
    p.add_argument('--preprocessing-workers',type=int,default=128)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--routing-seed',type=int,default=0)
    p.add_argument('--backend',choices=['sdpa','fa4'],default='fa4')
    p.add_argument('--expert-backend',choices=['reference','grouped'],default='reference')
    p.add_argument('--study',choices=['v2','v3','mlp','router'],default='v2')
    p.add_argument('--arms',nargs='+',choices=['D','P1-A','P1-B','R','N-A','N-B'])
    a=p.parse_args()
    if a.study in ('v3','mlp','router') and (a.arms or a.seed!=0 or a.routing_seed!=0 or a.tokens!=2_500_000_000):
        p.error('v3/mlp use their declared arm order, model/routing seeds and 2.5B budget')
    if (a.world_size < 1 or a.micro_batch < 1 or a.tokens < 524288
            or a.seed < 0 or a.routing_seed < 0 or a.preprocessing_workers < 1):
        p.error('Invalid batch, token budget or seed')
    if not re.fullmatch(r'[0-9a-fA-F]{40}',a.revision):p.error('revision must be a pinned dataset commit hash')
    per_micro=a.world_size*a.micro_batch
    if 512 % per_micro:p.error('world-size * micro-batch must divide 512 sequences/update')
    project=Path(__file__).resolve().parents[1]
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    manifest=Path(a.manifest).resolve() if a.manifest else out/'validation.json'
    steps=a.tokens//524288; jobs=[]
    runs=(ROUTER_RUNS if a.study=='router' else MLP_RUNS if a.study=='mlp' else V3_RUNS if a.study=='v3' else
          [(arm,a.seed) for arm in (a.arms or ['D','P1-A','P1-B'])])
    for arm,seed in runs:
        name=f'{arm}-s{seed}' if a.study in ('v3','mlp','router') else arm
        model=json.loads((project/'configs/bdot'/f'{arm}.json').read_text())
        model['bdot']['seed']=seed if a.study in ('v3','mlp','router') else a.routing_seed
        model['bdot']['expert_backend']=a.expert_backend
        config=out/f'{name}.model.json';config.write_text(json.dumps(model,indent=2)+'\n')
        args=dict(config_name=str(config),tokenizer_name=str(Path(a.tokenizer).resolve()),
            data_dir=str(Path(a.data).resolve()),output_dir=str(out/'runs'/name),
            eval_data_dir=str(Path(a.eval_data).resolve()),
            data_order='packed_shuffle',dataset_revision=a.revision,
            validation_manifest=str(manifest),document_end_token='<|endoftext|>',
            preprocessing_num_workers=a.preprocessing_workers,map_batch_size=1000,
            block_size=1024,attention_backend=a.backend,isolate_documents=True,reset_document_positions=True,
            seed=seed,data_seed=0,bf16=True,tf32=False,gradient_checkpointing=False,
            optim='adamw_torch',per_device_train_batch_size=a.micro_batch,
            gradient_accumulation_steps=512//per_micro,per_device_eval_batch_size=a.micro_batch,
            max_steps=steps,warmup_steps=math.ceil(.02*steps),learning_rate=3e-4,
            lr_scheduler_type='cosine_with_min_lr',lr_scheduler_kwargs={'min_lr_rate':.1},
            adam_beta1=.9,adam_beta2=.95,weight_decay=.1,max_grad_norm=1.,
            logging_steps=10,save_steps=250,save_total_limit=None,
            eval_strategy='steps',eval_steps=100,eval_max_sequences=1024,
            report_to='none',do_eval=True,disable_tqdm=True,
            dataloader_num_workers=8,dataloader_drop_last=True,train_sampling_strategy='sequential',
            ddp_timeout=21600,ddp_find_unused_parameters=True,ddp_broadcast_buffers=False)
        if a.study in ('v3','mlp','router'):args['keep_checkpoint_steps']=[1000,2500,steps]
        recipe=out/f'{name}.train.json';recipe.write_text(json.dumps(args,indent=2)+'\n')
        # Run directories are isolated by run_experiments. Use generated argument
        # JSON inside the existing native train entry point.
        jobs.append(dict(name=name,gpus=list(range(a.world_size)),
            argv=['{python}','-m','accelerate.commands.launch',
            '--config_file',str(Path(a.accelerate_config).resolve()),'--num_processes',str(a.world_size),
            '--main_process_port','0',
            str(project/'train.py'),str(recipe)],required_outputs=[
                dict(path=f'{name}.verified.json',json_equals={'status':'ok'})]))
    # A completion wrapper places a small receipt under the runner's run_dir.
    for job in jobs:
        arm=job['name']
        job['argv']=['{python}',str(project/'scripts/bdot_job.py'),
                     *(['--resume-existing','--attempts','2'] if a.study in ('v3','mlp','router') else []),
                     '--receipt','{run_dir}/'+arm+'.verified.json',
                     '--output',str(out/'runs'/arm),'--steps',str(steps),'--',*job['argv']]
    if a.study in ('v3','mlp','router'):
        ordered=[]
        for job,(arm,seed) in zip(jobs,runs):
            job['timeout_seconds']=10800
            ordered.append(job)
            name=job['name']
            config=dict(checkpoint=str(out/'runs'/name/f'checkpoint-{steps}'),
                eval_data_dir=str(Path(a.eval_data).resolve()),tokenizer_name=str(Path(a.tokenizer).resolve()),
                output_file='{run_dir}/'+name+'/result.json',attention_backend=a.backend,
                precision='bf16',batch_size=a.micro_batch,route_study=True,
                route_sequences=4096,route_artifact_sequences=256,pilot_diagnostics=True,
                routing_seed=seed)
            eval_argv=[]
            for key,value in config.items():
                eval_argv += ['--'+key,str(value).lower() if isinstance(value,bool) else str(value)]
            ordered.append(dict(name=name+'-eval',gpus=list(range(a.world_size)),timeout_seconds=1800,
                argv=['{python}','-m','accelerate.commands.launch','--config_file',
                      str(Path(a.accelerate_config).resolve()),'--num_processes',str(a.world_size),
                      '--main_process_port','0',str(project/'eval/packed_ppl.py'),*eval_argv],
                required_outputs=[dict(path=name+'/result.json',json_equals=dict(status='ok'))]))
        jobs=ordered
    (out/'jobs.json').write_text(json.dumps({'jobs':jobs},indent=2)+'\n')
    (out/'budget.json').write_text(json.dumps(dict(requested_tokens=a.tokens,steps=steps,
        actual_tokens=steps*524288,discarded_budget_tokens=a.tokens%524288,
        world_size=a.world_size,global_batch_sequences=512,warmup_steps=math.ceil(.02*steps),
        schedule='complete; LR and routing share max_steps; no stop_after'),indent=2)+'\n')
    print(out/'jobs.json')

if __name__=='__main__':main()
