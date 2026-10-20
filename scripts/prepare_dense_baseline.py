"""Prepare the authorized native 12-layer baseline and ordered verification queue."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--schedule-steps',type=int,default=30_000_000_000//524288)
    p.add_argument('--stop-after',type=int,default=5000)
    p.add_argument('--preprocessing-workers',type=int,default=128)
    a=p.parse_args()
    if not 50<a.stop_after<=a.schedule_steps:raise ValueError('Invalid full schedule/cutoff')
    selected=json.loads(a.selection.read_text());assets=json.loads(a.assets.read_text())
    if selected['status']!='passed' or assets['status']!='passed':raise ValueError('Missing verification gates')
    if selected['backend'] not in ('fa4','fa2'):raise ValueError('Unexpected backend')
    py=selected['python']
    if str(Path(sys.executable).resolve())!=str(Path(py).resolve()):raise ValueError('Prepare with the selected runtime')
    a.output.mkdir(parents=True,exist_ok=False)
    model=dict(assets['original_config']);model['num_hidden_layers']=12
    model['layer_types']=['full_attention']*12;model['use_cache']=False
    for key in ('_name_or_path','architectures','transformers_version','dtype','torch_dtype'):model.pop(key,None)
    if 'bdot' in model:raise ValueError('Native baseline cannot include routing')
    write(a.output/'model.json',model)
    base=dict(config_name=str(a.output/'model.json'),tokenizer_name=assets['tokenizer'],
        data_dir=str(Path(assets['subset_root'])/'train'),
        eval_data_dir=str(Path(assets['subset_root'])/'validation'),
        output_dir=str(a.output/'baseline'),dataset_revision=assets['contract']['revision'],
        data_order='packed_shuffle',block_size=1024,document_end_token='<|endoftext|>',
        isolate_documents=True,reset_document_positions=True,seed=0,data_seed=0,
        attention_backend=selected['backend'],bf16=True,tf32=False,
        max_steps=a.schedule_steps,stop_after=a.stop_after,
        learning_rate=3e-4,lr_scheduler_type='cosine_with_min_lr',lr_scheduler_kwargs={'min_lr_rate':.1},
        warmup_steps=math.ceil(.02*a.schedule_steps),adam_beta1=.9,adam_beta2=.95,
        weight_decay=.1,max_grad_norm=1.,optim='adamw_torch',gradient_checkpointing=False,
        per_device_train_batch_size=16,gradient_accumulation_steps=4,per_device_eval_batch_size=16,
        preprocessing_num_workers=a.preprocessing_workers,map_batch_size=1000,
        dataloader_num_workers=8,dataloader_drop_last=True,train_sampling_strategy='sequential',
        ddp_timeout=21600,ddp_find_unused_parameters=False,ddp_broadcast_buffers=False,
        save_strategy='steps',save_steps=250,save_total_limit=None,
        eval_strategy='steps',eval_steps=250,logging_steps=10,
        report_to='none',disable_tqdm=True,run_name='native-qwen3-12l-cx-en-seed0')
    write(a.output/'baseline.json',base)
    # Smoke uses a separate raw slice/output, so its cache and checkpoints cannot
    # become an accidental resume of the real baseline.
    from training_data import load_text
    first=Path(assets['subset_root'])/'train'/'shard_000000'
    raw,_=load_text(first)
    raw.save_to_disk(str(a.output/'smoke_text'))
    smoke={**base,'data_dir':str(a.output/'smoke_text'),'output_dir':str(a.output/'smoke'),
           'preprocessing_num_workers':8,'dataloader_num_workers':0,'eval_max_sequences':128,
           'save_steps':25,'eval_steps':25,'logging_steps':5,'stop_after':25,'run_name':'native-attention-smoke'}
    write(a.output/'smoke25.json',smoke)
    resume={**smoke,'stop_after':50,'resume_from_checkpoint':str(a.output/'smoke'/'checkpoint-25')}
    write(a.output/'smoke50.json',resume)
    # Validate HF schedule parsing without claiming a GPU test.
    from transformers import HfArgumentParser
    from train import ModelArguments,DataArguments,BaselineTrainingArguments
    check={**base,'use_cpu':True,'bf16':False}
    _,_,args=HfArgumentParser((ModelArguments,DataArguments,BaselineTrainingArguments)).parse_dict(check)
    if args.get_warmup_steps(a.schedule_steps)!=base['warmup_steps']:raise ValueError('Unexpected warmup mapping')
    jobs=[]
    def add(name,argv,path,equals,gpus=None,timeout=3600):
        jobs.append(dict(name=name,argv=list(map(str,argv)),gpus=[] if gpus is None else gpus,
            timeout_seconds=timeout,required_outputs=[dict(path=str(path),json_equals=equals)]))
    add('native-distributed-resume',[py,ROOT/'tests/ml/gpu_workflow_probe.py','--output','{run_dir}/resume-check',
        '--backend',selected['backend'],'--world-size','8'],'resume-check/complete.json',
        dict(status='passed',world_size=8),list(range(8)),1800)
    launch=[py,'-m','accelerate.commands.launch','--config_file',ROOT/'resources/accelerate_b200.yaml',
            '--main_process_port','0',ROOT/'train.py']
    for name,steps,start,file,output,timeout in [('real-smoke25',25,0,'smoke25.json','smoke',7200),
        ('real-smoke50',50,25,'smoke50.json','smoke',3600),
        ('baseline-5000',a.stop_after,0,'baseline.json','baseline',86400)]:
        # Results live outside the runner root; the dedicated checker writes the
        # small declared receipt there and checks native checkpoint completeness.
        argv=[py,ROOT/'scripts/run_checked_training.py','--arguments',a.output/file,
              '--receipt','{run_dir}/'+name+'.json','--expected-step',steps,'--start-step',start,
              '--schedule-steps',a.schedule_steps,'--',*launch,a.output/file]
        add(name,argv,name+'.json',dict(status='ok',global_step=steps),list(range(8)),timeout)
    write(a.output/'jobs.json',dict(jobs=jobs))
    from run_experiments import load_jobs
    load_jobs(a.output/'jobs.json')
    write(a.output/'plan.json',dict(status='prepared',selected=selected['selected'],python=py,
        backend=selected['backend'],initialization='random, native Qwen3, 12 layers',
        full_schedule_steps=a.schedule_steps,stop_after=a.stop_after,
        warmup_steps=base['warmup_steps'],global_batch_tokens=524288,
        screening_input_tokens=a.stop_after*524288,checkpoints='all; every 250 updates',
        data_note='Existing train/validation split; ordinary HF map packing and fixed seed-0 packed shuffle. No new holdout.'))
    print('BASELINE_QUEUE_PREPARED '+str(a.output/'jobs.json'),flush=True)


if __name__=='__main__':main()
