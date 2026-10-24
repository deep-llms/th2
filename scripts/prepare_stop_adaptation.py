"""Explicit warm starts after the completed 10k cosine schedule; never resume it."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def read(path): return json.loads(Path(path).read_text())
def write(path,value):
    with Path(path).open('x') as f: json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def initialize(checkpoint, destination, probability, *, stop_settings=None, exit_training=None):
    import torch
    from safetensors.torch import load_file, save_file
    from bdot.routing import RoutingConfig
    from bdot.stopping import StopConfig
    if read(checkpoint/'trainer_state.json')['global_step']!=10000:
        raise ValueError('STOP pilot requires the completed 10k parents')
    cfg=read(checkpoint/'config.json')
    cfg['bdot']['curriculum']='none'
    cfg['bdot_stop']={**dict(min_steps=2,train_random_prob=probability), **(stop_settings or {})}
    if exit_training is not None:
        cfg['bdot_exit_training']=dict(exit_training)
    StopConfig(**cfg['bdot_stop']).validate(RoutingConfig(**cfg['bdot']))
    weights=checkpoint/'model.safetensors'
    tensors=load_file(str(weights))
    if int(tensors['completed_steps'])!=10000:
        raise ValueError('Checkpoint/model step mismatch')
    resets={k:tensors[k].tolist() for k in
            ('completed_steps','routing_counter','training_totals','logged_training_totals')}
    for k in resets: tensors[k]=torch.zeros_like(tensors[k])
    tensors['stop_controller.norm.weight']=torch.ones(cfg['hidden_size'])
    tensors['stop_controller.head.weight']=torch.zeros(1,cfg['hidden_size'])
    tensors['stop_controller.bias']=torch.zeros(cfg['bdot']['S'])
    if exit_training is not None:
        tensors['exit_training_totals']=torch.zeros(3,dtype=torch.float64)
        tensors['logged_exit_training_totals']=torch.zeros(3,dtype=torch.float64)
    destination.mkdir(parents=True,exist_ok=False)
    save_file(tensors,str(destination/'model.safetensors'),metadata={'format':'pt'})
    write(destination/'config.json',cfg)
    # All original trainable weights, including balance bias and FiLM, survive.
    write(destination/'initialization.json',dict(parent=str(checkpoint),
        parent_weight_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
        reset_buffers=resets,optimizer='fresh AdamW',schedule='new 1000-step adaptation',
        preserved='all backbone/router weights and learned balance bias',
        changed_config=dict(curriculum='none',bdot_stop=cfg['bdot_stop'],
                            bdot_exit_training=cfg.get('bdot_exit_training'))))


def prepare(parent, output, source):
    prepared=output/'prepared';prepared.mkdir(exist_ok=False)
    jobs=[];branches={}
    common=['{python}','-m','accelerate.commands.launch','--config_file',
            str(source/'resources/accelerate_b200.yaml'),'--num_processes','8','--main_process_port','0']
    for arm in ('D','Partial-25-cosine'):
        old=read(parent/'prepared'/(arm+'-s0.train.json'))
        cp=Path(old['output_dir'])/'checkpoint-10000'
        if (old['max_steps']!=10000 or old['block_size']!=1024 or old['data_seed']!=0
                or old['train_sampling_strategy']!='sequential' or not old['dataloader_drop_last']
                or old['per_device_train_batch_size']*old['gradient_accumulation_steps']*8!=512):
            raise ValueError('Unexpected parent data-stream layout')
        for name,p in (('continue',0.),('STOP',.05)):
            label=arm+'-'+name;init=prepared/(label+'-init')
            initialize(cp,init,p)
            run=prepared/'runs'/label
            t=dict(old,model_name_or_path=str(init),config_name=str(init),output_dir=str(run),
                max_steps=1000,warmup_steps=20,learning_rate=3e-5,
                train_start_sequence=10000*512,train_parent_checkpoint=str(cp))
            t.pop('keep_checkpoint_steps',None)
            if t.get('resume_from_checkpoint') or t.get('stop_after'):
                raise ValueError('Adaptation must start a fresh optimizer and full schedule')
            path=prepared/(label+'.train.json');write(path,t)
            artifact=label+'.verified.json'
            jobs.append(dict(name=label,gpus=list(range(8)),timeout_seconds=5400,
                argv=['{python}',str(source/'scripts/bdot_job.py'),'--receipt','{run_dir}/'+artifact,
                    '--output',str(run),'--steps','1000','--',*common,str(source/'train.py'),str(path)],
                required_outputs=[dict(path=artifact,json_equals=dict(status='ok',global_step=1000))]))
            branches[label]=str(run/'checkpoint-1000')
    write(output/'branches.json',branches)
    write(output/'adaptation.json',dict(parent=str(parent),steps=1000,start_sequence=5120000,
        next_sequence=5632000,global_batch_sequences=512,input_tokens_per_branch=524288000,
        peak_lr=3e-5,warmup=20,cosine_min_lr=3e-6,branches=branches,
        meaning='Warm-start branches, not ordinary resumes; all four share the next packed data segment.'))
    return jobs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','output','source'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();write(a.output/'jobs.json',dict(jobs=prepare(a.parent,a.output,a.source)))


if __name__=='__main__':main()
