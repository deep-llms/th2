"""Six late-exit branches, with gates before all training; existing supervisor."""
import argparse
import hashlib
from pathlib import Path
import socket
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.prepare_stop_adaptation import initialize,read,write
from scripts.bdot_stop_launch import snapshot
from scripts.native_baseline_pipeline import burn_gate
from scripts.gpu_process import inspect

OVERLAYS=('bdot/model.py','bdot/trainer.py','bdot/stopping.py','bdot/exit_training.py',
          'resume_recipe.py','eval/stop_study.py','eval/stop_v2_study.py',
          'scripts/prepare_stop_adaptation.py','scripts/stop_v2_launch.py',
          'scripts/check_exit_training_gpu.py','scripts/stop_v2_eval_suite.py',
          'tests/bdot/test_exit_training.py','tests/bdot/test_stop_adaptation.py','tests/bdot/ddp_probe.py')
METHODS=(('LateDrop6',.5,None),('DualCE6',0.,dict(depth=6,ce_weight=.5,kl_weight=0.)),
         ('DualKD6',0.,dict(depth=6,ce_weight=.5,kl_weight=.5)))


def prepare(parent, previous, root, source):
    prepared=root/'prepared';prepared.mkdir()
    jobs=[];branches={};evaluations={}
    launch=['{python}','-m','accelerate.commands.launch','--config_file',
            str(source/'resources/accelerate_b200.yaml'),'--num_processes','8','--main_process_port','0']
    # Priority: distillation, explicit CE, then cheap late dropout; paired trunks.
    for method,p,aux in reversed(METHODS):
        for arm in ('D','Partial-25-cosine'):
            old=read(parent/'prepared'/(arm+'-s0.train.json'));cp=Path(old['output_dir'])/'checkpoint-10000'
            expected=dict(max_steps=10000,block_size=1024,data_seed=0,seed=0,
                train_sampling_strategy='sequential',dataloader_drop_last=True,
                per_device_train_batch_size=16,gradient_accumulation_steps=4,
                attention_backend='fa4',gradient_checkpointing=False,save_steps=250,save_total_limit=None)
            if any(old.get(k)!=v for k,v in expected.items()):raise ValueError('Unexpected parent recipe')
            label=arm+'-'+method;init=prepared/(label+'-init')
            initialize(cp,init,p,stop_settings=dict(min_steps=6,exit_steps=[6]),exit_training=aux)
            run=prepared/'runs'/label
            t=dict(old,model_name_or_path=str(init),config_name=str(init),output_dir=str(run),
                max_steps=1000,warmup_steps=20,learning_rate=3e-5,
                train_start_sequence=5120000,train_parent_checkpoint=str(cp))
            t.pop('keep_checkpoint_steps',None)
            if t.get('resume_from_checkpoint') or t.get('stop_after'):raise ValueError('Warm start only')
            recipe=prepared/(label+'.train.json');write(recipe,t)
            artifact=label+'.verified.json'
            jobs.append(dict(name=label,gpus=list(range(8)),timeout_seconds=7200,
                argv=['{python}',str(source/'scripts/bdot_job.py'),'--receipt','{run_dir}/'+artifact,
                      '--output',str(run),'--steps','1000','--',*launch,str(source/'train.py'),str(recipe)],
                required_outputs=[dict(path=artifact,json_equals=dict(status='ok',global_step=1000))]))
            branches[label]=str(run/'checkpoint-1000')
            evaluations[label]=dict(checkpoint=branches[label],arm=arm,method=method,progress500=True)
    for arm in ('D','Partial-25-cosine'):
        label=arm+'-continue'
        evaluations[label]=dict(checkpoint=str(previous/'prepared/runs'/label/'checkpoint-1000'),
                               arm=arm,method='continue',progress500=False)
    write(root/'branches.json',branches);write(root/'evaluation_manifest.json',evaluations)
    write(root/'adaptation.json',dict(parent=str(parent),previous=str(previous),steps=1000,
        start_sequence=5120000,next_sequence=5632000,input_tokens_per_branch=524288000,
        peak_lr=3e-5,warmup=20,min_lr=3e-6,methods=METHODS,branches=branches,
        auxiliary_note='Full LM + .5 exit CE [+ .5 forward KL], temperature 1; full-path router regularization and balance only.',
        comparisons='Equal data/updates; dual passes cost more training compute. Existing continue controls reused.'))
    return jobs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','parent','previous'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--host',required=True);p.add_argument('--session',required=True)
    a=p.parse_args()
    if socket.gethostname()!=a.host:raise ValueError('Wrong node')
    burn_gate(a.previous)
    if read(a.previous/'supervised/run/evaluation/summary.json')['status']!='ok':raise ValueError('Old study incomplete')
    original=a.previous/'source'
    for rel,h in read(original/'SOURCE_MANIFEST.json')['files'].items():
        if hashlib.sha256((original/rel).read_bytes()).hexdigest()!=h:raise ValueError('Source changed: '+rel)
    a.root.mkdir(parents=True,exist_ok=False)
    source=a.root/'source';snapshot(original,source,OVERLAYS)
    fits=prepare(a.parent,a.previous,a.root,source)
    config=read(source/'configs/bdot/Partial-25-cosine.json')
    config['bdot']['curriculum']='none'
    config['bdot_stop']=dict(train_random_prob=0.,min_steps=6,exit_steps=[6])
    config['bdot_exit_training']=dict(depth=6,ce_weight=.5,kl_weight=.5)
    write(a.root/'resume-model.json',config)
    jobs=[]
    def gate(name,argv,artifact,status='ok'):
        jobs.append(dict(name=name,gpus=list(range(8)),timeout_seconds=2400,argv=argv,
                         required_outputs=[dict(path=artifact,json_equals=dict(status=status))]))
    gate('numerics_and_capacity',['{python}',str(source/'scripts/check_exit_training_gpu.py'),
        '--root',str(a.root),'--output','{run_dir}/numerics.json'],'numerics.json')
    gate('ddp',['{python}','-m','torch.distributed.run','--standalone','--nproc_per_node=2',
        str(source/'tests/bdot/ddp_probe.py'),'{run_dir}/ddp','--exit-training','--device','cuda','--bf16'],
        'ddp/rank0.json','passed')
    gate('resume',['{python}',str(source/'tests/bdot/resume_gpu_probe.py'),'suite',
        '--output','{run_dir}/resume','--accelerate-config',str(source/'resources/accelerate_b200.yaml'),
        '--steps','8','--split','4','--soft-split','2','--curriculum','none',
        '--variant-config',str(a.root/'resume-model.json')],'resume/summary.json','passed')
    jobs.extend(fits)
    jobs.append(dict(name='evaluation',gpus=list(range(8)),timeout_seconds=10800,
        argv=['{python}',str(source/'scripts/stop_v2_eval_suite.py'),'--root',str(a.root),
              '--parent',str(a.parent),'--previous',str(a.previous),'--output','{run_dir}/evaluation'],
        required_outputs=[dict(path='evaluation/summary.json',json_equals=dict(status='ok'))]))
    write(a.root/'jobs.json',dict(jobs=jobs))
    burn_gate(a.previous);write(a.root/'preflight.json',inspect())
    subprocess.run([sys.executable,str(source/'scripts/bdot_b200_launch.py'),'--root',str(a.root),
        '--host',a.host,'--session',a.session,'--jobs-config',str(a.root/'jobs.json'),
        '--predecessor-burn-script',str(original/'resources/llm_pretrain_burn.py')],cwd=source,check=True)


if __name__=='__main__':main()
