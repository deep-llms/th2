"""Verify native Trainer completion for the generic sequential job runner."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def latest_complete_checkpoint(out):
    """A failed write must not hide the preceding complete optimizer save."""
    from scripts.training_results import file_records,weight_names
    candidates=sorted((p for p in out.glob('checkpoint-*') if p.name[11:].isdigit()),
                      key=lambda p:int(p.name[11:]),reverse=True)
    for path in candidates:
        if path.is_symlink():raise RuntimeError('Refusing symlinked checkpoint')
        try:
            world=json.loads((path/'training_recipe.json').read_text())['execution']['world_size']
            file_records(path,['trainer_state.json','config.json','train_config.json',
                'training_recipe.json','data_recipe.json','optimizer.pt','scheduler.pt',
                *weight_names(path),*(['rng_state.pth'] if world==1 else
                                     [f'rng_state_{i}.pth' for i in range(world)])])
            if json.loads((path/'trainer_state.json').read_text())['global_step']==int(path.name[11:]):
                return path
        except (OSError,ValueError,KeyError):
            continue
    raise RuntimeError('No complete checkpoint: refusing nonempty-run restart')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--receipt',required=True);p.add_argument('--output',required=True)
    p.add_argument('--resume-existing',action='store_true')
    p.add_argument('--attempts',type=int,default=1)
    p.add_argument('--steps',required=True,type=int);p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args();out=Path(a.output);receipt=Path(a.receipt)
    if a.attempts not in (1,2):p.error('At most one infrastructure retry is supported')
    if out.exists() and any(out.iterdir()) and not a.resume_existing:
        p.error('Pilot jobs require fresh output directories')
    if receipt.exists():p.error('Receipt already exists')
    command=a.command[1:] if a.command and a.command[0]=='--' else a.command
    if not command:p.error('Missing training command')
    receipt.parent.mkdir(parents=True,exist_ok=True)
    attempts=[]
    for attempt in range(a.attempts):
        actual_command=list(command);resumed_from=None
        if a.resume_existing and out.exists() and any(out.iterdir()):
            checkpoint=latest_complete_checkpoint(out)
            if not command[-1].endswith('.json'):raise RuntimeError('Resume requires a JSON Trainer recipe')
            config=json.loads(Path(command[-1]).read_text())
            config['resume_from_checkpoint']=str(checkpoint.resolve())
            resumed_from=str(checkpoint.resolve())
            resume_args=receipt.with_name(receipt.stem+f'.resume{attempt+1}.json')
            with resume_args.open('x') as handle:json.dump(config,handle,indent=2)
            actual_command[-1]=str(resume_args)
            # train.py owns completeness/recipe checks and native HF restore.
        if a.attempts==1:
            subprocess.run(actual_command,check=True)
            break
        log=receipt.with_name(receipt.stem+f'.attempt{attempt+1}.log')
        print('TRAIN_ATTEMPT '+str(log),flush=True)
        with log.open('x') as handle:
            run=subprocess.run(actual_command,stdout=handle,stderr=subprocess.STDOUT)
        attempts.append(dict(attempt=attempt+1,returncode=run.returncode,log=str(log),resumed_from=resumed_from))
        if run.returncode==0:break
        tail=log.read_text(errors='replace')[-16000:]
        print(tail,flush=True)
        infrastructure=run.returncode<0 or any(marker in tail for marker in (
            'torch.OutOfMemoryError', 'CUDA out of memory', 'SignalException',
            'NCCL error', 'ncclSystemError'))
        if not infrastructure or attempt+1>=a.attempts:
            raise RuntimeError('Training failed; batch stopped for review (see '+str(log)+')')
        from scripts.gpu_status import require_free
        require_free(list(range(8)))
    metrics=json.loads((out/'train_results.json').read_text())
    if metrics.get('global_step')!=a.steps or metrics.get('schedule_steps')!=a.steps:
        raise RuntimeError('Pilot must complete its declared full schedule')
    config=json.loads((out/'train_config.json').read_text())
    if config['training'].get('stop_after') is not None:
        raise RuntimeError('Gate pilots must not use stop_after')
    if not list(out.glob('*.safetensors')):raise RuntimeError('Missing final model weights')
    checkpoint=out/f'checkpoint-{a.steps}'
    from scripts.training_results import file_records,weight_names,check_metric
    world=json.loads((checkpoint/'training_recipe.json').read_text())['execution']['world_size']
    records=file_records(checkpoint,['trainer_state.json','optimizer.pt','scheduler.pt',
        'config.json','training_recipe.json','data_recipe.json','train_config.json',
        *weight_names(checkpoint),*(['rng_state.pth'] if world==1 else
                                  [f'rng_state_{rank}.pth' for rank in range(world)])])
    state=json.loads((checkpoint/'trainer_state.json').read_text())
    if state['global_step']!=a.steps or state['max_steps']!=a.steps:
        raise RuntimeError('Final checkpoint schedule/step mismatch')
    if config['training'].get('keep_checkpoint_steps') is not None:
        expected={f'checkpoint-{s}' for s in config['training']['keep_checkpoint_steps'] if s<=a.steps}
        if {p.name for p in out.glob('checkpoint-*')}!=expected:
            raise RuntimeError('Unexpected retained checkpoints')
    final=json.loads((out/'final_eval.json').read_text())
    validation=check_metric(final,'eval_')
    receipt.parent.mkdir(parents=True,exist_ok=True)
    with receipt.open('x') as handle:json.dump(dict(status='ok',global_step=a.steps,
        output_dir=str(out),attempts=attempts,checkpoint_files=records,validation=validation),handle)

if __name__=='__main__':main()
