"""Build the authorized expanded A/P6-iso 20k study using existing evaluators."""
import argparse
import json
from pathlib import Path

from run_experiments import load_jobs
from scripts.finetune_study import make_jobs as finetune_jobs, write

SUPERVISED=['glue_'+s for s in ('cola','sst2','mrpc','qqp','stsb','mnli_matched','mnli_mismatched','qnli','rte','wnli')]+['squad_v2']
GROUPS=['zero','five','math','continuation']  # Paloma is explicitly deferred: gated access unavailable.


def make(root,project,checkpoints,data_root,manifest,protocol,position,legacy):
    root,project=Path(root).resolve(),Path(project).resolve();run=root/'supervised/run';jobs=[]
    if set(checkpoints)!={'A10','A20','P20'}:raise ValueError('Need all three named checkpoints')
    def add(name,argv,outputs,gpus=None,timeout=172800):
        jobs.append(dict(name=name,argv=list(map(str,argv)),timeout_seconds=timeout,
                         required_outputs=outputs,**(dict(gpus=gpus) if gpus is not None else {})))
    def passed(path):return dict(path=path,json_equals=dict(status='passed'))
    for arm in ('A20','P20'):
        dest=f'numerics/{arm}.json'
        add('numerics-'+arm,['{python}','-m','scripts.check_downstream_eval','numerics','--checkpoint',checkpoints[arm],
            '--expected-step','20000','--output','{run_dir}/'+dest],[passed(dest)],[0],900)
    for phase in ('smoke','full'):
        dest='expanded-'+phase
        args=['{python}','-m','eval.suite','--checkpoints',checkpoints['A20'],checkpoints['P20'],
            '--output','{run_dir}/'+dest,'--dataset-root',data_root,'--manifest',manifest,
            '--attention-backend','fa4','--groups',*GROUPS,'--new-only','--workers','8']
        if phase=='smoke':args+=['--limit','2']
        add('expanded-'+phase,args,[passed(dest+'/summary.json')],list(range(8)))
    # Use the existing three-seed study, development-only LR selection and reload gates.
    fit=root/'finetune-plan'
    items=finetune_jobs(fit,project,data_root,manifest,{'A':checkpoints['A20'],'P6-iso':checkpoints['P20']},
        tasks=SUPERVISED,checkpoint_steps={'A':20000,'P6-iso':20000})
    for file in fit.rglob('*.json'):
        file.write_text(file.read_text().replace(str(fit/'supervised/run'),str(run/'finetune')))
    load_jobs(fit/'jobs.json')
    add('finetune-expanded',['{python}',project/'run_experiments.py','--config',fit/'jobs.json',
        '--project-dir',project,'--run-dir','{run_dir}/finetune'],[passed('finetune/summary.json')],list(range(8)),604800)
    args=['{python}','-m','eval.paired','plan','--protocol',protocol,'--calibration',project/'docs/PAIRED_LOGPROB_CALIBRATION_20261012.json',
        '--position-sample',position,'--a10',checkpoints['A10'],'--a20',checkpoints['A20'],'--p20',checkpoints['P20'],
        '--dataset-root',data_root,'--manifest',manifest,'--batch-size','16','--output','{run_dir}/paired',
        '--defer-paloma']
    if legacy:args+=['--legacy-f',legacy]
    add('paired-plan',args,[dict(path='paired/plan.json')],timeout=7200)
    add('paired-evaluate',['{python}','-m','eval.paired','run','--plan','{run_dir}/paired/plan.json'],
        [dict(path=f'paired/{a}/position/metadata.json',json_equals=dict(status='completed')) for a in checkpoints]+[dict(path='paired/precision.json')],list(range(8)),604800)
    for suite in ('F','Z','probes'):
        args=['{python}','-m','eval.paired','analyze','--root','{run_dir}/paired','--protocol',protocol,
              '--suite',suite,'--output','{run_dir}/paired-'+suite+'.json']
        if suite=='F' and legacy:args+=['--legacy-f',legacy]
        add('analyze-'+suite,args,[dict(path='paired-'+suite+'.json')],timeout=43200)
    add('analyze-position',['{python}','-m','eval.paired','analyze-position','--protocol',protocol,
        '--a10','{run_dir}/paired/A10/position','--a20','{run_dir}/paired/A20/position','--p20','{run_dir}/paired/P20/position',
        '--output','{run_dir}/paired-position.json'],[dict(path='paired-position.json')],timeout=43200)
    write(root/'jobs.json',dict(jobs=jobs));load_jobs(root/'jobs.json')
    write(root/'study-scope.json',dict(checkpoints=checkpoints,order=[j['name'] for j in jobs],
        new_benchmark_groups=GROUPS,supervised_tasks=SUPERVISED,fine_tune_seeds=[42,43,44],
        source_step=20000,reference_step=10000,fit_stages=len(items),
        deferred={'paloma':'Existing HF credential lacks gated dataset access; never silently replace this corpus'}))
    return jobs


def prepare(root,host,position_documents,position_selection=None):
    """Offline CPU preflight, while the approved burns keep running."""
    import importlib.metadata as md
    import shutil,socket,subprocess
    from accelerate.commands.config.config_args import default_yaml_config_file,load_config_from_file
    from transformers import AutoTokenizer
    from train import load_text
    from eval.paired import protocol,position,audit
    from scripts.materialize_expanded_inputs import materialize
    from scripts.train_then_burn import digest,approved_launcher,APPROVED_BURNS,BURN_HASH,GUARD_HASH
    from scripts.verified_gpu_reclaim import inspect,process
    root=Path(root);project=Path.cwd()
    if socket.gethostname()!=host:raise ValueError('Wrong host')
    versions={p:md.version(p) for p in ('torch','transformers','accelerate','datasets','lm_eval','flash-attn-4','nvidia-cutlass-dsl')}
    expected=['2.14.1','5.9.0','1.13.0','4.8.5','0.4.10','4.0.0b33','4.8.0']
    if [v.split('+')[0] for v in versions.values()]!=expected:raise ValueError(('Unexpected environment',versions))
    if shutil.disk_usage(root).free<1024*2**30:raise ValueError('Need at least 1 TiB free for retained fine-tuning checkpoints')
    data=Path('/mnt/local/_data/deep-llms_th2/expanded-20261012');manifest=root/'manifest.json'
    materialize(data,project/'resources/expanded_inputs_20261012.json',manifest)
    base=Path('/mnt/local/_outputs/deep-llms_th2')
    resumed=base/'q359-proxy-resume20k-seed1042-20261011-a01/supervised/run/seed-1042'
    original=base/'q359-proxy-10k-seed1042-20261009-a01/supervised/run/seed-1042'
    checkpoints=dict(A10=str(original/'A/checkpoint-10000'),A20=str(resumed/'A/checkpoint-20000'),P20=str(resumed/'P6-iso/checkpoint-20000'))
    for name,folder in checkpoints.items():
        ck=Path(folder);cfg=json.loads((ck.parent/'train_config.json').read_text())
        if (cfg['pilot']['arm']!=('P6-iso' if name=='P20' else 'A') or
            cfg['training']['seed']!=1042 or cfg['training']['data_seed']!=1042 or cfg['pilot']['attention_backend']!='fa4' or
            json.loads((ck/'trainer_state.json').read_text())['global_step']!=(10000 if name=='A10' else 20000)):
            raise ValueError('Wrong source checkpoint: '+name)
    cfg=json.loads((Path(checkpoints['A20']).parent/'train_config.json').read_text())
    tok=AutoTokenizer.from_pretrained(cfg['model']['tokenizer_name'],local_files_only=True)
    if tok.pad_token is None:tok.pad_token=tok.eos_token
    frozen_path=root/'paired-protocol.json';frozen=protocol.freeze(frozen_path,position_documents)
    write(root/'paired-data-audit.json',audit.data(data,manifest,frozen))
    position_path=root/'position-sample.json'
    if position_selection:
        from scripts.extended_position_sample import materialize as extended_sample
        extended_sample(position_selection,load_text(cfg['data']['eval_data_dir']),
            '/mnt/local/_data/deep-llms_th2/position-extension-20261012',
            '/mnt/local/_data/deep-llms_th2/cx_sampled_old/dataset_manifest.json',tok,position_path,frozen)
    else:
        position.freeze(load_text(cfg['data']['eval_data_dir']),tok,position_path,frozen)
    # Historical F logs lack tokenizer provenance required by the new protocol.
    # Recompute using the same canonical tasks/shots; never fabricate metadata.
    legacy_path=None
    write(root/'legacy-f-decision.json',dict(mode='recompute',
        reason='Historical F artifacts lack tokenizer_sha256 required by paired validation'))
    source=project/'resources/accelerate_config.yaml';dest=Path(default_yaml_config_file)
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():shutil.copy2(dest,root/'accelerate.previous.yaml')
    shutil.copy2(source,dest)
    if source.read_bytes()!=dest.read_bytes():raise ValueError('Accelerate copy differs')
    ac=load_config_from_file(str(dest)).to_dict()
    if ac['num_processes']!=8 or ac['mixed_precision']!='bf16' or ac['distributed_type']!='MULTI_GPU':raise ValueError('Wrong Accelerate configuration')
    subprocess.run(['accelerate','env'],check=True)
    make(root,project,checkpoints,str(data),str(manifest),str(frozen_path),str(position_path),legacy_path)
    status=inspect()
    if status['host']!=host or status['guard_disabled'] or len(status['gpus'])!=8 or any('B200' not in g['name'] for g in status['gpus']):raise ValueError('Unexpected GPU ownership/state')
    if digest('/mnt/local/_gpu_guard/gpu_guard.sh')!=GUARD_HASH:raise ValueError('Unknown GPU guard')
    approved={**APPROVED_BURNS,str(project/'resources/llm_pretrain_burn.py'):BURN_HASH}
    if any(not (approved_launcher(p,approved) or approved_launcher(process(p)['ppid'],approved)) for p in status['workers']):raise ValueError('Unknown GPU work; no signals sent')
    write(root/'inspection.json',status)
    write(root/'preflight.json',dict(status='passed',versions=versions,accelerate_cache=str(dest),accelerate_sha256=digest(dest),
        manifest_sha256=digest(manifest),checkpoints=checkpoints,protocol_sha256=frozen['protocol_sha256']))
    print('EXPANDED_STUDY_PREFLIGHT_PASSED',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--host',required=True)
    p.add_argument('--position-documents',type=int,required=True)
    p.add_argument('--position-selection')
    a=p.parse_args();prepare(a.root,a.host,a.position_documents,a.position_selection)
