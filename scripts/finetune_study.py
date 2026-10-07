"""Fixed downstream protocol: CUDA gates, HF Trainer jobs, dev-only LR selection."""
import argparse
import json
from pathlib import Path
import statistics
import sys

ARMS = ('A', 'P6', 'P7-simple')
SUPPORTED_ARMS = (*ARMS, 'P6-iso')
TASKS = ('paws', 'nli')
RATES = (1e-5, 3e-5)
SEEDS = (42, 43, 44)


def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)


def select(paths, output):
    candidates = []
    for path in paths:
        data = json.loads((Path(path)/'result.json').read_text())
        if data['status'] != 'completed' or data['smoke'] or data['evaluate_run']:
            raise ValueError('LR selection requires complete production training results')
        candidates.append((path, data))
    if len(candidates) != 2 or {d['learning_rate'] for _,d in candidates} != set(RATES):
        raise ValueError('Both predetermined learning rates are required')
    first = candidates[0][1]
    for _, other in candidates:
        for key in ('source', 'data', 'seed', 'global_step', 'world_size', 'max_length', 'attention_backend'):
            if other[key] != first[key]:
                raise ValueError('Unmatched LR candidates: '+key)
    # Development accuracy only; deterministic lower-LR tie break.
    path, best = max(candidates, key=lambda row: (row[1]['metrics']['eval_accuracy'], -row[1]['learning_rate']))
    write(output, dict(status='selected', arm=first['source']['arm'], task=first['data']['task'],
        learning_rate=best['learning_rate'], selected_run=str(path),
        dev_accuracy=best['metrics']['eval_accuracy'], candidates=[str(p) for p,_ in candidates]))


def resolve_eval(config, selection, output):
    # The selected seed-42 run is unknown when the queue is built. Resolve it
    # immediately before evaluation, after all fitting and selection are done.
    config = json.loads(Path(config).read_text())
    selected = json.loads(Path(selection).read_text())
    config['evaluate_run'] = selected['selected_run']
    config['learning_rate'] = selected['learning_rate']
    write(output, config)


def make_jobs(root, project, data_root, manifest, checkpoints):
    arms = tuple(checkpoints)
    if not arms or any(arm not in SUPPORTED_ARMS for arm in arms):
        raise ValueError('Select at least one supported fine-tuning arm')
    root, project = Path(root).resolve(), Path(project).resolve()
    run = root/'supervised/run'; configs=root/'configs'
    jobs=[]
    def add(name, argv, output, gpu=True, expected=None):
        jobs.append(dict(name=name, argv=argv, **({'gpus':list(range(8))} if gpu else {}),
            timeout_seconds=14400, required_outputs=[dict(path=output,json_equals=expected or {'status':'completed'})]))
    def config_for(arm, task, seed, rate, dest, smoke=False, evaluation=None, selection=None):
        return dict(checkpoint=checkpoints[arm], task=task, dataset_root=str(data_root), dataset_manifest=str(manifest),
            max_length=512, preprocessing_num_workers=8, attention_backend='fa4', smoke=smoke,
            evaluate_run=evaluation, selection_file=selection, output_dir=str(run/dest),
            bf16=True, tf32=True, seed=seed, data_seed=seed, report_to='none',
            per_device_train_batch_size=16, per_device_eval_batch_size=32, gradient_accumulation_steps=1,
            num_train_epochs=3 if task=='paws' else 1, max_steps=4 if smoke and not evaluation else -1,
            learning_rate=rate, weight_decay=.01, warmup_ratio=.06, lr_scheduler_type='linear',
            optim='adamw_torch_fused', max_grad_norm=1., gradient_checkpointing=False,
            eval_strategy='no' if evaluation else 'steps' if smoke else 'epoch', eval_steps=2 if smoke else None,
            save_strategy='no' if evaluation else 'steps' if smoke else 'epoch', save_steps=2 if smoke else 500,
            save_total_limit=1, save_only_model=True, load_best_model_at_end=not bool(evaluation),
            metric_for_best_model='accuracy', greater_is_better=True, logging_steps=25, logging_first_step=True,
            logging_nan_inf_filter=False, dataloader_num_workers=2, dataloader_pin_memory=True,
            ddp_find_unused_parameters=False, ddp_broadcast_buffers=False, disable_tqdm=True)
    def train_job(name, cfg):
        file=configs/(name+'.json');write(file,cfg)
        argv=['{python}','-m','accelerate.commands.launch','--config_file',str(project/'resources/accelerate_config.yaml'),
              '--main_process_port','29647','--module','eval.finetune',str(file)]
        add(name,argv,str(Path(cfg['output_dir']).relative_to(run)/'result.json'))
    for arm in arms:
        name='gate-'+arm;out='gates/'+arm+'.json'
        jobs.append(dict(name=name,gpus=[0],timeout_seconds=900,
            argv=['{python}','-u','-m','scripts.finetune_study','gate','--checkpoint',checkpoints[arm],'--output','{run_dir}/'+out],
            required_outputs=[dict(path=out,json_equals={'status':'passed','arm':arm})]))
    for arm in arms:
        dest='smoke/'+arm
        train_job('smoke-'+arm,config_for(arm,'paws',42,1e-5,dest,smoke=True))
        train_job('smoke-reload-'+arm,config_for(arm,'paws',42,1e-5,'smoke-eval/'+arm,
                                              smoke=True,evaluation=str(run/dest)))
    for task in TASKS:
        for arm in arms:
            paths=[]
            for rate in RATES:
                dest=f'search/{task}/{arm}/lr-{rate:g}'
                train_job(f'search-{task}-{arm}-{rate:g}',config_for(arm,task,42,rate,dest))
                paths.append(str(run/dest))
            out=f'selections/{task}-{arm}.json'
            add(f'select-{task}-{arm}',['{python}','-m','scripts.finetune_study','select','--runs',*paths,'--output','{run_dir}/'+out],
                out,False,{'status':'selected','arm':arm,'task':task})
    for task in TASKS:
        for arm in arms:
            selection=str(run/f'selections/{task}-{arm}.json')
            for seed in SEEDS[1:]:
                train_job(f'confirm-{task}-{arm}-{seed}',config_for(arm,task,seed,1e-5,
                    f'confirm/{task}/{arm}/seed-{seed}',selection=selection))
    # No test scoring until every development search/confirmation run completes.
    for task in TASKS:
        for arm in arms:
            selection=str(run/f'selections/{task}-{arm}.json')
            for seed in SEEDS:
                name=f'test-{task}-{arm}-{seed}';dest=f'test/{task}/{arm}/seed-{seed}'
                cfg=config_for(arm,task,seed,1e-5,dest,evaluation=str(run/f'confirm/{task}/{arm}/seed-{seed}'),selection=selection)
                if seed==42:
                    template=configs/(name+'-template.json');resolved=configs/(name+'.json');write(template,cfg)
                    # Resolve config inside a dedicated small launcher, without a shell.
                    argv=['{python}','-m','scripts.finetune_study','selected-test','--config',str(template),
                          '--selection',selection,'--resolved',str(resolved),'--project',str(project)]
                    add(name,argv,dest+'/result.json')
                else: train_job(name,cfg)
    add('summary',['{python}','-m','scripts.finetune_study','summary','--root','{run_dir}','--output','{run_dir}/summary.json','--arms',*arms],
        'summary.json',False,{'status':'passed'})
    write(root/'jobs.json',{'jobs':jobs})
    return jobs


def summary(root, output, arms=ARMS):
    import numpy as np
    if not arms or len(set(arms)) != len(arms) or any(arm not in SUPPORTED_ARMS for arm in arms):
        raise ValueError('Select distinct supported fine-tuning arms')
    root=Path(root); table={}
    expected_hashes={}
    for task in TASKS:
        table[task]={}
        labels_ref=None
        for arm in arms:
            scores=[]
            selection=json.loads((root/f'selections/{task}-{arm}.json').read_text())
            for seed in SEEDS:
                folder=root/f'test/{task}/{arm}/seed-{seed}'
                result=json.loads((folder/'result.json').read_text())
                if (result['status']!='completed' or result['source']['arm']!=arm or result['data']['task']!=task
                    or result['seed']!=seed or result['smoke'] or result['world_size']!=8
                    or result['attention_backend']!='fa4' or result['learning_rate']!=selection['learning_rate']):
                    raise ValueError('Invalid test provenance')
                n=2000 if task=='paws' else 5010
                if result['data']['rows']!={'test':n}:raise ValueError('Wrong test count')
                p=np.load(folder/'predictions.npz')
                if p['logits'].shape!=(n,2 if task=='paws' else 3) or not np.isfinite(p['logits']).all():
                    raise ValueError('Invalid predictions')
                if labels_ref is None:labels_ref=p['labels']
                if not np.array_equal(labels_ref,p['labels']):raise ValueError('Test label order differs')
                for key in ('train_order_sha256','dataset_manifest_sha256','tokenizer_sha256'):
                    expected_hashes.setdefault((task,key),result['data'][key])
                    if result['data'][key]!=expected_hashes[task,key]:raise ValueError('Unmatched data')
                expected_hashes.setdefault((task,'test_order'),result['data']['split_order_sha256']['test'])
                if result['data']['split_order_sha256']['test']!=expected_hashes[task,'test_order']:
                    raise ValueError('Unmatched test examples/order')
                acc=float((p['logits'].argmax(-1)==p['labels']).mean())
                if abs(acc-result['metrics']['test_accuracy'])>1e-10:raise ValueError('Accuracy mismatch')
                scores.append(acc)
            table[task][arm]=dict(seed_accuracies=scores,mean=statistics.mean(scores),stdev=statistics.stdev(scores),
                                 learning_rate=selection['learning_rate'])
    write(output,dict(status='passed',tasks=table,seeds=list(SEEDS),
                     caveat='Three fine-tuning seeds; one pretraining seed; exploratory arm selection.'))


def gate(checkpoint, output):
    import torch
    from eval.models import load_checkpoint
    from eval.finetune import PairClassifier, PairCollator, GradientCheck
    torch.manual_seed(42)
    adapter,_,source=load_checkpoint(checkpoint,'cuda',attention_backend='fa4')
    backbone=adapter.wrapped
    # Forward before opting into task gradients must be identical to after.
    model=PairClassifier(backbone,3,42).to('cuda')
    ids=torch.randint(10,10000,(2,64)).tolist()
    rows=[dict(input_ids=ids[0][:37],labels=0),dict(input_ids=ids[1],labels=2)]
    inputs={k:v.cuda() for k,v in PairCollator(0)(rows).items()}
    buffers={k:v.clone() for k,v in backbone.named_buffers()}
    calls=[]; backbone._fa4_observer=lambda *args:calls.append(1)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        out=model(**inputs)
    out['loss'].backward()
    check=GradientCheck();check.on_pre_optimizer_step(None,None,None,model=model)
    # Compare every trainable gradient to dense SDPA at identical weights/data.
    grads={k:p.grad.detach().cpu().clone() for k,p in model.named_parameters() if p.requires_grad}
    model.zero_grad(set_to_none=True)
    backbone.attention_backend='sdpa'
    with torch.autocast('cuda',dtype=torch.bfloat16):dense=model(**inputs)
    dense['loss'].backward()
    numer=denom=0.
    for k,p in model.named_parameters():
        if p.requires_grad:
            ref=p.grad.detach().cpu();numer+=float((grads[k]-ref).double().square().sum());denom+=float(ref.double().square().sum())
    relative=(numer/max(denom,1e-30))**.5
    logit_relative=float((out['logits']-dense['logits']).norm()/dense['logits'].norm().clamp_min(1e-8))
    if not calls or relative>=.05 or logit_relative>=.02:
        raise ValueError(f'FA4/SDPA disagreement: gradients={relative}, logits={logit_relative}')
    for k,v in backbone.named_buffers():
        if not torch.equal(v,buffers[k]):raise ValueError('Unexpected buffer mutation')
    write(output,dict(status='passed',arm=source['arm'],source=source,gradients=check.report,
        gradient_relative_l2=relative,logit_relative_l2=logit_relative,fa4_calls=len(calls),buffers_unchanged=True))


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='mode',required=True)
    s=sub.add_parser('select');s.add_argument('--runs',nargs='+',required=True);s.add_argument('--output',required=True)
    s=sub.add_parser('summary');s.add_argument('--root',required=True);s.add_argument('--output',required=True)
    s.add_argument('--arms',nargs='+',choices=SUPPORTED_ARMS,default=ARMS)
    s=sub.add_parser('gate');s.add_argument('--checkpoint',required=True);s.add_argument('--output',required=True)
    s=sub.add_parser('selected-test')
    for name in ('config','selection','resolved','project'):s.add_argument('--'+name,required=True)
    args=p.parse_args()
    if args.mode=='select':select(args.runs,args.output)
    elif args.mode=='summary':summary(args.root,args.output,args.arms)
    elif args.mode=='gate':gate(args.checkpoint,args.output)
    else:
        import subprocess
        resolve_eval(args.config,args.selection,args.resolved)
        subprocess.run([sys.executable,'-m','accelerate.commands.launch','--config_file',str(Path(args.project)/'resources/accelerate_config.yaml'),
            '--main_process_port','29647','--module','eval.finetune',args.resolved],check=True)


if __name__=='__main__':main()
