"""Fixed downstream protocol: CUDA gates, HF Trainer jobs, dev-only LR selection."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys

ARMS = ('A', 'P6', 'P7-simple')
SUPPORTED_ARMS = (*ARMS, 'P6-iso', 'P6-iso-sparse', 'P6-iso-short', 'P7-simple-sparse', 'P7-simple-short')
TASKS = ('paws', 'nli')
SUPPORTED_TASKS = (*TASKS, 'stsb', 'boolq')

def score_name(task):
    return 'correlation' if task == 'stsb' else 'accuracy'

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
    # Development metrics only; deterministic lower-LR tie break.
    metric = score_name(first['data']['task'])
    if any(not math.isfinite(d['metrics']['eval_'+metric]) for _, d in candidates):
        raise ValueError('Nonfinite selection metric')
    path, best = max(candidates, key=lambda row: (row[1]['metrics']['eval_'+metric], -row[1]['learning_rate']))
    write(output, dict(status='selected', arm=first['source']['arm'], task=first['data']['task'],
        learning_rate=best['learning_rate'], selected_run=str(path),
        selection_metric=metric, dev_score=best['metrics']['eval_'+metric], candidates=[str(p) for p,_ in candidates]))


def resolve_eval(config, selection, output):
    # The selected seed-42 run is unknown when the queue is built. Resolve it
    # immediately before evaluation, after all fitting and selection are done.
    config = json.loads(Path(config).read_text())
    selected = json.loads(Path(selection).read_text())
    config['evaluate_run'] = selected['selected_run']
    config['learning_rate'] = selected['learning_rate']
    write(output, config)


def make_jobs(root, project, data_root, manifest, checkpoints, tasks=TASKS):
    arms = tuple(checkpoints)
    if not tasks or len(set(tasks)) != len(tasks) or any(t not in SUPPORTED_TASKS for t in tasks):
        raise ValueError('Select distinct supported tasks')
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
            num_train_epochs=1 if task=='nli' else 3, max_steps=4 if smoke and not evaluation else -1,
            learning_rate=rate, weight_decay=.01, warmup_ratio=.06, lr_scheduler_type='linear',
            optim='adamw_torch_fused', max_grad_norm=1., gradient_checkpointing=False,
            eval_strategy='no' if evaluation else 'steps' if smoke else 'epoch', eval_steps=2 if smoke else None,
            save_strategy='no' if evaluation else 'steps' if smoke else 'epoch', save_steps=2 if smoke else 500,
            save_total_limit=1, save_only_model=True, load_best_model_at_end=not bool(evaluation),
            metric_for_best_model=score_name(task), greater_is_better=True, logging_steps=25, logging_first_step=True,
            logging_nan_inf_filter=False, dataloader_num_workers=2, dataloader_pin_memory=True,
            ddp_find_unused_parameters=False, ddp_broadcast_buffers=False, disable_tqdm=True)
    def train_job(name, cfg):
        file=configs/(name+'.json');write(file,cfg)
        argv=['{python}','-m','accelerate.commands.launch','--config_file',str(project/'resources/accelerate_config.yaml'),
              '--main_process_port','29647','--module','eval.finetune',str(file)]
        add(name,argv,str(Path(cfg['output_dir']).relative_to(run)/'result.json'))
    extended = any(t in ('stsb', 'boolq') for t in tasks)
    gate_tasks = tasks if extended else ('nli',)
    smoke_tasks = tasks if extended else ('paws',)
    for arm in arms:
        for task in gate_tasks:
            name='gate-'+arm+('-'+task if extended else '');out='gates/'+name+'.json' if extended else 'gates/'+arm+'.json'
            argv=['{python}','-u','-m','scripts.finetune_study','gate','--checkpoint',checkpoints[arm],'--output','{run_dir}/'+out]
            if extended:argv+=['--task',task,'--data-root',str(data_root),'--manifest',str(manifest)]
            jobs.append(dict(name=name,gpus=[0],timeout_seconds=900,argv=argv,
                required_outputs=[dict(path=out,json_equals={'status':'passed','arm':arm})]))
    for arm in arms:
        for task in smoke_tasks:
            tag=arm+('-'+task if extended else '')
            dest='smoke/'+tag;reload='smoke-eval/'+tag
            train_job('smoke-'+tag,config_for(arm,task,42,1e-5,dest,smoke=True))
            train_job('smoke-reload-'+tag,config_for(arm,task,42,1e-5,reload,
                                                  smoke=True,evaluation=str(run/dest)))
            if extended:
                out='smoke-checks/'+tag+'.json'
                add('verify-reload-'+tag,['{python}','-m','scripts.finetune_study','verify-reload',
                    '--trained',str(run/dest),'--reloaded',str(run/reload),'--output','{run_dir}/'+out],
                    out,False,{'status':'passed'})
    for task in tasks:
        for arm in arms:
            paths=[]
            for rate in RATES:
                dest=f'search/{task}/{arm}/lr-{rate:g}'
                train_job(f'search-{task}-{arm}-{rate:g}',config_for(arm,task,42,rate,dest))
                paths.append(str(run/dest))
            out=f'selections/{task}-{arm}.json'
            add(f'select-{task}-{arm}',['{python}','-m','scripts.finetune_study','select','--runs',*paths,'--output','{run_dir}/'+out],
                out,False,{'status':'selected','arm':arm,'task':task})
    for task in tasks:
        for arm in arms:
            selection=str(run/f'selections/{task}-{arm}.json')
            for seed in SEEDS[1:]:
                train_job(f'confirm-{task}-{arm}-{seed}',config_for(arm,task,seed,1e-5,
                    f'confirm/{task}/{arm}/seed-{seed}',selection=selection))
    # No test scoring until every development search/confirmation run completes.
    for task in tasks:
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
    add('summary',['{python}','-m','scripts.finetune_study','summary','--root','{run_dir}','--output','{run_dir}/summary.json',*(['--tasks',*tasks] if extended else []),'--arms',*arms],
        'summary.json',False,{'status':'passed'})
    write(root/'jobs.json',{'jobs':jobs})
    return jobs


def summary(root, output, arms=ARMS, tasks=TASKS):
    import numpy as np
    if not arms or len(set(arms)) != len(arms) or any(arm not in SUPPORTED_ARMS for arm in arms):
        raise ValueError('Select distinct supported fine-tuning arms')
    root=Path(root); table={}
    expected_hashes={}
    for task in tasks:
        table[task]={}
        labels_ref=None
        for arm in arms:
            scores=[]; measured=[]
            selection=json.loads((root/f'selections/{task}-{arm}.json').read_text())
            for seed in SEEDS:
                folder=root/f'test/{task}/{arm}/seed-{seed}'
                result=json.loads((folder/'result.json').read_text())
                if (result['status']!='completed' or result['source']['arm']!=arm or result['data']['task']!=task
                    or result['seed']!=seed or result['smoke'] or result['world_size']!=8
                    or result['attention_backend']!='fa4' or result['learning_rate']!=selection['learning_rate']):
                    raise ValueError('Invalid test provenance')
                n={'paws':2000,'nli':5010,'stsb':1379,'boolq':3270}[task]
                if result['data']['rows']!={'test':n}:raise ValueError('Wrong test count')
                p=np.load(folder/'predictions.npz')
                if p['logits'].shape!=(n, {'paws':2,'nli':3,'stsb':1,'boolq':2}[task]) or not np.isfinite(p['logits']).all():
                    raise ValueError('Invalid predictions')
                if p['labels'].shape!=(n,) or not np.isfinite(p['labels']).all():raise ValueError('Invalid label shape/values')
                if labels_ref is None:labels_ref=p['labels']
                if not np.array_equal(labels_ref,p['labels']):raise ValueError('Test label order differs')
                for key in ('train_order_sha256','dataset_manifest_sha256','tokenizer_sha256'):
                    expected_hashes.setdefault((task,key),result['data'][key])
                    if result['data'][key]!=expected_hashes[task,key]:raise ValueError('Unmatched data')
                expected_hashes.setdefault((task,'test_order'),result['data']['split_order_sha256']['test'])
                if result['data']['split_order_sha256']['test']!=expected_hashes[task,'test_order']:
                    raise ValueError('Unmatched test examples/order')
                from eval.finetune import metrics
                calculated=metrics((p['logits'],p['labels']))
                for name,value in calculated.items():
                    if abs(value-result['metrics']['test_'+name])>1e-10:raise ValueError('Metric mismatch: '+name)
                scores.append(calculated[score_name(task)]); measured.append(calculated)
            table[task][arm]=dict(metric=score_name(task),seed_scores=scores,mean=statistics.mean(scores),stdev=statistics.stdev(scores),
                                 learning_rate=selection['learning_rate'],
                                 metrics={name:dict(values=[m[name] for m in measured],
                                     mean=statistics.mean(m[name] for m in measured),
                                     stdev=statistics.stdev(m[name] for m in measured)) for name in measured[0]})
            if task != 'stsb':table[task][arm]['seed_accuracies']=scores
    write(output,dict(status='passed',tasks=table,seeds=list(SEEDS),
                     caveat='Three fine-tuning seeds; one pretraining seed; exploratory arm selection.'))


def numerical_check(model, inputs):
    """Compare production BF16 FA4 with an independent FP32 math reference.

    BF16 SDPA is another approximate implementation, not ground truth. The
    task-output, hidden-state and whole-gradient checks use the FP32 reference.
    """
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from eval.finetune import GradientCheck
    device = next(model.parameters()).device
    backbone = model.wrapped
    old_backend = backbone.attention_backend
    old_observer = getattr(backbone, '_fa4_observer', None)
    old_tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    buffers = {k:v.clone() for k,v in backbone.named_buffers()}
    pooled, calls = [], []
    hook = model.score.register_forward_pre_hook(lambda module, args: pooled.append(args[0].detach()))
    try:
        if any(p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError('FP32 reference requires FP32 master parameters')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        backbone.attention_backend = 'fa4'
        backbone._fa4_observer = lambda *args: calls.append(1)
        model.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type=='cuda'):
            out = model(**inputs)
        out['loss'].backward()
        check = GradientCheck(); check.on_pre_optimizer_step(None,None,None,model=model)
        gradients = {k:p.grad.detach().cpu().clone() for k,p in model.named_parameters() if p.requires_grad}
        actual_logits, actual_loss = out['logits'].detach(), float(out['loss'].detach())
        actual_hidden = pooled.pop()
        del out
        model.zero_grad(set_to_none=True)
        backbone.attention_backend = 'sdpa'
        # Disable autocast explicitly, including when called from an outer context.
        with sdpa_kernel(SDPBackend.MATH), torch.autocast(device.type, enabled=False):
            reference = model(**inputs)
        reference['loss'].backward()
        reference_check = GradientCheck()
        reference_check.on_pre_optimizer_step(None,None,None,model=model)
        expected_hidden = pooled.pop()
        numer = denom = 0.
        for k,p in model.named_parameters():
            if p.requires_grad:
                ref=p.grad.detach().cpu().double()
                numer+=float((gradients[k].double()-ref).square().sum())
                denom+=float(ref.square().sum())
        def relative(a,b):
            a,b=a.detach().double(),b.detach().double()
            return float((a-b).norm()/b.norm().clamp_min(1e-30))
        for k,v in backbone.named_buffers():
            if not torch.equal(v,buffers[k]):raise ValueError('Unexpected buffer mutation')
        return dict(gradient_relative_l2=(numer/max(denom,1e-30))**.5,
            logit_relative_l2=relative(actual_logits,reference['logits']),
            hidden_relative_l2=relative(actual_hidden,expected_hidden),
            logit_max_absolute=float((actual_logits-reference['logits'].detach()).abs().max()),
            loss_absolute=abs(actual_loss-float(reference['loss'].detach())),
            logits=actual_logits.cpu().tolist(),reference_logits=reference['logits'].detach().cpu().tolist(),
            gradients=check.report,reference_gradients=reference_check.report,
            task_kind='regression' if model.score.out_features==1 else 'classification',
            **({} if model.score.out_features==1 else classification_difference(actual_logits,reference['logits'])),
            fa4_calls=len(calls),buffers_unchanged=True)
    finally:
        hook.remove()
        backbone.attention_backend=old_backend
        backbone._fa4_observer=old_observer
        torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=old_tf32
        model.zero_grad(set_to_none=True)


def classification_difference(actual, reference):
    # Softmax is invariant to an arbitrary per-example common logit offset.
    # A relative norm of two raw logits does not have that property.
    actual,reference=actual.detach().double(),reference.detach().double()
    return dict(probability_max_absolute=float((actual.softmax(-1)-reference.softmax(-1)).abs().max()))


def numerical_limits(task_kind):
    limits=dict(gradient_relative_l2=.05,hidden_relative_l2=.02)
    if task_kind=='regression':limits['logit_relative_l2']=.02
    elif task_kind=='classification':limits.update(probability_max_absolute=.01,loss_absolute=.01)
    else:raise ValueError('Unknown numerical task kind')
    return limits


def check_numerical_limits(result):
    # Classification: at most 1 percentage point probability change and .01
    # cross-entropy change. Regression retains the original 2% output limit.
    # These task-independent limits are fixed before the new acceptance run.
    for name,limit in numerical_limits(result['task_kind']).items():
        value=result[name]
        if not math.isfinite(value) or value < 0 or value >= limit:
            raise ValueError(f'FA4/FP32 math SDPA disagreement: {name}={value}, limit={limit}')
    if not result['fa4_calls'] or not result['buffers_unchanged']:
        raise ValueError('FA4 not exercised or normalization buffers changed')
    for name in ('loss_absolute','logit_max_absolute','logit_relative_l2'):
        if not math.isfinite(result[name]):raise ValueError('Nonfinite numerical comparison: '+name)


def gate(checkpoint, output, task='nli', data_root=None, manifest=None):
    import torch
    from eval.models import load_checkpoint
    from eval.finetune import PairClassifier, PairCollator, encode_pairs
    torch.manual_seed(42)
    adapter,tokenizer_path,source=load_checkpoint(checkpoint,'cuda',attention_backend='fa4')
    model=PairClassifier(adapter.wrapped,{'nli':3,'paws':2,'boolq':2,'stsb':1}[task],42).to('cuda')
    # Preserve the original synthetic fixture, including its RNG order.
    ids=torch.randint(10,10000,(2,64)).tolist()
    batches={'synthetic':[dict(input_ids=ids[0][:37],labels=.5 if task=='stsb' else 0),
                           dict(input_ids=ids[1],labels=3.5 if task=='stsb' else 1)]}
    if (data_root is None) != (manifest is None):raise ValueError('Both data root and manifest are required')
    if data_root:
        from datasets import load_dataset
        from transformers import AutoTokenizer
        from eval.benchmarks import local_dataset_paths
        from eval.finetune import TASKS as task_definitions
        mapping=local_dataset_paths(data_root,manifest)[task_definitions[task][0]]
        raw=load_dataset(mapping['dataset_path'],**mapping['dataset_kwargs'])['train']
        rows=raw.select([0,1]).to_dict()
        if task=='stsb':rows['label']=[5*x for x in rows['score']]
        tokenizer=AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True)
        encoded=encode_pairs(rows,tokenizer,task_definitions[task][1],512,task_name=task)
        batches['real_training']=[dict(input_ids=x,labels=y) for x,y in zip(encoded['input_ids'],encoded['labels'])]
    report=dict(status='failed',arm=source['arm'],task=task,source=source,
        reference='fp32_math_sdpa_tf32_disabled',
        gate_version='fp32_task_v2',
        limits=numerical_limits('regression' if task=='stsb' else 'classification'),batches={})
    try:
        for name,rows in batches.items():
            inputs={k:v.cuda() for k,v in PairCollator(0,regression=task=='stsb')(rows).items()}
            result=numerical_check(model,inputs)
            result['lengths']=[len(r['input_ids']) for r in rows]
            report['batches'][name]=result
            check_numerical_limits(result)
        report.update(status='passed',buffers_unchanged=True,
            fa4_calls=sum(r['fa4_calls'] for r in report['batches'].values()),
            **{k:max(r[k] for r in report['batches'].values()) for k in
               ('gradient_relative_l2','logit_relative_l2','hidden_relative_l2')})
    except Exception as exc:
        report['error']=repr(exc)
        # Preserve finite failure evidence, instead of losing it in a traceback.
        def serializable(value):
            if isinstance(value,float) and not math.isfinite(value):return str(value)
            if isinstance(value,dict):return {k:serializable(v) for k,v in value.items()}
            if isinstance(value,list):return [serializable(v) for v in value]
            return value
        write(output,serializable(report))
        raise
    write(output,report)


def verify_reload(trained, reloaded, output):
    a=json.loads((Path(trained)/'result.json').read_text())
    b=json.loads((Path(reloaded)/'result.json').read_text())
    if not a['smoke'] or not b['smoke'] or a['status']!='completed' or b['status']!='completed':
        raise ValueError('Expected completed smoke results')
    for key in ('source','seed','learning_rate','global_step','weight_sha256','world_size','attention_backend'):
        if a[key]!=b[key]:raise ValueError('Reload differs: '+key)
    if a['data']['split_order_sha256']['validation']!=b['data']['split_order_sha256']['test']:
        raise ValueError('Reload example order differs')
    for key,value in a['metrics'].items():
        if key.startswith('eval_') and key not in ('eval_runtime','eval_samples_per_second','eval_steps_per_second','eval_model_preparation_time'):
            other=b['metrics']['test_'+key[5:]]
            if not math.isclose(value,other,rel_tol=1e-7,abs_tol=1e-7):raise ValueError('Reload metric differs: '+key)
    write(output,dict(status='passed',arm=a['source']['arm'],task=a['data']['task'],weight_sha256=a['weight_sha256']))


def audit_data(dataset_root, manifest, tokenizer_path, output):
    import tempfile
    from transformers import AutoTokenizer, TrainingArguments
    from eval.finetune import TaskArguments, prepare_data
    tokenizer=AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True)
    report={}
    with tempfile.TemporaryDirectory() as scratch:
        args=TrainingArguments(output_dir=scratch,use_cpu=True,report_to=[])
        for name in ('stsb','boolq'):
            task=TaskArguments(checkpoint='unused',task=name,dataset_root=dataset_root,
                dataset_manifest=manifest,preprocessing_num_workers=1,attention_backend='sdpa')
            _,train=prepare_data(task,tokenizer,args)
            task.evaluate_run='audit-final'
            _,final=prepare_data(task,tokenizer,args)
            # Dataset cache fingerprints include paths; compare portable content.
            train.pop('fingerprints');final.pop('fingerprints')
            report[name]=dict(training=train,final=final)
    write(output,dict(status='passed',tasks=report))


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='mode',required=True)
    s=sub.add_parser('select');s.add_argument('--runs',nargs='+',required=True);s.add_argument('--output',required=True)
    s=sub.add_parser('summary');s.add_argument('--root',required=True);s.add_argument('--output',required=True)
    s.add_argument('--arms',nargs='+',choices=SUPPORTED_ARMS,default=ARMS)
    s.add_argument('--tasks',nargs='+',choices=SUPPORTED_TASKS,default=TASKS)
    s=sub.add_parser('gate');s.add_argument('--checkpoint',required=True);s.add_argument('--output',required=True)
    s.add_argument('--task',choices=SUPPORTED_TASKS,default='nli')
    s.add_argument('--data-root');s.add_argument('--manifest')
    s=sub.add_parser('verify-reload')
    for name in ('trained','reloaded','output'):s.add_argument('--'+name,required=True)
    s=sub.add_parser('audit-data')
    for name in ('dataset-root','manifest','tokenizer','output'):s.add_argument('--'+name,required=True)
    s=sub.add_parser('selected-test')
    for name in ('config','selection','resolved','project'):s.add_argument('--'+name,required=True)
    args=p.parse_args()
    if args.mode=='select':select(args.runs,args.output)
    elif args.mode=='summary':summary(args.root,args.output,args.arms,args.tasks)
    elif args.mode=='gate':gate(args.checkpoint,args.output,args.task,args.data_root,args.manifest)
    elif args.mode=='verify-reload':verify_reload(args.trained,args.reloaded,args.output)
    elif args.mode=='audit-data':audit_data(args.dataset_root,args.manifest,args.tokenizer,args.output)
    else:
        import subprocess
        resolve_eval(args.config,args.selection,args.resolved)
        subprocess.run([sys.executable,'-m','accelerate.commands.launch','--config_file',str(Path(args.project)/'resources/accelerate_config.yaml'),
            '--main_process_port','29647','--module','eval.finetune',args.resolved],check=True)


if __name__=='__main__':main()
