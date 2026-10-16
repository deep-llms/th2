"""Small useful-memory gate; no proxy sweep or automatic long training."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time
from types import SimpleNamespace

from scripts.train_then_burn import digest
from scripts.probe_p7_teacher import save, read, selected_rows

CHECKPOINT = '/mnt/local/_outputs/deep-llms_th2/q359-proxy-12layer-screen22-20261013-a01/supervised/run/seed-42/A/checkpoint-2500'
CHECKPOINT_HASH = 'ecc2643c3b13ec32bf70aecd29c945b495df8c0296a2aea942e37b3ad8dd15c7'
PREDECESSOR = '/mnt/local/_outputs/deep-llms_th2/q359-p7-tuning-eval-20261016-a01/supervised'
HOST = 'thiennh-p6-q359-worker-0'
STEPS, TRAIN_ROWS, EVAL_ROWS = 256, 4096, 512


def predecessor_ready(root=PREDECESSOR, *, retry=False):
    from scripts.train_then_burn import burn_progress
    from scripts.verified_gpu_reclaim import ownership
    from scripts.gpu_status import snapshot
    root=Path(root); state=read(root/'supervisor.json'); burn=read(root/'burn-verified.json')
    completed = (state.get('training_status')=='ok' and state.get('training_returncode')==0
                 and (root/'run/complete.json').is_file())
    if retry:
        run=read(root/'run/run.json')
        completed=(state.get('training_status')=='failed' and run['status']=='failed'
                   and len(run['jobs'])==1 and run['jobs'][0]['name']=='smoke-shallow'
                   and run['jobs'][0]['status']=='failed'
                   and read(root.parent/'preflight.json')['status']=='passed'
                   and not (root/'run/full-shallow').exists()
                   and not (root/'run/full-deep').exists())
    if (not completed or not state.get('finished_at')
            or not burn.get('collective_progress_verified')
            or state.get('burn',{}).get('session')!=burn['session']
            or Path('/mnt/local/_gpu_guard/DISABLED').exists()):
        raise ValueError('Previous evaluation/burn handoff is incomplete')
    progress=burn_progress((root/'burn.log').read_text())
    if (not progress or time.time()-(root/'burn.log').stat().st_mtime>30
            or ownership(snapshot(list(range(8))))!=ownership(burn['gpus'])):
        raise ValueError('Previous verified burns are no longer live/owned')


def prepare(args):
    import importlib.metadata as md
    import shutil
    import socket
    import subprocess
    from accelerate.commands.config.config_args import default_yaml_config_file,load_config_from_file
    from transformers import AutoTokenizer
    from train import load_text
    from deep_kv.packing import preprocess_dataset
    from scripts.train_then_burn import approved_launcher,APPROVED_BURNS,BURN_HASH,GUARD_HASH
    from scripts.verified_gpu_reclaim import inspect,process
    from run_experiments import load_jobs
    root=Path(args.root)
    if root.exists() or socket.gethostname()!=HOST:
        raise ValueError('Wrong host or reused output')
    predecessor_ready(Path(args.predecessor)/'supervised',retry=True) if args.predecessor else predecessor_ready()
    expected=dict(torch='2.14.1',transformers='5.9.0',accelerate='1.13.0',datasets='4.8.5',
                  **{'flash-attn-4':'4.0.0b33','nvidia-cutlass-dsl':'4.8.0'})
    versions={k:md.version(k) for k in expected}
    if any(versions[k].split('+')[0]!=v for k,v in expected.items()):
        raise ValueError(('Environment differs',versions))
    ck=Path(CHECKPOINT); recipe=read(ck.parent/'train_config.json')
    if (digest(ck/'model.safetensors')!=CHECKPOINT_HASH or read(ck/'trainer_state.json')['global_step']!=2500
            or recipe['pilot']['arm']!='A' or recipe['model_config']['num_hidden_layers']!=12
            or recipe['pilot']['attention_backend']!='fa4' or not recipe['data']['isolate_documents']):
        raise ValueError('Baseline identity differs')
    root.mkdir(parents=True)
    tokenizer=AutoTokenizer.from_pretrained(recipe['model']['tokenizer_name'],local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token
    data=recipe['data']; dummy=SimpleNamespace(main_process_first=lambda **kw:nullcontext())
    # Reuse train.py's existing HF cache, EOS packing, document isolation and
    # shuffle. Only small fixed subsets are snapshotted inside this pilot root.
    train=preprocess_dataset(load_text(data['data_dir']),tokenizer,data['block_size'],dummy,
        num_proc=data['preprocessing_num_workers'],isolate_documents=True).shuffle(seed=recipe['training']['data_seed'])
    val=preprocess_dataset(load_text(data['eval_data_dir']),tokenizer,data['block_size'],dummy,
        num_proc=1,isolate_documents=True).select(range(data['eval_rows']))
    if train._fingerprint!=recipe['train_fingerprint'] or val._fingerprint!=recipe['eval_fingerprint']:
        raise ValueError('Original packed dataset fingerprints differ')
    indices=selected_rows(len(val),EVAL_ROWS)
    train.select(range(TRAIN_ROWS)).save_to_disk(str(root/'train'))
    val.select(indices).save_to_disk(str(root/'validation'))
    files={str(p.relative_to(root)):digest(p) for name in ('train','validation') for p in sorted((root/name).rglob('*')) if p.is_file()}
    save(root/'scope.json',dict(checkpoint=CHECKPOINT,checkpoint_sha256=CHECKPOINT_HASH,
        recipe_sha256=digest(ck.parent/'train_config.json'),versions=versions,
        source_train_fingerprint=train._fingerprint,source_eval_fingerprint=val._fingerprint,
        train_rows=TRAIN_ROWS,eval_rows=EVAL_ROWS,eval_indices=indices,files=files,
        steps=STEPS,seed=42,branch_seed=43,microbatch=2,world_size=8,sequence_length=2048,
        consumer=4,target=8,width=128,heads=2,learning_rate=3e-4,warmup_steps=16,
        schedule='constant_with_warmup',stage='useful external deep-memory teacher gate only'))
    if args.predecessor and read(root/'scope.json')!=read(Path(args.predecessor)/'scope.json'):
        raise ValueError('Retry changed frozen scientific scope or data')
    source=Path('resources/accelerate_config.yaml');destination=Path(default_yaml_config_file)
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():shutil.copy2(destination,root/'accelerate.previous.yaml')
    shutil.copy2(source,destination);config=load_config_from_file(str(destination)).to_dict()
    if (destination.read_bytes()!=source.read_bytes() or config['num_processes']!=8
            or config['mixed_precision']!='bf16'):
        raise ValueError('Accelerate differs')
    subprocess.run(['accelerate','env'],check=True)
    status=inspect();approved={**APPROVED_BURNS,str(Path.cwd()/'resources/llm_pretrain_burn.py'):BURN_HASH}
    if (status['host']!=HOST or status['guard_disabled'] or len(status['gpus'])!=8
            or any('B200' not in g['name'] for g in status['gpus'])
            or digest('/mnt/local/_gpu_guard/gpu_guard.sh')!=GUARD_HASH
            or any(not(approved_launcher(p,approved) or approved_launcher(process(p)['ppid'],approved)) for p in status['workers'])):
        raise ValueError('Unknown GPU workload or guard; no signals sent')
    save(root/'inspection.json',status)
    jobs=[]
    for phase,steps in (('smoke',2),('full',STEPS)):
        for source_kind in ('shallow','deep'):
            name=phase+'-'+source_kind
            jobs.append(dict(name=name,gpus=list(range(8)),timeout_seconds=900,
                argv=['accelerate','launch','--config_file',str(destination),'--main_process_port','29661',
                      '-m','scripts.teacher_first_pilot','fit','--root',str(root),'--output','{run_dir}/'+name,
                      '--source',source_kind,'--steps',str(steps)],
                required_outputs=[dict(path=name+'/result.json',json_equals={'status':'passed'})]))
    jobs.append(dict(name='summarize',argv=['{python}','-m','scripts.teacher_first_pilot','summarize',
        '--root',str(root),'--output','{run_dir}'],required_outputs=[dict(path='summary.json',json_equals={'status':'passed'})]))
    save(root/'jobs.json',dict(jobs=jobs));load_jobs(root/'jobs.json')
    save(root/'preflight.json',dict(status='passed',accelerate_cache=str(destination),versions=versions,
        accelerate_sha256=digest(destination),checkpoint_sha256=CHECKPOINT_HASH))
    print('TEACHER_FIRST_PREFLIGHT_PASSED',flush=True)


def fit(args):
    import numpy as np
    import torch
    from datasets import load_from_disk
    from transformers import TrainingArguments,TrainerCallback
    from deep_kv.teacher_first import TeacherFirst,make_trainer_class
    from deep_kv.packing import isolated_data_collator
    from eval.models import load_checkpoint
    from scripts.evaluate_proxy_gates import state_hash
    from deep_kv.training import DeepKVTrainer
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    root=Path(args.root);out=Path(args.output);scope=read(root/'scope.json')
    for name,sha in scope['files'].items():
        if digest(root/name)!=sha:raise ValueError('Frozen probe data changed')
    if digest(Path(scope['checkpoint'])/'model.safetensors')!=scope['checkpoint_sha256']:
        raise ValueError('Source weights changed')
    training_args=TrainingArguments(output_dir=str(out),max_steps=args.steps,
        per_device_train_batch_size=2,per_device_eval_batch_size=2,gradient_accumulation_steps=1,
        learning_rate=scope['learning_rate'],warmup_steps=min(16,args.steps),lr_scheduler_type='constant_with_warmup',
        weight_decay=.01,max_grad_norm=1.,bf16=True,tf32=False,seed=42,data_seed=42,
        remove_unused_columns=False,save_strategy='no',eval_strategy='no',report_to=[],
        logging_steps=16,ddp_find_unused_parameters=False,dataloader_num_workers=0,
        disable_tqdm=True,optim='adamw_torch',label_names=['labels'])
    if training_args.world_size!=8:raise ValueError('Require all eight ranks')
    adapter,_,meta=load_checkpoint(scope['checkpoint'],training_args.device,attention_backend='fa4')
    model=TeacherFirst(adapter.wrapped,source=args.source,consumer=4,target=8,seed=43)
    del adapter
    train=load_from_disk(str(root/'train'));val=load_from_disk(str(root/'validation'))
    if args.steps==2:val=val.select(range(16))
    class Budget(TrainerCallback):
        def on_train_begin(self,args,state,control,**kw):self.start=time.monotonic()
        def on_step_end(self,args,state,control,**kw):
            elapsed=time.monotonic()-self.start
            if state.global_step==16 and elapsed/state.global_step*args.max_steps>600:
                raise RuntimeError('Budget gate: projected fit exceeds ten minutes')
    trainer=make_trainer_class()(model=model,args=training_args,train_dataset=train,eval_dataset=val,
        data_collator=isolated_data_collator,callbacks=[Budget()])
    main=trainer.is_world_process_zero()
    before=state_hash(model.base) if main else None
    # Check the real trained backbone's FA4 path against dense SDPA with a
    # nonzero branch, including gradients through the actual frozen tail.
    tiny=isolated_data_collator([{k:v[:64] for k,v in val[i].items()} for i in range(2)])
    tiny=DeepKVTrainer.context({k:v.to(training_args.device) for k,v in tiny.items()})
    original={k:v.detach().clone() for k,v in model.branch.state_dict().items()}
    with torch.random.fork_rng(devices=[training_args.device.index]):
        torch.manual_seed(123)
        torch.nn.init.normal_(model.branch.out.weight,std=.01)
    numerical=[]
    from torch.nn.attention import sdpa_kernel,SDPBackend
    for backend in ('fa4','sdpa','fp32_math'):
        model.base.attention_backend='sdpa' if backend=='fp32_math' else backend
        model.zero_grad(set_to_none=True)
        reference=sdpa_kernel(SDPBackend.MATH) if backend=='fp32_math' else nullcontext()
        with reference,torch.autocast('cuda',dtype=torch.bfloat16,enabled=backend!='fp32_math'):
            r=model(tiny);loss=r['nll'].sum()/r['counts'].sum()
        loss.backward()
        gradient=torch.cat([p.grad.float().flatten() for p in model.branch.parameters()])
        numerical.append((loss.detach().float(),gradient))
    comparisons={}
    # Use the project's established trained-checkpoint tolerances (.01 nats,
    # .03 gradient relative L2), and additionally require BOTH BF16 paths to
    # agree with independent full-FP32 math SDPA. Initial pilot used an overly
    # strict, new .001 absolute loss threshold; retain its failed evidence.
    for name,i,j in [('fa4_vs_sdpa',0,1),('fa4_vs_fp32_math',0,2),('sdpa_vs_fp32_math',1,2)]:
        gap=float((numerical[i][0]-numerical[j][0]).abs())
        rel=float((numerical[i][1]-numerical[j][1]).norm()/numerical[j][1].norm().clamp_min(1e-12))
        comparisons[name]=dict(loss_gap=gap,gradient_relative_l2=rel)
        if gap>.01 or rel>.03 or not np.isfinite([gap,rel]).all():
            raise ValueError(('Independent numerical gate failed',name,gap,rel))
    model.base.attention_backend='fa4';model.branch.load_state_dict(original,strict=True)
    model.zero_grad(set_to_none=True)
    del original,numerical
    # Real checkpoint identity before updates; both paths must equal A.
    initial=trainer.predict(val.select(range(16))).predictions
    if not np.array_equal(initial[:,0],initial[:,1]):raise ValueError('Zero branch changes baseline')
    if main:
        save(out/'initial.json',dict(status='passed',baseline_identity_exact=True,metadata=meta,
             numerical_checks=comparisons,tolerances=dict(loss_absolute=.01,gradient_relative_l2=.03),
             trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
    start=time.monotonic();trained=trainer.train();elapsed=time.monotonic()-start
    prediction=trainer.predict(val).predictions
    model.base.attention_backend='sdpa'
    dense_prediction=trainer.predict(val).predictions
    model.base.attention_backend='fa4'
    if (not np.isfinite(prediction).all() or (prediction[:,2]<=0).any()
            or not np.isfinite(dense_prediction).all()
            or not np.array_equal(prediction[:,2],dense_prediction[:,2])
            or trainer.state.global_step!=args.steps or any(p.grad is not None for p in model.base.parameters())):
        raise ValueError('Invalid training/evaluation or backbone gradient')
    if main:
        if before!=state_hash(model.base):raise ValueError('Frozen backbone changed')
        torch.save(model.branch.state_dict(),out/'branch.pt')
        # Strict branch restore is checked independently of optimizer state.
        model.branch.load_state_dict(torch.load(out/'branch.pt',weights_only=True,map_location=training_args.device),strict=True)
    trainer.accelerator.wait_for_everyone()
    replay=trainer.predict(val.select(range(16))).predictions
    if not np.array_equal(prediction[:16],replay):raise ValueError('Branch reload/replay differs')
    if main:
        result=dict(status='passed',source=args.source,steps=args.steps,elapsed_seconds=elapsed,
            checkpoint_sha256=scope['checkpoint_sha256'],scope_sha256=digest(root/'scope.json'),
            frozen_backbone_unchanged=True,baseline_identity_exact=True,reload_exact=True,
            input_tokens=args.steps*8*2*2048,train_metrics=trained.metrics,
            nll=float(prediction[:,0].sum()/prediction[:,2].sum()),
            baseline_nll=float(prediction[:,1].sum()/prediction[:,2].sum()),
            rows=prediction.tolist(),sdpa_rows=dense_prediction.tolist(),branch_sha256=digest(out/'branch.pt'))
        save(out/'result.json',result)
        print('TEACHER_FIRST_RESULT',json.dumps({k:v for k,v in result.items() if k not in ('rows','sdpa_rows')}),flush=True)


def summarize(args):
    import numpy as np
    out=Path(args.output);scope=read(Path(args.root)/'scope.json')
    shallow,deep=(read(out/('full-'+s)/'result.json') for s in ('shallow','deep'))
    for source,result in zip(('shallow','deep'),(shallow,deep)):
        if (result['status']!='passed' or result['source']!=source or result['steps']!=STEPS
                or result['scope_sha256']!=digest(Path(args.root)/'scope.json')
                or result['checkpoint_sha256']!=CHECKPOINT_HASH or len(result['rows'])!=EVAL_ROWS
                or not all(result[k] for k in ('frozen_backbone_unchanged','baseline_identity_exact','reload_exact'))):
            raise ValueError('Incomplete or mismatched fit')
    s,d=np.array(shallow['rows']),np.array(deep['rows'])
    ds,dd=np.array(shallow['sdpa_rows']),np.array(deep['sdpa_rows'])
    if (not np.array_equal(s[:,1:],d[:,1:]) or not np.array_equal(ds[:,1:],dd[:,1:])
            or not np.array_equal(d[:,2],dd[:,2])):raise ValueError('Paired evaluation differs')
    counts=d[:,2];samples=np.random.default_rng(42).integers(0,len(d),(2000,len(d)))
    def compare(delta):
        boot=delta[samples].sum(1)/counts[samples].sum(1)
        return dict(delta_nats=float(delta.sum()/counts.sum()),ci95=np.quantile(boot,[.025,.975]).tolist(),
            halves=[float(delta[h::2].sum()/counts[h::2].sum()) for h in (0,1)])
    comparisons=dict(deep_vs_A=compare(d[:,0]-d[:,1]),shallow_vs_A=compare(s[:,0]-s[:,1]),
                     deep_vs_shallow=compare(d[:,0]-s[:,0]))
    dense_comparisons=dict(deep_vs_A=compare(dd[:,0]-dd[:,1]),shallow_vs_A=compare(ds[:,0]-ds[:,1]),
                           deep_vs_shallow=compare(dd[:,0]-ds[:,0]))
    # A deliberately modest, predeclared effect gate, not proof of publication value.
    passed=all(c[k]['delta_nats']<-.001 and c[k]['ci95'][1]<0
               and max(c[k]['halves'])<0 for c in (comparisons,dense_comparisons)
               for k in ('deep_vs_A','deep_vs_shallow'))
    result=dict(status='passed',scientific_gate='promising_teacher' if passed else 'stop_no_teacher_advantage',
        comparisons=comparisons,sdpa_comparisons=dense_comparisons,
        baseline_nll=deep['baseline_nll'],deep_nll=deep['nll'],shallow_nll=shallow['nll'],
        training_seconds={s:r['elapsed_seconds'] for s,r in zip(('shallow','deep'),(shallow,deep))},
        scope=scope,uncertainty='Exploratory packed-row bootstrap; one training seed, not an independent document or seed CI.',
        next_action='Review teacher benefit before any proxy distillation; no automatic longer training.')
    save(out/'summary.json',result);print('TEACHER_FIRST_SUMMARY',json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('prepare','fit','summarize'));p.add_argument('--root',required=True)
    p.add_argument('--output');p.add_argument('--source',choices=('shallow','deep'));p.add_argument('--steps',type=int)
    p.add_argument('--predecessor')
    args=p.parse_args();globals()[args.mode](args)
