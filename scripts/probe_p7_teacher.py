"""Small frozen-weight consumer-teacher test; eight independent GPU workers."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from scripts.train_then_burn import digest


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def selected_rows(length, count):
    if count < 8 or count % 2 or count > length:
        raise ValueError('Require an even number of distinct held-out rows')
    return [i*length//count for i in range(count)]


def predecessor_ready(root, guard=Path('/mnt/local/_gpu_guard/DISABLED')):
    root = Path(root)/'supervised'
    state, run, burn = (read(root/p) for p in ('supervisor.json','run/run.json','burn-verified.json'))
    result = read(root/'run/downstream/full/summary.json')
    if (state.get('training_status') != 'ok' or state.get('training_returncode') != 0
            or not state.get('finished_at') or run['status'] != 'ok'
            or len(run['jobs']) != 21 or any(j['status'] != 'ok' for j in run['jobs'])
            or not (root/'run/complete.json').is_file() or guard.exists()
            or not burn.get('collective_progress_verified')
            or not state.get('burn', {}).get('collective_progress_verified')
            or state['burn']['session'] != burn['session']
            or result['status'] != 'passed' or result['diagnostic'] is not False
            or len(result['checkpoints']) != 4):
        raise ValueError('Consumer queue and verified burn handoff must be complete')


def prepare(args):
    import importlib.metadata as md
    import shutil
    import socket
    import subprocess
    from accelerate.commands.config.config_args import default_yaml_config_file, load_config_from_file
    from transformers import AutoTokenizer
    from deep_kv.packing import preprocess_dataset
    from train import load_text
    from scripts.train_then_burn import approved_launcher, APPROVED_BURNS, BURN_HASH, GUARD_HASH
    from scripts.verified_gpu_reclaim import inspect, process
    from run_experiments import load_jobs

    root = Path(args.root)
    if socket.gethostname() != args.host:
        raise ValueError('Wrong machine')
    predecessor_ready(args.predecessor)
    expected = dict(torch='2.14.1', transformers='5.9.0', accelerate='1.13.0', datasets='4.8.5',
                    **{'flash-attn-4':'4.0.0b33','nvidia-cutlass-dsl':'4.8.0'})
    versions = {k:md.version(k) for k in expected}
    if any(versions[k].split('+')[0] != v for k,v in expected.items()):
        raise ValueError(('Environment changed', versions))
    spec = read('resources/p7_teacher_probe_20261015.json')
    configs = []
    for item in spec['checkpoints']:
        ck = Path(item['checkpoint']); cfg = read(ck.parent/'train_config.json')
        if (digest(ck/'model.safetensors') != item['sha256']
                or read(ck/'trainer_state.json')['global_step'] != 2500
                or cfg['model_config']['num_hidden_layers'] != 12
                or cfg['training']['seed'] != 42 or cfg['training']['data_seed'] != 42
                or cfg['pilot']['attention_backend'] != 'fa4'
                or not cfg['data']['isolate_documents']):
            raise ValueError('Unexpected checkpoint: '+str(ck))
        configs.append(cfg)
    first = configs[0]
    for cfg in configs:
        if (cfg['eval_fingerprint'] != first['eval_fingerprint'] or cfg['data'] != first['data']
                or cfg['model']['tokenizer_name'] != first['model']['tokenizer_name']):
            raise ValueError('Checkpoint data identities differ')
    tokenizer = AutoTokenizer.from_pretrained(first['model']['tokenizer_name'], local_files_only=True)
    if tokenizer.pad_token is None:tokenizer.pad_token = tokenizer.eos_token
    # Reuse train.py packing/cache. No CUDA context, Trainer, optimizer or new split.
    dataset = preprocess_dataset(load_text(first['data']['eval_data_dir']), tokenizer,
        first['data']['block_size'], SimpleNamespace(main_process_first=lambda **kw:nullcontext()),
        num_proc=1, isolate_documents=True).select(range(first['data']['eval_rows']))
    if dataset._fingerprint != first['eval_fingerprint']:
        raise ValueError('Original validation fingerprint does not match')
    indices = selected_rows(len(dataset), spec['rows'])
    rows = [dict(index=i, data=dataset[i]) for i in indices]
    if any(len(r['data']['input_ids']) != spec['sequence_length'] for r in rows):
        raise ValueError('Unexpected context length')
    save(root/'rows.json', rows)
    save(root/'scope.json', dict(**spec, rows_sha256=digest(root/'rows.json'),
        eval_fingerprint=dataset._fingerprint, versions=versions))
    source = Path('resources/accelerate_config.yaml'); destination = Path(default_yaml_config_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():shutil.copy2(destination, root/'accelerate.previous.yaml')
    shutil.copy2(source, destination)
    config = load_config_from_file(str(destination)).to_dict()
    if (source.read_bytes() != destination.read_bytes() or config['num_processes'] != 8
            or config['mixed_precision'] != 'bf16'):
        raise ValueError('Accelerate configuration differs')
    subprocess.run(['accelerate','env'], check=True)
    status = inspect()
    if (status['host'] != args.host or status['guard_disabled'] or len(status['gpus']) != 8
            or any('B200' not in g['name'] for g in status['gpus'])
            or digest('/mnt/local/_gpu_guard/gpu_guard.sh') != GUARD_HASH):
        raise ValueError('Unexpected GPU ownership/state')
    approved = {**APPROVED_BURNS, str(Path.cwd()/'resources/llm_pretrain_burn.py'):BURN_HASH}
    if any(not (approved_launcher(p,approved) or approved_launcher(process(p)['ppid'],approved))
           for p in status['workers']):
        raise ValueError('Unknown GPU workload; no signals sent')
    save(root/'inspection.json', status)
    jobs = [dict(name=stage, argv=['{python}','-m','scripts.probe_p7_teacher','run',
        '--root',str(root),'--output','{run_dir}/'+stage,'--stage',stage],
        gpus=list(range(8)), timeout_seconds=3600,
        required_outputs=[dict(path=stage+'/summary.json',json_equals={'status':'passed'})])
        for stage in ('smoke','full')]
    save(root/'jobs.json', dict(jobs=jobs));load_jobs(root/'jobs.json')
    save(root/'preflight.json', dict(status='passed', versions=versions,
        accelerate_cache=str(destination), accelerate_sha256=digest(destination),
        rows_sha256=digest(root/'rows.json'), eval_fingerprint=dataset._fingerprint))
    print('TEACHER_PROBE_PREFLIGHT_PASSED', flush=True)


def worker(args):
    import torch
    from deep_kv.model import Context
    from deep_kv.packing import isolated_data_collator
    from eval.models import load_checkpoint
    from eval.p7_teacher import capture, intervention, validate_capture
    from scripts.evaluate_proxy_gates import state_hash
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = Path(args.root); scope = read(root/'scope.json')
    if digest(root/'rows.json') != scope['rows_sha256']:
        raise ValueError('Frozen evaluation rows changed')
    spec = scope['checkpoints'][args.case]
    adapter, _, metadata = load_checkpoint(spec['checkpoint'], 'cuda', attention_backend='fa4')
    m = adapter.wrapped; m.eval()
    if (metadata['checkpoint_sha256'] != spec['sha256'] or metadata['step'] != 2500
            or list(m.layers) != scope['sites'] or torch.cuda.device_count() != 1):
        raise ValueError('Wrong model, sites or GPU assignment')
    initial = state_hash(m)
    rows = read(root/'rows.json')[args.half::2]
    if args.stage == 'smoke':rows = rows[:2]
    records, checks = [], []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for begin in range(0,len(rows),8):
            selected = rows[begin:begin+8]
            inputs = {k:v.cuda() for k,v in isolated_data_collator([r['data'] for r in selected]).items()}
            ctx = Context(inputs['input_ids'],inputs['attention_mask'].bool(),inputs['position_ids'],
                          inputs['segments'],inputs['labels'])
            hidden, sites = capture(m,ctx)
            base, counts, _ = m.lm_statistics(ctx,hidden)
            scores = {'baseline':base.double().cpu().tolist()}
            if begin == 0:
                plain = m.hidden_states(ctx)[0]
                torch.testing.assert_close(plain,hidden,atol=0,rtol=0)
                for site in scope['sites']:
                    with intervention(m,sites,site,0):identity = m.hidden_states(ctx)[0]
                    torch.testing.assert_close(identity,hidden,atol=0,rtol=0)
                checks.append(dict(identity_exact=True, consumer_errors=validate_capture(m,sites,ctx)))
            for site in scope['sites']:
                for alpha in scope['alphas']:
                    with intervention(m,sites,site,alpha):changed = m.hidden_states(ctx)[0]
                    losses, targets, _ = m.lm_statistics(ctx,changed)
                    if not torch.equal(targets,counts) or not bool(torch.isfinite(losses).all()):
                        raise ValueError('Invalid intervention loss or target count')
                    scores[f'{site}:{alpha}'] = losses.double().cpu().tolist()
            for j,row in enumerate(selected):
                records.append(dict(row=row['index'], targets=int(counts[j]),
                    nll={k:v[j] for k,v in scores.items()}))
            print('BATCH_DONE',spec['label'],args.half,begin+len(selected),'/',len(rows),flush=True)
    if initial != state_hash(m) or digest(Path(spec['checkpoint'])/'model.safetensors') != spec['sha256']:
        raise ValueError('Model state or source checkpoint changed')
    save(Path(args.output)/'result.json', dict(status='passed',case=args.case,half=args.half,
        stage=args.stage,checkpoint_sha256=spec['sha256'],rows_sha256=scope['rows_sha256'],
        rows=records, checks=checks, model_state_unchanged=True))


def summarize(scope, reports, source_rows, stage):
    import numpy as np
    result = {}
    for case,spec in enumerate(scope['checkpoints']):
        pair = sorted([r for r in reports if r['case']==case],key=lambda r:r['half'])
        if [r['half'] for r in pair] != [0,1]:raise ValueError('Missing/duplicate workers')
        for half,report in enumerate(pair):
            expected = source_rows[half::2]
            if stage=='smoke':expected=expected[:2]
            if (report['status']!='passed' or report['stage']!=stage
                    or report['checkpoint_sha256']!=spec['sha256']
                    or report['rows_sha256']!=scope['rows_sha256']
                    or not report['model_state_unchanged']
                    or not report['checks'] or not all(c['identity_exact'] for c in report['checks'])
                    or [r['row'] for r in report['rows']]!=[r['index'] for r in expected]):
                raise ValueError('Worker identity, checks or coverage differs')
        rows = [r for report in pair for r in report['rows']]
        counts = np.array([r['targets'] for r in rows]);base=np.array([r['nll']['baseline'] for r in rows])
        if (counts<=0).any() or not np.isfinite(base).all():raise ValueError('Invalid baseline')
        scores = {}
        # Exploratory paired-row bootstrap, not multiplicity-adjusted or a seed CI.
        samples = np.random.default_rng(42).integers(0,len(rows),(2000,len(rows)))
        for site in scope['sites']:
            for alpha in scope['alphas']:
                key=f'{site}:{alpha}';delta=np.array([r['nll'][key] for r in rows])-base
                if not np.isfinite(delta).all():raise ValueError('Invalid delta')
                boot=delta[samples].sum(1)/counts[samples].sum(1)
                halves=[sum(r['nll'][key]-r['nll']['baseline'] for r in p['rows'])/
                        sum(r['targets'] for r in p['rows']) for p in pair]
                scores[key]=dict(delta_nats_per_token=float(delta.sum()/counts.sum()),
                    halves=halves, exploratory_row_ci95=np.quantile(boot,[.025,.975]).tolist())
        result[spec['label']]=dict(checkpoint=spec['checkpoint'],checkpoint_sha256=spec['sha256'],
            baseline_nats_per_token=float(base.sum()/counts.sum()),target_tokens=int(counts.sum()),
            rows=len(rows),interventions=scores)
    return dict(status='passed',stage=stage,results=result,
        scope='Single-site counterfactual, teacher frozen from unchanged trajectory; not recurrent or an oracle bound.',
        interpretation='Negative delta improves LM loss. Compare +0.1 with -0.1 and both row halves; 1.0 tests compatibility at full strength.',
        uncertainty='Exploratory paired packed-row bootstrap; not independent training seeds or multiplicity-adjusted claims.')


def run(args):
    from eval.suite import execute
    root=Path(args.root);output=Path(args.output);scope=read(root/'scope.json')
    output.mkdir(parents=True,exist_ok=False)
    jobs=[]
    for case in range(len(scope['checkpoints'])):
        for half in (0,1):
            dest=output/f'{case}-{half}'
            jobs.append(dict(output=str(dest),argv=[sys.executable,'-m','scripts.probe_p7_teacher','worker',
                '--root',str(root),'--output',str(dest),'--stage',args.stage,'--case',str(case),'--half',str(half)]))
    execute(jobs,workers=8)
    report=summarize(scope,[read(Path(j['output'])/'result.json') for j in jobs],read(root/'rows.json'),args.stage)
    save(output/'summary.json',report)
    print('TEACHER_PROBE_RESULT',json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('prepare','worker','run'));p.add_argument('--root',required=True)
    p.add_argument('--host');p.add_argument('--predecessor');p.add_argument('--output')
    p.add_argument('--stage',choices=('smoke','full'));p.add_argument('--case',type=int)
    p.add_argument('--half',type=int,choices=(0,1))
    a=p.parse_args();globals()[a.mode](a)
