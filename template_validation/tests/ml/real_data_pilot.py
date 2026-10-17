"""Prepare a bounded real-data pilot using existing local Qwen assets.

Creates a fixed text slice and a manifest for the unchanged native training
entry point. No GPU management, downloads, or environment changes.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def make_jobs(model_config, tokenizer, train, validation, accelerate, cache):
    example = json.loads((ROOT / 'jobs.training.example.json').read_text())
    jobs = []
    for backend in ('sdpa', 'fa4'):
        for original in example['jobs']:
            job = copy.deepcopy(original)
            name = original['name']
            job['name'] = backend + '-' + name
            replacements = {'/absolute/model/config.json': str(model_config),
                '/absolute/tokenizer': str(tokenizer), '/absolute/data/train': str(train),
                '/absolute/data/validation': str(validation)}
            a = [replacements.get(x, x).replace('{project_dir}', str(ROOT)).replace('{run_dir}', '{run_dir}/' + backend)
                 .replace('/checkpoint-2', '/checkpoint-25').replace('/checkpoint-4', '/checkpoint-50')
                 for x in job['argv']]
            def option(key, value):
                if key in a: a[a.index(key)+1] = str(value)
                else: a.extend([key, str(value)])
            if name in ('train', 'resume'):
                # HF 5.9 maps the deprecated ratio onto warmup_steps, even if zero.
                index = a.index('--warmup_ratio')
                del a[index:index+2]
                settings = {'--use_cpu':'false', '--attention_backend':backend, '--block_size':2048,
                    '--max_steps':28600, '--stop_after':25 if name=='train' else 50,
                    '--per_device_train_batch_size':16, '--gradient_accumulation_steps':4,
                    '--per_device_eval_batch_size':2, '--learning_rate':0.0003,
                    '--lr_scheduler_type':'cosine_with_min_lr',
                    '--warmup_steps':1430, '--lr_scheduler_kwargs':'{"min_lr_rate": 0.1}',
                    '--weight_decay':0.1, '--adam_beta1':0.9, '--adam_beta2':0.95,
                    '--logging_steps':5, '--eval_steps':25, '--bf16':'true',
                    '--gradient_checkpointing':'true', '--skip_memory_metrics':'false',
                    '--data_cache_dir':str(cache), '--preprocessing_num_workers':8,
                    '--document_end_token':'<|endoftext|>', '--eval_max_sequences':128,
                    '--dataloader_num_workers':4, '--ddp_timeout':1800}
                for key, value in settings.items(): option(key, value)
            elif name in ('verify-train', 'verify-resume'):
                option('--expected-step', 25 if name=='verify-train' else 50)
                option('--schedule-steps',28600)
                option('--start-step',0 if name=='verify-train' else 25)
            elif name == 'evaluate':
                option('--use_cpu','false'); option('--precision','bf16')
                option('--max_sequences',128)
            if name in ('train','resume','evaluate'):
                job['gpus'] = list(range(8))
                a = ['{python}','-m','accelerate.commands.launch','--config_file',str(accelerate),
                     '--main_process_port','0'] + a[1:]
            job['argv'] = a
            job['timeout_seconds'] = 3600
            for spec in job['required_outputs']:
                spec['path'] = backend + '/' + spec['path'].replace('/checkpoint-2','/checkpoint-25').replace('/checkpoint-4','/checkpoint-50')
                for key, value in list(spec.get('json_equals',{}).items()):
                    if key=='global_step': spec['json_equals'][key] = {2:25,4:50}[value]
                    elif key in ('max_steps','schedule_steps'): spec['json_equals'][key]=28600
            jobs.append(job)
    return {'jobs':jobs}



def validate_schedule(jobs):
    # Parse with the installed HF version before reclaiming GPUs. Device and
    # precision are changed only for this CPU validation, not in the job argv.
    from transformers import HfArgumentParser
    from train import ModelArguments, DataArguments, BaselineTrainingArguments
    for job in jobs['jobs']:
        if job['name'] not in ('sdpa-train', 'sdpa-resume', 'fa4-train', 'fa4-resume'):
            continue
        argv = job['argv']
        argv = argv[argv.index(str(ROOT/'train.py'))+1:]
        argv[argv.index('--use_cpu')+1] = 'true'
        argv[argv.index('--bf16')+1] = 'false'
        _, _, args = HfArgumentParser((ModelArguments, DataArguments,
            BaselineTrainingArguments)).parse_args_into_dataclasses(argv)
        expected = 25 if job['name'].endswith('-train') else 50
        if args.max_steps != 28600 or args.get_warmup_steps(args.max_steps) != 1430 or args.stop_after != expected:
            raise ValueError('Native HF schedule differs from the declared pilot recipe')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-config',type=Path,required=True)
    p.add_argument('--accelerate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    import os
    os.environ.update(HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='offline')
    from training_data import discover_sources,load_text
    config=json.loads(args.source_config.read_text())
    for key in ('config_name','tokenizer_name','data_dir','eval_data_dir'):
        if not Path(config[key]).exists(): raise FileNotFoundError(config[key])
    receipt={'initialization':'random, existing full Qwen config and tokenizer', 'splits':{}}
    for split,key,limit in [('train','data_dir',100000),('validation','eval_data_dir',256)]:
        kind,source=discover_sources(config[key])[0]
        raw,records=load_text(source,cache_dir=str(args.output/'input-cache'))
        selected=raw.select(range(min(limit,len(raw))))
        target=args.output/split
        selected.save_to_disk(str(target))
        receipt['splits'][split]={'source_root':config[key],'first_source':str(source),
            'source_records':records,'selection':'ordered prefix of first sorted source',
            'rows':len(selected),'fingerprint':selected._fingerprint,
            'first_text_sha256':hashlib.sha256(selected[0]['text'].encode()).hexdigest(),
            'last_text_sha256':hashlib.sha256(selected[-1]['text'].encode()).hexdigest(),
            'output':str(target)}
    receipt['model_config_sha256']=hashlib.sha256(Path(config['config_name']).read_bytes()).hexdigest()
    (args.output/'data-selection.json').write_text(json.dumps(receipt,indent=2)+'\n')
    jobs=make_jobs(config['config_name'],config['tokenizer_name'],args.output/'train',
        args.output/'validation',args.accelerate,args.output/'hf-cache')
    validate_schedule(jobs)
    manifest=args.output/'jobs.json';manifest.write_text(json.dumps(jobs,indent=2)+'\n')
    from run_experiments import load_jobs
    load_jobs(manifest)
    print('REAL_DATA_PILOT_PREPARED',json.dumps(receipt),flush=True)


if __name__=='__main__': main()
