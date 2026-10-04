"""C1/C3 full-objective train/eval backend smoke; no optimizer updates."""
import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
import os
from pathlib import Path

os.environ.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  HF_HUB_DISABLE_TELEMETRY='1', WANDB_MODE='offline')
import torch
from transformers import AutoConfig, TrainingArguments, set_seed
from datasets import load_from_disk
from deep_kv import ARMS
from deep_kv.model import DeepKV
from deep_kv.training import DeepKVTrainer
from deep_kv.sdpa_audit import SDPAAudit, runtime_metadata
from scripts.benchmark_document_training import Collator, write
from scripts.check_trained_attention import fingerprint


def capture(trainer, inputs, training):
    model = trainer.model
    model.train(training)
    def execute(audit=None):
        model.zero_grad(set_to_none=True)
        with (nullcontext() if training else torch.no_grad()), torch.autocast("cuda", dtype=torch.bfloat16), trainer.compute_loss_context_manager():
            loss, outputs = trainer.compute_loss(model, inputs, return_outputs=True)
        if training:
            if audit is not None:
                loss.register_hook(audit.backward_marker)
            loss.backward()
        return loss, outputs
    # Warm up the actual objective/path before observing backend choices.
    execute()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    audit = SDPAAudit(model)
    with audit.observe():
        loss, outputs = execute(audit)
    torch.cuda.synchronize()
    result = audit.result()
    result.update(loss=float(loss.detach()), document_id_dtype=None if inputs.get('segments') is None else str(inputs['segments'].dtype),
        statistics={k: float(v.detach()) for k,v in outputs.items() if k != 'statistics'},
        peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
        finite_gradients=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()))
    counts = {site: row['forward_calls'] for site,row in result['sites'].items()}
    expected = {f'backbone.layer_{i}':1 for i in range(model.backbone.config.num_hidden_layers)}
    if model.aux is not None:
        expected['auxiliary.branch'] = 1
    if model.consumer_aware:
        expected['auxiliary.detached_consumer'] = 1
    result['expected_forward_counts'] = expected
    result['counts_match'] = counts == expected
    result['complete_attribution'] = not result['unattributed_calls'] and not result['unattributed_backward']
    # A fused training attention call must also have a recorded backward; math
    # attention's backward is decomposed and has no single fused SDPA operator.
    result['backward_sites_complete'] = (not training or all(
        row['backward_operators'] or any('attention_math' in op for op in row['forward_operators'])
        for row in result['sites'].values()))
    result['passed'] = (result['counts_match'] and result['complete_attribution']
                        and result['finite_gradients'] and result['backward_sites_complete'])
    model.zero_grad(set_to_none=True)
    return result


def worker(args):
    torch.set_num_threads(2)
    set_seed(42)
    source = Path(args.source)
    root = Path(args.output)
    recipe = json.loads((source/'data.json').read_text())['recipe']
    data = {split:list(load_from_disk(str(source/split)).select(range(16))) for split in ('data','eval')}
    if args.index == 0:
        trainer_receipt_smoke(root)
    for arm in ARMS[args.index::8]:
        for checkpoint in (False, True):
            cfg = AutoConfig.from_pretrained(recipe['config_name'], local_files_only=True)
            cfg._attn_implementation='sdpa';cfg.use_cache=False
            model = DeepKV.from_scratch(cfg,arm,seed=42,checkpoint_layers=checkpoint,
                checkpoint_lm=checkpoint,checkpoint_aux=checkpoint,lm_chunk=128).cuda()
            initial = fingerprint(model)
            settings=TrainingArguments(output_dir=str(root/'trainer'/f'{arm}-{int(checkpoint)}'),
                bf16=True,report_to=[],per_device_train_batch_size=16,per_device_eval_batch_size=16,
                gradient_accumulation_steps=4,remove_unused_columns=False)
            trainer=DeepKVTrainer(model=model,args=settings)
            for isolated in (False,True):
                mode='sdpa_isolated' if isolated else 'sdpa_cross'
                for phase in ('train','eval'):
                    name=f'{arm}-ckpt{int(checkpoint)}-{mode}-{phase}'
                    destination=root/(name+'.json')
                    if destination.exists():raise ValueError(f'Refuse overwrite {destination}')
                    inputs={k:v.cuda() for k,v in Collator(mode)(data['data' if phase=='train' else 'eval']).items()}
                    print('PROFILE_START',name,flush=True)
                    result=capture(trainer,inputs,phase=='train')
                    result.update(arm=arm,mode=mode,phase=phase,checkpointing=checkpoint,
                        microbatch=16,sequence_length=2048,world_size=1,optimizer_updates=0,
                        runtime=runtime_metadata(),model_config=cfg.to_dict(),
                        parameter_sha256=initial,parameters_unchanged=fingerprint(model)==initial,
                        input_sha256=hashlib.sha256(inputs['input_ids'].cpu().numpy().tobytes()).hexdigest(),
                        initialization='random seed 42; unchanged auxiliary zero initialization',
                        profile_scope='independent per-GPU full-objective microbatch; not DDP throughput')
                    result['passed'] &= result['parameters_unchanged']
                    write(destination,result)
                    print('PROFILE_DONE',name,'passed',result['passed'],'math',result['has_math'],
                          'peak_gib',result['peak_allocated_gib'],flush=True)
                    if not result['passed']:raise RuntimeError(f'Incomplete or invalid profile: {destination}')
                    del inputs,result
            del trainer,model
            gc.collect();torch.cuda.empty_cache()


def trainer_receipt_smoke(root):
    """Exercise automatic receipts through a real Trainer train/evaluate loop."""
    from datasets import Dataset
    from tests.test_deep_kv import config
    model=DeepKV.from_scratch(config(),'Consumer-Aware-Align',consumer=2,deep_target=4,
        checkpoint_layers=True,checkpoint_lm=True,checkpoint_aux=True,lm_chunk=3)
    rows=[dict(input_ids=[3,4,31,8,9,10,31],attention_mask=[1]*7,
        labels=[3,4,31,8,9,10,31],segments=[0,0,0,1,1,1,1],position_ids=[0,1,2,0,1,2,3])]*4
    data=Dataset.from_list(rows)
    args=TrainingArguments(output_dir=str(root/'receipt-smoke'),bf16=True,report_to=[],
        per_device_train_batch_size=2,per_device_eval_batch_size=2,max_steps=2,
        eval_on_start=True,save_strategy='no',remove_unused_columns=False,disable_tqdm=True)
    trainer=DeepKVTrainer(model=model,args=args,train_dataset=data,eval_dataset=data)
    trainer.train();trainer.evaluate()
    if len(trainer._sdpa_receipts)!=2 or trainer._sdpa_seen!={'train','eval'}:
        raise RuntimeError('Expected exactly one automatic receipt per phase')
    write(root/'trainer-receipt-smoke.json',dict(status='passed',receipts=trainer._sdpa_receipts,
        scope='tiny fixture, two optimizer updates; validates automatic Trainer integration only'))


def summarize(args):
    root=Path(args.output);cases=[]
    for arm in ARMS:
        for ckpt in (0,1):
            for mode in ('sdpa_cross','sdpa_isolated'):
                for phase in ('train','eval'):
                    name=f'{arm}-ckpt{ckpt}-{mode}-{phase}'
                    row=json.loads((root/(name+'.json')).read_text())
                    if not row['passed']:raise ValueError(name)
                    cases.append(dict(name=name,arm=arm,mode=mode,phase=phase,checkpointing=ckpt,
                        sites=row['sites'],has_math=row['has_math'],peak_allocated_gib=row['peak_allocated_gib']))
    write(root/'summary.json',dict(status='passed',cases=cases,math_cases=[r['name'] for r in cases if r['has_math']],
        note='Math use is a performance finding, not a correctness failure. No production isolation/pinning enabled.'))
    print('ALL_PROFILES_COMPLETE',len(cases),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['worker','summarize'])
    parser.add_argument('--source',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--index',type=int,choices=range(8))
    args=parser.parse_args()
    if args.command=='worker' and args.index is None:parser.error('worker needs --index')
    {'worker':worker,'summarize':summarize}[args.command](args)
