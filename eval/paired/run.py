"""Opt-in evaluation runner, reusing the existing checkpoint and harness adapters."""
import json
import os
from pathlib import Path
import subprocess
import sys

from . import artifacts, protocol


def token_hash(tokenizer):
    return protocol.digest(dict(backend=tokenizer.backend_tokenizer.to_str(),
        special_tokens=tokenizer.special_tokens_map,padding_side=tokenizer.padding_side,
        truncation_side=tokenizer.truncation_side,bos=tokenizer.bos_token_id,eos=tokenizer.eos_token_id,
        pad=tokenizer.pad_token_id,add_bos_token=getattr(tokenizer,'add_bos_token',None),
        add_eos_token=getattr(tokenizer,'add_eos_token',None)))


def load_model(checkpoint,device,path):
    import torch
    from transformers import AutoTokenizer
    from eval.models import load_checkpoint, file_hash
    dtype=torch.float32 if path=='reference' else torch.bfloat16
    backend='sdpa' if path in ('reference','sdpa-bf16') else None
    if path=='reference':
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    model,source,meta=load_checkpoint(checkpoint,device,dtype,backend)
    if not hasattr(model,'wrapped'):
        raise ValueError('Paired study requires project A/P6 checkpoints and their training recipe')
    if path=='reference': model.autocast_dtype=None
    tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True)
    if tokenizer.pad_token is None: tokenizer.pad_token=tokenizer.eos_token
    meta.update(tokenizer_sha256=token_hash(tokenizer),precision_path=path,
                weights_dtype=str(next(model.parameters()).dtype),softmax_dtype='float32',
                autocast_dtype=str(model.autocast_dtype),matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_tf32=torch.backends.cudnn.allow_tf32,
                attention_runtime=getattr(model.wrapped,'attention_runtime',dict(backend=meta['attention_backend'])))
    return model,tokenizer,meta


def evaluate(checkpoint,task,suite,output,frozen_path,dataset_root,manifest,
             device='cuda',batch_size=16,path='production',selection=None):
    import lm_eval
    from importlib.metadata import version
    from eval.benchmarks import local_dataset_paths
    from eval.eval_checkpoint import json_default
    from eval.generation import harness_model_class
    from eval.models import file_hash
    frozen=protocol.read(frozen_path)
    shots=protocol.tasks(suite)[task]
    if batch_size<1: raise ValueError('Positive batch size required')
    output=Path(output)
    if output.exists(): raise ValueError('Choose a fresh output directory')
    mapping=local_dataset_paths(dataset_root,manifest)
    configs=protocol.configs(task,mapping)
    model,tokenizer,meta=load_model(checkpoint,device,path)
    class AuditedLM(harness_model_class()):
        def _loglikelihood_tokens(self,requests,*args,**kwargs):
            for _,context,continuation in requests:
                length=len(context)+len(continuation)
                self.context_audit['requests']+=1
                self.context_audit['truncated_requests']+=int(length>self.max_length+1)
                self.context_audit['maximum_request_tokens']=max(length,self.context_audit['maximum_request_tokens'])
            return super()._loglikelihood_tokens(requests,*args,**kwargs)
    lm=AuditedLM(pretrained=model,tokenizer=tokenizer,batch_size=batch_size,
        device=device,backend='causal',max_length=min(2048,model.config.max_position_embeddings),
        softmax_dtype='float32',logits_cache=not task.startswith('paloma_'))
    lm.context_audit=dict(requests=0,truncated_requests=0,maximum_request_tokens=0,
                         max_length=lm.max_length,policy='unchanged harness left truncation; full rolling windows for Paloma')
    meta.update(protocol_sha256=frozen['protocol_sha256'],task=task,suite=suite,
        task_config_sha256=protocol.digest(protocol.configs(task)),batch_size=batch_size,
        seed=42,num_fewshot=shots,prefix_token_id=lm.prefix_token_id,selection=selection,
        dataset_manifest_sha256=file_hash(manifest),status='started',
        packages={p:version(p) for p in ('lm_eval','torch','transformers','datasets','accelerate')})
    output.mkdir(parents=True)
    protocol.write_json(output/'eval_metadata.json',meta)
    result=lm_eval.simple_evaluate(model=lm,tasks=configs,task_manager=protocol.manager(),
        num_fewshot=shots,random_seed=42,numpy_random_seed=42,torch_random_seed=42,
        fewshot_random_seed=42,apply_chat_template=False,log_samples=True,
        samples=selection,bootstrap_iters=0)
    # 0.4.10 reports full effective counts with `samples=`. Keep the original
    # metadata and repair only this documented subset-reporting bug, after
    # checking the exact requested ID set; full-run counts are never repaired.
    if selection is not None:
        meta['harness_n_samples']=json.loads(json.dumps(result['n-samples']))
        for sub,ids in selection.items():
            rows=result['samples'][sub]
            if {s['doc_id'] for s in rows}!=set(ids) or len(rows)!=len(ids):
                raise ValueError('Harness did not evaluate the frozen precision selection')
            result['n-samples'][sub]['effective']=len(rows)
    # Round-trip exactly as the saved artifact before validating.
    result=json.loads(json.dumps(result,default=json_default,allow_nan=False))
    artifacts.validate(result,result['samples'],task,shots,selection)
    meta['context_audit']=lm.context_audit
    with (output/'eval_samples.jsonl').open('x') as stream:
        for name,rows in result.pop('samples').items():
            for row in rows: stream.write(json.dumps(dict(task=name,**row),allow_nan=False)+'\n')
    protocol.write_json(output/'eval_benchmarks_full.json',result)
    meta['status']='completed'
    temporary=output/'metadata.pending.json'; protocol.write_json(temporary,meta)
    os.replace(temporary,output/'eval_metadata.json')


def freeze_selection(frozen,dataset_root,manifest):
    """Select without examining model outputs; independent seed per subtask."""
    import numpy as np
    from lm_eval.tasks import get_task_dict
    from eval.benchmarks import local_dataset_paths
    mapping=local_dataset_paths(dataset_root,manifest); selection={};sizes={}
    for task in protocol.PRECISION:
        loaded=get_task_dict(protocol.configs(task,mapping),task_manager=protocol.manager())
        selection[task]={};sizes[task]={}
        for name in protocol.inventory(task):
            n=len(loaded[name].eval_docs)
            count=frozen['precision']['per_mmlu_subject' if task=='mmlu_continuation' else 'per_task']
            if n<count: raise ValueError('Precision subset exceeds dataset: '+name)
            seed=int(protocol.digest([42,name])[:16],16)
            selection[task][name]=sorted(np.random.default_rng(seed).choice(n,count,replace=False).tolist())
            sizes[task][name]=n
    return selection,sizes


def plan(output,protocol_path,checkpoints,dataset_root,manifest,position_sample,calibration,
         batch_size=16,device='cuda',legacy_f=None,defer_paloma=False):
    from eval.models import file_hash
    from . import stats, position
    from . import audit as audits
    from .audit import data
    frozen=protocol.read(protocol_path);output=Path(output).resolve()
    if output.exists(): raise ValueError('Choose a fresh study output directory')
    if batch_size<1: raise ValueError('Positive batch size required')
    sample=position.read_sample(position_sample,frozen)
    calibration_result=artifacts.load_json(calibration)
    if (not calibration_result.get('full_protocol') or not calibration_result.get('passed') or
        calibration_result.get('statistics_sha256')!=file_hash(stats.__file__) or
        calibration_result.get('calibration_code_sha256')!=file_hash(audits.__file__)):
        raise ValueError('Full passing calibration of this statistics implementation is required')
    if set(checkpoints)!={'A10','A20','P20'}: raise ValueError('Require A10, A20 and P20 checkpoints')
    identities={}
    for arm,path in checkpoints.items():
        checkpoint=Path(path).resolve()
        recipe=next((p/'train_config.json' for p in (checkpoint,checkpoint.parent) if (p/'train_config.json').is_file()),None)
        if recipe is None:raise ValueError('Missing training recipe')
        saved=artifacts.load_json(recipe);state=artifacts.load_json(checkpoint/'trainer_state.json')
        if saved['pilot']['arm']!=('P6-iso' if arm=='P20' else 'A') or state['global_step']!=(10000 if arm=='A10' else 20000):
            raise ValueError('Wrong checkpoint arm or step for '+arm)
        identities[arm]=dict(path=str(checkpoint),sha256=file_hash(checkpoint/'model.safetensors'),recipe=str(recipe),recipe_sha256=file_hash(recipe))
    audit=data(dataset_root,manifest,frozen)
    selection,sizes=freeze_selection(frozen,dataset_root,manifest)
    value=dict(protocol=str(Path(protocol_path).resolve()),protocol_sha256=frozen['protocol_sha256'],
        checkpoints=identities,dataset_root=str(Path(dataset_root).resolve()),manifest=str(Path(manifest).resolve()),
        manifest_sha256=file_hash(manifest),batch_size=batch_size,device=device,selection=selection,sizes=sizes,
        position_sample=str(Path(position_sample).resolve()),sample_sha256=sample['sample_sha256'],
        calibration=str(Path(calibration).resolve()),calibration_sha256=file_hash(calibration),
        legacy_f=str(Path(legacy_f).resolve()) if legacy_f else None,legacy_f_sha256=file_hash(legacy_f) if legacy_f else None,
        deferred_suites=['paloma'] if defer_paloma else [],
        order=['precision','F','Z','probes','paloma','position'],output=str(output))
    value['plan_sha256']=protocol.digest(value)
    output.mkdir(parents=True);protocol.write_json(output/'plan.json',value);protocol.write_json(output/'data_audit.json',audit)
    return output/'plan.json'


def execute(plan_path):
    """Sequential, restartable at completed tasks; incomplete outputs fail closed."""
    from eval.models import file_hash
    from .analysis import precision
    from .position import read_sample
    value=artifacts.load_json(plan_path);body={k:v for k,v in value.items() if k!='plan_sha256'}
    if protocol.digest(body)!=value['plan_sha256']: raise ValueError('Plan was modified')
    if value.get('deferred_suites',[]) not in ([],['paloma']):raise ValueError('Only descriptive Paloma may be deferred')
    frozen=protocol.read(value['protocol']);root=Path(value['output'])
    if frozen['protocol_sha256']!=value['protocol_sha256'] or file_hash(value['manifest'])!=value['manifest_sha256']:
        raise ValueError('Protocol or dataset manifest changed since planning')
    for obj in value['checkpoints'].values():
        if file_hash(Path(obj['path'])/'model.safetensors')!=obj['sha256'] or file_hash(obj['recipe'])!=obj['recipe_sha256']:
            raise ValueError('Checkpoint or recipe changed since planning')
    if file_hash(value['calibration'])!=value['calibration_sha256']:raise ValueError('Calibration report changed')
    sample=read_sample(value['position_sample'],frozen)
    if sample['sample_sha256']!=value['sample_sha256']:raise ValueError('Position sample changed since planning')
    legacy=None
    if value['legacy_f']:
        if file_hash(value['legacy_f'])!=value['legacy_f_sha256']:raise ValueError('Historical mapping changed')
        legacy=artifacts.load_json(value['legacy_f'])
    def job(arm,task,suite,path,output,selection=None):
        if not output.exists():
            cmd=[sys.executable,'-m','eval.paired','evaluate','--checkpoint',value['checkpoints'][arm]['path'],
                 '--task',task,'--suite',suite,'--output',str(output),'--protocol',value['protocol'],
                 '--dataset-root',value['dataset_root'],'--manifest',value['manifest'],'--path',path,
                 '--device',value['device'],'--batch-size',str(value['batch_size'])]
            if selection is not None:
                selected=root/(task+'.selection.json')
                if not selected.exists(): protocol.write_json(selected,selection)
                if artifacts.load_json(selected)!=selection: raise ValueError('Changed selection file')
                cmd+=['--selection',str(selected)]
            subprocess.run(cmd,check=True)
        obj=artifacts.read(output,task,protocol.tasks(suite)[task],frozen,selection=selection)
        meta=obj['metadata']
        if (meta['checkpoint_sha256']!=value['checkpoints'][arm]['sha256'] or
            meta['recipe_sha256']!=value['checkpoints'][arm]['recipe_sha256'] or
            meta['dataset_manifest_sha256']!=value['manifest_sha256'] or
            meta['precision_path']!=path or meta['batch_size']!=value['batch_size']):
            raise ValueError('Existing output does not match queued job')
        return obj
    checks={}
    for task in protocol.PRECISION:
        objects={}
        for path in ('reference','production','repeat'):
            for arm in ('A20','P20'):
                objects[path,arm]=job(arm,task,'Z','production' if path=='repeat' else path,
                    root/'precision'/path/arm/task,value['selection'][task])
        checks[task]=precision(*(objects[path,arm] for path in ('reference','production','repeat')
                               for arm in ('A20','P20')),sum(value['sizes'][task].values()))
    selected='production' if all(x['passed'] for x in checks.values()) else 'reference'
    check=dict(tasks=checks,suite_Z_path=selected,protocol_sha256=frozen['protocol_sha256'],plan_sha256=value['plan_sha256'])
    report=root/'precision.json'
    if report.exists():
        if artifacts.load_json(report)!=check: raise ValueError('Precision decision changed')
    else: protocol.write_json(report,check)
    for suite in ('F','Z','probes','paloma'):
        if suite in value.get('deferred_suites',[]):continue
        for arm in (('A10','A20','P20') if suite in ('Z','probes') else ('A20','P20')):
            for task in protocol.tasks(suite):
                if suite=='F' and legacy:
                    obj=artifacts.read(legacy[arm][task],task,protocol.F[task],frozen,legacy=True)
                    if obj['metadata']['checkpoint_sha256']!=value['checkpoints'][arm]['sha256']:
                        raise ValueError('Historical checkpoint differs')
                else:job(arm,task,suite,selected if suite in ('Z','probes') else 'production',root/arm/suite/task)
    # Recheck after the benchmark jobs as well: inputs may change during a long queue.
    sample=read_sample(value['position_sample'],frozen)
    if sample['sample_sha256']!=value['sample_sha256']:raise ValueError('Position sample changed since planning')
    for arm in ('A10','A20','P20'):
        output=root/arm/'position'
        if not output.exists():
            subprocess.run([sys.executable,'-m','eval.paired','position','--checkpoint',value['checkpoints'][arm]['path'],
                '--sample',value['position_sample'],'--output',str(output),'--protocol',value['protocol'],
                '--path',selected,'--device',value['device']],check=True)
        meta=artifacts.load_json(output/'metadata.json')
        if (meta['status']!='completed' or meta['checkpoint_sha256']!=value['checkpoints'][arm]['sha256'] or
            meta['recipe_sha256']!=value['checkpoints'][arm]['recipe_sha256'] or meta['batch_size']!=1 or
            meta['tokenizer_sha256']!=sample['tokenizer_sha256'] or
            meta['sample_sha256']!=value['sample_sha256'] or meta['protocol_sha256']!=frozen['protocol_sha256'] or
            meta['precision_path']!=selected or meta['T8_max_abs_error']>1e-4 or
            file_hash(output/'losses.npy')!=meta['losses_sha256']):raise ValueError('Position job does not match plan')
