"""Offline A2 stages. Inspection/plan commands never launch GPU work implicitly."""
import os
os.environ.update(HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                  HF_HUB_DISABLE_TELEMETRY='1',WANDB_MODE='offline')
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from efficient_llm.diagnostic_data import sha256
from .data import write_json,PackedRows,prepare,load_pool,prepare_secondary,training_order
from .protocol import DATASETS,PARENTS,FIXED_SETS,Run,matrix,choose_s2,layers_for
from .model import load_parent,prune
from .measure import rank_world,score,survey
from .results import selected_lrs,write_report,estimate_workload


def read(path):
    text=Path(path).read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()] if Path(path).suffix=='.jsonl' else json.loads(text)


def provenance(args):
    import importlib.metadata
    import platform
    commit=args.code_commit
    if commit is None or len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('Supply the reviewed source commit via --code-commit or A2_CODE_COMMIT')
    files={f'a2/{p.name}':sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    project=Path(__file__).resolve().parents[2]
    for name in ('packed_evaluation.py','packed_training.py','training_data.py','efficient_llm/diagnostic_data.py'):
        files[name]=sha256(project/name)
    return dict(spec_sha256=sha256(args.spec),assets_sha256=sha256(args.assets),
        data_sha256=sha256(Path(args.data)/'manifest.json'),code_commit=commit,code_files=files,
        software={name:importlib.metadata.version(name) for name in
                  ('torch','transformers','accelerate','datasets','numpy','safetensors')},python=platform.python_version())


def eval_sets(root):
    manifest=read(Path(root)/'manifest.json')
    return dict(final=PackedRows(root,'final'),tuning=PackedRows(root,'tuning'),
        small=PackedRows(root,'tuning',count=len(manifest['small_indices']),order=manifest['small_indices']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    plan=sub.add_parser('plan'); plan.add_argument('--results',type=Path); plan.add_argument('--output',type=Path,required=True)
    secondary=sub.add_parser('prepare-secondary')
    secondary.add_argument('--source',type=Path,required=True); secondary.add_argument('--tokenizer',required=True)
    secondary.add_argument('--source-provenance',type=Path,required=True); secondary.add_argument('--output',type=Path,required=True)
    data=sub.add_parser('prepare'); data.add_argument('--pool',type=Path,required=True)
    data.add_argument('--qwen-tokenizer',required=True); data.add_argument('--tokenizer',required=True)
    data.add_argument('--pool-contract',type=Path,required=True)
    data.add_argument('--dataset-revision',required=True); data.add_argument('--output',type=Path,required=True)
    data.add_argument('--spec',type=Path,default=Path('docs/A2_pilot_spec.md'))
    data.add_argument('--tokenizer-provenance',type=Path,required=True)
    for name in ('survey','pre','check','throughput','heal'):
        p=sub.add_parser(name)
        p.add_argument('--code-commit',default=os.environ.get('A2_CODE_COMMIT'))
        p.add_argument('--secondary-map',type=Path)
        p.add_argument('--assets',type=Path,required=True); p.add_argument('--data',type=Path,required=True)
        p.add_argument('--spec',type=Path,default=Path('docs/A2_pilot_spec.md'))
        p.add_argument('--dataset',choices=DATASETS,required=True); p.add_argument('--output',type=Path,required=True)
        if name in ('survey','pre','check','throughput'): p.add_argument('--parent',choices=PARENTS,required=True)
        if name in ('pre','check','heal'): p.add_argument('--sets',type=Path,required=True)
        if name=='throughput': p.add_argument('--microbatch',type=int,default=1)
        if name=='heal':
            p.add_argument('--run',type=Path,required=True); p.add_argument('--pre-values',type=Path,required=True)
            p.add_argument('--preflight',type=Path,required=True)
            p.add_argument('--microbatch',type=int,default=1); p.add_argument('--chunk',type=int,default=128)
            p.add_argument('--estimated-seconds',type=float,required=True)
    sets=sub.add_parser('sets'); sets.add_argument('--surveys',type=Path,required=True)
    sets.add_argument('--identity-tests',type=Path,required=True); sets.add_argument('--output',type=Path,required=True)
    rep=sub.add_parser('report'); rep.add_argument('--results',type=Path,required=True)
    rep.add_argument('--parents',type=Path,required=True); rep.add_argument('--output',type=Path,required=True)
    est=sub.add_parser('estimate'); est.add_argument('--costs',type=Path,required=True); est.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise ValueError('Choose a fresh stage output; no overwrite/resume')
    if args.action=='plan':
        chosen=selected_lrs(read(args.results)) if args.results else {}
        write_json(args.output,dict(runs=[r.record() for r in matrix(chosen)],launch=False)); return
    if args.action=='estimate': write_json(args.output,estimate_workload(read(args.costs))); return
    if args.action=='report': write_report(args.output,read(args.results),read(args.parents)); return
    if args.action=='sets':
        surveys=read(args.surveys); tests=read(args.identity_tests); result={}; selections={}
        if isinstance(surveys,list):surveys={f'{p["dataset"]}/{p["parent"]}':p for p in surveys}
        for ds in DATASETS:
            usable=[p for p in PARENTS if tests[f'{ds}/{p}']['T1'] and (p in ('P300','P1T') or tests[f'{ds}/{p}']['T3'])]
            selections[ds]=choose_s2({p:surveys[f'{ds}/{p}']['group_bi'] for p in usable},usable)
            result[ds]=dict(FIXED_SETS,S2=selections[ds]['groups'])
        write_json(args.output,dict(sets=result,selection=selections,source_sha256=sha256(args.surveys),tests_sha256=sha256(args.identity_tests))); return
    if args.action=='prepare-secondary':
        from transformers import AutoTokenizer
        from training_data import load_text
        ds,records=load_text(args.source)
        prepare_secondary(ds,AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True),args.output,
            provenance=dict(source=read(args.source_provenance),files=records))
        return
    if args.action=='prepare':
        from transformers import AutoTokenizer
        contract=read(args.pool_contract)
        if (contract.get('repo')!='nht10/cx_sampled' or contract.get('subset')!='qwen3_0.6b_base_en_30B'
            or contract.get('revision')!=args.dataset_revision):
            raise ValueError('Dataset contract/revision mismatch')
        pool,identity=load_pool(args.pool)
        if identity['manifest_sha256']!=contract['manifest_sha256']:
            raise ValueError('Pool manifest differs from pinned Hub contract')
        if len(args.dataset_revision)!=40: raise ValueError('Pinned HF dataset commit required')
        tp=read(args.tokenizer_provenance)
        for name,path in (('qwen',args.qwen_tokenizer),('gpt_neox',args.tokenizer)):
            if len(tp[name].get('revision',''))!=40 or not {'tokenizer.json','tokenizer_config.json'}<=tp[name]['files'].keys():
                raise ValueError('Pinned tokenizer revision and file checksums required')
            for file,digest in tp[name]['files'].items():
                if sha256(Path(path)/file)!=digest: raise ValueError('Tokenizer provenance mismatch')
        prepare(pool['train'],pool['validation'],
            AutoTokenizer.from_pretrained(args.qwen_tokenizer,local_files_only=True),
            AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True),args.output,
            provenance=dict(identity,dataset='nht10/cx_sampled',subset='qwen3_0.6b_base_en_30B',
                revision=args.dataset_revision,tokenizers=tp,pool_contract_sha256=sha256(args.pool_contract),spec_sha256=sha256(args.spec))); return
    # Initializes process groups only for explicit GPU/CPU model stages.
    from accelerate import PartialState
    state=PartialState(); device=state.device; rank,world=rank_world()
    assets=read(args.assets); prov=provenance(args)
    data_manifest=read(args.data/'manifest.json')
    if data_manifest['provenance']['spec_sha256']!=prov['spec_sha256']:
        raise ValueError('Data prepared under a different preregistration')
    if data_manifest['streams']['heal']['length']!=4096:
        raise ValueError('Production A2 requires 4096 tokens per sequence')
    evaluation=eval_sets(args.data)
    secondary_status='not supplied; acquisition or explicit skip reason required before pilot'
    if args.secondary_map:
        item=read(args.secondary_map)[args.dataset]
        if item.get('path'):
            evaluation['secondary']=PackedRows(item['path'],'final')
            if sha256(Path(item['path'])/'manifest.json')!=item['manifest_sha256']:
                raise ValueError('Secondary manifest identity mismatch')
            secondary_status='may overlap pretraining; no decision attached'
        elif not item.get('skip_reason'):raise ValueError('Secondary set needs a path or explicit skip reason')
        else:secondary_status=item['skip_reason']
        prov['secondary_map_sha256']=sha256(args.secondary_map)
    if args.action=='survey':
        model=load_parent(assets,args.dataset,args.parent,device)
        values=survey(model,evaluation['small'])
        values.update(provenance=prov,dataset=args.dataset,parent=args.parent,secondary_status=secondary_status,
            final=score(model,evaluation['final']),tuning=score(model,evaluation['tuning']))
        if 'secondary' in evaluation:values['secondary']=score(model,evaluation['secondary'])
        del model
        teacher=load_parent(assets,args.dataset,args.parent,device,teacher=True)
        values['teacher_bf16_small']=score(teacher,evaluation['small'])
        del teacher
        other=load_parent(assets,args.dataset,args.parent,device,theta=100000)
        values['other_theta_small']=score(other,evaluation['small'])
        del other
        values['T1_precision_passed']=values['precision_gap']<.002
        values['diagnostic_flags']=dict(other_theta_not_worse=values['other_theta_small']['loss']<=values['small']['loss'],
            position_loss_not_decreasing=not all(a>=b for a,b in zip(values['small']['bucket_losses'],values['small']['bucket_losses'][1:])))
        if rank==0:write_json(args.output,values)
        return
    if args.action=='throughput':
        from .check import throughput
        model=load_parent(assets,args.dataset,args.parent,device);prune(model,layers_for(FIXED_SETS['S1']))
        teacher=load_parent(assets,args.dataset,args.parent,device,teacher=True)
        result=throughput(model,teacher,PackedRows(args.data,'heal',count=12*64),
            args.output.with_suffix('.work'),microbatch=args.microbatch)
        if rank==0:write_json(args.output,dict(result,provenance=prov))
        return
    sets=read(args.sets)['sets'][args.dataset]
    if args.action=='check':
        from .check import numerical_checks
        model=load_parent(assets,args.dataset,args.parent,device)
        result=numerical_checks(model,evaluation['small'],sets,args.output.with_suffix('.work'))
        if rank==0:write_json(args.output,dict(result,provenance=prov))
        if not result['passed']:raise ValueError('Native numerical check failed')
        return
    if args.action=='pre':
        from .model import bypass
        model=load_parent(assets,args.dataset,args.parent,device)
        baseline={k:score(model,rows) for k,rows in evaluation.items()}
        values={}
        for name,groups in sets.items():
            with bypass(model,layers_for(groups)):
                values[name]={k:score(model,rows) for k,rows in evaluation.items()}
        if rank==0:write_json(args.output,dict(provenance=prov,parent=baseline,sets=values,sets_sha256=sha256(args.sets)))
        return
    gate=read(args.preflight)
    if not args.secondary_map:raise ValueError('Record secondary acquisition/skip decisions before healing')
    if gate.get('status')!='passed' or gate.get('provenance')!=prov or gate.get('sets_sha256')!=sha256(args.sets):
        raise ValueError('Real-parent/GPU preflight required for this exact spec/data/assets/code')
    required=('T4','T5','T6','T7','T8','T10_formula','T11','T12','T13')
    if not all(gate.get('tests',{}).get(t,{}).get('passed') is True for t in required):
        raise ValueError('Missing pre-heal correctness evidence')
    raw=read(args.run); run=Run(**{k:raw[k] for k in ('dataset','arm','set','budget','lr','suffix')})
    if run.dataset!=args.dataset: raise ValueError('Run dataset differs')
    pre_values=read(args.pre_values)
    if (pre_values.get('provenance')!=prov or pre_values.get('sets_sha256')!=sha256(args.sets)
        or pre_values.get('parent')!=run.parent or pre_values.get('teacher')!=run.teacher):
        raise ValueError('Unhealed baseline provenance differs')
    for parent in {run.parent,run.teacher}:
        if not gate.get('parents',{}).get(f'{run.dataset}/{parent}',{}).get('passed'):
            raise ValueError('Affected parent failed T1/T2/T3; other valid parents remain eligible')
    count=run.steps*64
    order=training_order(count,run.suffix)
    train_rows=PackedRows(args.data,'heal',count=count,order=order)
    student=load_parent(assets,run.dataset,run.parent,device); prune(student,layers_for(sets[run.set]))
    teacher=load_parent(assets,run.dataset,run.teacher,device,teacher=True)
    from .train import heal
    heal(student,teacher,train_rows,evaluation,args.output,run=run,microbatch=args.microbatch,
        chunk=args.chunk,estimated_seconds=args.estimated_seconds,provenance=prov,pre_values=pre_values,rerun_of=raw.get('rerun_of'))

if __name__=='__main__': main()
