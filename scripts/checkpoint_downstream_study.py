"""Compose the existing few-shot and supervised protocols for named checkpoints."""
import argparse
import json
from pathlib import Path
import shlex

from run_experiments import load_jobs
from scripts.finetune_study import make_jobs as finetune_jobs

GROUPS = {5:['xnli_en','xstorycloze_en','paws_en','piqa','arc_easy','winogrande'],
          10:['hellaswag'],25:['arc_challenge']}
FITS = {'pairs':('paws','nli'), 'extended':('stsb','boolq')}


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False)


def make(root, project, spec, data_roots, manifests):
    root,project=Path(root).resolve(),Path(project).resolve()
    run=root/'supervised/run'; items=[]
    checkpoints=list(spec)
    assert len(checkpoints)==3 and len({v['label'] for v in spec.values()})==3
    spec_path=root/'checkpoint-spec.json';write(spec_path,spec)
    for checkpoint,entry in spec.items():
        label=entry['label'];dest=f'numerics/{label}.json'
        items.append(dict(name='numerics-'+label,gpus=[0],timeout_seconds=900,
            argv=['{python}','-m','scripts.check_downstream_eval','numerics','--checkpoint',checkpoint,
                  '--expected-step',str(entry['step']),'--output','{run_dir}/'+dest],
            required_outputs=[dict(path=dest,json_equals=dict(status='passed',arm=entry['arm'],step=entry['step']))]))
    for phase,limit,batch in [('smoke',8,4),('full',None,8)]:
        commands=[]; outputs=[]
        for shot,gpus in [(5,[0,1,2]),(10,[3,4,5]),(25,[6,7])]:
            dest=f'{phase}-{shot}shot'
            argv=['{python}','-u','-m','eval.eval_parallel','--checkpoints',*checkpoints,
                '--bench-only','--english-only','--tasks',*GROUPS[shot],
                '--num-gpus',str(len(gpus)),'--gpu-ids',*map(str,gpus),
                '--batch-size',str(batch),'--num-fewshot',str(shot),'--seed','42',
                '--dataset-root',str(data_roots['pairs']),'--dataset-manifest',str(manifests['pairs']),
                '--output-dir','{run_dir}/'+dest,'--log','{run_dir}/'+dest+'-launcher.log']
            if limit:argv+=['--limit',str(limit)]
            commands.append(shlex.join(argv)+' >'+shlex.quote('{run_dir}/'+dest+'-pool.log')+' 2>&1')
            outputs.append(dict(path=dest+'/checkpoints.json'))
        shell='set -euo pipefail\n'
        for i,command in enumerate(commands):shell+=command+f' &\nTASK_PID_{i}=$!\n'
        shell+='TASK_STATUS=0\n'
        for i in range(len(commands)):shell+=f'wait "$TASK_PID_{i}" || TASK_STATUS=1\n'
        shell+='exit "$TASK_STATUS"'
        items.append(dict(name=phase+'-fewshot',gpus=list(range(8)),timeout_seconds=14400,
            argv=['bash','-c',shell],required_outputs=outputs))
        for shot in GROUPS:
            dest=f'{phase}-{shot}shot';out=dest+'-validation.json'
            argv=['{python}','-m','scripts.check_downstream_eval','validate','--directory','{run_dir}/'+dest,
                '--data-check',str(root/f'data-validation-{shot}.json'),'--count','3','--num-fewshot',str(shot),
                '--seed','42','--fewshot-audit',str(root/f'prompt-audit-{shot}.json'),
                '--checkpoint-spec',str(spec_path),'--output','{run_dir}/'+out]
            if limit:argv+=['--limit',str(limit)]
            items.append(dict(name='validate-'+dest,argv=argv,
                required_outputs=[dict(path=out,json_equals=dict(status='passed',limit=limit,num_fewshot=shot))]))
    # Separate study directories allow two checkpoints of arm A without aliasing
    # architecture names or changing the established LR-selection logic.
    for checkpoint,entry in spec.items():
        for group,tasks in FITS.items():
            label,arm=entry['label'],entry['arm']
            plan=root/'finetune-plans'/label/group
            finetune_jobs(plan,project,data_roots[group],manifests[group],{arm:checkpoint},
                          tasks=tasks,checkpoint_steps={arm:entry['step']})
            old=str(plan/'supervised/run');dest=f'finetune/{label}/{group}'
            new=str(run/dest)
            for path in plan.rglob('*.json'):
                # Replace only the generated output prefix; inputs are unchanged.
                path.write_text(path.read_text().replace(old,new))
            load_jobs(plan/'jobs.json')
            items.append(dict(name=f'finetune-{label}-{group}',gpus=list(range(8)),timeout_seconds=43200,
                argv=['{python}',str(project/'run_experiments.py'),'--config',str(plan/'jobs.json'),
                      '--project-dir',str(project),'--run-dir','{run_dir}/'+dest],
                required_outputs=[dict(path=dest+'/summary.json',json_equals=dict(status='passed'))]))
    items.append(dict(name='final-summary',argv=['{python}','-m','scripts.checkpoint_downstream_study','summary',
        '--root','{run_dir}','--spec',str(spec_path)],
        required_outputs=[dict(path='downstream-summary.json',json_equals=dict(status='passed'))]))
    write(root/'jobs.json',dict(jobs=items));load_jobs(root/'jobs.json')
    return items


def summary(root, spec):
    root=Path(root); report=dict(status='passed',checkpoints=spec,fewshot={},finetune={})
    for shot in GROUPS:
        value=json.loads((root/f'full-{shot}shot-validation.json').read_text())
        assert value['status']=='passed' and value['limit'] is None
        report['fewshot'][str(shot)]=value['arms']
    matched={}
    for checkpoint,entry in spec.items():
        label,arm=entry['label'],entry['arm']
        for group,tasks in FITS.items():
            directory=root/'finetune'/label/group
            validated=json.loads((directory/'summary.json').read_text())
            assert validated['status']=='passed' and validated['seeds']==[42,43,44]
            for task in tasks:
                report['finetune'].setdefault(task,{})[label]=validated['tasks'][task][arm]
                for seed in (42,43,44):
                    result=json.loads((directory/f'test/{task}/{arm}/seed-{seed}/result.json').read_text())
                    source=result['source']
                    assert source['checkpoint']==checkpoint and source['arm']==arm and source['step']==entry['step']
                    assert source['checkpoint_sha256']==entry['checkpoint_sha256']
                    data=result['data']
                    identity={k:data[k] for k in ('train_order_sha256','dataset_manifest_sha256','tokenizer_sha256')}
                    identity['test_order']=data['split_order_sha256']['test']
                    assert matched.setdefault(task,identity)==identity,'Unmatched downstream data'
    write(root/'downstream-summary.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['summary'])
    parser.add_argument('--root',required=True);parser.add_argument('--spec',required=True)
    args=parser.parse_args();summary(args.root,json.loads(Path(args.spec).read_text()))
