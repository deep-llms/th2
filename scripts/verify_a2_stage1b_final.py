"""Read-only recomputation of final FWE artifacts after native terminal handoff.

No model forwards, GPU acquisition, process changes or output-file writes. Run
native verification with the frozen native runtime; batch mode uses the pinned
CPU harness runtime and its canonical task objects.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import socket
import subprocess

from efficient_llm.diagnostic_data import sha256
from eval.a2_run import lock
from eval.a2_request_bridge import digest
from eval.a2_stage1b_final_plan import validate_plan,admit_final_regression,PREPARED_SHA
from eval.a2_stage1b_stats import PART_MODELS,paired_report
from eval.a2_stage1_stats import aligned_accuracy_arrays
from eval.a2_suite import collect_files
from eval.downstream_suite import COUNTS,PRIMARY
from scripts.a2_remaining import read_locked


def read(path):
    return json.loads(Path(path).read_text())


def verify_gate(gate,identity,prepared):
    if (gate['passed'] is not True or gate['checkpoint_identity_sha256']!=digest(identity)
            or gate['prepared_sha256']!=digest(prepared) or gate['production_batch_size']!=2
            or gate['precision']!='fp32_parameters_bf16_cuda_autocast'):
        raise ValueError('Wrong actual-model gate identity')
    a,b=gate['batch1_token_predictions'],gate['batch2_token_predictions']
    selected=gate['selected_request_indices']
    if len(a)!=len(b) or len(a)!=len(selected) or len(set(selected))!=len(selected):
        raise ValueError('Incomplete representative token comparison')
    total=0;mismatch=0
    for i,x,y in zip(selected,a,b):
        n=len(prepared['requests'][i]['continuation'])
        if len(x)!=n or len(y)!=n:raise ValueError('Wrong compared-token count')
        total+=n;mismatch+=sum(v!=w for v,w in zip(x,y))
    if total<=0:raise ValueError('No compared tokens')
    original=gate['original_gate'];manual=original['manual_max_nats_per_token']
    batch=max(original['batch_max_nats_per_token'],original['production_max_nats_per_token'])
    expected=dict(manual_max_nats_per_token=manual,batch_max_nats_per_token=batch,
        greedy_token_mismatches=mismatch,compared_tokens=total,greedy_token_mismatch_rate=mismatch/total)
    if any(gate[k]!=v for k,v in expected.items()):raise ValueError('Recorded gate disagrees with token evidence')
    if not (0<=manual<=2e-5 and 0<=batch<=.1 and mismatch/total<=.001):
        raise ValueError('Recomputed revised numerical gate failed')
    if gate['thresholds']!=dict(manual_nats_per_token=2e-5,batch_nats_per_token=.1,greedy_token_mismatch_rate=.001):
        raise ValueError('Gate thresholds changed')
    if any(not 0<=v<=2e-5 for v in gate['token_diagnostic_ll_errors'].values()):
        raise ValueError('Argmax diagnostic did not reproduce scorer')
    return expected


def verify_terminal(root,source_sha):
    root=Path(root).resolve();manifest=read(root/'source-manifest.json')
    if manifest['code_commit']!=source_sha:raise ValueError('Wrong execution commit')
    for rel,value in manifest['files'].items():
        p=(root/'source'/rel).resolve()
        if not p.is_relative_to(root/'source') or sha256(p)!=value:raise ValueError('Frozen source changed')
    supervisor=read(root/'supervised/supervisor.json')
    terminal=read(root/'supervised/run/complete.json');burn=read(root/'supervised/burn-verified.json')
    if (supervisor.get('training_status')!='ok' or supervisor.get('training_returncode')!=0
            or not supervisor.get('finished_at') or len(supervisor.get('free_after_training',[]))!=8
            or any(g['pids'] for g in supervisor['free_after_training'])
            or terminal.get('status')!='ok' or len(terminal.get('jobs',[]))!=1
            or burn.get('collective_progress_verified') is not True):
        raise ValueError('Native successful terminal handoff missing')
    job=terminal['jobs'][0]
    if job.get('name')!='stage1b-downstream-final' or job.get('status')!='ok' or job.get('returncode')!=0:
        raise ValueError('Wrong or unsuccessful native job')
    base=root/'supervised/run';bound=set()
    for artifact in job['artifacts']:
        p=(base/artifact['path']).resolve()
        if not p.is_relative_to(base) or sha256(p)!=artifact['sha256']:raise ValueError('Native terminal artifact changed')
        bound.add(str(p.relative_to(base)))
    if not {'downstream/complete.json','downstream/results.json','downstream/batch_decision.json','downstream/paired_accuracy.json'}<=bound:
        raise ValueError('Native completion does not bind all final outputs')
    return manifest,supervisor,terminal,burn


def batch_recompute(root):
    from lm_eval.tasks import get_task_dict
    from eval.benchmark_data import local_dataset_paths,task_configs,benchmark_task_manager
    from eval.a2_stage1b_batch_report import compare
    root=Path(root);plan=read(root/'plan.json');out=root/'supervised/run/downstream'
    decision=read(out/'batch_decision.json');inputs=decision['inputs']
    prepared=read_locked(inputs['prepared'])
    if inputs['prepared']['sha256']!=PREPARED_SHA:raise ValueError('Changed canonical requests')
    replays={k:read_locked(inputs['replays'][k]) for k in ('1','2')}
    # Independently verify aggregate task accuracies from every binary item.
    arrays=aligned_accuracy_arrays(replays,models=('1','2'))
    diffs={t:float(arrays[t]['2'].mean()-arrays[t]['1'].mean()) for t in COUNTS}
    mc=sum(v for t,v in diffs.items() if t!='lambada_openai')/8
    accept=abs(mc)<=.001+1e-15 and abs(diffs['lambada_openai'])<=.001+1e-15
    if (decision['production_batch_size']!=(2 if accept else 1)
            or decision['batch2_accepted']!=accept):raise ValueError('Batch decision disagrees with item accuracy')
    for task in COUNTS:
        for batch in ('1','2'):
            if abs(decision['per_task'][task]['batch'+batch]-float(arrays[task][batch].mean()))>1e-12:
                raise ValueError('Reported per-task accuracy disagrees with items')
    paths=local_dataset_paths(plan['dataset_root'],plan['manifest']['path'])
    tasks=get_task_dict(task_configs(list(COUNTS),paths),task_manager=benchmark_task_manager())
    if set(tasks)!=set(COUNTS):raise ValueError('Canonical task selection changed')
    actual=compare(inputs,{t:tasks[t].doc_to_choice for t in COUNTS if t!='lambada_openai'})
    rows=actual['likelihood'].pop('rows')
    row_lock=decision['likelihood']['per_request']
    if rows!=read_locked(row_lock):raise ValueError('Per-request LL/padding diagnostics differ')
    actual['likelihood']['per_request']=row_lock
    if actual!=decision or actual['hard_batch_passed'] is not True:
        raise ValueError('Full batch diagnostics/decision failed independent recomputation')
    return dict(status='batch_verified',production_batch_size=decision['production_batch_size'],
        requests=78259,decision=lock(out/'batch_decision.json'),per_request=row_lock,
        prediction_flips={t:v['flips'] for t,v in decision['prediction_flips'].items()},
        max_abs_ll=decision['likelihood']['summary']['max_abs'],mean_abs_ll=decision['likelihood']['summary']['mean_abs'])


def verify(root,source_sha):
    root=Path(root).resolve();manifest,sv,terminal,burn=verify_terminal(root,source_sha)
    plan=read(root/'plan.json');ids=validate_plan(plan)
    admission=admit_final_regression(plan['regression'],root/'source',plan['policy'])
    if admission!=read(root/'regression-admission.json'):raise ValueError('Regression admission changed')
    if set(ids)!=set(PART_MODELS['final']):raise ValueError('Missing final model')
    out=root/'supervised/run/downstream';done=read(out/'complete.json')
    result=read_locked(done['results']);decision=read_locked(done['batch_decision'])
    if (done['status']!='ok' or done['models']!=6 or done['time_policy']!=plan['time_policy']
            or result['status']!='ok' or result['diagnostic'] is not False
            or set(done['scores'])!=set(ids) or set(done['replays'])!=set(ids)
            or set(done['revised_gates'])!=set(ids)):
        raise ValueError('Incomplete final screen')
    prepared=read_locked(done['prepared']);batch=decision['production_batch_size']
    if (done['prepared']['sha256']!=PREPARED_SHA or len(prepared['requests'])!=78259
            or batch not in (1,2) or done['production_batch_size']!=batch
            or done['scores']['P300-parent']!=decision['inputs']['scores'][str(batch)]
            or done['replays']['P300-parent']!=decision['inputs']['replays'][str(batch)]):
        raise ValueError('Canonical requests or production parent reuse changed')
    gates={n:verify_gate(read_locked(item),ids[n],prepared) for n,item in done['revised_gates'].items()}
    if any(read_locked(item)['batch_size']!=batch for item in done['scores'].values()):
        raise ValueError('Mixed production batch sizes')
    actual=collect_files(ids,done['prepared'],done['scores'],done['replays'])
    actual.update(production_batch_size=batch,batch_decision=done['batch_decision'],revised_gates=done['revised_gates'],
                  policy=plan['policy'],failed_gate=plan['failed_gate'])
    if actual!=result:raise ValueError('Final results differ from native score/replay artifacts')
    paired=paired_report({n:read_locked(item) for n,item in done['replays'].items()},'final')
    if paired!=read_locked(done['additional_results']['paired_accuracy']):raise ValueError('Paired intervals differ')
    command=[plan['eval_python'],'-m','scripts.verify_a2_stage1b_final','--root',str(root),'--mode','batch']
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1',
         'TRANSFORMERS_OFFLINE':'1','PYTHONPATH':str(root/'source'),'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1'}
    child=subprocess.run(command,cwd=root/'source',env=env,check=True,capture_output=True,text=True)
    lines=[x[len('BATCH_RECOMPUTED '):] for x in child.stdout.splitlines() if x.startswith('BATCH_RECOMPUTED ')]
    if len(lines)!=1:raise ValueError('Missing independent full batch recomputation')
    comparison=json.loads(lines[0])
    if comparison['decision']!=done['batch_decision']:raise ValueError('Wrong independently checked decision')
    return dict(status='replay_verified',at=datetime.now(timezone.utc).isoformat(),source=source_sha,
        source_files=len(manifest['files']),models=6,requests=78259,tasks=9,production_batch_size=batch,
        scores=done['scores'],replays=done['replays'],results=done['results'],
        paired_accuracy=done['additional_results']['paired_accuracy'],resamples=10000,seed=0,
        batch_comparison=comparison,recomputed_gates=gates,
        live_burn_handoff_requires_separate_fresh_receipt=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--source-sha')
    p.add_argument('--mode',choices=('all','batch'),default='all');a=p.parse_args()
    if socket.gethostname()!='thiennh-p6-6y3o-worker-0':raise ValueError('Wrong verification host')
    if a.mode=='all':
        if not a.source_sha or len(a.source_sha)!=40:raise ValueError('Reviewed actual execution source required')
        print('INDEPENDENT_STAGE1B_FINAL_REPLAY_VERIFIED',json.dumps(verify(a.root,a.source_sha)),flush=True)
    else:print('BATCH_RECOMPUTED',json.dumps(batch_recompute(a.root)),flush=True)


if __name__=='__main__':main()
