#1 +180+a
#th2-6y3o-a2-stage1b-fwe-failed-batch-replay-20261031-a01
set -euo pipefail
cd /mnt/local/_outputs/deep-llms_th2/6y3o-a2-stage1b-fwe-downstream-final-20261030-a01/source
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 PYTHONPATH="$PWD"
/mnt/local/conda-py311/envs/eval/bin/python3.11 -u - <<'PY'
import json,hashlib,socket
from pathlib import Path
from datetime import datetime,timezone
from eval.a2_stage1b_batch_report import compare
from eval.a2_stage1_stats import aligned_accuracy_arrays
from eval.benchmark_data import local_dataset_paths,task_configs,benchmark_task_manager
from eval.downstream_suite import COUNTS
from lm_eval.tasks import get_task_dict
from scripts.a2_remaining import read_locked
assert socket.gethostname()=='thiennh-p6-6y3o-worker-0'
root=Path.cwd().parent;out=root/'supervised/run/downstream'
read=lambda p:json.loads(p.read_text())
manifest=read(root/'source-manifest.json')
assert manifest['code_commit']=='f0c8154808bef431c76553a47fd9727d3ba8bd53'
for rel,sha in manifest['files'].items():assert hashlib.sha256((root/'source'/rel).read_bytes()).hexdigest()==sha,rel
plan=read(root/'plan.json');decision=read(out/'batch_decision.json');inputs=decision['inputs']
assert inputs['prepared']['sha256']=='85cf7f2119e35e81a79aea48f58080bdc9ba5371b27accbda0eac52aa6770bbf'
replays={k:read_locked(inputs['replays'][k]) for k in ('1','2')}
arrays=aligned_accuracy_arrays(replays,models=('1','2'))
for task in COUNTS:
 for k in ('1','2'):assert abs(float(arrays[task][k].mean())-decision['per_task'][task]['batch'+k])<1e-12
paths=local_dataset_paths(plan['dataset_root'],plan['manifest']['path'])
tasks=get_task_dict(task_configs(list(COUNTS),paths),task_manager=benchmark_task_manager())
assert set(tasks)==set(COUNTS)
actual=compare(inputs,{t:tasks[t].doc_to_choice for t in COUNTS if t!='lambada_openai'})
rows=actual['likelihood'].pop('rows');rowlock=decision['likelihood']['per_request'];assert rows==read_locked(rowlock)
actual['likelihood']['per_request']=rowlock;assert actual==decision
assert decision['hard_batch_passed'] is False and decision['batch2_accepted'] is True
exceed=[r for r in rows if abs(r['nats_per_token'])>.1];assert exceed
worst=max(rows,key=lambda r:abs(r['nats_per_token']))
prepared=read_locked(inputs['prepared']);req=prepared['requests'][worst['index']]
result=dict(status='failed_batch_independently_reproduced',at=datetime.now(timezone.utc).isoformat(),source=manifest['code_commit'],source_files=len(manifest['files']),requests=len(rows),decision_sha256=hashlib.sha256((out/'batch_decision.json').read_bytes()).hexdigest(),hard_batch_passed=False,accuracy_rule_batch2_accepted=True,full_max_nats_per_token=decision['full_max_nats_per_token'],full_greedy_token_mismatch_rate=decision['full_greedy_token_mismatch_rate'],mc_mean_difference_pp=decision['mc_mean_difference_pp'],lambada_difference_pp=decision['lambada_difference_pp'],above_point1_count=len(exceed),above_point1_rows=exceed,worst_request=dict(row=worst,context_tokens=len(req['context']),continuation_tokens=len(req['continuation'])),successful_final_replay=False)
print('FAILED_BATCH_INDEPENDENT_REPLAY',json.dumps(result),flush=True)
PY
