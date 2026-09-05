#1 +60+a
#th2-readonly-tiered-groupreduce-eval-finetune-status-20260905-a02
set -euo pipefail
date -u
TASK_BASE=/mnt/local/_outputs/@PROJECT@
TASK_RUN=tiered_c512_groupreduce_lb_10k_20260905_a01
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
/mnt/local/conda-py311/envs/eval/bin/python3.11 - "$TASK_BASE" "$TASK_RUN" <<'PY'
from pathlib import Path
import json,math,subprocess,sys
base=Path(sys.argv[1]); run=sys.argv[2]
for arm in ('tiered_ranklift_lb_t4_c512','groupreduce_matched_lb_t4'):
    checkpoint=base/arm/'checkpoint-10000'
    for name in ('eval_ppl.json','eval_benchmarks.json'):
        p=checkpoint/name
        if p.is_file():
            data=json.loads(p.read_text())
            print('EVAL_RESULT',arm,name,'entries',len(data))
        else: print('EVAL_RESULT_MISSING',arm,name)
ft=base/f'finetune_{run}'
count=0
for task in ('hellaswag','arc_easy','xnli'):
    for arm in ('tiered_ranklift_c512_s10000','groupreduce_lb_t4_s10000'):
        complete=[];pending=[]
        for seed in (42,123,456):
            stem=f'{task}_{arm}_seed{seed}';p=ft/(stem+'.json')
            if p.is_file():
                r=json.loads(p.read_text())
                valid=(r.get('task')==task and r.get('seed')==seed and r.get('epochs')==3 and bool(r.get('eval_results')) and all(math.isfinite(float(v['acc'])) for v in r['eval_results'].values()))
                print('FINETUNE_RESULT',stem,'valid',valid,'train_seconds',r.get('train_time_s'))
                if valid: complete.append(seed);count+=1
            else:
                pending.append(seed)
                log=ft/(stem+'.log')
                if log.is_file():
                    with log.open('rb') as f:
                        f.seek(max(0,log.stat().st_size-2500)); text=f.read().decode(errors='replace')
                    print('PENDING_LOG',stem,'mtime',log.stat().st_mtime,'tail',text[-1100:])
        print('FINETUNE_COUNTS',task,arm,'completed',complete,'pending',pending)
print('TOTAL_VALID_FINETUNE_RESULTS',count,'/18')
marker=base/f'eval_finetune_{run}.complete'
print('COMPLETION_MARKER',marker.read_text() if marker.is_file() else 'ABSENT')
for p in Path('/proc').iterdir():
    if not p.name.isdigit() or int(p.name)<=1: continue
    try:
        args=(p/'cmdline').read_bytes().split(b'\0')
        if any(a.endswith((b'finetune/train.py',b'finetune/run_all.py',b'eval_checkpoint.py',b'eval_parallel.py',b'llm_pretrain_burn.py',b'run_tiered_groupreduce_eval_finetune_burn.sh')) for a in args):
            print('ACTIVE_PROCESS',p.name,b' '.join(args).decode(errors='replace'))
    except (OSError,ProcessLookupError): pass
PY
tail -45 "$TASK_BASE/logs/eval_finetune_${TASK_RUN}.log"
for TASK_BURN_LOG in "$TASK_BASE/logs/burn_final_${TASK_RUN}.log" "$TASK_BASE/logs/burn_final_${TASK_RUN}_recovery.log"; do
    if [[ -s "$TASK_BURN_LOG" ]]; then
        echo "$TASK_BURN_LOG"
        tail -5 "$TASK_BURN_LOG"
    fi
done
echo 'TH2 READONLY PAIRED EVAL FINETUNE STATUS COMPLETE'
