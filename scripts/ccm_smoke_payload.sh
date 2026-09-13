#!/usr/bin/env bash
# Foreground child of the reviewed handoff. Does not manage any GPU burns.
set -euo pipefail
TASK_DATA=/mnt/local/_data/deep-llms_th2/ccm/smoke_20260913_a01
TASK_OUTPUT=/mnt/local/_outputs/deep-llms_th2/ccm_smoke_20260913_a01
TASK_ASSETS=/mnt/local/_models/deep-llms_th2/Qwen3-0.6B-Base-da87bfb608c14b7cf20ba1ce41287e8de496c0cd
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
TASK_TEST_PIDS=()
for TASK_GPU in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES="$TASK_GPU" python tests/cuda_pilot_smoke.py --pipeline > "$TASK_OUTPUT/pipeline_gpu${TASK_GPU}.log" 2>&1 &
  TASK_TEST_PIDS+=("$!")
done
TASK_TEST_FAILURE=0
for TASK_PID in "${TASK_TEST_PIDS[@]}"; do
  if ! wait "$TASK_PID"; then TASK_TEST_FAILURE=1; fi
done
test "$TASK_TEST_FAILURE" -eq 0
sleep 3
python scripts/gpu_status.py --gpus 0 1 2 3 4 5 6 7 --require-free
python -m torch.distributed.run --standalone --nproc_per_node=8 --module ccm train \
  --data "$TASK_DATA/corpus" --output "$TASK_OUTPUT/common" --phase common --arm base \
  --model-config "$TASK_DATA/model_config" --device cuda --engineering \
  --activation-checkpointing --microbatch-segments 8 --loss-chunk 1024 --save-every 8 --log-every 1
python -m torch.distributed.run --standalone --nproc_per_node=8 tests/real_data_performance_smoke.py \
  --data "$TASK_DATA/corpus" --vocabulary "$TASK_DATA/vocabulary.npz" \
  --model-config "$TASK_ASSETS" --asset-manifest resources/qwen3_base_assets.json \
  --output "$TASK_OUTPUT/performance" --steps 4 --microbatch-segments 8 --loss-chunk 1024
python - "$TASK_DATA" "$TASK_OUTPUT" <<'PY'
import json, math, sys
from pathlib import Path
from ccm.contracts import read_json, write_json, file_hash
from ccm.data import Corpus
from ccm.runtime import checkpoint_meta
data,out=map(Path,sys.argv[1:])
corpus=Corpus(data/'corpus')
complete=read_json(out/'common/complete.json')
assert complete['success'] and complete['step']==8 and complete['input_tokens']==2097152
ck=checkpoint_meta(out/'common/checkpoint-8')
assert ck['corpus_hash']==corpus.meta['manifest_hash'] and ck['step']==8 and ck['engineering']
assert file_hash(out/'common/checkpoint-8/optimizer.pt')==ck['optimizer_sha256']
rows=[json.loads(x) for x in (out/'common/train.jsonl').read_text().splitlines()]
assert len(rows)==8 and all(math.isfinite(r['nll']) and math.isfinite(r['grad_norm']) for r in rows)
perf=read_json(out/'performance/complete.json')
assert perf['success'] and perf['world_size']==8 and len(perf['cases'])==5
assert all(r['success'] and r['post_warmup_input_tokens_per_second']>0 for r in perf['cases'])
write_json(out/'payload_verified.json',dict(success=True,corpus_hash=corpus.meta['manifest_hash'],
           common_steps=8,common_input_tokens=2097152,model_sha256=ck['model_sha256'],performance_cases=5))
print('CCM_B200_PAYLOAD_VERIFIED',flush=True)
PY
