# Deep-KV F/G: deep-key precision note and proposed B200 smoke — 2026-09-28

**Original status: review notes only.** The user subsequently authorized F/G
training. The precision cast is now implemented and locally tested; the proposed
D/F/G smoke passed on B200 before the long F/G queue started. See the
execution record below.
The notes below describe the pre-adjustment review. Written
after reviewing the F/G implementation (`f6ce7e5`, `a8f4fd6`) against
`deep_route_kl_variant.md`. That review found no implementation bug; all 59
local CPU tests pass. Section 1 is an optional precision refinement.
Section 2 proposes the GPU check still needed before the long F/G runs.

## 1. Deep keys enter the routing KL in FP32; predicted keys and query in BF16

### What happens

Under BF16 autocast, the tensors passed to `DeepKV.routing_alignment` have
these dtypes (measured on the test model with CPU BF16 autocast):

| Tensor | Source | dtype |
|---|---|---|
| Shallow query (detached) | `rotated_query.to(dtype)`, `deep_kv/model.py:96` | bfloat16 |
| Predicted key | `rotated_k.to(dtype)`, `deep_kv/model.py:96` | bfloat16 |
| Predicted value | auxiliary `v` | bfloat16 |
| **Deep key** | `kr` from `apply_rotary_pos_emb`, `deep_kv/model.py:141`, used at `:183` | **float32** |
| Deep value | block-21 `v_proj` output | bfloat16 |

The predicted side is correct: it is exactly what the live auxiliary attention
uses. The deep key is not cast the same way. Qwen3's `k_norm` multiplies a
float32 weight by the BF16 projection output, and RoPE keeps float32, so `kr`
stays float32. `routing_alignment` then upcasts everything to float32, so the
deep key carries more precision than the predicted key.

Spec §3 asks for keys "in the exact form used by attention" for both sides.
The deep reference is therefore slightly more precise than the form the
predicted keys can represent.

### Size of the effect

Measured KL between the routing produced by float32 keys and the same keys
rounded to BF16: one 2048-token context, 16 heads, `head_dim=128`, strict-past
mask, BF16 query.

| q/k scale (grows if learned q/k norm weights grow) | KL(FP32 keys ‖ BF16-rounded keys) |
|---|---|
| 1 | 1.4e-06 |
| 3 | 4.5e-05 |
| 6 | 1.8e-04 |

Even if P_K reproduced the deep keys perfectly (to BF16 rounding), the route
loss could not go below roughly this level. That floor is negligible compared
with route-loss values of practical size. It does not affect the forward pass,
the LM loss, Arm D, or any A–E result.

### Optional fix

Cast the deep key to the same dtype as the other attention inputs before
using it as the target, for example in `hidden_states`:

```python
target = (kr.detach().to(v.dtype), v.detach())
```

This adjustment was applied before any F/G run. It only changes F/G loss
numerics; the earlier A-E results and forward computation are unchanged.

## 2. Proposed B200 smoke for F/G

### Why

F/G add FP32 routing computations at the consumer layer. Per 128-query chunk
these are two score matrices over all 2048 sources, softmax/log-softmax, and
for G two message products, all recomputed in backward. CPU tests and the
eight-process Gloo resume check verified correctness, but not GPU memory,
speed, or NCCL behavior at the production shape. Arm D ran at about
3.3–3.5 s/update (≈2h25m for the 2,500-update cutoff); F/G cost is unknown.

### Design

Follow the verified pattern of `scripts/smoke_deep_kv_b200.sh`, with these
changes:

- **Arms:** `D F G`. D gives a same-node, same-subset reference for speed and
  memory; F/G are the new losses.
- **Model and batch unchanged:** full 28-layer Qwen3, context 2048, BF16,
  eight GPUs, microbatch 16, accumulation 4 (1,048,576 tokens/update).
- **Data:** a fresh text subset (for example 30,000 train / 2,000 eval
  documents), preprocessed by the normal HF pipeline. It must pack to at
  least `max_steps × 512` contexts (6,144 for 12 steps).
- **Short schedule:** `max_steps=12`, `warmup_steps=1`, cutoff 10, then resume
  each arm to 12. `logging_steps=1`, `save_steps=5`, `eval_steps=5`,
  `eval_rows=129` (uneven across eight ranks), `monitor_rows=16`.
- **Memory metrics:** set `skip_memory_metrics=false` so Trainer records GPU
  memory in the train metrics. Also sample `nvidia-smi` once during an F/G arm.
- **Queue:** `python -m deep_kv make-jobs --arms D F G --stop-after 10 ...`,
  generated on B200, then `run_experiments.py` into a fresh root. Then rerun
  each arm's command with `--stop_after 12` to exercise resume.

Operational prerequisites are unchanged from earlier launches: fresh GPU
ownership inspection, only explicitly authorized workload stops, repository
Accelerate config copied to the actual default path and checked with
`accelerate env`, and `commands.sh` returned to `#0` afterwards.

### Pass criteria

1. All three arms exit zero at step 10 and after resuming to step 12; no OOM
   or NCCL error.
2. Every logged `step_loss_route` is finite and positive. F has
   `eval_loss_msg == 0`; G has a finite positive `eval_loss_msg`.
3. `eval_route_queries == 129 × 2047` for F/G, and
   `eval_loss == eval_lm_loss + 0.3·eval_loss_route (+ 0.3·eval_loss_msg for G)`.
4. Peak GPU memory for F/G leaves clear headroom below device capacity.
5. Seconds per update for F/G relative to D, averaged over steps 3–10 to skip
   startup, are recorded. Use this ratio to estimate the 2,500-update runtime
   (D ≈ 2h25m).

The smoke's losses are not scientific evidence. Its purpose is to establish
that F/G run stably at the production shape and to price the long runs.

## Launch implementation

`scripts/smoke_deep_kv_functional.py` runs the smoke under the existing
`train_then_burn` supervisor. Smoke-only callbacks measure synchronized steps
3-10 separately from evaluation/checkpoint overhead and record peak allocated
and reserved memory on every rank, plus an eight-GPU nvidia-smi sample. The gate
requires over 8 GiB headroom on every GPU. Step-10 artifacts are preserved before
resuming; final validation/reporting is rerun at step 12, including eight RNG
states and optimizer/scheduler files. A finite KL within numerical tolerance
of zero is allowed; strict positivity is not a universal correctness condition.
The precision measurements above illustrate rounding discrepancies, not a proven
lower bound on the best routing KL attainable by learned predicted keys.

Only a successful smoke receipt unlocks fresh F/G production training, using
the unchanged 28600-step schedule / 2500 cutoff / 1430 warmup. On smoke or
training failure, the queue stops and the verified supervisor restores burns
after owned-worker cleanup and free-GPU checks.

## Execution record

Launch 26e8e7f passed this gate on B200 at 2026-09-28 20:07:48 UTC.
All D/F/G initial and resumed runs exited successfully. Query counts and
weighted objectives matched; every rank passed the memory-headroom gate.
Measured synchronized seconds/update: D 4.278, F 4.918, G 5.262; ratios
F/D 1.150 and G/D 1.230. Peak reserved memory stayed below 24 GiB per GPU.
Production F then started fresh, followed by queued G, each stopping at 2500
updates of the unchanged full schedule. See CURRENT_TASK.md for exact paths,
receipt hash, verification timestamp and automatic final burn supervision.
