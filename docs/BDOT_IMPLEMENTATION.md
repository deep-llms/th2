# B-DOT Phase A implementation

Implements the Phase A subset and pilot recipe of [spec v1.4](BDOT_v2_spec.md). No dataset download,
remote deployment or research training is part of this implementation turn.

## What is reused

`train.py`, native HF Trainer/Accelerate, the packed collator, SDPA/optional FA4
adapter, periodic LM evaluation, optimizer/scheduler checkpoints, strict resume
receipts and `run_experiments.py` remain the training infrastructure. Plain dense
Qwen3 still uses the native model. The matched dense control adds only step FiLM.

## New code

- `bdot/model.py`: native Qwen3 stem/coda with a routed middle; Q/q_norm/O/MLP belong
  to modules, while attention-input norm/K/k_norm/V belong to steps. Supports
  fixed, learned and random routing, soft top-two, FiLM and step KV caches.
- `bdot/routing.py`: Phase A configuration, curricula and exact no-reuse tables.
- `bdot/trainer.py`: global router-loss denominator, one balance update per
  optimizer step, router metrics and ordinary DDP unused-parameter handling.
- `bdot/data.py`: shared packed-validation indices and a one-time exact
  train/validation document-overlap check. Legacy document-stream helpers remain
  available. Both use ordinary HF caches; there is no token-export step.

Other modes/windows/KV designs, STOP and re-entry fail rather than silently falling
back to Phase A. They are later phases, conditional on the pilot result.

## Training configurations

`configs/bdot/` contains native dense Qwen3, matched dense D, P1-A, P1-B, R, N-A and
N-B. All use the 12-layer Qwen3 dimensions and random initialization. D includes
FiLM, while `dense_native.json` is the unmodified reference for T1.

To prepare a sequential queue (this command does not launch anything):

```bash
python scripts/make_bdot_pilot.py \
  --tokenizer /absolute/local/Qwen3-tokenizer \
  --data /absolute/local/qwen3_0.6b_base_en_30B/train \
  --eval-data /absolute/local/qwen3_0.6b_base_en_30B/validation \
  --revision DATASET_COMMIT_HASH \
  --manifest /absolute/local/cx-packed-validation.json \
  --accelerate-config /absolute/local/accelerate.yaml \
  --world-size 8 --micro-batch 16 \
  --output /absolute/new/pilot-plan
```

For the validated B200 grouped implementation, add `--expert-backend grouped`;
the generator default remains `reference`. Keep this setting fixed across the
comparison and on resume.

The default queue contains D → P1-A → P1-B. R and the matching N follow selection
of the curriculum; they are not selected automatically. The generator supports
`--arms R N-A` or `--arms R N-B`. It records the rounded token budget and separate
model/routing seeds. Eight ranks × 16 sequences × accumulation 4 gives 512 × 1024
input tokens/update. The default 2.5B request rounds down to 4,768 updates
(2,499,805,184 tokens), with 96 warmup updates. LR and routing share that complete
schedule. The generator does not set `stop_after`; the pilot job rejects an
early-cutoff run. Tests/exploratory runs may still use the inherited cutoff.
Checkpoint retention is unlimited unless explicitly changed.

The generated `jobs.json` works with the existing runner. Device allocation,
runner authorization and availability must be established separately. `commands.sh`
remains inactive. Every job checks for fresh output and verifies the final step
and saved weights before declaring completion.

The local data needs a `text` column. All arms reuse the existing training and
validation splits, HF maps and seed-0 packed shuffle, with a sequential sampler.
Defaults preserve the native baseline's preprocessing: 128 workers and map batch
size 1,000. Keep these settings fixed; `--preprocessing-workers` is explicit.
FA4 is the generator default; `--backend sdpa` remains available.

The first arm creates the shared validation manifest automatically. It hashes
the raw text to detect exact document overlap, removes affected validation
sequences without changing training, and freezes the first 1,024 eligible
sequences for periodic evaluation and first 10,240 for final evaluation (or all
if fewer exist). Later arms reuse the manifest and caches. Changed source
fingerprints are rejected. The full manifest is embedded in checkpoint training
receipts, so standalone evaluation does not need the training corpus. Actual
pool disjointness is only established when this check runs on the real data.
Router regularizers never enter reported validation loss/PPL.

## Checkpoints and evaluation

Load routed checkpoints through `train.build_model()` or
`BDOTForCausalLM.from_pretrained()`. Their config embeds `bdot` options in the
Qwen3 config; an unmodified AutoModel loader alone does not reconstruct the router.
The project's training and evaluation entry points use the correct loader.

`eval/packed_ppl.py` loads B-DOT checkpoints and their frozen packed-validation
selection; `--eval_data_dir` points to the existing validation split.
Legacy document-stream checkpoints still take their original corpus path and
document manifest. `--routing_override fixed` or `random` performs the specified
route intervention; `--routing_seed` selects a repeatable evaluation stream.
Off-default share and repeated-module fraction are included in the result.
Reuse means the fraction of tokens whose path repeats any module, as in the
spec. Training logs also contain router entropy, per-step off-default shares
and per-module counts. These fields explicitly describe the last optimizer
update; they are not averages over the entire logging interval.

### Training and validation metrics

The scientific-review behavior, including the v1.4 curriculum changes:

- Forced single-candidate choices contribute zero z-loss. The denominator is
  still all input-token × routing-step decisions across accumulation and ranks.
  Entirely forced routers are bypassed; home-only router parameters have no
  gradient and receive no AdamW decay or momentum update.
- Periodic `eval_loss`, `eval_curriculum_loss`, and `eval_spike_loss` measure the
  current curriculum with exploration disabled. `eval_target_loss` also records
  the target hard policy; identical policies share one scoring pass.
  Every widening/soft-to-hard transition triggers validation on the same weights
  under both phases, including when the transition is between periodic evaluations.
  The transition jump is logged and the spike minimum resets to the new loss.
  Trainer's checkpointed log history preserves that minimum on resume.
  Final validation and standalone route studies use only the target policy.
- `objective_loss` is HF's logged optimization objective. `lm_loss` is the
  token-weighted LM aggregate since the previous training log, with `lm_tokens`
  and `input_tokens` recorded. `router_z_loss` uses the same interval and the
  router denominator above. The `*_last_update` fields keep their narrower scope.
- `train_lm_loss` is cumulative across resume, using checkpointed sums and a
  checkpointed logging cursor. `train_objective_loss` preserves HF's native
  invocation/rank/resume aggregation semantics; use held-out loss for model
  comparisons. The artifact validator accepts these explicit names and the
  unchanged native-baseline names.
- `input_tokens_per_second_interval_wall` includes validation/checkpoint work
  between logs. `peak_allocated_bytes` and `peak_reserved_bytes` are interval
  maxima across GPUs (null on CPU). T8 remains a separate training benchmark,
  against native Qwen for the v1.4 gate, with matched D also reported.

Schedule checks reject warmup extending to the first curriculum boundary.
`tests/bdot/test_pilot.py` covers T13 and a small CPU T14 pair over 500 updates;
the full-model B200 pipeline comparison is a pre-launch check. It passed on
2026-10-20; see [pilot launch evidence](results/bdot_pilot_launch_20261020.json).
`scripts/bdot_pipeline_check.py` implements that check through unchanged
`train.py`: exact seed-initialization mapping, two fresh 500-update runs,
last-100-update LM loss and periodic validation comparison. It runs a fresh native
numerical control if the 0.01-nat gate fails, then stops for review. The authorized
`scripts/bdot_pilot_pipeline.py` reuses the existing cache helper, sequential
runner and burn supervisor to order T12 → T14 → D → P1-A → P1-B. A failed check
prevents pilot training. CPU cache preparation performs the one-time exact
train/validation overlap check before GPU reclamation. The resume
test covers a 1,000-step horizon stopped at 200, resuming at 100 across B's
soft-to-hard boundary. The GPU resume probe supports the same setup with
`--schedule-steps 1000 --steps 200 --split 100 --soft-split 20`.

Validation restores training modes, routing state and RNG. Logging sums/cursors
are model buffers saved by HF; native optimizer/scheduler saving is unchanged.
These changes intentionally do not bypass the strict source/state resume guard
for older checkpoints. Use fresh runs with the corrected policy.

### Route study

Use the same evaluator with `--route_study true`, on one device or through
Accelerate. It loads the checkpoint and held-out data once. Each rank scores
disjoint whole sequences without padding or duplicate rows.

```bash
accelerate launch --config_file resources/accelerate_b200.yaml \
  eval/packed_ppl.py \
  --checkpoint /absolute/pilot/P1-B/checkpoint-4768 \
  --eval_data_dir /absolute/local/qwen3_0.6b_base_en_30B/validation \
  --output_file /absolute/new/route-study/final_eval.json \
  --precision bf16 --batch_size 16 --route_study true
```

The backend, tokenizer, document boundaries, position resets and validation
manifest come from the checkpoint receipts. Keep the training backend for the
study. The output contains:

- Full selected-validation LM loss/PPL for routing as trained, forced default
  paths and random paths. No exploration or soft routing is used. Positive
  `default_path_loss_increase` means learned deviations help this checkpoint.
  These interventions are **not** the separately trained D/R controls needed
  for Gate 0a. Identical passes for fixed/random controls are reused.
- Route diagnostics on the first **4,096** validation sequences: off-default
  share (overall/per step), module counts, reuse, distinct paths and path length.
  Routing denominators include all input tokens, including EOS and document
  starts. LM loss still excludes the usual masked prediction targets.
- `distinct_paths_per_10000_tokens`: mean distinct paths over complete,
  contiguous 10,000-token windows in validation order. The final partial window
  is excluded and its length recorded; with no complete window the value is
  null. The global distinct count is also reported.
- `routes_sample.npz`: paths for the first **256** sequences, with shape
  `[sequence, token, routing_step]`, zero-based module IDs, input IDs, sequence
  indices, and hashes covering tokens, masked labels and positions. The file
  also records checkpoint, routing configuration, available training seeds,
  precision/backend and evaluation layout. Loading requires no pickle.

The study refuses insufficient validation data instead of silently shrinking
the sample. For small smoke tests set `--route_sequences` and
`--route_artifact_sequences` explicitly. `--max_sequences` limits **all three**
loss evaluations; omit it for the full-validation Gate 0a report. Random-route
replay requires the same evaluation seed, world size and per-rank batch size,
because it uses the existing rank/batch-indexed evaluation stream.
Both standalone B-DOT evaluation and the study now visit all-boundary batches.
Their random streams and whole-validation routing denominators therefore match
for identical layouts. `routing_input_tokens` states the latter denominator;
the separate diagnostic-prefix statistics retain their explicitly smaller scope.

To compare another seed, add `--compare_routes /previous/routes_sample.npz`, or
compare two existing artifacts without GPU inference:

```bash
python -m eval.route_study /seed0/routes_sample.npz /seed1/routes_sample.npz \
  /absolute/new/route_agreement.json
```

Agreement is measured over the **exported sample**, with both per-step and
complete-path agreement. Token identities/order/boundaries, model layout,
tokenizer, backend and precision must match. Set `--route_artifact_sequences 4096`
for both exports to compare the entire diagnostic prefix instead of 256 rows.
An identical-checkpoint comparison is a repeatability check, not independent-seed
evidence. Existing JSON/NPZ outputs are never overwritten.

Local verification on 2026-10-18 passed 32 B-DOT tests and 8 existing evaluation
tests. This covers route-count oracles, state/RNG preservation, matching LM
scoring, artifact identity guards, fixed-control pass reuse, two-rank CLI runs
with uneven diagnostic shards and an empty rank, and FA4 adapter wiring with a
CPU reference kernel. [Receipt](results/bdot_route_study_review_20261018.json).
This exporter has not yet been run on a scientific pilot checkpoint.

`eval/bdot_harness.py` passes the constructed model to the native lm-eval HFLM
adapter and keeps the harness's task/scoring definitions. Each invocation uses one
explicit device; independent task/checkpoint invocations can be scheduled in
parallel. It requires locally cached tasks and uses SDPA for ordinary padded
likelihood inputs. Packed FA4 is not silently applied to these different inputs.

## Correctness and remaining launch checks

Follow-up review on 2026-10-18 found and fixed a checkpoint-loading defect:
HF's meta-device loader replaces nonpersistent buffers with uninitialized
storage. B-DOT now explicitly reconstructs quota tables and zeros update
accumulators in its HF initialization hook. Persistent weights, balance bias
and routing counters remain loaded from the checkpoint. This matters especially
for the N-A/N-B no-reuse models; the original save/load test covered reuse-on only.

The expanded suite passes **25 tests**, including checkpoint reloads for all six
pilot configurations, CPU BF16 autocast, sparse dispatch against an independent
dense reference, 10,000 random paths for each tested mask, and FA4 wiring using
a documentwise CPU reference kernel. Two-rank tests also pass with nonreentrant
checkpointing in both hard routing (including unused experts) and soft top-two
routing; the largest gradient difference is 5.96e-8. See the
[review evidence](results/bdot_phase_a_review_20261018.json).
The FA4 wiring test does not execute or validate the actual CUDA kernel.

Initial local validation on 2026-10-18: **21 B-DOT tests**, a **two-rank CPU DDP probe**,
and **5 native PPL regression tests** passed in the rebuilt train_env. The DDP
probe's maximum gradient difference from its full-batch reference was
5.96e-8. An earlier broader run passed 42 inherited ML tests in the compatible
reference environment; 37 utility tests also passed. Dependency checks and
compilation passed. [Machine-readable evidence](results/bdot_phase_a_local_20261018.json)
records versions, source hashes, local log paths and limits of this validation.

The local test suite covers fixed-route logits, gradients and AdamW equivalence;
causality/isolation; no-reuse path counts; LM router gradients with z-loss disabled;
soft routing; checkpoint recomputation; global accumulation normalization;
save/load; cached decoding; and 200-step save/resume parity. A two-process CPU
probe compares actual Trainer/DDP gradients, unused experts, balance counts and
an optimizer update with a full-batch reference.

```bash
conda activate train_env
python -m unittest discover -s tests/bdot -v
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  tests/bdot/ddp_probe.py temp/bdot_ddp
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  tests/bdot/ddp_probe.py temp/bdot_ddp_soft --checkpointing --soft
python -m unittest discover -s tests/ml -v
```

`envs/train.txt` pins the training libraries; `envs/eval.txt` adds the harness.
The recreated local environment uses PyTorch 2.14.1, Transformers 5.9.0,
Accelerate 1.13.0 and Datasets 4.8.5. This development machine's driver cannot run
its CUDA 13 build, so tests in that environment use CPU. Separate compatible
environments were used for the local GPU checks below. No driver changes were made.

The subsequent [local GPU report](BDOT_GPU_VERIFICATION_20261018.md) records
CUDA numerical, NCCL and SDPA throughput checks. The later
[B200 verification](BDOT_B200_VERIFICATION_20261018.md) validates the pinned
training runtime, real FA4 backward and eight-GPU complete-update throughput.
`scripts/benchmark_bdot.py` now uses the actual PackedTrainer/BDOTTrainer for
complete forward/backward/clipping/AdamW/scheduler/routing updates, with explicit
backend, accumulation and curriculum phase. It supports single-device or torchrun
DDP and records the world size; local timings are not eight-GPU results. Dispatch currently
uses sorted token groups and PyTorch operations. Both matched D and native dense timings are reported; the throughput
gate is passed by the synthetic eight-B200 test at 524,288 input tokens/update.

An opt-in `bdot.expert_backend: "grouped"` uses PyTorch's CUDA BF16
`torch.nn.functional.grouped_mm` for expert Q/O and MLP projections. The default
is `"reference"`. It preserves the original parameters and state-dict names;
only occupied experts are stacked, so unused expert gradients remain absent
and AdamW does not start updating unused parameters. Shared step K/V, attention,
normalization, routing and loss definitions are unchanged. Weight packing is
included in complete-update benchmarks. Single-expert steps use ordinary linear
layers. CPU/FP32 multi-expert execution must use the reference backend. The
backend is part of the strict resume recipe: do not silently switch existing
runs. The [completed B200 comparison](BDOT_GROUPED_GEMM_20261020.md) passes
numerical, distributed and A/B resume checks, with +5.1% hard and +2.1% soft
throughput in repeated full-update measurements. It remains opt-in.

The route-study exporter and artifact comparisons are implemented above. They
still require actual trained checkpoints for a scientific Gate 0a report.
The subsequent [eight-B200 FA4 resume test](BDOT_FA4_RESUME_20261018.md)
passes exactly on all ranks with fresh processes and the real train.py entry point.
Actual benchmark scores require trained checkpoints and cached task data.
No scientific result is inferred from tiny local fixtures.

The subsequent [scientific code review](BDOT_PHASE_A_CODE_REVIEW_20261018.md)
identified and reproduced curriculum-A prior cancellation, an evaluation-policy
ambiguity and training-loss labeling. All three are now fixed as described
above, together with the minor evaluation and handoff issues. See
[fix evidence](results/bdot_phase_a_fixes_20261018.json). Revalidate these training
changes on B200/FA4 before a scientific pilot; earlier GPU receipts cover the
previous implementation. Decode/prefill speed measurement remains unimplemented.
