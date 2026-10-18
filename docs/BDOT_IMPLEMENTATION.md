# B-DOT Phase A implementation

Implements the Phase A subset of [spec v1.3](BDOT_v2_spec.md). No dataset download,
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
- `bdot/data.py`: frozen validation document IDs, document shuffle seed 0, and
  continuous packing with only the final remainder dropped. Uses HF caches;
  no separate token-export command is required.

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
  --data /absolute/local/fineweb-edu-sample-10BT \
  --revision DATASET_COMMIT_HASH \
  --manifest /absolute/local/fineweb-validation.json \
  --accelerate-config /absolute/local/accelerate.yaml \
  --world-size 8 --micro-batch 16 \
  --output /absolute/new/pilot-plan
```

The default queue contains D → P1-A → P1-B. R and the matching N follow selection
of the curriculum; they are not selected automatically. The generator supports
`--arms R N-A` or `--arms R N-B`. It records the rounded token budget and separate
model/routing seeds. Eight ranks × 16 sequences × accumulation 4 gives 512 × 1024
input tokens/update. A 1B request rounds down to 1,907 updates (999,817,216 tokens).
The full schedule has 2% warmup; the inherited `stop_after` cutoff remains available.
Checkpoint retention is unlimited unless explicitly changed.

The generated `jobs.json` works with the existing runner. Device allocation,
runner authorization and availability must be established separately. `commands.sh`
remains inactive. Every job checks for fresh output and verifies the final step
and saved weights before declaring completion.

The local raw dataset must contain `text` and unique, nonnull `id` columns. The
manifest fixes the revision, tokenizer and validation documents. Training uses a
sequential prefix after document shuffling, independent of model seed. A budget
larger than the packed corpus is rejected instead of repeating the corpus.
Periodic evaluation uses a fixed approximately 1M-input-token prefix; final
`final_eval.json` scores the complete frozen validation split. Router regularizers
never enter reported validation loss/PPL.

## Checkpoints and evaluation

Load routed checkpoints through `train.build_model()` or
`BDOTForCausalLM.from_pretrained()`. Their config embeds `bdot` options in the
Qwen3 config; an unmodified AutoModel loader alone does not reconstruct the router.
The project's training and evaluation entry points use the correct loader.

`eval/packed_ppl.py` supports the document-stream manifest and B-DOT checkpoints.
For these checkpoints, `--eval_data_dir` points to the original raw corpus; the
manifest selects held-out documents. `--validation_manifest` relocates the same
manifest if needed. `--routing_override fixed` or `random` performs the specified
route intervention; `--routing_seed` selects a repeatable evaluation stream.
Off-default share and repeated-module fraction are included in the result.
Reuse means the fraction of tokens whose path repeats any module, as in the
spec. Training logs also contain router entropy, per-step off-default shares
and per-module counts. These fields explicitly describe the last optimizer
update; they are not averages over the entire logging interval.

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
uses sorted token groups and PyTorch operations, without a custom grouped-GEMM
kernel. Both matched D and native dense timings are reported; the throughput
gate is passed by the synthetic eight-B200 test at 524,288 input tokens/update.

The full route-study exporter (4,096-sequence path statistics and the
256-sequence routes_sample.npz artifact), cross-seed comparison reporting remain to finish before the pilot's final Gate 0a
report. Routed FA4 resume parity also needs a GPU check; existing exact resume
evidence is CPU-only.
Actual benchmark scores require trained checkpoints and cached task data.
No scientific result is inferred from tiny local fixtures.
