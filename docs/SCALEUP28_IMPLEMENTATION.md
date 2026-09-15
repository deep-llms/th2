# 28-layer / 12B implementation

Date: 2026-09-15. Implements [the v3 scale-up plan](ccm_28layer_scaleup_plan_v3_12B.md).
This is a new experiment version, not a retroactive modification of the frozen
12-layer pilot. No B200 job is launched by creating these files.

## Protocols and scope

| Explicit training study | Backbone | Common / Stage-1 / Stage-2 updates | Meaning |
| --- | --- | --- | --- |
| `pilot12` (default) | 12 layers | 15,259 / 977 / 3,815 | Historical protocol, unchanged |
| `pilot12-shallow-followup` | Existing 12L common | Stage-2 Shallow only: 3,815 | Post-hoc follow-up, seeds 17 and 29 |
| `scaleup28-12b-v3` | 28 layers, from scratch | 38,147 / 977 / 7,629 | New 10B common + 2B continuation study |

The scale-up has 11,999,903,744 matched optimization input tokens. Its 1B-token
compiler pass is additional forward compute. Read location remains block 2;
the writer is block 28 **before final RMSNorm**. The reader, precision,
optimizer groups and exact bigram capacity are unchanged. Delta is prohibited
in the new study; Shallow is required in both new memory-training panels.

`ccm/model.py` exposes `deep_pre_final_norm` and the depth-specific
`r28_pre_final_norm`. Historical 12L captures retain `r12_pre_final_norm`.
The shared upstream Qwen implementation is retained; no blocks are rewritten.

## Data extension and holdout isolation

Use `prepare-scaleup`, **not** ordinary `prepare` with a larger budget. The
extension builder reserves new roles in this fixed order:

1. Preserve every selected historical document/content-hash group.
2. Allocate the fresh 20M-token `val28` holdout.
3. Allocate the 999,817,216-token continuation extension.
4. Allocate the 5,999,951,872-token common extension.

New roles take the first eligible source rows in the verified, sorted shard/row
stream, in three ordered passes. This allocation convention is fixed before
results; it is not a new random split of the historical data. Exact-text groups
stay in one role, including later duplicate rows. A quota-truncated document's
unused tail is not reassigned. SQLite keeps identity/deduplication state on disk.

Tokenization, EOS, immutable segment cuts and per-update input-token counting
reuse the pilot conventions. The historical prefix is **not re-tokenized**.
For every data seed, training yields the historical seed-shuffled optimizer
batches first, then separately seed-shuffled extension batches. Globally
reshuffling old + new would violate the prefix requirement and is not done.
The exact historical and extension batch-index permutations for data seeds
1017/1029/1043 are stored in the checksummed manifest before training. Loading
rejects a changed ordering convention; extensions use those saved orders.

`compile`, `adapt`, and `dev` delegate to the checksum-verified historical
corpus. `val` in the new corpus maps **only** to the fresh `val28` payload.
The old `val` is excluded from all extensions and is not the 28L holdout.

The existing vocabulary file is reused without metadata rebinding or recounting.
Both its full semantic hash and SHA256 of ordered little-endian int64 key bytes
(array index = slot ID) are frozen in the new manifest. Every memory entry point
requires those historical hashes. Thus capacity, mapping and compile data stay
identical, although the writer and full corpus hashes change.

The new corpus references the historical corpus by absolute path plus manifest
hash. **Keep both directories for backup/portability**; the extension directory
alone is not a self-contained dataset. Historical files are never edited.
A `complete.json` bound to the new manifest is required before consumption.
An incomplete output is not resumed or deleted automatically.

Example preparation, after verifying these historical paths still exist:

```bash
python -m ccm prepare-scaleup \
  --historical-data /mnt/local/_data/deep-llms_th2/ccm/pilot_v1_20260913_a01/corpus \
  --vocabulary /mnt/local/_data/deep-llms_th2/ccm/pilot_v1_20260913_a01/vocabulary.npz \
  --raw-dir /mnt/local/_data/deep-llms_th2/data/raw \
  --source-manifest resources/culturax_raw_manifest.tsv \
  --dataset-revision 6a8734bc69fefcbb7735f4f9250f43e4cd7a442e \
  --tokenizer-path /mnt/local/_models/deep-llms_th2/Qwen3-0.6B-Base-da87bfb608c14b7cf20ba1ce41287e8de496c0cd \
  --tokenizer-manifest resources/qwen3_base_assets.json \
  --output /mnt/local/_data/deep-llms_th2/ccm/scaleup28_v3_a01
python -m ccm validate-data \
  --data /mnt/local/_data/deep-llms_th2/ccm/scaleup28_v3_a01
```

All acquisition remains controller `#i`/`#d` only; this command uses local files.
Source shard hashes and official Base asset hashes are checked before selection.

## Stability check and common training

`train --study scaleup28-12b-v3 --phase common --arm base` chooses 28 blocks
and requires the explicit extended corpus. No 12L weights are loaded.

`--stability-steps N` runs a strict prefix with the **full 38,147-update
schedule**, saves its checkpoint and a separate `stability.json`, and does not
publish a full-training completion marker. The generated queue defaults to
32 updates: this is an early safety smoke, **not a test at peak LR**, whose
warmup endpoint is update 763. A longer predetermined smoke may be chosen
before training, without shortening the scientific schedule.

Full scientific common training requires `--stability-report` from a successful
matching check (data, config, seed, LR, code, precision environment, world size
and batching). It restarts from fresh initialization; it does not resume the
smoke. The default peak LR is 3e-4. A 2e-4 fallback requires
`--lr-failure-report` recording objective non-finite loss/gradient/optimizer
failure at 3e-4, then a fresh successful 2e-4 smoke. OOM, infrastructure failures
or an unattractive early loss curve are not LR-fallback evidence.

The implementation does not automatically erase, retry, or resume failed runs.
Preserve failure evidence and use a new output root for an authorized restart.
Common `checkpoint-38147` is the `theta_10B_28L` used for every memory arm.

## Distributed compilation

`compile --validate-distributed` and `compile --distributed` are torchrun paths.
They shard **whole reference forward batches** by batch-index modulo world
size. They do not change padding/forward shapes via length regrouping.
Each rank accumulates CPU FP32 sums, int64 counts and FP64 squared-norm sums.
Chunked collectives merge sufficient statistics before any mean or variance
is computed. Only rank 0 writes artifacts and constructs Isolated/Shuffled.

Run the bounded validation against the completed 28L writer first:

```bash
torchrun --standalone --nproc_per_node=8 -m ccm compile \
  --data "$TASK_DATA28" --vocabulary "$TASK_VOCAB" \
  --checkpoint "$TASK_COMMON28" --device cuda \
  --microbatch-segments 4 --validate-distributed \
  --validation-batches 16 --validation-eval-batches 4 \
  --output "$TASK_OUTPUT/compiler-gate"
torchrun --standalone --nproc_per_node=8 -m ccm compile \
  --data "$TASK_DATA28" --vocabulary "$TASK_VOCAB" \
  --checkpoint "$TASK_COMMON28" --device cuda \
  --microbatch-segments 4 --distributed \
  --compiler-validation "$TASK_OUTPUT/compiler-gate/validation.json" \
  --output "$TASK_OUTPUT/tables"
```

The variables above must be set to inspected, resolved local paths first.
Outputs must be fresh. Acceptance tolerances are frozen in code before the
scientific compiler runs: mean and variance `atol=rtol=1e-5`, exact counts and
mapping, and absolute NLL difference ≤1e-5 nats/target token on a small dev
subset. The lookup test uses a fixed **nonzero** value projection; a zero reader
would conceal table errors. Unobserved subset rows are zero in both validation
paths; full production compilation still requires every vocabulary slot's
count to equal its historical compile count.

Production refuses a gate from a different writer/data/vocabulary/code/world/
microbatch/device environment. Do not edit research code between the gate and
full compilation. A failed gate stops the queue. The explicit fallback is the
unchanged reference path (`compile` without distributed flags), not relaxed
tolerances or changed constructors. Generate a new reference panel queue if
that fallback is needed; do not reuse a partially populated output root.

Reported accounting separates wall time, active-task GPU-hours, reserved-node
GPU-hours, aggregate process CPU time, input token bytes and output bytes.
Active-task time includes transfer/collective waits; it is not GPU-kernel-only
time. Token bytes exclude index/checkpoint reads. Reduced wall time is not a
claim of reduced total compute.

## Sequential queues and parallel branches

`python -m ccm scaleup-jobs --help` generates JSON compatible with the existing
`run_experiments.py`. It does **not** execute or push anything.

| Queue | Behavior |
| --- | --- |
| `--queue common` | Short stability smoke → fresh full 10B common → checkpoint/optimizer/log validation |
| `--queue shallow12` | Verify historical common → dev coverage → fresh 3,815-update Shallow → validate → dev eval → Deep−Shallow report |
| `--queue panel` | Verify 28L common → coverage → compiler gate/compile → five Stage-1 arms + Base eval → six Stage-2 arms → dev reports → replication decision |

Required generation inputs include `--data`, `--vocabulary`, `--gpus`, and
`--output`; common additionally needs `--model-config`. Panel and Shallow need
`--checkpoint`; Shallow also needs `--tables`, `--contextual-eval` and its seed.
`--compiler reference` selects the explicit known reference fallback.

Example generation, after assigning the path variables:

```bash
python -m ccm scaleup-jobs --queue common \
  --data "$TASK_DATA28" --vocabulary "$TASK_VOCAB" \
  --model-config "$TASK_ASSETS" --gpus 0 1 2 3 \
  --output temp/common28.jobs.json
python run_experiments.py --config temp/common28.jobs.json --list
```

Common and the two 12L Shallow queues can run concurrently on explicitly
disjoint subsets (for example 0–3, 4–5 and 6–7), retaining the same global
token batch. This split is not a B200 throughput recommendation: validate it
before deployment. Every torchrun uses an independent standalone rendezvous.

The generated manifests leave final holdout access and later backbone seeds
out of the automatic queue. They use the existing runner's artifact/free-GPU
checks and never stop existing workloads. **A live launch still needs an outer
reviewed GPU handoff**, including the current burn-observer disarm, safe worker
reclamation, waits and idle-workload restoration from `GPU_SAFETY.md`.
Single-device eval/reference compilation leaves other assigned GPUs unused;
the deployment handoff must apply the operator's idle policy without colliding
with the next multi-GPU job. The generator itself is not that handoff.

`validate-run` checks exact endpoint/token counts, checkpoint/config/optimizer
hashes, finite saved tensors and sequential finite training logs. A directory
name or exited PID is not accepted as successful training.

## Heldout locks and interpretation

After both post-hoc Shallow results and interpretation choices are frozen:

```bash
python -m ccm lock-final --study pilot12-shallow-followup \
  --checkpoints "$TASK_17_BASE" "$TASK_17_CONTEXT" "$TASK_17_ISOLATED" \
  "$TASK_17_SHUFFLED" "$TASK_17_SHALLOW" "$TASK_17_GRAD" \
  "$TASK_29_BASE" "$TASK_29_CONTEXT" "$TASK_29_ISOLATED" \
  "$TASK_29_SHUFFLED" "$TASK_29_SHALLOW" "$TASK_29_GRAD" \
  --confirm-choices-locked --cluster doc_id --cross-seed-coupling independent \
  --output "$TASK_12L_FINAL_LOCK"
```

This explicitly requires six arms for both seeds 17/29 and labels the two-seed
post-hoc protocol amendment. Default `pilot12` still requires its original
three-seed panel. Use the historical corpus for those evaluations.

For the new study, `lock-final --study scaleup28-12b-v3` requires the full
six-arm panel for seed 17, or seeds 17/29/43; a seed-17-only scientific lock
additionally requires `--single-seed-terminal` (decision 2a, 2026-09-15). Reduced-arm later replication
needs a separately declared protocol extension; it is not silently accepted.
Use the **extended** corpus with `evaluate --role val --final-evaluation
--final-lock ...`; that role resolves to fresh `D_val_28` only. Existing
one-evaluation-per-named-checkpoint claim files remain in force. Use
`compare --final-report` for final results with the lock's statistical policy.

Interpretation caveat: extension roles (including `D_val_28`) are allocated as
first-eligible rows of the sorted source stream, not hash-randomized draws
like the historical roles. The convention is frozen before results; still, do
not over-interpret absolute NLL level differences between `D_dev` and
`D_val_28` — the locked holdout exists to confirm paired contrasts.

The dev `scaleup-decision` rechecks paired records and saved 10,000-replicate
document-bootstrap reports. Both upper confidence limits below zero means
`replicate`; one means `ambiguous`; neither means `diagnose`. It reports a
decision but launches no additional seed. Deep−Shallow is secondary, not a
replacement for either primary contrast. All panels include Grad as a quality/
optimizer-memory reference; no claim of superiority is assumed.

## Pre-launch decisions — 2026-09-15

Recorded from the plan owner's explicit answers (see
[SCALEUP28_PRELAUNCH_DECISIONS_20260915.md](SCALEUP28_PRELAUNCH_DECISIONS_20260915.md)
for the full facts and options), before any 28L result exists:

1. **Stability smoke = 32-update pipeline smoke, then 1,024-update stability
   smoke.** The `common` queue now generates `common-pipeline` (32 updates) and
   `common-stability` (1,024 updates, past the 763-update warmup at peak LR);
   the full run consumes only the 1,024-update certificate, and the generator
   refuses a stability smoke of ≤763 updates. Objective instability remains
   defined as non-finite loss/gradients/optimizer state only; no
   finite-divergence monitor exists or is added.
2. **`D_val_28` is untouched until the full replication panel is frozen.**
   A seed-17-only final lock is permitted only if the study stops at seed 17,
   for honest one-backbone reporting. Enforced in code: a scientific
   seeds=[17] scale-up lock now requires the explicit `--single-seed-terminal`
   flag and records that declaration in the lock.
3. **Sparse scale-up checkpoints:** common every 5,000 updates + final,
   Stage 2 every 2,000 + final, Stage 1 final-only; smokes save only their
   endpoint. Estimated panel footprint drops from ~0.8–1 TB to roughly
   0.2 TB. The 12L Shallow follow-up keeps the historical every-1,000
   convention. **No resume path is added**; restart-from-scratch remains the
   failure policy for the ~10B common run.

These decisions change queue generation only; research-core training,
compilation, evaluation and statistics code are unchanged by them.

Two further review fixes, same date: the reference compiler no longer marks
scale-up contextual tables `online_capable` (online writing is excluded from
this study; the distributed path already recorded `false`, so both paths now
emit identical metadata semantics), and the single-seed lock enforcement
above. Neither changes any numeric result path.

## Verification and remaining launch gates

Final local regression run on 2026-09-15 (after the pre-launch decisions and
review fixes): **136 tests passed in 90.89 seconds** with CUDA hidden, using
the command below. This includes actual two-process Gloo reference/merge/
lookup checking, the complete tiny-width 28-block arm pipeline, the historical
Shallow follow-up, queue assertions locking the decided smoke lengths and save
intervals, and the single-seed lock enforcement test. No B200 process was
queried, stopped or launched, and neither the execution checkout nor
`commands.sh` was modified by this task.

Tested `ccm/` source hash, computed as `ccm.cli.code_hash()` (`digest_json` of
per-file SHA256 over `ccm/*.py`):
`214d57c4816b37298000051da8a0ef16fabdfce1f2c870e8e5262729e35b1582`.
Hashes previously recorded here (`ce87aff7...`, then `eba9ecef...`) belong to
superseded trees; the `ce87aff7...` value additionally predated same-day edits
and did not identify any reviewed tree. Use only the current value for
deployment verification.

`tests/test_scaleup.py` reuses the model acceptance suite at 28 blocks with
tiny width, adds explicit final-RMSNorm hook checking, exact historical prefix/
mapping/disjointness tests, two-process CPU/Gloo compiler validation and full
tiny Stage-1/Stage-2 training/evaluation with a fresh heldout lock. It also
checks queue parsing, explicit post-hoc locking and failed compiler statistics.

CPU test command:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/users/thien/miniconda3/envs/sparse_emb/bin/python -m pytest -q tests
```

These tests do not establish B200/NCCL equivalence, full-width memory fit,
throughput, full-scale extension availability or scientific quality. Before an
expensive launch, verify prepared real manifests, full-width 28L CUDA behavior,
the stability check, GPU ownership, free node disk against the decided
checkpoint footprint (roughly 0.2 TB for the panel under the 3a intervals,
plus data/table artifacts), and the live handoff. Before distributed
1B compilation, its matched B200 equivalence gate must pass. No new capacity,
online writing or third 12L backbone is automatically scheduled.
