# B/F/G optimized continuation to 10,000 updates

## Initial gate outcome and checkpoint-only retry

Initial a01 gate completed all six one-update runs; all48 rank receipts verified
exact restored model/optimizer/scheduler/RNG and matched next microbatches.
However, the numerical gate rejected combined optimized B: parameter relative L2
1.1260900673368356e-5 exceeds the predeclared1e-5 limit (control update relative
L2 0.0018496647; difference is about0.61% of the update). Evaluation deltas were
B -6.76e-6, F +1.36e-5, G +2.89e-6. This is a numerical tolerance failure, not a
state/data restoration failure. Do not report the original gate as passed.
Production never started. Queue failed16:14:53 UTC; automatic burn recovery
started and all8 burn workers were live in the16:15:59 audit. The final handoff
receipt/guard release must be rechecked before retry.

Keep every tolerance unchanged. New a02 recipe disables only decoder, LM-loss
and auxiliary-loss checkpointing, retaining original causal_attention=False and
lm_chunk128. Both additional switches remain available in train.py, but are not
selected for this continuation. The a02 root replaces a01 in the production
paths below. Reuse the successful original control checkpoints/receipts read-only;
run three fresh optimized one-update resumes and compare all states/data and
numerics against those controls. A gate receipt records its control directory.
Only passing this unchanged gate permits fresh original5000 copies to resume
B/F/G to10000. No original or failed-attempt outputs are overwritten/deleted.

Initial audit68c5aa7:73 small artifacts verified under
artifacts/optimized-resume-20260929-a01/. Raw log SHA256:
a5b6ef983e2276e4925497418a5526c980445cb50472b1bb339d844b93c347d6.

The following records the initial proposal and common resume protocol:

Authorized 2026-09-29. Continue each completed step-5,000 arm on all eight B200
GPUs, sequential B/F/G. Keep microbatch16, accumulation4, sequence2048,
1,048,576 input tokens/update, full schedule28,600 and warmup1,430. Dataset,
EOS packing, sampler seeds, optimizer and scientific arm/loss settings stay fixed.

`train.py` exposes the previously tested model switches: checkpoint_layers,
checkpoint_lm, checkpoint_aux, causal_attention and lm_chunk. Defaults preserve
old execution. Optimized continuation disables all three recomputation switches,
uses implicit causal SDPA on fully packed/unsegmented inputs, and LM chunks512.
The production loop remains native HF Trainer/Accelerate.

Resume configuration matching remains strict except these five settings, and
changing any requires explicit `--allow_performance_change_on_resume true`.
Legacy missing flags mean the historical defaults. Each permitted transition
archives the complete previous/requested recipe and changed values in
`resume-transition-<step>-<UTC>.json`. This record describes requested execution;
only successful Trainer loading/training establishes that resume actually worked.
Other settings, data fingerprints, optimizer schedule, global/per-device batch,
seeds, ignore_data_skip and arm must match. Do not use this flag to change them.
Small floating-point differences are expected; bitwise equivalence is not claimed.

Before the long queue, `scripts/check_optimized_resume.py` observes native HF
restoration from full copies of each real checkpoint5000. Control and optimized
runs each take exactly one update and save/evaluate normally. All ranks verify
restored model, complete AdamW state, scheduler and RNG against the source files;
the two paths must consume the identical ordered microbatches. The checker does
not replace native loading, RNG handling, sampler skipping or optimization.
Predeclared acceptance bounds: parameter relative L2<=1e-5 and difference<=5%
of the control update; first/second moment relative L2<=3%; final LM loss delta
<=0.001. Scheduler states and optimizer step counters must match exactly.
The gate writes `resume-smoke/verified.json` only after every arm passes.

Production resumes fresh checksum-verified copies of the original5000 checkpoints,
not the smoke outputs. Source directory:
`/mnt/local/_outputs/deep-llms_th2/deep-kv-BFG-5000-20260929-a01/production/run`.
New root: `/mnt/local/_outputs/deep-llms_th2/deep-kv-BFG-10000-20260929-a01`.
Production arms: `production/run/continuation/{B,F,G}` beneath that root.
Stop at10,000 total updates /10,485,760,000 input tokens, keeping the full schedule.

Launch copies the resource Accelerate YAML to the actual HF cache, compares bytes,
checks8 processes/BF16/MULTI_GPU and runs `accelerate env`. Existing supervisor
rechecks approved burn identities, signals only verified worker PIDs, waits and
requires all eight GPUs free. The same supervisor restores and verifies enhanced
burns after queue success or failure, after cleaning only its own descendants.
No environment, driver, CUDA or package installation is part of this launch.

Local validation:30 existing/extended CPU tests passed, including all-arm model
checks and B/F/G native Trainer state/data/optimizer continuation, rejection of
unauthorized metadata changes, and numerical comparisons. Two additional targeted
tests passed: the observational gate on a real tiny saved G checkpoint plus queue
contracts, and production causal-path rejection of padded/segmented contexts.
Remote gate and launch status must be checked separately; this document does not
assert that they have run or passed yet.
