# Seed-17 memory dependency diagnostics (A1/A2)

Post-hoc diagnostics only, following `ccm_next_steps_after_seed17.md`.
No new training, constructor change, seed replication, or locked `D_val` access.

Scope: completed Stage-2 Contextual, Isolated, Grad, step 3815; full 20M-input-token
`D_dev`. Preserve original checkpoint files and evaluation outputs. Base-Continue
is the already-validated evaluation on the same documents/target labels.

## Measurement

`scripts/memory_diagnostics.py` adds a temporary external reader hook; it does
not modify `ccm/`. Each original eight-segment batch is evaluated twice with
loss chunk 1024, in evaluation mode without gradients:

- Normal: reader output returned unchanged.
- Memory-off: replace the reader contribution with zeros, leaving trained
  weights, slots, eligibility, targets, and attention masks unchanged.

Normal replay must match the original evaluation: each segment/population NLL
within 2e-5, token-weighted absolute overall segment discrepancy within 2e-6.
Actual discrepancies are reported, not assumed zero. Full input/target counts,
ordered segment identities, checkpoint/data/vocabulary hashes must agree.
Miss losses can change because previous hit positions affect later states.

On original hit targets in the normal pass, compute FP32 L2 norms of the actual
reader contribution and the block-2 hidden state **before** memory addition.
Report mean, median, p10, p90 of each norm and the per-token norm ratio. The
ratio denominator is clamped at 1e-12; count affected denominators. All hit
targets enter these summaries; padded/ignored targets do not.

Report normal/off/Base NLL and paired `off − normal`, `off − Base` intervals
for overall, hit, miss, and eligible-miss populations. Bootstrap source `doc_id`
clusters, 10,000 replicates, seed 20260913, conditional on fixed backbone seed 17.
Positive `off − normal` means disabling memory worsens NLL. This is an
inference-dependency test, not definitive causal attribution of gains to
backbone learning. Memory-off still computes the reader, so its runtime is not
a reader-free performance benchmark. A3 frequency × variance is separate.

## Deployment and safety

Approved run root:
`/mnt/local/_outputs/deep-llms_th2/ccm_memory_diagnostics_seed17_20260914_a01`.

`scripts/launch_memory_diagnostics.sh` activates `train_env` inside the persistent
tmux session. No Accelerate config is needed for independent evaluation jobs.
`scripts/pilot_memory_diagnostics.py`:

1. Checks original burn ownership and the frozen scientific code/data/checkpoints.
2. Requires seed-29 common workflow completion and a fresh successful heartbeat.
3. Sets only that completed observer's `STOP_IDLE_WATCH`; waits for its exact
   process to exit voluntarily. Never signals the observer, sleeper, or PID 1.
4. Stops only verified burn-worker pidfds; waits/rechecks all GPUs twice (30s each).
5. Starts an original five-GPU burn group on GPUs 3–7. Runs two-batch real-data
   smoke diagnostics on GPUs 0–2, then full diagnostics on those same GPUs.
6. Reaps all three children, checks outputs, waits 30s and checks GPUs 0–2 free.
   Stops the verified spare burn group, waits 30s and checks all GPUs free.
7. Restores the original `/tmp/llm_pretrain_burn.py` on all eight GPUs, then
   computes/validates the CPU bootstrap report while burns remain active.
8. Writes `complete.json` only after the full report and burn checks succeed.
   Continues observing burns until this new root's `STOP_IDLE_WATCH` is created
   by a later authorized handoff.

The original source checkpoints/data/results are read-only. Smoke and full
outputs are separate fresh directories. Failed children are not automatically
retried or killed; all launched diagnostic children are reaped before normal
recovery. Unknown GPU owners prevent burn replacement. Runner/AWS faults are
reported for operator repair, not treated as research-code failures.

Retrieve `status.json`, `complete.json`, `full/summary.json`, each arm's
`diagnostics.json`, and `normal/` + `memory_off/` records/metrics through `#2`.
Every arm diagnostic records the new result hashes and original metrics hash.
