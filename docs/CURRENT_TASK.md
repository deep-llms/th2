# SharedLocal G16 and independent low-rank output diagnostic

Status: implementation and local validation complete; remote training has not
been launched.

## Locked 10k screen

- **Current hardware constraint (2026-08-22): th2 is the only active remote**, a
  single 8× B200 node. The two screens must run sequentially, not in parallel.
- First on th2: rank-128 low-rank input plus an independent rank-128 low-rank
  output initialized from the same factor values (runner index 12).
- Then on th2, after the first run is stopped/finished and GPUs and output paths
  are verified: SharedLocal tied, shared rank 64, local rank 64, 16 vocabulary
  groups (runner index 11).
- Both retain the full one-epoch learning-rate schedule and must be stopped by
  `run_experiments.py --stop-at-step 10000`; do not set `max_steps` in the
  production scripts.
- On the 183 GB B200s, both use 16 sequences/device and 4 gradient-accumulation
  steps across 8 GPUs. The effective batch remains the H100-era value of 512
  sequences (about 1M tokens) per optimizer step, so the update count, warmup,
  and learning-rate schedule remain directly comparable.
- The current runner's Python 3.11 conda prefix is `/mnt/local/conda-py311`;
  activate `sparse_emb` from that prefix and invoke its `python3.11` explicitly
  because this installation does not provide a `bin/python` symlink.

## Implementation completed (2026-08-17)

- `IndependentLowRankHead` owns separate `V x r` token factors and `d x r`
  projection weights. Both are cloned from the input factors without consuming
  RNG or sharing storage. Forward is `(hidden @ W_out) @ X_out.T`.
- The output has no bias. Qwen's native classifier is bias-free, and copying
  the input rank-to-hidden bias would only add the same scalar to every class.
- `--independent_lowrank_output` is valid only with `--arm lowrank` and is
  mutually exclusive with `--tie_output`.
- Periodic and final checkpoints save `output_head.pt` beside `embedding.pt`.
  Evaluation and finetuning reconstruct it through the common compositional
  loader. Missing/stale artifacts fail loudly, and sidecar tensors must exactly
  match their corresponding tensors in the HF checkpoint.
- Resume validates sidecars, the saved compositional configuration, and the
  actual input/output tensor topology in the HF model state before Hugging
  Face's permissive checkpoint loader runs. It also requires matching
  `trainer_state.json`, optimizer, scheduler, and every DDP rank's RNG state,
  including an exact world-size match, preventing a partial checkpoint from
  silently restarting the schedule.
- New scripts and runner entries:
  - index 11: `shared_local_tied_g16`
  - index 12: `lowrank_independent_output_r128`
- Both new runner entries require a fresh output directory. The runner now
  exits nonzero for failed/skipped jobs or a process that remains alive.

## Validation completed

- `63 passed` across the compositional, output-head, and runner-safety suites.
- Covered factorized algebra, identical initial classifier factors, no aliasing,
  unchanged RNG state, gradients, optimizer ownership, generation, G16's
  divisible batched-GEMM path, strict HF-state resume, periodic/final sidecars,
  parent/configless loading, BF16 casting, complete single/8-rank Trainer
  state, sidecar/HF value integrity, and corruption failure paths.
- An actual tiny Qwen Trainer run completed step 1, saved both sidecars,
  automatically resumed to step 2, reloaded through the evaluation loader, and
  produced finite logits (`TRAINER_SAVE_RESUME_LOAD_SMOKE_OK`).
- Four-A100 validation completed on the dev machine. A real four-process BF16
  DDP run through `train_compositional.py` trained to step 2, saved all four RNG
  states plus both sidecars, automatically resumed to step 3, and reloaded
  through the eval path with successful generation. A second four-GPU smoke at
  the exact production vocabulary/hidden/rank dimensions completed finite
  forward/backward and synchronized updates. Finally, the complete
  Qwen3-0.6B configuration completed a BF16 causal-loss backward pass with
  479,626,240 parameters and about 1,947 MiB peak allocated memory.

## Next actions

1. Commit shared code on `main` and merge it into active branch `h100-1` only.
   Do not update or push `h100-2`/th3 while it is inactive.
2. Put the index-12 command on th2 with GPU-count/process, dataset/model, and
   fresh-output preflight checks. Run to checkpoint 10k and evaluate it with
   `eval/eval_parallel.py`.
3. Interpret the index-12 PPL screen, then stop/clean only that run and verify
   all eight th2 GPUs are free before launching index 11 on the same machine.
4. Run and evaluate index 11 to checkpoint 10k using the same protocol. Do not
   start a full training run or finetuning until both PPL screens are compared.
