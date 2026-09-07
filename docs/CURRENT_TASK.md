# Current task

Status: creating `swt` environments on dev/B200 from each host's `sparse_emb`,
verifying B200 code/environment and preparing frozen English/GPT-2 data.
No Stagewise training or GPU-process termination is authorized.

## User request and authorized scope

- Objective: create `swt` on both machines; verify code/environment and prepare data.
- Allowed remote actions: necessary setup/download/CPU-preprocessing commands and
  pushes to both confirmed repositories, with monitoring through Dropbox.
- Constraints / do not touch: no training, GPU cancellation, or modification of
  existing environments, datasets or checkpoints. Work in this project,
  not the old sparse-embedding project or generic template.

## Configuration and evidence

- Development: `/disk/thuat/stagewise_widening_transformer`, `main`, origin
  `nguyenhuuthuat09/stagewise_widening_transformer`.
- Execution: `deep-llms/th2`, `main`; deployment checkout
  `/disk/thuat/th2_runner_clean_probe` (remote `second`; development alias `runner`).
- Machine: `thiennh-p6-oish-worker-0`, 8 B200s. Last read-only observation at
  2026-09-07 22:21 UTC: all GPUs occupied (155010 MiB, 100% utilization each).
- Environment/interpreter: `/home/users/thien/miniconda3/envs/sparse_emb/bin/python`
  for local CPU tests; separate dependencies in `requirements-capacity.txt`.
- Input manifests and output/run directory:
- Exact command and job name:
- Latest verified status: 2026-09-07, 40 tests passed in
  `temp/capacity_all_tests.log`; BF16 CPU, two-rank Gloo model smoke, two-rank
  Trainer and standalone-NLL consistency checks passed. No destination GPU test.
- Completion criteria / expected artifacts: all six model arms, offline frozen
  packs, Trainer CLI, standalone PPL, documentation and correctness tests.

## Next action

Publication uses `#0`: updating GitHub alone does not prove code has been synced
to the B200 filesystem. A later authorized `#1` job performs that sync.

See `CAPACITY_EXPERIMENTS.md` for implemented arms, offline preparation, tests and
the training CLI. Run the destination-GPU smoke test and freeze real input
provenance/splits and optimization settings before authorizing research training.

## Handoff

Record what completed, failed, remains uncertain, and what is authorized next.
Move durable results/decisions to `PROJECT_NOTES.md`; mark old plans historical.
