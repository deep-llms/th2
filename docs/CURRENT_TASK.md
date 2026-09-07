# Current task

Status: `swt` created and verified on both hosts; preparing frozen English/GPT-2
data with the user's revised target of at least 10B training tokens.
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
- Environments: `/home/users/thien/miniconda3/envs/swt` (dev),
  `/mnt/local/conda-py311/envs/swt` (B200), offline clones of each host's
  existing `sparse_emb`; source versions matched. B200 torch is 2.14.0,
  dev torch 2.7.1+cu118; both Transformers 5.9.0. GPU runtime not tested here.
- Input manifests: `resources/culturax_en_source_b200.json` and
  `resources/gpt2_tokenizer_607a30d_manifest.json`.
- Output: `/mnt/local/_data/deep-llms_th2/swt/english_gpt2_10b_seed0_20260907_a01`.
- Preparation command: `bash scripts/prepare_english_b200.sh`.
- Latest verified status: 2026-09-07, 40 tests passed in
  `temp/capacity_all_tests.log`; BF16 CPU, two-rank Gloo model smoke, two-rank
  Trainer and standalone-NLL consistency checks passed. No destination GPU test.
- Completion criteria / expected artifacts: all six model arms, offline frozen
  packs, Trainer CLI, standalone PPL, documentation and correctness tests.

## Next action

B200 verified the three main Stagewise file hashes and 40 CPU tests at
2026-09-07 22:48:14 UTC (`temp/remote_logs/swt_clone_a01.log`, runner commit
`52c73b3`). Tokenizer download `94f1cf7` reported all five files successful;
the preparation job will independently verify their hashes. Latest dev review:
42 tests passed, plus pinned real GPT-2 long-document/EOS/dedup packing smoke.
Random-row token-yield preview across all 50 English shards estimates 43.07B
tokens before document sampling/exact dedup. Fraction 0.30 estimates 12.87B
train and 25.84M each held-out split, with actual minimums enforced at completion.
Submitted offline preparation in th2 commit `4e6e553` at 2026-09-07 22:57 UTC,
job `th2-swt-english-gpt2-10b-prepare-20260907-a01`. Await the matching remote
log to verify startup and progress; no training or GPU changes.

See `CAPACITY_EXPERIMENTS.md` for implemented arms, offline preparation, tests and
the training CLI. Run the destination-GPU smoke test and freeze real input
provenance/splits and optimization settings before authorizing research training.

## Handoff

Record what completed, failed, remains uncertain, and what is authorized next.
Move durable results/decisions to `PROJECT_NOTES.md`; mark old plans historical.
