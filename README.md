# General remote-project template

A small, standard-library foundation for projects using a Git-triggered GPU
runner. No model, dataset, hardware allocation, or experiment is configured.
No Git repository is initialized and no remote action is submitted by setup.

## Start a new project

1. Copy this template to a new folder. Do not copy someone else's credentials
   or populated `temp/` folder.
2. Read `docs/AGENT_GUIDE.md`, then configure `project.local.json` from
   `project.example.json`. Populate the project notes and current task.
3. Configure and verify the development and execution Git remotes separately
   using `docs/GIT_PUSH.md`. Leave `commands.sh` at `#0` until a job is ready.
4. Choose project dependencies. Use runner `#i` to install, `#d` to download,
   and `#1` to verify/process/run. Follow `docs/commands.md`.
5. Run the local tests and harmless CPU example below before adding real jobs.

## Local verification (no network or GPU use)

```bash
python3 -m unittest discover -s tests -v
python3 run_experiments.py --config jobs.example.json --list
python3 run_experiments.py --config jobs.example.json --run-dir temp/example-run-001
```

The example writes and verifies one JSON artifact, then publishes
`temp/example-run-001/complete.json`. It refuses an existing run directory;
use a new run ID for a retry. It does not train, download, or use GPUs.

## Included utilities

| File | Purpose |
|---|---|
| `run_experiments.py` | Sequential arbitrary commands, logs, timeouts, required artifacts, success marker |
| `scripts/gpu_status.py` | Read-only physical GPU/PID inspection; optional fail-if-busy check |
| `scripts/verify_manifest.py` | SHA-256/size manifest creation and verification for local files |
| `scripts/dropbox_access.py` | Dev-machine non-recursive shared-folder listing and safe file download |
| `scripts/example_job.py` | Tiny CPU example; not a project training script |
| `envs/runtime.txt` | Minimal runner environment example |
| `resources/accelerate_config.example.yaml` | Optional, deliberately single-process/no-mixed-precision example |
| `resources/llm_pretrain_burn.py` | Optional PyTorch/NCCL GPU stress workload; see `docs/GPU_SAFETY.md` before use |

See `docs/JOBS.md` for the job manifest and `docs/DROPBOX_ACCESS.md` for
credentials and result retrieval. Optional ML training, benchmark evaluation,
finetuning, and data sampling must be implemented for the actual project.
Do not copy old architectures or benchmark defaults into the core template.

## Salvaged reference code (July 2026 template, added 2026-09-22)

Working reference implementations carried over from the previous template
generation; adapt before use — they are starting points, not the protocol:

| File | Purpose | Caveats |
|---|---|---|
| `prepare_data.py` | The exact sampler that built `nguyenhuuthuat09/CulturaX_sampled` (6 languages, seeded selection, token-count targets, disjoint train/eval) | Reads `HF_TOKEN` from the environment; run through controller `#d`/`#1` conventions, never direct downloads on GPU nodes |
| `eval/ppl.py`, `eval/benchmarks.py`, `eval/eval_checkpoint.py`, `eval/eval_parallel.py` | Sliding-window perplexity, lm-eval-harness multilingual benchmarks, and a one-checkpoint-per-GPU parallel eval runner (pairs with the `eval` env spec) | Check dataset/task paths for the new project |
| `train.py` | HF `run_clm.py`-style causal-LM training skeleton | Contains old-project ("EmbHub") wrapper code that must be stripped; see its docstring |
| `scripts/train_qwen3_0.6b_baseline.sh` | Hyperparameter record for from-scratch Qwen3-0.6B on the sampled data (~31.5B tokens, 1M-token steps) | REFERENCE ONLY: stale `/opt/dlami/nvme` paths and a missing `smoke_train.py` entry point; do not run as-is |
