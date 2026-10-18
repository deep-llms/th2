# Template validation history

These are historical infrastructure tests, not a new project's results, an
active run or permission to reuse any listed repository/machine. Configure and
validate each new project's own runtime, data and model. Nothing runs on copy.

- [Tiny B200 checks](B200_SMOKE_20261017.md): SDPA/FA4, single/two-GPU
  train/resume/evaluate, exact resumed/uninterrupted weights and optimizer,
  scheduler/RNG checks, real-kernel numerical and document-isolation tests.
- [Full-model B200 pilot](B200_REAL_DATA_PILOT_20261017.md): full Qwen3-0.6B,
  eight GPUs, both backends, checkpoint at 25 and resume to 50, full scheduler
  and warmup verified, finite gradients and matching periodic/final evaluation.
  It does not establish long-run model quality or bitwise backend equivalence.
- Structured receipts are retained under results/. Paths and machine names
  inside those reports are historical provenance, not runnable defaults.

The tested B200 runtime was Python 3.11.15, torch 2.14.1+cu130,
transformers 5.9.0, accelerate 1.13.0, datasets 4.8.5 and optional
flash-attn-4 4.0.0b33. This is evidence of a tested combination, not a portable
CUDA/driver lockfile. Create a project-specific ML environment specification;
the template does not bundle one.

Run the standard-library suite with `python3 -m unittest discover -s tests -v`.
In a compatible ML environment, `python -m unittest discover -s tests/ml -v`
runs offline CPU tests, including real tiny-model save/resume/evaluate workflows.
GPU probes require explicitly allocated free GPUs; see TRAINING.md.

## New-project portability review — 18 October 2026

The active task/notes were reset to unconfigured project placeholders. Local
credentials, populated temp, old run history and five corpus-specific or
incompatible legacy utilities were archived outside the template. The README
now documents a clean copy and the tested warmup-option precedence. Training,
packing, SDPA/FA4, resume and evaluation core files are byte-identical to the
B200-tested snapshot; no remote job was launched for this review.

Local verification passed: 37 utility tests, all 42 ML CPU tests, the complete
train/resume/evaluate/summary integration, a clean-copy CPU workflow, syntax,
JSON, documentation links and credential scan (excluding a synthetic test URL).
The template remains unconfigured, with commands.sh inactive and no credentials.
