# Downstream evaluation of the completed proxy screen

Scope: A plus all ten completed P4–P7 variants at checkpoint-2500, seed 42,
FA4, zero-shot, no chat template. Nine English tasks, existing lm-eval 0.4.10
prompts and scoring. One checkpoint per GPU, up to eight concurrently. No training.

Launch: `5ac2fbf`, job `th2-tjx3-downstream-2500-20261007-a01`.
Root: `/mnt/local/_outputs/deep-llms_th2/downstream-2500-20261007-a01`.

## Data and environment

31 pinned files, 104,884,425 bytes, downloaded through controller job `f025c70`.
Location: `/mnt/local/_data/deep-llms_th2/downstream-english-20261007`.
Raw file hashes and repository revisions: resources/downstream_english_20261007.json.
All file hashes and processed document hashes verified on B200 at
16:19:32 Singapore, 7 October. Reference: resources/downstream_english_reference_20261007.json.
No Hub requests are made from GPU-shell jobs. Dataset loading uses explicit
local files because incomplete multilingual repository snapshots cannot rely
on Hugging Face's automatic config discovery. Task prompts/scoring are unchanged.

| Task | Scored examples per checkpoint |
|---|---:|
| hellaswag | 10,042 |
| xnli_en | 2,490 |
| belebele_eng_Latn | 900 |
| xstorycloze_en | 1,511 |
| paws_en | 2,000 |
| piqa | 1,838 |
| arc_easy | 2,376 |
| arc_challenge | 1,172 |
| winogrande | 1,267 |

XNLI uses validation per the installed task template; ARC uses test;
HellaSwag/PIQA/WinoGrande use validation; XStoryCloze uses eval;
Belebele/PAWS-X use test. All training splits are available for template
initialization but zero-shot evaluation uses no demonstrations.

Separate pinned `eval_fa4` environment installed via `4e07abc`. Core packages
match training: torch 2.14.1+cu130, Transformers 5.9.0, datasets 4.8.5,
Accelerate 1.13.0, FA4 4.0.0b33, CUTLASS DSL 4.8.0; lm_eval 0.4.10.
Pip dependency check and FA4 import passed. Training environments untouched.
Accelerate config copied/byte-verified; accelerate env confirms eight processes,
MULTI_GPU, BF16. Evaluation itself uses independent single-GPU processes.

## Acceptance gates and outputs

Local checks: ten existing groups passed; expanded real-template CLI test covers
all nine tasks. Additional local-file/hash-failure test passed after its fixture
was corrected to provide the training split required by the task template.
Real benchmark data/labels/counts validated locally and identically on B200.

GPU sequence: eleven checkpoint numerical checks (training-forward loss,
FA4 execution, document isolation, padding, active proxies, unchanged buffers,
SDPA same-weight logits), then eight examples per task for all eleven models,
then smoke validation, then full evaluation and final completeness validation.
The final validator checks step/backend/data/seed/few-shot settings, all expected
sample counts, finite accuracies and identical document/prompt/target hashes
across checkpoints. Smoke and full outputs are separate.

The existing supervisor stops only reverified authorized burn workers, waits and
requires free GPUs before work. It cleans only owned children and restores
communicating burns on success/failure. At 16:19:32 it had identified/stopped
workers 308656–308663 and was in the mandatory wait; later gate results pending.
Monitor evidence: artifacts/downstream-monitor-20261007-a01/.
Do not treat smoke metrics as scientific results.

## CUDA/smoke acceptance and full launch verified

Monitor `963c243`, 7 October 16:27:02 Singapore, confirms all eleven numerical
gates passed, followed by all eleven real-data smoke evaluations and their
completeness/provenance validator. Maximum evaluation-vs-training loss difference
9.536743164e-07; padding relative L2 zero in all eleven checks; maximum same-weight
FA4/SDPA logit relative L2 0.002962695202. All proxy consumers active, document
isolation checks exact, normalization buffers unchanged. These are adapter
acceptance checks, not task-quality measurements.

Full evaluation began at **16:26:22 Singapore**, with A/P6-iso/P7-simple/P7/
P7-mlp/P7-ems/P7-kq/P6 in the first eight GPU slots; P4/P5/P4-iso queued.
Each checkpoint scores 23,596 examples, nine tasks, no limit, batch size 8.
Smoke used batch size 4 and eight examples per task in separate directories.
Final full metrics and final burn handoff remain pending.

Evidence: artifacts/downstream-monitor-20261007-a04/ (49 hash-verified JSON
artifacts), temp/downstream-monitor-20261007-a04.log SHA256
`1e7ac062a681f59ebafbab728327cb90fb18810d79822f165c4fa994a3606045`.
Full outputs: `supervised/run/full/`; completion validator:
`supervised/run/full-validation.json`; queue marker:
`supervised/run/complete.json`; final burns: `supervised/burn-verified.json`.

## Final completion and results — 7 October 2026

Monitor `764d872` at **16:41:20 Singapore** verifies successful completion of
all 15 stages: eleven numerical gates, all-model smoke, smoke validation,
all-model full evaluation, and full validation. All 99 model–task evaluations
passed completeness/provenance checks, 23,596 scored examples per model
(259,556 total), zero-shot, no limit. Automatic communicating burns verified
at **16:41:13**, and the monitor confirmed the same workers 314038–314045
at 100% utilization on all eight GPUs. No further evaluation is queued.

Pulled/source-hash-verified 66 JSON artifacts to
`artifacts/downstream-monitor-20261007-a08/`. Full metrics are in
`full-validation.json`; per-checkpoint metrics and provenance are under `full/`.
Original per-example JSONL files remain in the B200 full-evaluation folders.
Log SHA256: `67641c3e716ce3c48a8c73fc4f3fc2c8248cb33c8d3cce8d46fd41b675b976bd`.
No model weights, optimizer states or training data were altered.

### Descriptive macro-averages

All values are percentages. Each task receives equal weight. Raw uses `acc`
on every task; the second summary uses `acc_norm` where the harness reports it
(ARC-C/E, Belebele, HellaSwag, PIQA), and `acc` elsewhere. These summaries are
explicit descriptive choices, not a standardized benchmark composite.

| Arm | Raw accuracy mean | Normalized where available mean | Difference from A (normalized, pp) |
|---|---:|---:|---:|
| A | 40.947 | 41.113 | +0.000 |
| P6 | 40.836 | 41.039 | -0.074 |
| P7-ems | 40.788 | 41.026 | -0.088 |
| P7-simple | 40.713 | 40.990 | -0.124 |
| P7-kq | 40.598 | 40.839 | -0.274 |
| P6-iso | 40.773 | 40.807 | -0.307 |
| P4-iso | 40.690 | 40.765 | -0.348 |
| P5 | 40.670 | 40.734 | -0.379 |
| P4 | 40.504 | 40.668 | -0.445 |
| P7-mlp | 40.587 | 40.637 | -0.476 |
| P7 | 40.510 | 40.635 | -0.478 |

A has the highest average under both choices. P6-iso's better held-out LM loss
did not translate into a downstream macro-average improvement in this screen.
There are individual-task gains (for example P7-simple on XNLI), but these are
single-training-seed results across multiple variants/tasks, with no paired
significance or additional-seed confirmation. Several tasks are near chance,
which limits the strength of architectural conclusions from this short run.

### Per-task scores: normalized where available

| Arm | ARC-C | ARC-E | Belebele | HellaSwag | PAWS-X | PIQA | WinoGrande | XNLI | XStoryCloze |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 21.50 | 37.29 | 22.89 | 29.18 | 50.25 | 61.70 | 51.93 | 39.36 | 55.92 |
| P6 | 21.93 | 36.41 | 22.67 | 29.08 | 52.00 | 61.81 | 50.28 | 40.72 | 54.47 |
| P7-ems | 21.16 | 36.78 | 23.11 | 29.09 | 50.55 | 61.26 | 51.85 | 39.96 | 55.46 |
| P7-simple | 21.93 | 37.12 | 23.11 | 28.99 | 47.40 | 61.32 | 50.99 | 42.33 | 55.72 |
| P7-kq | 21.67 | 36.99 | 23.00 | 29.40 | 46.30 | 61.86 | 51.78 | 41.69 | 54.86 |
| P6-iso | 21.42 | 36.95 | 23.11 | 29.13 | 49.30 | 61.37 | 51.14 | 39.84 | 55.00 |
| P4-iso | 21.59 | 36.78 | 23.11 | 28.75 | 47.70 | 60.72 | 50.83 | 42.61 | 54.80 |
| P5 | 21.16 | 37.04 | 23.00 | 29.27 | 48.65 | 60.88 | 50.43 | 41.65 | 54.53 |
| P4 | 21.33 | 36.20 | 22.89 | 29.21 | 47.55 | 61.32 | 50.99 | 41.61 | 54.93 |
| P7-mlp | 21.93 | 36.62 | 23.00 | 28.98 | 46.85 | 61.70 | 51.54 | 39.60 | 55.53 |
| P7 | 21.50 | 36.95 | 23.00 | 29.50 | 49.35 | 61.59 | 51.14 | 37.55 | 55.13 |

### Per-task scores: raw accuracy

| Arm | ARC-C | ARC-E | Belebele | HellaSwag | PAWS-X | PIQA | WinoGrande | XNLI | XStoryCloze |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 17.66 | 39.90 | 22.89 | 28.26 | 50.25 | 62.35 | 51.93 | 39.36 | 55.92 |
| P6 | 18.09 | 38.93 | 22.67 | 28.07 | 52.00 | 62.30 | 50.28 | 40.72 | 54.47 |
| P7-ems | 17.06 | 38.72 | 23.11 | 28.02 | 50.55 | 62.35 | 51.85 | 39.96 | 55.46 |
| P7-simple | 17.41 | 39.98 | 23.11 | 27.94 | 47.40 | 61.53 | 50.99 | 42.33 | 55.72 |
| P7-kq | 16.81 | 40.99 | 23.00 | 28.04 | 46.30 | 61.92 | 51.78 | 41.69 | 54.86 |
| P6-iso | 18.60 | 39.81 | 23.11 | 28.07 | 49.30 | 62.08 | 51.14 | 39.84 | 55.00 |
| P4-iso | 17.41 | 40.11 | 23.11 | 28.11 | 47.70 | 61.53 | 50.83 | 42.61 | 54.80 |
| P5 | 17.06 | 39.81 | 23.00 | 27.88 | 48.65 | 63.00 | 50.43 | 41.65 | 54.53 |
| P4 | 18.09 | 38.72 | 22.89 | 28.17 | 47.55 | 61.59 | 50.99 | 41.61 | 54.93 |
| P7-mlp | 18.52 | 39.77 | 23.00 | 28.18 | 46.85 | 62.30 | 51.54 | 39.60 | 55.53 |
| P7 | 17.83 | 40.07 | 23.00 | 28.16 | 49.35 | 62.35 | 51.14 | 37.55 | 55.13 |
