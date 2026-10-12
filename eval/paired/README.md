# Paired likelihood evaluation (opt-in)

The existing `eval.eval_checkpoint`, `eval.suite`, few-shot, fine-tuning and
PPL commands retain their scoring and defaults. This package adds a **separate**
`python -m eval.paired` entry point. It reuses the existing model loader, full-prefix
harness adapter, portable dataset manifests and lm-eval **0.4.10**. It does not
change a model's proxy computation, training or packing.

The protocol is revision 3 plus the review corrections below, named `paired-v3.1`.
Historical five-shot continuation/Social IQa results are not relabeled as the
new zero-shot results. The original benchmark reporting code still uses its
original rules, including raw SciQ and MMLU-continuation accuracy.

## Protocol

| Suite | Tasks | Shots / scoring |
|---|---|---|
| Z | ARC-E/C, HellaSwag, PIQA, WinoGrande, XNLI-en, XStoryCloze-en, LAMBADA, SciQ, OpenBookQA, Social IQa normalized, CommonsenseQA continuation, MMLU continuation (57 subjects), BLiMP (67 paradigms) | All zero-shot; exact rules in `protocol.RULES` |
| F | Existing eight English tasks | ARC-C 25, HellaSwag 10, remaining tasks 5; descriptive |
| Probes | bAbI qa1/qa2/qa3 | Two fixed demonstrations; story-cluster inference |
| Paloma | All 16 supported sources | Zero-shot rolling likelihood; descriptive BPB reduction |
| Position | First 2,000 qualifying original validation documents | First 2,048 tokens, no appended EOS; six frozen buckets |

M1 is gold-answer log likelihood per character; M2 is the softmax-normalized
choice score; M3 is the gold-minus-best-wrong score margin. M2 is not a calibrated
probability of correctness. ACC retains the declared raw/character-normalized
decision. LAMBADA has M1 and greedy-match ACC only. Positive paired differences
favor P20; loss/BPB reports use positive **reductions** instead.

Suite Z has fixed, equal task weights (subjects/paradigms pooled by example).
H1 uses standardized M1 differences on all 14 tasks; H2 uses M2 on the 13
multiple-choice tasks only if H1 rejects. Tests are two-sided at 0.05, with
10,000 resamples and analysis seed 20261012. Evaluator seeds are all 42.
Admission and the label-symmetric subset remain descriptive. Output includes
unrounded checkpoint means, accuracy, natural effects, standardized effects,
intervals and average rows. These are checkpoint-level comparisons; the report
discloses that previous accuracies/fine-tuning/PPL informed this research.
Admission/filter tables apply to Suite Z only. bAbI uses story-cluster statistics
and does not report independent-question standard errors.

Corrections retained from the review:

- Probe specificity uses a **direct difference of contrasts**, jointly resampling
  the shared A20 reference. Significant versus nonsignificant is insufficient.
- A10→A20 progress ratios are descriptive. An unstable denominator produces
  `estimable: false`, not an extreme numerical ratio. They do not predict effects
  from the validation-PPL ratio.
- Precision samples are seeded without replacement, stratified by MMLU subject.
  Repeatability checks maximum absolute error, mean and RMSE, including M2;
  constant offsets cannot pass simply because their SD is zero.
- Sign-flip inference requires joint pair-swap invariance at the sampling unit.
  Calibration is evidence about simulated cases, not a general validity proof.
- All-zero differences have effect zero and p=1. Constant nonzero differences
  fail because their standardized effect is undefined. Nonfinite standardized
  bootstrap intervals are explicitly unavailable; they are never silently zero.

## Local implementation checks

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -m unittest tests.test_paired_eval -v
```

Tests exercise real 0.4.10 templates/artifacts on local fixtures, metric
reconstruction, strict pairing/completeness, story clusters, exact small-sample
sign flips, joint covariance, numerical gates, a saved tiny P6-iso checkpoint,
and agreement with the native training LM loss. Fixture results are not real
benchmark results or evidence of BF16/FA4 equivalence on B200.

## Prepare and freeze

Run commands in the existing pinned evaluation environment. No package upgrade
is required. Paths below are examples; these commands do not submit runner jobs.

```bash
mkdir -p artifacts/paired
python -m eval.paired freeze --output artifacts/paired/protocol.json

python -m scripts.prepare_expanded_benchmarks \
  --profile paired --output /data/paired-benchmarks \
  --source-lock /data/paired-benchmarks.sources.json \
  --tasks arc_easy arc_challenge hellaswag piqa winogrande xnli_en \
    xstorycloze_en lambada_openai sciq openbookqa social_iqa_norm \
    commonsense_qa_cf mmlu_continuation blimp paws_en \
    babi_cf_qa1 babi_cf_qa2 babi_cf_qa3 paloma

python -m eval.paired calibrate --workers 4 \
  --output artifacts/paired/calibration.json

python -m eval.paired freeze-position --data /data/original-validation/en \
  --tokenizer /models/Qwen3-0.6B --protocol artifacts/paired/protocol.json \
  --output artifacts/paired/position-sample.json
```

Dataset preparation uses the original snapshot exporter with an opt-in task
profile. Source commits, parquet file hashes and task-to-split mappings are
locked. Existing standard-profile behavior is unchanged. Paloma requires the
operator's existing dataset access; this code does not accept licence agreements.

The protocol freezes task definitions **and implementation hashes**. Freeze it
after implementation changes and before opening new paired outputs (including
historical Suite F for analysis). Subsequent implementation changes require a
fresh protocol/study; existing output directories are never overwritten.
Position input accepts one language's original Arrow dataset or sorted parquet
files, never a packed token cache. It fails if 2,000 qualifying documents cannot
be obtained. Document identities and token hashes travel with every loss array.
If an input-only audit finds fewer eligible documents, explicitly freeze the
available count using `freeze --position-documents N` before opening results.
This changes the declared position sample size only; it does not change Suite Z.
Never duplicate documents or silently lower the requested count during scoring.

Full calibration uses 1,000 datasets per case, B=2,000 for T4a and B=1,000 for
the other cases. `--datasets`/`--resamples` can make a smoke check shorter, but
such a report cannot satisfy the study gate. Worker count does not change RNGs.
The ignored local `temp/` logs record development checks; do not substitute
small-fixture checks for production precision validation.

A full passing local calibration is saved in
[`docs/PAIRED_LOGPROB_CALIBRATION_20261012.json`](../../docs/PAIRED_LOGPROB_CALIBRATION_20261012.json).
It can be reused while the recorded statistics/calibration source hashes match;
planning checks this. The correct procedures met the predeclared rejection and
coverage bands. The deliberately invalid example-level clustered control had
28.6% rejection, compared with 5.6% for story-level inference. These simulations
are not measurements of either trained checkpoint's performance.

## Parallel study

```bash
python -m eval.paired plan \
  --protocol artifacts/paired/protocol.json \
  --calibration artifacts/paired/calibration.json \
  --position-sample artifacts/paired/position-sample.json \
  --a10 /checkpoints/A/checkpoint-10000 \
  --a20 /checkpoints/A/checkpoint-20000 \
  --p20 /checkpoints/P6-iso/checkpoint-20000 \
  --dataset-root /data/paired-benchmarks \
  --manifest /data/paired-benchmarks/manifest.json \
  --batch-size 16 --output artifacts/paired/study

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m eval.paired run \
  --plan artifacts/paired/study/plan.json --workers 8
```

Planning checks arm/step identities, requires a full passing calibration, audits
the five custom tasks and freezes precision example IDs before scoring.
Execution checks checkpoint, recipe, dataset and protocol hashes. The CLI defaults
to one persistent worker per GPU in the explicit `CUDA_VISIBLE_DEVICES` allocation;
`--workers 1` retains the serial reference. Each spawned worker sees only its GPU.
Independent checkpoint/task jobs run concurrently, with long groups scheduled
first and preference for each worker's resident checkpoint. Main scoring reuses
that model; precision/repeat jobs deliberately reload to keep checks independent.
Precision jobs finish and pass artifact validation before the main scoring backend
is selected. The three position jobs run concurrently after benchmark scoring.
Task prompts, shots, seeds, batch sizes, example IDs, metrics and statistics are
unchanged. No task uses distributed reductions or changes its batch composition.
`parallel-status.json` records GPU assignment, task time and model loads; each task
has a sibling `.worker.log`. On failure/cancellation only owned worker processes
are stopped. Completed matching outputs are reused;
partial or mismatched outputs stop the queue. Move an incomplete diagnostic
directory aside deliberately before restarting; the runner never deletes data.
The frozen position sample is verified before the first job and again before
position extraction. Reused position outputs must also match the recipe,
tokenizer and batch size.
If Paloma access is unavailable, `plan --defer-paloma` records that descriptive
corpus as deferred. All Z, F, bAbI and position stages still run; a deferred
Paloma stage is never reported as a completed corpus evaluation.

The first stage evaluates both arms twice in production precision and once in
FP32 SDPA on the five frozen precision tasks. Production means the saved backend
(including FA4) with BF16 autocast and FP32 softmax. Reference explicitly disables
autocast and TF32; `--path sdpa-bf16` is also available for attribution tests.
The gate verifies these settings in the saved metadata and requires repeated
production evaluations to use the same numeric settings and attention runtime.
The strict revision-3 variance/bias gates are preserved. If any gate fails,
Suite Z and both mechanism probes use FP32 SDPA; Suite F remains descriptive
production scoring. Passing five tasks is not an all-task numerical guarantee.

The remaining order is F, Z, bAbI, Paloma, position. A10 runs Z/bAbI/position;
A20 and P20 run everything. Position extraction independently checks the exact
same inputs against the training code's LM sum within 1e-4 nats (T8). This
does **not** reproduce packed-validation PPL. A precision or T8 discrepancy
is reported/fails as specified; weights are never changed to make a check pass.
Analysis enforces the study's precision decision for both Z and bAbI. Position
comparisons reject mixed numeric settings and non-finite native-loss checks.

The queue itself does not manage GPU burns, Accelerate configuration, environment
installation, or deployment. Those remain the existing authorized runner's job.
Do not submit this full study merely to test the implementation.

## Saved results and analysis

Each task writes `eval_metadata.json`, `eval_benchmarks_full.json` and
`eval_samples.jsonl`. Metadata includes the empty-context prefix token, backend,
precision, seeds, batch size, hashes and a context-truncation audit. Raw arguments
and log likelihoods remain available for independent recomputation.

The adapter checks counts, the full subtask inventory, logged metric
reconstruction, example IDs, documents, prompt/target hashes, gold indices and
complete arguments. lm-eval 0.4.10 incorrectly reports full effective counts
for `samples=`; only explicit precision subsets get corrected, after checking
their exact requested IDs. Original harness counts are retained in metadata.

```bash
for suite in F Z probes paloma; do
  python -m eval.paired analyze --root artifacts/paired/study \
    --protocol artifacts/paired/protocol.json --suite "$suite" \
    --output "artifacts/paired/${suite}-analysis.json"
done
python -m eval.paired analyze-position \
  --a10 artifacts/paired/study/A10/position \
  --a20 artifacts/paired/study/A20/position \
  --p20 artifacts/paired/study/P20/position \
  --protocol artifacts/paired/protocol.json \
  --output artifacts/paired/position-analysis.json
```

To reuse saved Suite F files, supply `--legacy-f mapping.json` to `plan` and to
the F `analyze` command. The JSON maps `A20`/`P20` to each of the eight task names
and its historical output directory (combined shot-group directories work).
Historical files retain their original provenance; they are never stamped as
prospectively generated. Missing sample logs, hashes, counts or mismatched
prompts require reevaluation, not reconstruction from aggregate accuracy.

For a bounded standalone diagnostic, `evaluate` accepts one `--task`, `--suite`,
`--checkpoint`, `--path`, the frozen protocol and manifest paths, and a fresh
output directory. An optional `--selection` JSON maps every subtask to explicit
document IDs. Such subsets cannot pass the full-suite completeness checks.

Source documents: `docs/downstream_paired_logprob_analysis_spec.md`,
`docs/paired_logprob_review_response_rev3.md`, and
`docs/PAIRED_LOGPROB_REVIEW_AND_PLAN_20261012.md`. Their reference script is
preserved as supplied; this package is the tested project implementation.


The first deployment of parallel scheduling refreshes the frozen execution hashes
before paired planning starts. `scripts/refresh_paired_schedule.py` requires the
queue still be in fine-tuning, checks the exact reviewed source delta, preserves
old protocol/sample files as backups, and rejects any scientific-rule or document
change. The position rows, tokenizer, order and token IDs remain identical. This
is an explicit pre-scoring protocol revision, not reuse of scored results under
new provenance. Existing calibration remains valid because statistics and its
calibration implementation are unchanged.
