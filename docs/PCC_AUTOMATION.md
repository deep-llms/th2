# Sequential PCC experiments

The PCC workflow can now run with one command. `run_experiments.py` can also
queue it alongside other experiments. All stages run in the foreground and
sequentially; no background workers, remote submissions, or automatic retries
are added. Current validation remains local CPU-only; B200 sampling is untouched.

## One PCC experiment

Copy `pcc.pipeline.example.json` to `temp/pcc.pipeline.local.json` and fill in
the exact local model and completed sampled-data paths. The JSON config contains:

- `model_path`: the pinned pretrained snapshot.
- `train_data`: existing English training directory (`train/en`, containing shards).
- `val_data`: existing English validation Dataset directory (`eval/en`).
- `test_data`: optional independent fixed test Dataset; omit for validation-only work.
- `microbatch`: defaults to 1; must divide 16 contexts/update.

Relative input paths are interpreted relative to the config file's directory,
not the working directory. Absolute paths are also accepted. Budgets, seeds,
pair candidates, losses, and scientific gates remain fixed by the contract.

Preview the ordered stages without loading torch, model weights, data, or
querying GPUs. This does not create an output directory or verify input readiness:

```bash
conda run -n train_env python -m pcc pipeline \
  --config temp/pcc.pipeline.local.json --dry-run
```

When sampling has completed and the configured inputs are ready, run:

```bash
conda run --no-capture-output -n train_env python -u -m pcc pipeline \
  --config temp/pcc.pipeline.local.json --output temp/pcc-pipeline-001
```

To check the configured model and data first, use the same command with
`--check-only` and a fresh output directory:

```bash
conda run --no-capture-output -n train_env python -u -m pcc pipeline \
  --config temp/pcc.pipeline.local.json --output temp/pcc-check-001 --check-only
```

Unlike `--dry-run`, this loads the local model/tokenizer, runs its synthetic
correctness preflight, and tokenizes the fixed train/validation datasets. It
checks full-run token budgets, vocabulary bounds, split separation/policies,
and token-identical screen prefixes. It does not train experimental adapters
or inspect test data. Synthetic preflight uses disposable adapters for gradient
checks. A successful check writes `complete.json` with `decision=inputs_validated`;
this establishes input compatibility, not a scientific result or GPU capacity.

Normal pipelines automatically perform the same data checks before starting
the screen. This catches a short full validation split before spending compute
on the four pairs. No separate check job is required. A check-only run has its
own caches; it is not a resumable training run or an export prerequisite.

Without `--check-only`, this performs real scientific adapter training. CPU is the default.
The model is loaded once and shared between the screen and full probe. The
pipeline executes:

```text
screen correctness checks → four-pair training/evaluation → selection
    ├─ negative: save completed negative result; skip full probe
    └─ eligible: fresh teacher → calibration → shallow/PCC students
                 → dev evaluation → conditional permuted control
                 → freeze decisions → test once if supplied and all dev gates pass
```

Without `test_data`, a positive dev result completes as
`validation_complete_test_not_supplied`; it does not claim test confirmation or
recommend scaling. The existing validation split is never reused as test data.

All arms use identical ordered contexts, initialization within each stage,
tokenization, packing, and global token batches. Trainers stop and save at exactly
128 optimizer updates per screen arm and 610 per full-probe arm. The generic
runner's timeout is a wall-clock failure limit, not a fair training cutoff; a
killed process is not a completed experiment.

The screen and probe retain their own correctness checks and artifact validation.
The full probe rederives screen selection from saved paired losses; the pipeline
does not bypass that check or continue trained screen adapter weights.

## Queue multiple experiments with the existing runner

`jobs.pcc.example.json` wraps the entire PCC pipeline as one sequential job.
After configuring the local pipeline JSON, these are the runner commands:

```bash
conda run -n train_env python run_experiments.py \
  --config jobs.pcc.example.json --list

conda run --no-capture-output -n train_env python -u run_experiments.py \
  --config jobs.pcc.example.json --project-dir "$PWD" \
  --run-dir temp/pcc-suite-001
```

The runner uses the same Python interpreter as its parent, so launching it
inside `train_env` also runs PCC in `train_env`. Its `jobs` array is ordered:
add additional explicitly configured experiments or validators after the PCC
entry, with distinct names and output directories. Each job's `argv` is an
argument list, not a shell command. See [JOBS.md](JOBS.md) for the generic schema.

The example verifies `pcc/complete.json` with `status=ok` and
`pretraining_authorized=false`. A **negative scientific result is successful
experiment completion**: it stops dependent PCC stages, but the generic runner
may continue to the next independent job. A software failure, nonzero exit,
missing completion artifact, or failed output validation stops the entire queue.
Do not attach an unconditional dependent scaling/training job after PCC. A
positive PCC result only recommends a separately specified pilot; it does not
authorize or launch one.

The standalone `screen` and `probe` commands remain available if separate
processes are preferable. The pipeline command combines them to avoid repeated
model loading and manual stage handoff.

## Outputs and operational limits

Inside each pipeline output directory:

| Artifact | Meaning |
|---|---|
| `plan.json`, `provenance.json` | Input configuration, planned order, runtime/model identity |
| `input-check.json` | Full/screen input and target counts, source metadata, ordered-context SHA256 fingerprints; test untouched |
| `screen-started.json`, `screen-finished.json` | Screen progress boundaries |
| `screen/` | Screen checkpoints, losses, decision, completion marker |
| `data-cache/` | Internal tokenization/packing/order Arrow caches, shared between stages |
| `probe-started.json`, `probe-finished.json` | Full-probe boundaries, only when selected |
| `probe/` | Full training/evaluation artifacts and development-locked test records |
| `complete.json` | Final scientific decision and completed/skipped stages; written last |
| `failure.json` | Caught pipeline error and stages finished before failure |

Compare `streams.<name>.context_sha256` in input reports to verify identical
model inputs across runs. The hash covers token IDs, masks, positions, and
optional segments in order; it is independent of integer storage width. It
does not authenticate the source corpus or prove that sampling has finished.

When wrapped by the runner, the suite root also contains `run.json`,
`jobs.snapshot.json`, per-job logs (including `pcc-pipeline.log`), and a suite
`complete.json` only after all jobs succeed. Model/config errors can occur before
pipeline creation; the runner still records the nonzero exit and captures stderr.

Every run directory must be fresh. There is no automatic resume, no directory
cleanup, and no retry of failed test evaluation. Creating another output root
must not be used to repeatedly examine the same locked test set or tune against
it. Queue only experiments permitted by the chosen protocol.

For a future authorized local GPU run, standalone PCC uses `--physical-gpu N`.
When wrapping it with the generic runner, declare the same physical index in
the job's `gpus: [N]` and its PCC argv `--physical-gpu N`. PCC's omission means
CPU, even if the parent process has visible GPUs. No GPU allocation is configured
in the example. This documentation is not a B200 launch request.

The pipeline loads fixed sampler outputs directly and tokenizes each split once
per run. Screen inputs are prefixes of the same ordered full-probe stream;
there are no separately exported screen files or new document splits. Cached
Arrow files live under the run directory, leaving source datasets unchanged.
Initial preprocessing covers the entire supplied split before context shuffling
and can take substantial time/disk space. Subsequent stages reuse that cache.
See [PCC_DATA_HANDOFF.md](PCC_DATA_HANDOFF.md) for packing and token-budget details.
Directory existence alone does not establish sampling completion.
