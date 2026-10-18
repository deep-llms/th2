# Generic job orchestration

`run_experiments.py` is a sequential runner for any project-specific executable.
It has no LLM dependency, implicit dataset, GPU burn, training schedule, or
model-specific checkpoint knowledge. Python 3.11+, POSIX are required.

## Manifest

Use `jobs.example.json` as the smallest working example. A config has one
top-level `jobs` list. Each job declares:

| Field | Meaning |
|---|---|
| `name` | Unique letters/digits/underscore/hyphen identifier |
| `argv` | Command argument list; executed without an implicit shell |
| `gpus` | Optional explicit physical GPU index list; absence selects CPU mode |
| `timeout_seconds` | Optional positive wall-clock limit |
| `required_outputs` | Nonempty list of expected nonempty files under the run root |

Each output has `path` and optional `sha256` and/or `json_equals` (expected
top-level JSON fields). Outputs are relative, confined to the run directory,
and cannot be runner metadata/logs. A nested directory or sharded checkpoint
needs a project validator job that checks its contents and writes a validation
JSON; directory existence is not sufficient.

Each declared output must be absent immediately before its job starts. The
runner refuses an existing file, directory, or dangling symlink; it never
deletes an earlier stage's output to make room. Use distinct paths for each
stage, such as `train/summary.json` and `eval/results.json`. A validator may
read an earlier artifact as input, but must declare its own newly written
validation report as output. Reusing or overwriting a declared output path
between jobs is not supported.

Only these literal placeholders are replaced inside argv elements:
`{python}` (current interpreter), `{project_dir}`, and `{run_dir}`. No implicit
shell/environment expansion takes place. To use a shell script, explicitly
invoke `bash scripts/your_job.sh ...`; that script must stay in the foreground,
propagate failures, and not detach/background the workload. Never put secrets
in argv/config: the job manifest is saved with the run. Pass credentials only
through operator-managed environment variables or ignored local files.

```bash
python3 run_experiments.py --config jobs.example.json --list
python3 run_experiments.py --config jobs.example.json \
  --project-dir "$PWD" --run-dir temp/example-run-001
```

`--list` does not create logs or execute jobs. `--run-dir` must not exist.
The runner creates its directories before logs, records `jobs.snapshot.json`
and `run.json`, and writes each job's stdout/stderr to `<name>.log`.
It stops at the first failure and exits nonzero. Existing run directories
are never overwritten or deleted automatically.

CPU jobs get `CUDA_VISIBLE_DEVICES` empty. GPU jobs use `CUDA_DEVICE_ORDER=PCI_BUS_ID`
and their declared physical indices, with fail-if-busy checks before launch and after successful
completion. These checks do not lock a shared allocation, enforce a sandbox,
or identify authorized existing workloads; coordinate ownership externally.
The runner never tries to free occupied GPUs by killing unrelated processes.

On timeout or interruption, only the child session started by this runner is
eligible for termination. Do not use it to wrap operator sleepers, tmux servers,
detached workers or other infrastructure. The post-job GPU check may fail if
CUDA cleanup is still in progress; inspect/recheck before proceeding rather
than falsely publishing completion or killing residual processes blindly.

## Completion contract

`complete.json` appears only after all jobs have passed their output-freshness
preflight, exited zero, passed declared artifact checks, and GPU jobs passed
their post-job free check.
It contains UTC times and artifact size/hash records. `run.json` records a
failure instead. No marker is written on missing files, bad hashes/JSON,
nonzero exit, timeout, or interruption.

This generic check cannot judge scientific validity or detect stale external
inputs. Run roots are fresh, but your validator must still check dataset
provenance, checkpoint step, metrics, expected tasks/seeds and no NaNs as
appropriate. Do not declare only a log file and call that validation.

For multi-stage workflows, add train → validate → eval → validate → downstream
jobs explicitly. The optional train.py baseline and eval/packed_ppl.py held-out evaluator are
available. Downstream benchmarks and finetuning must be implemented for the
new project; no legacy benchmark suite is bundled.
The included train.py supports --max_steps for the full schedule and --stop_after
for a graceful optimizer-step cutoff with a full checkpoint save. For other
trainers, implement and verify an equivalent cutoff. A wall-clock timeout is a
failure condition, not successful training completion.

## Training → resume → evaluation example

`jobs.training.example.json` connects the baseline and packed evaluator to the
existing runner. It is a **CPU example for tiny local assets**, not a GPU launch
configuration or a useful full-model training recipe. No dependencies, data or
models are downloaded. In a copy of the JSON, replace all occurrences of:

- `/absolute/model/config.json`: a local full-attention Qwen3 config.
- `/absolute/tokenizer`: its matching local tokenizer directory.
- `/absolute/data/train`: local training text (Parquet or saved Dataset).
- `/absolute/data/validation`: separate held-out text in the same supported format.

Run with the interpreter containing the baseline's ML dependencies:

```bash
python run_experiments.py --config /absolute/configured-jobs.json --list
python run_experiments.py --config /absolute/configured-jobs.json \
  --project-dir "$PWD" --run-dir /absolute/outputs/fresh-run-id
```

The six foreground jobs are:

1. Train with the full 8-step LR schedule, stopping at step 2. Periodic validation
   runs every 2 optimizer updates. The checkpoint state must report step 2 of 8.
2. Check the expected cutoff/schedule, root/checkpoint agreement, model/shards,
   tokenizer, optimizer, scheduler, all ranks' RNG files, CUDA FP16 scaler when
   applicable, and finite metrics. Publish `checks/train.json` only on success.
3. Resume explicitly from checkpoint 2 to step 4, retaining the 8-step schedule.
4. Verify step 4 and its explicit resume source; publish `checks/resume.json`.
5. Load checkpoint 4 and run actual packed held-out evaluation, writing
   `evaluation.json` with loss/PPL/counts and provenance.
6. Check the report chain, unchanged checkpoint evidence, and final evaluation's
   checkpoint/data/weight metadata and metric consistency. If training monitored
   validation, require the same held-out data, packing and selection. Write
   `summary.json` and `summary.md` with stage steps, validation loss/PPL and time.

The generic runner and native HF training/restore logic are unchanged. A failed
verification stops the queue before later work and prevents `complete.json`.
Each stage declares a fresh artifact. Training overwrites its normal run-root
metrics during resume, so verification reports snapshot each stage first.
Keep both cutoff checkpoints until summarization finishes; the example leaves
checkpoint retention unlimited and uses `save_strategy=no` (the cutoff callback
still saves complete checkpoints). Summaries do not choose a winning model.

Times are `train_runtime` per Trainer invocation, including periodic validation;
they exclude preprocessing, final root-model saving, and standalone evaluation.
They are stage durations, not repeated totals for resumed training. The summary
labels the actual validation step, which may precede the cutoff if evaluation
intervals do not align. Use held-out loss/PPL for comparisons, not the native
resume invocation's reported average training loss.

The validator checks nonempty binary files and their size/mtime; JSON metadata
also has SHA256 checks. It does **not** deserialize optimizer tensors, hash all
large weights or prove scientific validity. Native resume/evaluation loads the
actual states. Preserve immutable inputs/checkpoints and trust the original
training process, not arbitrary user-edited reports.

When adapting the example, keep both training commands' shared settings identical.
Update cutoff numbers, checkpoint paths and validator expectations together.
To use authorized GPUs, add explicit `gpus` lists to training/evaluation jobs,
remove CPU mode, set matching precision and use the verified Accelerate launch
configuration. Validator/summary jobs remain CPU-only. This example neither
allocates GPUs nor stops/starts burns. Do not copy CPU smoke batch/step settings
into a production experiment accidentally.

The existing runner requires a fresh run directory. It does not automatically
restart a failed queue or delete partial outputs; preserve evidence and configure
an explicit recovery instead. No new queue-resume mechanism is added here.

Real integration check (offline, tiny CPU model and synthetic text):

```bash
python -m unittest discover -s tests/ml -p test_training_workflow.py -v
```

It runs this exact manifest with local test paths, verifies all six jobs, compares
resumed weights with uninterrupted training and checks that retrying the existing
run directory is rejected without changing results. Standard-library tests also
cover missing state/shards, changed artifacts, invalid metrics/evaluation identity
and stopping the queue after a validator failure.

## File manifests

Create a trusted manifest from an explicit list of known files (on the dev
machine or on existing validated inputs):

```bash
python3 scripts/verify_manifest.py create --root data/raw \
  --manifest temp/raw-manifest.json --files shard01.parquet shard02.parquet
python3 scripts/verify_manifest.py verify --root data/raw \
  --manifest temp/raw-manifest.json
```

Manifest creation refuses to overwrite a file. Verification checks SHA256,
size, missing files and escaping paths. `--strict` also rejects extra files;
keep the manifest outside the verified root when using that option. Hashes
must come from a trusted producer/source or verified download, not merely be
generated from possibly damaged data and treated as independent proof.
Small non-sensitive manifests may be committed to `resources/`; actual data
does not belong in Git. File-byte identity does not prove sample quality,
absence of train/eval leakage, or validity of a model's output.
