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
jobs explicitly. Train/eval code is intentionally absent from the template.
To stop at an iteration, implement a graceful project-specific trainer cutoff
that saves and verifies the desired checkpoint. A wall-clock timeout is a
failure condition, not successful training completion.

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
