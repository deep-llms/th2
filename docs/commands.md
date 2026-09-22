# Remote runner command guide

Project-neutral runner reference. Confirm the actual operator specification in
local `temp/INSTRUCTION.md` before first use. Never push that file. The syntax
below describes the runner used to develop this template; another runner may
have different limits, log locations or capabilities.

Protocol reference reviewed: **2026-09-06**. An operator instruction file can
be an older snapshot. Preserve later explicit operator updates, including
asynchronous downloads, `#d list` / `#d stop`, and single-file model URLs;
do not remove those capabilities merely because an older file omits them.
Use the latest confirmed operator specification for parser behavior and these
project guides for project policy. Record the source/date of differences in
`PROJECT_NOTES.md`; ask the operator before acting on an unresolved conflict.

## Control file

The execution repository contains `commands.sh`. Pushing that repository can
trigger its action. Leave the template at `#0` until a job is configured.

| Line 1 | Action |
|---|---|
| `#0` | No action |
| `#i envs/runtime.txt` | Install environment(s) |
| `#i envs/runtime.txt +a` / `+N` | Install, keep full / last N log lines |
| `#d` | Start dataset/model downloads |
| `#d +a` / `+N` | Downloads with full / last N saved log lines |
| `#d list` | List active background download IDs |
| `#d stop <id>` / `#d stop all` | Stop specified background download(s) |
| `#1` | Sync code and run shell body |
| `#1 +60` | Run; request a log snapshot after 60 seconds |
| `#1 +60+200` / `#1 +60+a` | Run; snapshot last 200 lines / full log |
| `#2` / `#2 +a` / `#2 +N` | Pull latest log, last 500 / all / last N lines |
| `#2 -1` / `#2 -0-3` | Previous log / offsets 0, 1, 2 (exclusive slice end) |
| `#2 -f-<paths>` | Export exact files/folders to the configured result share |
| `#3` | List available logs and indices |

Modes are exclusive within one revision. For `#1`, `#2`, `#3`, line 2 is a
unique job name starting with `#` in column 1. For `#1`, shell starts on line 3.
`#2`/`#3` do not execute a shell body. `#i` names its env files on line 1.
For `#d`, directives begin on line 2 and end at the **first blank line**.

Text after ` #` on line 1 is ignored by the parser but can still retrigger a
job when changed/pushed. Never repush `#1` just to refresh a log. A log-snapshot
delay is not a timeout or a signal to stop a long job.

## Paths and offline execution

- Code/cwd: `/mnt/local/<owner>_<repo>/`.
- Raw/prepared data: `/mnt/local/_data/<project>/`.
- Models: `/mnt/local/_models/<project>/`.
- Outputs/logs/checkpoints: `/mnt/local/_outputs/<project>/` (or another
  operator-approved output root outside the Git-synced source tree).
- `@PROJECT@` in `commands.sh` expands to the runner's `owner_repo` identifier.
  `$PROJECT` is exported for called scripts. Helpers/JSON files are not assumed
  to receive literal `@PROJECT@` substitution; pass resolved CLI paths.
- Python 3.11 conda: `/mnt/local/conda-py311/envs/<env>/bin/python3.11`.
- Python 3.10 conda: `/mnt/local/conda/envs/<env>/bin/python`.
- uv: `/mnt/local/uvenvs/<env>/bin/python`.

Verify these paths on a new node. Do not inherit old `/opt/dlami/nvme` paths,
fixed GPU counts or another project's environment name. Local node storage is
temporary; coordinate backup of large checkpoints before node replacement.

Normal `#1` jobs may have no public internet access. Acquire external inputs
through controller `#d`, then load local data/config/tokenizer/model paths.
A token exported inside `#1` does not prove that controller downloads have
credentials. Give private/gated access credentials to the operator, not Git.

## 1. Install environments

An env specification's basename becomes the environment name: this template's
`envs/runtime.txt` installs `runtime`. Retain the headers:

```text
# manager: conda
# fresh: true
# python: 3.11
pip
```

Add project dependencies one per line. `-package` requests uninstallation.
An exact prebuilt `.whl` filename can request an operator-provided wheel;
verify its Python/PyTorch/CUDA compatibility. Do not commit wheel binaries.
Do not assume a historical exact torch pin resolves on the current internal
index. `fresh: true` rebuilds an env; do not use it on an env a live job needs.

```bash
#i envs/runtime.txt +a
#project-install-runtime-YYYYMMDD-a01
```

Multiple env files are supported, e.g. `#i envs/runtime.txt envs/eval.txt +a`
after creating the second specification. Inspect each item's outcome, not only the
overall summary. Then verify the exact interpreter/imports with `#1`:

```bash
#1 +60+a
#project-verify-runtime-YYYYMMDD-a01
set -euo pipefail
TASK_PYTHON=/mnt/local/conda-py311/envs/runtime/bin/python3.11
test -x "$TASK_PYTHON"
"$TASK_PYTHON" -c 'import sys; print(sys.version); print(sys.executable)'
```

For scripts requiring activation:

```bash
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate runtime
test "$CONDA_DEFAULT_ENV" = runtime
command -v python
```

Verify actual project imports and CUDA support separately before GPU jobs.
Generic utilities do not require torch, transformers, fastText or lm-eval.

## 2. Download inputs

The following are examples, not a selected dataset or model for your project:

```bash
#d +a
#datasets
--hf-dataset org/dataset /mnt/local/_data/@PROJECT@/raw
--url https://example.com/shard.parquet /mnt/local/_data/@PROJECT@/raw
#models
--hf org/model /mnt/local/_models/@PROJECT@/model
```

There must be no blank line between sections. Dataset item formats are
`--hf-dataset <repo> <dest>`, `--url <url> <dest>`, or a controller-local
`<path> <dest>`. A single HF dataset file can use a direct
`https://huggingface.co/datasets/<org>/<repo>/resolve/<revision>/<file>` URL;
the revision is a branch/tag/commit, not a token. Prefer immutable revisions
or a trusted file manifest when exact reproducibility matters.

Model formats are `--hf <repo> <dest-dir>` or `--url <url> <dest-file>`.
For the latter, give the exact destination file, not a directory. Single-file
model URLs are copied directly to the pod by the controller, without S3 staging.

For gated/private HF content, the preferred credential is supplied **once to
the operator's monitor**, using its `-hf <token>` startup option or `HF_TOKEN`
environment variable. These are monitor settings, not `#d` item syntax or a
`#1` shell setup step. The configured credential is auto-attached as an
authentication header only to `huggingface.co` URL requests. A local
`temp/HF_TOKEN.txt` can hold the credential, but the runner does not
automatically read that dev-machine file.

The runner also supports per-item `--hf-token <token>` in a **private**
repository. This is a documented exception, not the template's default:
require explicit approval and confirmed private repository/log access before
using it, because the token would persist in Git history and logs. Never put a
real token in examples, ordinary download lines, or a public repository.
Prefer operator-managed credentials outside Git; report authentication failure
to the operator rather than silently switching to an embedded token.

`#d` is asynchronous. A later `#1` or `#i` may run while downloads continue,
but must not consume unfinished data. `#d list` writes download IDs to a
`_download_tasks_<time>.log`; use an ID for an explicitly authorized stop.
One failed item does not necessarily stop the other items. Require each
needed item to report successful completion (e.g. `OK | download:`), then
verify the actual files, hashes and layout on the node.

For local file identity checks:

```bash
python3 scripts/verify_manifest.py verify \
  --root /mnt/local/_data/@PROJECT@/raw \
  --manifest resources/data_manifest.json
```

`resources/data_manifest.json` must be created for the actual project; it is
not supplied by the template. Sampling/preprocessing must be offline, seeded,
and produce validated outputs. Do not infer completion from directory existence.

## 3. Run approved work

Use unique run IDs and fresh output roots. This CPU example uses no GPU:

```bash
#1 +60+a
#project-cpu-example-YYYYMMDD-a01
set -euo pipefail
date -u
hostname
git rev-parse HEAD
TASK_PYTHON=/mnt/local/conda-py311/envs/runtime/bin/python3.11
test -x "$TASK_PYTHON"
"$TASK_PYTHON" run_experiments.py --config jobs.example.json \
  --project-dir "$PWD" \
  --run-dir /mnt/local/_outputs/@PROJECT@/cpu-example-YYYYMMDD-a01
```

For GPU work, first follow `GPU_SAFETY.md`, declare physical GPU indices in
the job manifest, and verify data/model/env/output settings. Honor requested
wait/recheck intervals; after copying the actual distributed config, compare
its bytes and check availability again immediately before launch.

The optional Accelerate config in this template is only a single-process,
no-mixed-precision example. If the project needs Accelerate, configure and
validate its GPU count and precision first, then explicitly pass its path to
`accelerate launch --config_file ...` or copy it to the interpreter's actual
HF Accelerate cache path. On the known runner this is commonly
`/mnt/local/.cache/huggingface/accelerate/default_config.yaml`. Do not copy the
unmodified example as an eight-GPU training configuration.

The generic job runner supports wall-clock timeouts, not trainer-specific
step counting. Implement a graceful stop-at-step in the actual trainer,
including checkpoint save and verified final step. Do not change a production
optimizer schedule with `max_steps` merely to impose a screening cutoff.
Do not send SIGTERM based solely on a metric CSV and call that a saved checkpoint.

## 4. Monitor and retrieve results

```bash
#2 +a
#project-pull-latest-YYYYMMDD-a01
```

```bash
#3
#project-list-logs-YYYYMMDD-a01
```

Request exact small files, using the **resolved** project identifier for `#2`
paths rather than assuming placeholder expansion in this mode:

```bash
#2 +a -f-/mnt/local/_outputs/OWNER_REPO/RUN_ID/complete.json,/mnt/local/_outputs/OWNER_REPO/RUN_ID/run.json
#project-pull-results-YYYYMMDD-a01
```

Replace `OWNER_REPO` and `RUN_ID`. A trailing `/` requests a folder; no trailing
slash requests a file. `+a` avoids truncated JSON/log files. A bare filename
without `/` refers to the runner's log directory. File requests take priority
over log-index selectors. Missing paths may be skipped while others succeed.

`#2` exports to the runner's configured share; it does not automatically place
files in this dev checkout. Use `DROPBOX_ACCESS.md` to list/download them and
verify SHA256 plus expected result counts. Require current timestamps and the
matching job/commit. A successful `#2` response does not prove the workload succeeded.

The known operator limits are 25 MB per pulled file/log, last 20 log items,
last 10 pulled folders, and up to 5 `#2` pulls per 10 minutes / 20 per hour.
Confirm the current limits in `temp/INSTRUCTION.md`. Export compact results
instead of trying to pull multi-GB checkpoints through this channel.
Keep the pushed source repository below the known runner's 25 MB code limit;
confirm that limit with the operator too. Never upload from the GPU node via
HF, Git, scp/rsync, or S3 as a workaround.

## Failure and retry policy

- Do not relaunch just because a snapshot lacks a completion marker. Check the
  correct job log/status and allow the documented task to finish.
- If AWS/controller/runner/credential-refresh fails, report it and wait for the
  operator. Do not kill, clean or change project code to "fix" infrastructure.
- If a project command fails, preserve its log/output and fix that diagnosed
  failure. A retry gets a new run ID or a deliberately tested resume path.
- A next stage starts only after successful exit, project-specific validation
  and required GPU checks. Failures must not publish a success marker.
