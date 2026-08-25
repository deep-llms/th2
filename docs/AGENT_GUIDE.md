# Agent Guide

How this project is structured, how to work in it, and what to keep in mind.

## Recommended reading order

1. `docs/AGENT_GUIDE.md` — current infrastructure and working rules.
2. `docs/PROJECT_NOTES.md` — durable experiment history, results, and decisions.
3. `docs/CURRENT_TASK.md` — the active foreground experiment and next actions.
4. `docs/commands.md` — exact remote-runner syntax and monitoring workflow.
5. `docs/GIT_PUSH.md` — branch/worktree-to-remote mapping and push safety.

Dated design and implementation documents preserve historical plans. They do
not override the active-machine state in this guide or `CURRENT_TASK.md`.

## Project Structure

```
sparse_embedding/
├── commands.sh                          # Remote runner — submit jobs to training machine
├── .gitignore
├── prepare_data.py                      # Download + sample CulturaX multilingual data
├── train.py                             # HF Trainer training script (needs custom model wrapper)
├── run_experiments.py                   # Sequential experiment runner with GPU management
│
├── eval/                                # Evaluation pipeline (generic, no project-specific deps)
│   ├── ppl.py                           # Perplexity (sliding window)
│   ├── benchmarks.py                    # Benchmarks via lm-evaluation-harness
│   ├── eval_checkpoint.py               # Single checkpoint: PPL + benchmarks
│   └── eval_parallel.py                 # Parallel eval across GPUs
│
├── scripts/
│   ├── setup_env.sh                     # Legacy/manual bootstrap; current runner uses mode #i
│   ├── dropbox_downloader.py            # Dropbox API helpers for remote logs/results
│   ├── push_all.sh                      # Legacy multi-remote helper; do not use while th2-only
│   └── train_qwen3_0.6b_baseline.sh     # Baseline training launch template
│
├── resources/
│   └── accelerate_config.yaml           # 8-GPU accelerate config
│
├── docs/
│   ├── AGENT_GUIDE.md                   # This file
│   ├── PROJECT_NOTES.md                 # Single source of truth for project background + results
│   ├── CURRENT_TASK.md                  # What we're working on right now
│   └── commands.md                      # Remote runner syntax
│
└── temp/                                # GITIGNORED — results, credentials, scratch
    ├── dropbox_credentials.txt          # Dropbox app key/secret/refresh token
    └── dropbox_folders.txt              # Current th2 and any retained legacy shared links
```

## Machines

### Dev machine (where you run)

Code development and testing. This is the machine Claude Code sessions run on.

- **GPUs:** 4× A100 (40GB) — use for testing, not training
- **Conda envs:** `sparse_emb` (main), `fasttext_env`, `eval`

### Active training machine: th2 (one 8× B200 node)

As of 2026-08-22, **th2 is the only active remote training machine**. It is one
node with 8× NVIDIA B200 GPUs (183,359 MiB reported per GPU). Submit training, evaluation,
downloads, environment installs, and GPU checks only to th2 unless the user
explicitly announces another active machine.

- **Git target:** local branch `h100-1` → remote `second` (`deep-llms/th2`) →
  remote branch `main`. The branch name is historical; it now targets B200.
- **Runner filesystem:** code at `/mnt/local/<owner>_<repo>/`, datasets at
  `/mnt/local/_data/<project>/`, and models at `/mnt/local/_models/<project>/`.
  In `commands.sh`, use `@PROJECT@` so the runner substitutes the project name.
- **Conda envs:** `/mnt/local/conda/envs/<env>/`; install them with runner mode
  `#i` and verify the Python path before launching a long job.
- **HF_TOKEN:** already exported by default on the training machine before any command runs. No need to set it in `commands.sh` or worry about HuggingFace authentication for downloading models/datasets on the training machine.
- **Access:** there is no direct SSH workflow. Use `commands.sh`, then inspect
  the th2 Dropbox status/log/result files.
- **How to run things:** see `docs/commands.md` for the current runner syntax.

### Inactive historical setup: th3 and older H100/H200 machines

The `h100-2` branch, `third` remote, th3 Dropbox entry, old two-machine
staggering rules, and `/opt/dlami/nvme/` paths are retained for historical
experiments and possible future reactivation. **th3 is currently inactive and
its old Dropbox link may be stale. Do not push jobs to `third`, include th3 in
availability claims, or use `scripts/push_all.sh` unless the user explicitly
reactivates th3 and provides/validates its current runner and Dropbox details.**

### HuggingFace token

- **Training machine:** `HF_TOKEN` is pre-exported. No action needed.
- **Dev machine:** read from `temp/HF_TOKEN.txt` (gitignored). To use it: `export HF_TOKEN=$(cat temp/HF_TOKEN.txt)`

## Remote status and Dropbox retrieval

The remote runner uploads status, logs, and requested files to Dropbox. The
canonical local credential/configuration files are inside the gitignored
`temp/` directory:

- `temp/dropbox_folders.txt`: the active th2 URL is labelled `h100-1`. An
  `h100-2`/th3 entry may remain for history, but must not be assumed current.
- `temp/dropbox_credentials.txt`: Dropbox `app_key`, `app_secret`, and
  `refresh_token`, automatically read by `scripts/dropbox_downloader.py`.

Never print, commit, or paste the credential-file contents into commands or
logs. The shared-folder URLs should also remain in the gitignored temp file.
Before using Dropbox, verify the credential file exists without displaying it:

```bash
test -s temp/dropbox_credentials.txt
test -s temp/dropbox_folders.txt
```

### Check uploaded status and logs

Extract the appropriate machine URL and list the folder:

```bash
TASK_DROPBOX_URL="$(sed -n 's/^h100-1: //p' temp/dropbox_folders.txt)"  # th2
python scripts/dropbox_downloader.py list "$TASK_DROPBOX_URL"
```

Root files include `_RUN_STATUS_.log` and timestamped
`_run-...-<job-name>.log` files. Match the th2 job name, timestamp, hostname,
commit, and expected command marker before trusting a result. Only use an
`h100-2` URL after th3 has been explicitly reactivated and its link revalidated.

A missing Git result commit does **not** mean the runner failed: first check the
machine's Dropbox folder. Mode `#1 +W+a` waits `W` seconds and uploads the log;
it does not necessarily create a new commit in the local repository.

### Run-system and infrastructure errors

If `_RUN_STATUS_.log` reports an AWS, remote-runner, controller, credential
refresh, upload/synchronization, or other run-system error, treat it as an
infrastructure failure rather than a defect in this project's code or in
`commands.sh`. For example, `Failed to force refresh the credentials` is a
run-system error even when it appears beside a command commit.

Do not modify the experiment code or command, repeatedly resubmit mode `#1`,
kill processes, or clean outputs in response to such an error. Report the exact
error, machine, job name, commit, and timestamp to the user, state that the
requested machine state or job result could not be freshly verified, and wait.
The user will repair the run system and explicitly say when it is ready to try
again. Only attribute a failure to our code or command when the delivered job
log shows that the command actually ran and failed inside the project.

### Download one specific file

For the current shared-folder links, the high-level
`dropbox_downloader.py download --path ...` route asks Dropbox for an unsupported
recursive listing. Use the script's existing direct-download helpers instead:

```bash
python - "$TASK_DROPBOX_URL" "/_run-...log" "temp/remote_logs/th2/job.log" <<'PY'
import sys
from pathlib import Path
from scripts.dropbox_downloader import clean_shared_link, download_file, get_token

shared_url, remote_path, output_path = sys.argv[1:]
output = Path(output_path)
output.parent.mkdir(parents=True, exist_ok=True)
if not download_file(get_token(), clean_shared_link(shared_url), remote_path, output):
    raise SystemExit(1)
print(f"saved {output} ({output.stat().st_size} bytes)")
PY
```

This uses the refresh-token credentials automatically and does not require
modifying `dropbox_downloader.py`.

### Current th2-only safety and legacy multi-machine rules

1. Give every active command a unique `th2-...` job name.
2. Push only `h100-1:main` to `second`; do not push `h100-2`/`third` while th3
   is inactive.
3. Do not re-push an unchanged mode-`#1` command to refresh its log; that runs
   the command again. List/download the Dropbox log, or submit a new uniquely
   named mode-`#2` pull.
4. `scripts/push_all.sh` still pushes origin, th2, and th3. It is therefore
   unsafe for the current th2-only setup and must not be used.
5. If a second machine is reactivated later, restore machine-unique `th2-...`
   / `th3-...` names, separate Dropbox links, and 2–5 minute staggered pushes.
   Verify critical files by content and checksum.

## Documentation — keep it updated

You need to maintain and update project documentation to track progress across sessions. This project tends to exhaust the context window, so documentation is the only way a new session can understand what was done, what worked, what didn't, and what to do next. We use two files:

### `docs/PROJECT_NOTES.md`

Single source of truth across sessions. A new session starts by reading this file. Contains:
- TL;DR of current state
- What the project is and why
- Timeline of what was tried and what the results were
- Code structure
- Data pipeline, training, evaluation details
- Machine setup

**Update this file** when asked. Distill findings, results, and conclusions into the timeline. This is the permanent record.

### `docs/CURRENT_TASK.md`

The active foreground work. Contains:
- What we're doing right now
- Step-by-step progress
- Decisions made
- Dead ends (so we don't repeat them)

**Update this file** when asked. When a task finishes, move its conclusion into `PROJECT_NOTES.md` and clear this file for the next task.

## Implementation Guidelines

### What matters most

1. **Correctness.** The code must work correctly. Test it before considering it done.
2. **Clarity.** Clean, readable code that's easy to maintain and understand.
3. **Standard patterns.** Follow established conventions (HuggingFace, PyTorch).

### Technical preferences (not restrictions)

These are starting points, not rules (this is how I code/run/setup in my other previous projects). Choose whatever is cleaner and more correct for the task at hand.

- **HuggingFace Transformers** as the base. Use `AutoModelForCausalLM`, `Trainer`, standard HF patterns.
- **`register_forward_hook`** for injecting custom layers into existing models — avoids modifying model source code, keeps the base model unchanged if able or no necessary to change.
- **Separate model wrapper** (`model_wrapper.py` or similar) for inject/save/load of custom components. Keep the custom layer definition in its own file.
- **`accelerate launch`** for multi-GPU training.
- **Architecture-independent evaluation.** Eval scripts should work with any model checkpoint — load the model, run forward pass, measure. Don't bake architecture-specific logic into eval code.

### Testing

Write tests for new code and run them on the dev machine before considering the work done. Use a tiny model config for fast iteration (e.g. 2-4 layers, small hidden dim). The dev machine has 4× A100 GPUs — use them. Test the full pipeline: imports, model creation, forward pass, save/load round-trip, and the actual logic (correct outputs, edge cases).

### Git

Git operations (commit, push) are handled in a separate session. Don't commit or push unless explicitly asked.

### What NOT to do

- Don't hardcode paths. Use CLI arguments with sensible defaults.
- Don't store secrets in code or tracked files. Use environment variables or
  the project's designated gitignored credential file in `temp/`.
- Don't skip testing. Run the code on the dev machine (even with a tiny model) before deploying to the training machine.
- Don't leave stale documentation. If the code changes, update the docs.

## Sequential Experiment Runner

`run_experiments.py` runs experiments one at a time with GPU cleanup between runs. An "experiment" is a variant or version of a model architecture, a different hyperparameter setting, or an idea to test — each is a training run that produces checkpoints to evaluate and compare.

Edit `EXPERIMENT_COMMANDS` to define experiments:

```python
EXPERIMENT_COMMANDS = [
    {
        "name": "baseline",
        "cmd": "accelerate launch train.py --config_name Qwen/Qwen3-0.6B --bf16 --output_dir /path/to/outputs/baseline",
        "output_dir": "/path/to/outputs/baseline",   # optional
        "monitor_csv": "smoke_metrics.csv",                  # optional
    },
]
```

Only `name` and `cmd` are required. `output_dir` + `monitor_csv` enable auto-stop at a target step (`--stop-at-step`).

## Eval Pipeline

Generic — works with any HF checkpoint:

```bash
# Single checkpoint
python eval/eval_checkpoint.py --checkpoint /path/to/ckpt --eval-dir /path/to/eval --bf16

# Multiple checkpoints in parallel (one per GPU)
python eval/eval_parallel.py --checkpoints ckpt1 ckpt2 ckpt3 --eval-dir /path/to/eval --bf16

# PPL only / benchmarks only
python eval/eval_checkpoint.py --checkpoint /path/to/ckpt --eval-dir /path/to/eval --bf16 --ppl-only
python eval/eval_checkpoint.py --checkpoint /path/to/ckpt --bf16 --bench-only
```

## Data Pipeline

`prepare_data.py` downloads and samples CulturaX:

```bash
python prepare_data.py download sample --tokenizer-name Qwen/Qwen3-0.6B --num-workers 4
```

Produces `data/{tokenizer}/train/{lang}/` and `data/{tokenizer}/eval/{lang}/` with HF Dataset format. Train and eval are non-overlapping, sampled sequentially from the same document ordering.
