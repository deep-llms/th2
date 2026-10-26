# Efficient LLM architecture

New research project exploring similar model quality with lower inference
latency, compute or KV-cache memory. The architecture is not selected yet.
Start with [PROJECT_IDEA.md](docs/PROJECT_IDEA.md),
[CURRENT_TASK.md](docs/CURRENT_TASK.md) and
[PROJECT_NOTES.md](docs/PROJECT_NOTES.md).

Initialized by copying `/disk/thuat/new_project_template` on 26 October 2026 SGT.
The helper code below is unchanged. This project is separate from B-DOT's dynamic
layer ordering and the earlier deep-to-shallow proxy project.

Reusable runner utilities and an optional Hugging Face causal-LM baseline.
No dataset, model, GPU allocation, repository or experiment is configured.
Copying the folder does not install anything or submit work.

## Start a project

Copy the **clean template folder**, including hidden files and local guides,
into a new directory. Do not copy a used project's credentials, populated
`temp/`, caches, outputs or Git history.

```bash
TASK_NEW_PROJECT=/absolute/path/to/new-project
# Fails if the destination already exists; replace both example paths.
mkdir -- "$TASK_NEW_PROJECT" && cp -a /absolute/path/to/clean-template/. "$TASK_NEW_PROJECT/"
```

1. Add your idea/design document (any filename), then ask the agent to read it,
   [AGENTS.md](AGENTS.md) and [the agent guide](docs/AGENT_GUIDE.md).
   Implement the idea using the existing helper code where applicable; the
   supplied baseline is a starting point, not a required research design.
2. Fill in [CURRENT_TASK.md](docs/CURRENT_TASK.md) with the idea document path,
   objective and next action, and [PROJECT_NOTES.md](docs/PROJECT_NOTES.md) with
   project decisions.
3. Configure repositories and remote resources only when allocated. Keep
   `commands.sh` at `#0` until an execution command is approved.
4. Select dependencies and local model/data inputs. For the remote runner,
   create a project-specific environment file before using `#i`; use `#d`
   for external inputs. See [commands.md](docs/commands.md).
5. Run the CPU example below, then the relevant training tests before real work.

`project.example.json` is an optional checklist. A local copy named
`project.local.json` is ignored by Git, but **no program reads either file**.
You can use `PROJECT_NOTES.md` alone. Actual runtime configuration comes from
CLI arguments, job manifests, environment files and the Accelerate YAML.

## Local checks

These utilities need Python 3.11+ on POSIX, with no ML packages, network or GPUs:

```bash
python3 -m unittest discover -s tests -v
python3 run_experiments.py --config jobs.example.json --list
python3 run_experiments.py --config jobs.example.json --run-dir temp/example-run-001
```

The example verifies a JSON artifact and writes `complete.json`. Existing run
directories are refused; choose a new run ID for retries. It does not train.
For the optional ML suite, activate a compatible ML environment and run:

```bash
python -m unittest discover -s tests/ml -v
```

## Training baseline

- Native HF Trainer/Accelerate for training, optimizer, scheduling and checkpoints.
- Local Parquet or saved HF text datasets; tokenization and packing use HF caches.
- EOS at document boundaries, isolated causal attention and reset positions.
- Dense SDPA by default; optional pinned FA4. The isolated path is validated for
  **full-attention Qwen3 with default RoPE**, not arbitrary model architectures.
- `max_steps` sets the full LR schedule; `stop_after` is an absolute cutoff.
  Native resume restores optimizer/scheduler/RNG/data position under strict checks.
- Periodic and standalone packed validation report token-weighted loss/PPL.
- No corpus, language mixture, downstream benchmark suite or finetuner is selected.

Use [TRAINING.md](docs/TRAINING.md) for dependencies, commands, packing details,
FA4 requirements, warmup, resume restrictions, custom-model/loss extension points
and evaluation. The complete sequential example is
[jobs.training.example.json](jobs.training.example.json), explained in
[JOBS.md](docs/JOBS.md#training--resume--evaluation-example). Its defaults are
small CPU test settings; replace paths and recipe before real experiments.

No ML environment file is bundled. `requirements.txt` documents that utilities
are standard-library-only; it does not install training dependencies.
Historical [validation evidence](docs/VALIDATION.md) lists tested runtimes and
limits. Those tests are not this project's results or current machine status.

## Reference

| Document / file | Use |
|---|---|
| [TRAINING.md](docs/TRAINING.md) | Training, data, resume, validation and test commands |
| [JOBS.md](docs/JOBS.md) | Sequential jobs, artifact checks and summaries |
| [commands.md](docs/commands.md) | Runner installs, downloads, execution and result exports |
| [GIT_PUSH.md](docs/GIT_PUSH.md) | Development versus execution pushes |
| [GPU_SAFETY.md](docs/GPU_SAFETY.md) | GPU ownership and authorized workload handoffs |
| [DROPBOX_ACCESS.md](docs/DROPBOX_ACCESS.md) | Local credentials setup and result retrieval |
| [VALIDATION.md](docs/VALIDATION.md) | Historical template test evidence |
| [scripts/gpu_status.py](scripts/gpu_status.py) | Read-only GPU inspection |
| [scripts/verify_manifest.py](scripts/verify_manifest.py) | Local SHA-256/size manifests |
| [scripts/training_results.py](scripts/training_results.py) | Checkpoint/stage checks and summaries |
| [resources/accelerate_config.example.yaml](resources/accelerate_config.example.yaml) | Single-process, no-mixed-precision example; configure before GPU use |
| [resources/llm_pretrain_burn.py](resources/llm_pretrain_burn.py) | Optional stress workload; requires explicit allocation and authorization |
