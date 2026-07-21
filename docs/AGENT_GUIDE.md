# Agent Guide

How this project is structured, how to work in it, and what to keep in mind.

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
│   ├── setup_env.sh                     # Installs conda envs (sparse_emb, fasttext_env, eval)
│   └── train_qwen3_0.6b_baseline.sh     # Baseline training launch template
│
├── resources/
│   └── accelerate_config.yaml           # 8-GPU accelerate config
│
├── docs/
│   ├── AGENT_GUIDE.md                   # This file
│   ├── PROJECT_NOTES.md                 # Single source of truth for project background + results
│   ├── CURRENT_TASK.md                  # What we're working on right now
│   └── commands.md                      # Remote runner documentation
│
└── temp/                                # GITIGNORED — results, pulled files, scratch
```

## Machines

### Dev machine (where you run)

Code development and testing. This is the machine Claude Code sessions run on.

- **GPUs:** 4× A100 (40GB) — use for testing, not training
- **Conda envs:** `sparse_emb` (main), `fasttext_env`, `eval`

### Training machine (remote, 8× H200)

Training and evaluation at scale. You cannot SSH into it directly — all interaction is through `commands.sh`.

- **GPUs:** 8× H200 (141GB each), 192 CPUs, CUDA 13.2
- **Storage:** `/opt/dlami/nvme/` — 28TB fast NVMe for data + checkpoints
- **Conda envs:** same names as dev machine, installed via `scripts/setup_env.sh`
- **HF_TOKEN:** already exported by default on the training machine before any command runs. No need to set it in `commands.sh` or worry about HuggingFace authentication for downloading models/datasets on the training machine.
- **How to run things:** see `docs/commands.md` for full documentation on how to submit commands, pull logs, and pull files from the training machine.

### HuggingFace token

- **Training machine:** `HF_TOKEN` is pre-exported. No action needed.
- **Dev machine:** read from `temp/HF_TOKEN.txt` (gitignored). To use it: `export HF_TOKEN=$(cat temp/HF_TOKEN.txt)`

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
- Don't store secrets (tokens, keys) in code files. Use environment variables.
- Don't skip testing. Run the code on the dev machine (even with a tiny model) before deploying to the training machine.
- Don't leave stale documentation. If the code changes, update the docs.

## Sequential Experiment Runner

`run_experiments.py` runs experiments one at a time with GPU cleanup between runs. An "experiment" is a variant or version of a model architecture, a different hyperparameter setting, or an idea to test — each is a training run that produces checkpoints to evaluate and compare.

Edit `EXPERIMENT_COMMANDS` to define experiments:

```python
EXPERIMENT_COMMANDS = [
    {
        "name": "baseline",
        "cmd": "accelerate launch train.py --config_name Qwen/Qwen3-0.6B --bf16 --output_dir /opt/dlami/nvme/outputs/baseline",
        "output_dir": "/opt/dlami/nvme/outputs/baseline",   # optional
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
