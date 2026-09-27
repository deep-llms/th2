# Deep-KV: prepared-data recipe coupling and wandb output location — 2026-09-27

**Status: both implemented.** These came from reviewing commits
`170bb9b` (HF Trainer/Accelerate migration) and `1c10a46`. Neither changes the
model or loss. Prepared-data checks now compare only data-defining recipe
fields and exclude microbatch from source-configuration matching. Full recipe
and configuration matching still applies to run resume and arm comparison.
W&B runs now explicitly use the arm output directory (or `WANDB_DIR`) and close
on success or failure. The findings below describe the original behavior.

## Issue 1: prepared data is tied to operational recipe fields

### Original behavior

`prepare` writes `prepared/complete.json` containing the **whole** `Recipe`
(`deep_kv/data.py:74`, `asdict(recipe)`). `TokenStream` then rejects the
directory unless that stored recipe equals the current `Recipe()` exactly
(`deep_kv/data.py:92-93`):

```python
if self.manifest.get("format") != "deep-kv-data-v2" or self.manifest["recipe"] != asdict(recipe):
    raise ValueError("Prepared stream recipe mismatch")
```

After the migration, `Recipe` also holds fields that only control logging and
execution: `logging_every`, `checkpoint_every`, `eval_every`,
`dataloader_workers`, `warmup`, `learning_rate`, `weight_decay`, `seed`, and
others.

### Why it matters

With the same sources, tokenizer and packing code, only five Recipe fields
determine the prepared files' content:

| Field | Affects |
|---|---|
| `updates`, `tokens_per_update`, `context` | Number of train contexts (`train_rows`) and context length |
| `eval_rows` | Number of eval contexts |
| `data_seed` | The fixed context permutation (`*.order.npy`) |

Changing any other field leaves `train.bin`, `eval.bin` and the order files
byte-identical, but the directory is still rejected. Example: after a full
`prepare` (~120 GB; tokenization runtime has not been measured), changing
`logging_every` from 10 to 1 makes every arm fail at startup with "Prepared
stream recipe mismatch". The only ways around it are re-preparing, or
hand-editing `complete.json`, which defeats the check.

This matters most when a prepared directory is reused through
`make-jobs --data-dir`, for example for a later queue with different checkpoint
or monitoring intervals. Changing only `--stop-after` already allowed reuse:
the cutoff does not alter the full recipe or prepared token budget.

### Resolution

1. **No code change.** Freeze every `Recipe` field before `prepare` and do
   not edit it afterwards.
2. **Compare only data-defining fields (recommended).** In `TokenStream`,
   compare `updates`, `tokens_per_update`, `context`, `eval_rows` and
   `data_seed` between the manifest and the current recipe. Keep storing the
   full recipe in `complete.json` for provenance. The existing shape,
   packing-policy and SHA256 checks stay as they are.

Option 2 is implemented, without changing the v2 manifest format or existing
prepared files. The startup provenance check also excludes `microbatch`, which
was another operational setting stored in the original source configuration.
Source paths and model-config checksum checks remain mandatory.

This does not weaken matching between arms. Each arm's run identity still
contains the full `Recipe` (`build_identity`), and the report rejects arms
whose identities differ. Resuming an existing arm also still requires an
identical full recipe, because resume compares run identities. The change
only allows the same prepared token files to feed a *new* run whose
operational settings differ.

## Issue 2: wandb offline files land in the synced source tree

### Original behavior

Training enables wandb in offline mode (`report_to="wandb"`,
`deep_kv/training.py:90`; `WANDB_MODE=offline` at `training.py:256`,
`deep_kv/__main__.py:12`, `scripts/train_deep_kv.sh:7`). Transformers'
`WandbCallback` calls `wandb.init(project=..., name=...)` without a `dir`
argument. wandb therefore writes to `$WANDB_DIR` if set, otherwise to
`./wandb` in the current working directory.

`run_experiments.py` starts every job with `cwd=project_dir`
(`run_experiments.py:176`). On the runner that is the Git-synced source
checkout, `/mnt/local/<owner>_<repo>/` (`docs/commands.md`). All four arms
would therefore write their offline wandb runs into the source tree.

### Why it matters

- Run artifacts are mixed with synced source, contrary to the project rule
  that outputs live outside the source tree (`docs/commands.md`, "Paths and
  offline execution").
- The runner's code synchronization may overwrite or remove them, or they may
  interfere with later syncs.
- The four arms' wandb runs are not stored next to their own checkpoints and
  `metrics.json`, which makes retrieval through `#2` harder.

### Resolution

`train()` now opens a scoped `offline_wandb_run(args)` after the output directory
exists. Rank zero calls `wandb.init(mode="offline", dir=...)`, using the explicit
`WANDB_DIR` override if present, otherwise the absolute arm output directory.
The standard HF callback attaches to this run. Other ranks and normal CPU tests
do not initialize W&B. The context closes the run on success or exception.

Using an explicit `dir` instead of a persistent `os.environ.setdefault` avoids
both environment leakage and W&B's cached settings when several arms are trained
in the same Python process. Offline files default to `<run-dir>/<arm>/wandb/`.
An already active W&B run is rejected rather than mixing two experiments. This
fix is needed before training; it does not affect preparation or token files.

## Verification

All 15 focused CPU tests passed (13.609s), including operational recipe reuse,
data-defining mismatch rejection, microbatch/source provenance checks, strict
resume rejection, and real offline W&B logging through the HF callback across
multiple arms, an operator directory override and an injected training failure.
Compilation and diff checks passed. No GPU work, preparation or launch.
Log: `temp/deep-kv-data-wandb-tests-20260927-final.log`.
