# Deep-KV: NCCL setting on the queue path and a test import — 2026-09-28

**Status: both fixed (2026-09-28).** `train.py` now defaults
`NCCL_NVLS_ENABLE=0` before HF/Accelerate imports, covering direct and queue
launches while preserving an explicit operator override. Existing shell exports
and offline W&B/HF behavior are unchanged. The sibling test import now uses
`tests.test_deep_kv`. A fresh-process regression checks the launch defaults and
explicit override. Validation passed in the pinned local environment:

- `python -m unittest tests.test_train -v`: 6 tests passed.
- `python -m unittest discover -s tests -v`: 48 tests passed.

The original findings below describe the pre-fix code.

Found while reviewing the refactor
(`7b59f57`) and the verified B200 smoke (`b886e56`) against
`anticipatory_deep_kv_four_arm_pilot_v2.md`. Neither affects the four-arm
mechanism, which was unchanged by the refactor. The full CPU suite passes
(47 tests with `python -m unittest discover -s tests`).

## Issue 1: `NCCL_NVLS_ENABLE=0` is missing from the generated queue

### What happens

The baseline workflow disables NCCL NVLS, and so did the verified smoke:

- `scripts/train_deep_kv.sh:4` exports `NCCL_NVLS_ENABLE=0`.
- `scripts/smoke_deep_kv_b200.sh:7` exports it before calling
  `run_experiments.py`, so every smoke arm inherited it.

The production queue does not set it. `python -m deep_kv make-jobs` writes
each arm's command as `accelerate launch --config_file
resources/accelerate_config.yaml train.py ...`, without the wrapper script.
`run_experiments.py:166` passes the runner's own environment to each job, and
`train.py:7-8` sets only the offline Hugging Face and W&B variables. Before
the refactor, `deep_kv/__main__.py` also set
`os.environ.setdefault("NCCL_NVLS_ENABLE", "0")` (added in `170bb9b`). That
line was removed in `7b59f57`.

### Why it matters

Unless the `#1` launch job exports the variable itself, the real queue runs
with NCCL's default NVLS behavior. That is a different communication setup
from the one the smoke verified on eight B200s. It applies equally to all four
arms, so it does not break matching between arms. It does leave the
production runtime configuration unverified.

### Fix

Choose one:

1. **In `train.py` (recommended).** Add `NCCL_NVLS_ENABLE` to the existing
   environment setup, before any distributed import:
   ```python
   os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
   ```
   `setdefault` keeps an explicit operator override. This covers the queue,
   `scripts/train_deep_kv.sh` and any direct launch the same way.
2. **In the launch job.** Export `NCCL_NVLS_ENABLE=0` in the `#1` shell body
   before `run_experiments.py`, as the smoke script does. This must then be
   repeated in every future launch and resume command.

## Issue 2: `tests/test_train.py` imports a sibling test module by bare name

### What happens

`tests/test_train.py:142`:

```python
from test_deep_kv import model
```

This works only when `tests/` is on `sys.path`. `unittest discover -s tests`,
the command documented in `README.md` and `docs/AGENT_GUIDE.md`, adds it, so
the full suite passes. Running the module directly does not:

```text
$ python -m unittest tests.test_train
ERROR: test_native_gradient_accumulation_scaling
ModuleNotFoundError: No module named 'test_deep_kv'
```

This test checks that HF Trainer's gradient accumulation (microbatch mean
÷ accumulation steps, then DDP averaging) matches one whole-batch gradient
for all four arms. It is worth being able to run it on its own.

### Fix

Import it through the package path, which works under both invocations
(`tests/__init__.py` is not needed for `python -m unittest tests.test_train`):

```python
from tests.test_deep_kv import model
```

`tests/deep_kv_resume_worker.py:12` (`from test_train import fixture`) is not
affected. It is run as a script, so Python puts `tests/` on `sys.path`
automatically.
