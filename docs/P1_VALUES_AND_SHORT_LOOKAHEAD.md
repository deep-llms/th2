# P1 values-only and shorter-lookahead experiments

Implementation added 6 October 2026. These are explicit variants of P1-block;
the default P1/P3 architectures and saved recipes remain unchanged. No new
training is authorized by this document.

Use the existing `train.py`, HF Trainer, Accelerate, packing, normalization,
auxiliary loss, and checkpoint/resume paths. Both SDPA and FA4 use the same
modified projection path, before attention. Actual FA4 CUDA acceptance is a
separate GPU smoke test; CPU reference tests do not establish CUDA correctness.

## Independent changes

| Variant | `proxy_kv_mode` | `proxy_lookahead` | Proxy layers |
|---|---|---|---|
| Current P1 control | `kv` | 4 | 2, 4, …, 24 |
| Values-only | `v` | 4 | 2, 4, …, 24 |
| Short lookahead, first choice | `kv` | 2 | 2, 4, …, 24 |
| Short lookahead, optional alternative | `kv` | 3 | 2, 4, …, 24 |

Values-only computes **all keys from the native normalized input**. Queries
are unchanged; only the final selected KV groups' values use the proxy-enhanced
input. Key normalization, RoPE, document isolation, positions and GQA mapping
are unchanged. There is still one attention operation per layer. This keeps
routing native within each attention operation; earlier proxy outputs can
still affect later layers' hidden states and hence their queries/keys.

Shorter lookahead changes only the supervised target: for proxy block `l`,
sum the MLP outputs of blocks `l` through `l+k-1`, including block `l`.
The target is detached and standardized with the existing r7 normalization.
Set `proxy_layers` explicitly to preserve all twelve original proxy layers.
Without this override, the historical automatic placement remains unchanged:
a shorter lookahead can add eligible late proxy layers. Explicit layers must
be unique, increasing, even, and allow the entire target window to fit.

Use `proxy_alpha_init=1` to match the current alpha-one control, with learned
per-channel gates. Auxiliary weight remains 0→0.1 over 250 steps. These are
separate ablations: values-only retains k=4; k=2/3 retain K/V injection.
All other recipe settings stay identical, including seed, full LR schedule,
2,500-step cutoff, global batch, data order and attention backend.

## CLI examples

Append one of these to the existing P1-block training invocation:

```bash
# Values-only; same four-block target as the control.
--proxy_kv_mode v --proxy_lookahead 4 --proxy_alpha_init 1

# Two-block target; explicitly preserve the control's injection layers.
--proxy_lookahead 2 --proxy_layers 2 4 6 8 10 12 14 16 18 20 22 24 --proxy_alpha_init 1

# Three-block target, with the same placement.
--proxy_lookahead 3 --proxy_layers 2 4 6 8 10 12 14 16 18 20 22 24 --proxy_alpha_init 1
```

Nondefault routing and explicit placement are recorded in `train_config.json`.
Resume refuses changes to routing, placement, lookahead or gate initialization.
Defaults are omitted from saved metadata for compatibility with old runs.
Gate evaluation reconstructs these settings from the saved recipe, including
old checkpoints that predate the options. Cross-variant comparison should
explicitly identify the changed fields; the ordinary matched-recipe report
continues to reject mismatched recipes instead of silently ignoring them.

## Sequential jobs using the existing runner

Generate on the execution checkout so the existing job generator resolves the
correct `train.py` and Accelerate config paths. Generation does not launch work.
Use a fresh preparation directory; put run outputs outside the source tree.
This example queues the first two experiments; add the commented entry only
if the three-block alternative is wanted. Reuse the established supervised
launch and GPU handoff procedure, with a smoke test before real training.

```python
import json
from pathlib import Path
from deep_kv.__main__ import jobs

prepared = Path('/mnt/local/_outputs/deep-llms_th2/p1-variants-prepared')
prepared.mkdir(parents=True, exist_ok=False)
base = json.loads(Path('baseline_a_fa4.b200.json').read_text())
variants = [('values-only', {'proxy_kv_mode': 'v', 'proxy_lookahead': 4}),
            ('lookahead-2', {'proxy_kv_mode': 'kv', 'proxy_lookahead': 2})]
# variants.append(('lookahead-3', {'proxy_kv_mode': 'kv', 'proxy_lookahead': 3}))
queue = []
for name, overrides in variants:
    recipe = {**base, 'arm': 'P1-block', 'proxy_alpha_init': 1.,
              'proxy_layers': list(range(2, 25, 2)), **overrides}
    path = prepared / (name + '.json')
    path.write_text(json.dumps(recipe, indent=2))
    for job in jobs(path, stop_after=2500, arms=['P1-block'], seeds=[42])['jobs']:
        job['name'] = name + '-' + job['name']
        job['argv'] = [arg.replace('{run_dir}', '{run_dir}/' + name) for arg in job['argv']]
        for output in job['required_outputs']:
            output['path'] = name + '/' + output['path']
        queue.append(job)
(prepared / 'jobs.json').write_text(json.dumps({'jobs': queue}, indent=2))
```

Each variant uses all eight GPUs, separate fresh outputs, and the original
`run_experiments.py` sequential runner. The generator also adds the existing
per-variant completion reports. This is preparation, not a live launch manifest.
