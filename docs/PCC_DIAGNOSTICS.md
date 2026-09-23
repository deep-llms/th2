# First PCC diagnostics

The research contract in
`cross_depth_anticipation_research_ideas_v2_13_launch_final_20260923_en.md`,
sections 11–13, defines the experiment. Current user authorization is local
development/testing only. Do not run anything on B200 or submit runner jobs.

There are two different questions:

1. **Does the implementation preserve the pretrained model and causality?**
   Run the correctness preflight, first on a tiny random Qwen on CPU, then on
   the exact local pretrained snapshot once available and authorized.
2. **Does privileged deep feedback help?** Train the small Privileged-Deep and
   Shallow-ExtraAttn adapters in the four-pair screen. An untrained branch is
   exactly zero and cannot answer this scientific question. The backbone
   remains frozen in both stages.

## Development environment and tests

The project environment is `train_env`, cloned from `sparse_emb` at the user's
request. Transformers is pinned to 4.57.1 because the frozen-tail implementation
uses its Qwen decoder-layer interface. Only the clone is changed. The required
ML dependencies are recorded in `envs/pcc.txt`; the file does not submit a job.

```bash
conda run -n train_env python -m unittest discover -s tests -v
conda run -n train_env python -m pcc preflight --tiny \
  --output temp/pcc-tiny-preflight-001.json
```

`--tiny` is CPU-only, uses randomly initialized reduced-width Qwen3 with all 28
blocks and 128-dimensional heads, and never downloads anything. Passing it is
not evidence of pretrained-model gains. Existing output paths are refused.

The tests exercise fp32 and bf16 paths, all candidate hooks, zero-init logits,
nonzero-branch future perturbations, isolated packed segments, padding/empty
sources, the analytic Qwen RoPE convention, clean source preservation, tail-only
execution, frozen backbone bytes, teacher detachment, correction-scale reference,
checkpoint roundtrip, weighted paired bootstrap, and a reduced synthetic screen.
The tail is independently compared with a nonzero intervention in a full model
forward, including gradients and activation recomputation. LM-only gradients
are tested separately from alignment gradients: an alignment loss must not hide
a detached LM path. Mutation tests deliberately inject a detached tail, a small
future leak, and cross-segment visibility and require preflight failure. A
batched update is also compared in fp32 with the same examples accumulated one
at a time, including unequal eligible-target counts to catch incorrect averaging.
The synthetic screen's tiny budgets exist only inside the test, not as CLI
overrides for the scientific screen.

## Acceptance coverage and remaining work

Passing local tests or `preflight.status=ok` is **not completion of all 17
acceptance requirements** in research-contract section 11.13. The contract
requires those checks before probe data; local synthetic checks do not establish
pretrained-model or real-data acceptance. Full-probe code is described in
[PCC_EXPERIMENT.md](PCC_EXPERIMENT.md).

| Contract requirements | Current evidence / limitation |
|---|---|
| 1–8, 10–12: model equivalence, masks, hooks, gradients, RoPE, empty sources | Tiny-model tests and all 42 pinned-model preflight checks passed on local A100; see the dated diagnostic report. |
| 9: target permutation | Global-update bucketed derangement implemented; deterministic mapping, no self indices, bucket preservation, and singleton rejection tested. |
| 13: correction scale | Exact-prefix calibration and fp64 reference tested with reduced fixtures; one scalar reused across all student arms. Actual 1M-token calibration not run. |
| 14: preprocessing equivalence | Shared-loader tokens match actual single-process legacy dataset maps/shuffling on synthetic fixtures; completed CulturaX inputs and their chosen policy remain unverified. |
| 15: paired batches | The screen shares each clean batch across paired arms; synthetic fp32 accumulated/batched updates are compared. Actual sampled-data provenance still needed. |
| 16: fresh full-run initialization | Seed 2901, byte-identical shared initialization, and absence of screen-checkpoint loading tested in full orchestration. |
| 17: data fairness | Fixed full budgets, screen/full slices, matched streams/policies, dev/test counts, and locked-test control flow implemented and synthetically tested; actual split provenance remains unverified. |

The four-pair runner below is implemented for development validation; it does
not establish that the complete contract is ready for a scientific launch.
The locked research contract remains unchanged.

## Pretrained preflight

```bash
conda run -n train_env python -m pcc preflight \
  --model-path /LOCAL/PATH/Qwen3-0.6B-Base-ddc928429ed09d9ad603fd762053d0434c15e865 \
  --output temp/pcc-pretrained-preflight-001.json
```

CPU is the default. There is no network/model fallback. The model/tokenizer
must already be local; config assertions are enforced. The snapshot directory
must be the pinned revision or end in `-<revision>`. This directory label is
operator-supplied provenance, not proof of weight identity: retain the verified
download provenance. The loader rejects missing, unexpected, mismatched, or
errored weight loads rather than accepting randomly initialized missing weights.
This does not independently authenticate a snapshot's claimed revision.
The revision in the current sampling assets (`da87bfb…`)
differs from the research contract (`ddc928…`) and has not been substituted.

The JSON records each check, runtime versions, config, revision, available code
commit, and a before/after backbone hash. Zero-init comparisons require exact
logit equality. Future-token, isolated-segment and padding invariance also
require exact equality for the same-shape eager forwards. Vanilla equivalence
uses atol=0.02, rtol=0.01 in bf16 and atol=1e-6, rtol=1e-5 in fp32; tolerances
are recorded. The loose vanilla tolerance is not used to excuse causal leaks.
These correctness tolerances are not thresholds for scientific NLL improvements.
Runtime invariants use explicit exceptions and remain active under `python -O`.

The optional `--physical-gpu INDEX` selects one local physical GPU after a
read-only free check. Its presence is not authorization to use any machine;
local A100 diagnostics were authorized and run on 2026-09-23. See
[PCC_DEV_DIAGNOSTICS_20260923.md](PCC_DEV_DIAGNOSTICS_20260923.md) for results and
the singleton-control blocker. No B200 access or deployment was performed.

## Fixed sampled-data inputs

PCC now reads the sampler's existing `train/en` shards and `eval/en` validation
Dataset directly. No raw-text export or new document split is required. See
[PCC_DATA_HANDOFF.md](PCC_DATA_HANDOFF.md) for the shared deterministic packing,
cache behavior, and the sampler's approximate eval-budget caveat. Sampling must
finish before its outputs are consumed; current B200 work remains untouched.

For a combined local model/data check, use `pipeline --check-only` with the
pipeline JSON; see [PCC_AUTOMATION.md](PCC_AUTOMATION.md). It checks full-run
budgets, split policies, vocabulary, and screen-prefix equality and records
ordered-context fingerprints without opening test data. Normal pipelines run
the data checks automatically before training; a successful check is not evidence
of gain or a full-context memory/throughput benchmark.

The screen uses the first 4,194,304 ordered training input tokens and 2,000,000
validation input tokens. These are prefixes of the same streams used by the
full probe. The last evaluation context may be right-padded. Every valid causal
next-token target is scored, excluding padding. Tokenization adds no special
tokens or separators, with ordinary causal attention across packed documents.
The low-level NPZ interface remains supported for existing packed inputs and
fixtures, including explicit block-isolated segment masks.

## Scientific layer-pair screen (implemented; not run)

```bash
conda run -n train_env python -m pcc screen \
  --model-path /LOCAL/PATH/Qwen3-0.6B-Base-ddc928429ed09d9ad603fd762053d0434c15e865 \
  --train-data /LOCAL/PATH/data/TOKENIZER_SLUG/train/en \
  --val-data /LOCAL/PATH/data/TOKENIZER_SLUG/eval/en \
  --microbatch 1 --output temp/pcc-screen-001
```

This command is a real adapter-training run, not a smoke test. It always runs
the preflight first and then the four fixed pairs `(4,16)`, `(4,20)`, `(8,20)`,
`(8,24)`, with 128 updates of 32,768 input tokens, paired initialization seed
1701, and the locked optimizer schedule. One device plus gradient accumulation
is supported; microbatch must divide 16 contexts/update. No distributed path is
implemented. FP32 adapter master parameters keep AdamW moments in fp32; bf16
autocast handles matrix operations, with fp32 normalization/softmax/loss
reductions. Tail activation checkpointing and chunked LM-head losses bound
activation memory. Production memory and throughput remain unmeasured.

Keep one microbatch setting across the compared arms, as this runner does.
Changing microbatch size can change bf16 rounding and Adam updates even with
identical ordered examples and global token counts. Local review observed this
in the tiny model; the fp32 accumulation test isolates normalization correctness
and does not promise numerically identical bf16 checkpoints across batch shapes.

For each pair, a clean pass is shared by both arms on each ordered microbatch.
Only source states differ. The screen saves adapter checkpoints, per-sequence
loss sums and target counts, training logs, timing/memory, and `decision.json`.
Base is evaluated on the same dev contexts. All bootstrap contrasts reuse the
same 1,000 sequence resamples and weight NLL by target counts. Both upper 95%
CI bounds must be negative before a pair is eligible; ranking and tie breaks
follow the contract. A negative screen is a successfully completed experiment.
Provenance is saved before training. JSON records, including the completion
marker, are published atomically without overwriting existing files. Failures
after creation of a fresh run directory write `failure.json` and never publish
`complete.json`. Invalid inputs/model loading can fail earlier, before a run
directory or report exists. Failure-report creation itself can fail on storage
errors; the original error is preserved.

The screen has no test input and cannot unlock the test split or launch the
full probe. The separate `probe` command implements full adapter training,
calibration, conditional permutation, and development-locked test evaluation;
see [PCC_EXPERIMENT.md](PCC_EXPERIMENT.md). Incremental inference KV caching
and actual pretrained/data acceptance remain outstanding.
