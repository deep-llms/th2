# A2 implementation and validation

Protocol: [A2_pilot_spec.md](A2_pilot_spec.md), **v2.1**. Research rationale:
[A2_novelty_and_feasibility.md](A2_novelty_and_feasibility.md).

This is a new depth-pruning study, not a continuation of the KV-sharing fits.
Existing `train.py`, KV code, runner, Accelerate resources and `commands.sh`
are unchanged. No full model/dataset download or B200 job has been submitted.

## Code

| Component | Implementation |
|---|---|
| Fixed parents, matrix, schedules, S2 and decision boundaries | `efficient_llm/a2/protocol.py` |
| Whole-pool sampling, document disjointness, EOS packing and hashes | `efficient_llm/a2/data.py` |
| Exact main code + branch weights, native layer removal, pure KL, BF16 export | `efficient_llm/a2/model.py` |
| CE, positional losses, BI/update norms, bypass and Gaussian diagnostics | `efficient_llm/a2/measure.py` |
| HF Trainer: sequential sampling, accumulation, AdamW and update-indexed LR | `efficient_llm/a2/train.py` |
| Actual-model numerical checks and discarded eight-GPU throughput probe | `efficient_llm/a2/check.py` |
| Adaptive LR grids, prescribed retries, phase caps and sequential children | `efficient_llm/a2/queue.py` |
| Effective-run selection, decision tables, Markdown and figures | `efficient_llm/a2/results.py` |
| Offline stage commands | `python -m efficient_llm.a2 --help` |
| Public Hub metadata lock, **dev only**, no weight download | `python -m scripts.a2_asset_manifest --help` |

There are 127 scheduled fits before retries: 58 in 0A (including T14), 69 in
0B. Each token budget has a **fresh optimizer and complete schedule**, not an
early cutoff of a longer fit. The last update uses LR zero. All 64 × 4096
positions contribute to forward KL; CE scoring uses T−1 predictions. No CE
term is added to training. All remaining student parameters train.

The head/loss is checkpointed by position chunk to avoid retaining full
sequence × vocabulary logits for backward. Decoder activations are **not**
checkpointed. CPU tests verify this head recomputation against full logits.
The teacher is frozen, outside DDP, with no vocabulary-wide logits retained.

For data, the published `document_index.parquet` contains `tokens_with_eod`.
The original sampler counts text tokens **plus one EOS**. A2 verifies the shard
hashes, checks this convention against the Qwen tokenizer on fixed documents,
and reuses these counts. It retokenizes reserved documents with GPT-NeoX only.
There is no long-document filter. Packing carries remainders across documents;
only the final partial sequence is discarded. No document-isolation mask is
used in training; the isolated mask exists only for T9.

## Local validation

Review on 28 October: the follow-up receipt is under
`docs/results/a2_review_20261028/` (the original 26-test receipt remains intact).
The revised suite has 30 passing tests plus the two-process comparison and 11
unchanged supervisor checks. Distributed scoring also covers uneven row counts
and fewer rows than ranks. Public Hub metadata for all 12 pinned parents passed
T2/T3 identity checks; that does not verify downloaded model bytes or CUDA.

Corrections from this review:
- Final/secondary scoring errors stop the queue; they cannot trigger a lower-LR
  retry. Only training numerical failure or tuning divergence selects a retry.
- Run receipts must match all registered fields, retry identity and baseline
  provenance; a failed non-divergence receipt cannot count as completion.
- Secondary parent/teacher/pre/post losses and damage contrasts are retained.
  Secondary summaries use the same tuning-selected effective runs and carry no
  decision labels. Curves also use only validated effective runs.
- T3 branch identity failures are recorded per branch, preserving eligible
  annealed parents. Metadata records the resolved annealed-72k comparison commit.

`tests/a2/ddp_probe.py <fresh-root> --cuda` is a separate bounded random-model
test: one GPU followed by eight, BF16 head recomputation, native removal checks,
exact sample coverage, BF16 export/reload and a 4096-token scoring probe. It uses
native OpenSci code with reduced width/depth/vocabulary; it is not T1/T8 for the
released 1.7B checkpoints. `scripts/launch_a2_check.py` wraps those two jobs in
the existing supervisor, preserving Accelerate verification and burn handoff.
CUDA evidence must be read from the separate review receipt, not inferred from
these CPU tests or the existence of the launcher.

Use a separate environment. Local testing used an ignored venv at
`temp/a2_test_env`, inheriting Torch from the existing environment but installing
Transformers 4.57.1, Hub 0.36.0 and Tokenizers 0.22.1 locally in the venv.
The existing `train_env` was not changed. `envs/a2_env.txt` defines a separate
runner environment; its installation was requested during the 28 October review.
Installation and CUDA-test status are recorded in CURRENT_TASK and the review receipt.

The native-model tests require these three small files from the pinned DCLM
P300 commit `9d02563017b25fd0a1e1849b7cd2bfd02b30d4e0`:
`modeling_opensci.py`, `configuration_opensci.py`, `config.json`.
Place them in an ignored local directory and set `A2_OPENSCI_CODE` to it.
Tests verify the source-file digests and instantiate tiny native OpenSci models;
no released weights are downloaded. The full released parameter count is also
checked on the meta device without allocating the 1.7B model.

```bash
A2_OPENSCI_CODE=temp/a2_model_code CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
  temp/a2_test_env/bin/python -m unittest discover -s tests/a2 -v

A2_OPENSCI_CODE=temp/a2_model_code PYTHONPATH=. OMP_NUM_THREADS=1 \
  temp/a2_test_env/bin/python tests/a2/ddp_probe.py temp/a2_ddp_probe

A2_OPENSCI_CODE=temp/a2_model_code PYTHONPATH=. OMP_NUM_THREADS=1 \
  temp/a2_test_env/bin/python -m torch.distributed.run --standalone \
  --nproc_per_node=2 tests/a2/ddp_probe.py temp/a2_ddp_probe
```

Use a fresh probe directory for a new comparison. Tests cover loss/gradients,
64-vs-8 accumulation, real Trainer accumulation, exact data order, original
layer identities, BF16 export/reload, Gaussian restore, real pool file layout,
LR selection and decision boundaries, retries and queue freshness.
CPU/DDP evidence is not a substitute for released-weight CUDA/BF16 checks.
See `docs/results/a2_implementation/validation.json` for the recorded receipt.

## Offline workflow

1. On the **dev** machine, create the public model lock with
   `python -m scripts.a2_asset_manifest --node-asset-root <local-node-assets> --output <fresh-lock-dir>`.
   This checks pinned main/branch configuration and shared-trunk metadata.
   Download actual assets through controller `#d` later, not from a GPU job.
   Each parent lives at `<asset-root>/<dataset>/<parent>/`; branch parents use
   Python code from that dataset's `P1T` directory. The loader checks actual
   model/config SHA256 against the lock and rejects missing/unexpected weights.
2. Pin the dataset revision and tokenizer identities. The pool contract has
   `repo`, `subset`, `revision`, `manifest_sha256`; tokenizer provenance has
   `qwen` and `gpt_neox`, each with repo/revision and `files: {filename: SHA256}`.
   Run `prepare --help` for explicit local paths. Output includes binary tokens,
   real document-start flags, document records, selected indices and hashes.
3. Prepare secondary shards with `prepare-secondary`, or record a genuine
   best-effort skip reason per corpus. The secondary-map JSON maps `dclm`,
   `fwe`, `nem` to either `{path, manifest_sha256}` or `{skip_reason}`.
4. Run `survey` for each eligible parent. This includes T1 precision comparison,
   pure-BF16 teacher scoring, T9, the other-theta T10 diagnostic, bypass, BI,
   relative updates and Gaussian sensitivity. Supply the reviewed source commit
   via `--code-commit` or `A2_CODE_COMMIT`; GPU source copies need no `.git`.
5. Aggregate surveys as `parents.jsonl`. Freeze sets using `sets`; its identity
   JSON maps `dataset/parent` to `T1` and `T3` booleans. T15 loss observations
   **never** affect eligibility. S2 records both raw and adjusted selections.
6. Run `check` for native numerical verification and `pre` for the four
   unhealed baselines. Run `throughput` on a disposable fresh student; it
   requires eight GPUs and the real 64 × 4096 workload. Add timings for **all**
   measurements, evaluations, loading and saving with `estimate`.
7. Review the actual preflight results before any heal, as the spec requires.
   The `heal` command refuses a missing/failed preflight or mismatched source,
   data, assets, sets or unhealed baselines. The preflight must record real
   evidence for T4–T8, T10 formula, T11–T13 and per-parent T1/T2/T3 eligibility;
   `T14` and `T16` are checked during fits. Do not fabricate a passing gate from
   the local CPU receipt. Aggregating/reviewing the actual GPU evidence and
   constructing its launch configuration remain part of the future rollout.
8. Run each phase through `queue.py` under the **existing** train-then-burn
   supervisor. Its launcher is an explicit argv ending in
   `-m efficient_llm.a2` (or Accelerate's `--module efficient_llm.a2`).
   The coordinator supplies `heal` and run-specific arguments. It does not own
   GPU/burn management. Before any GPU stage: copy the resource Accelerate
   config, check `accelerate env`, reclaim only authorized verified burns, and
   verify free GPUs. Preserve the existing automatic burn handoff.
9. Feed the combined `runs.jsonl` and `parents.jsonl` to `report`. This writes
   separate decision JSON files, per-dataset tables, healing/profile plots and
   `REPORT.md`. Include the test receipts, timings and deviations with it.

`plan --output <fresh.json>` only writes the 127-run matrix. It never launches.
Unresolved LRs remain null until the corresponding three-point grid completes.
No deferred phase is launched by any decision label.

## Queue input

`python -m efficient_llm.a2.queue --config <json>` accepts:
- `phase`: `0A` or `0B`; `root`: a fresh output directory;
- `launcher`: explicit Python/Accelerate argv; `common`: offline CLI arguments
  for assets, data, sets, spec, preflight, secondary map, source commit and the
  microbatch validated by T8;
- `pre`: maps `dataset/parent` to its exact `pre` output;
- `estimated`: measured whole-run seconds for each of `0.03B`, `0.05B`, `0.2B`,
  `0.8B`, including evaluation and saving;
- `spent_seconds`: the actual phase cost already spent on measurements/checks.

The outer supervisor must enforce phase-level timeout and perform descendant
cleanup after a child timeout. Missing artifacts, OOMs and infrastructure errors
stop the queue; they do not become divergence retries. Only registered numerical
divergence gets one half-LR retry. Grid retries never enter LR selection.
T14b's retry keeps its special data permutation. Existing outputs are refused;
there is no implicit mid-run resume. Failed/ineligible parents are recorded and
skipped; resulting missing comparisons remain INCOMPLETE.

## Remaining before the real pilot

The 12 actual checkpoints, full data preparation, numerical/precision gates on
B200, secondary acquisition, T15 logged-LR availability, measured workload budget,
pre-launch review and supervisor launch/handoff verification remain **pending**.
No real A2 training result or B200 throughput has been measured yet.
