# Full frozen-backbone PCC probe

This implements the next stage of research-contract sections 11–13. It trains
small adapters on **pretrained, frozen Qwen3-0.6B-Base**. It does not train a
language model from scratch. Current execution remains local development tests
only; no B200 or remote run has been submitted.

## Implemented workflow

For automatic sequential execution of the screen followed by this full probe,
use `python -m pcc pipeline`; see [PCC_AUTOMATION.md](PCC_AUTOMATION.md) for a
JSON config, dry run, and integration with `run_experiments.py`.

1. Read a successfully completed layer screen; recompute its selection from
   the nine saved per-sequence loss records. Verify the pinned revision and
   backbone fingerprint. A negative screen stops before loading full-run data.
2. Select the eligible pair and initialize all full-run adapters identically
   with seed 2901. Screen checkpoints are never loaded. Train Privileged-Deep
   for 610 updates of 32,768 input tokens, with the 31-update warmup and locked
   optimizer. Freeze it and clear its gradients.
3. Calibrate one correction RMS on exactly the first 1,000,000 training input
   tokens. A partial final context is masked; only eligible prediction positions
   contribute to the fp64 squared-sum/count calculation. Reject unusable scales.
4. Train Shallow-ExtraAttn (LM only) and Student-PCC (LM + normalized SmoothL1,
   beta=1, lambda=1) for the same 610 updates, with identical initialization,
   ordered batches, and scalar calibration. Teacher targets are detached.
5. Evaluate four arms on the fixed 10M-token development set, with 2,000 paired
   sequence bootstrap replicates. If PCC and distillation gates pass, train
   Target-Permuted for the same budget, even if another gate already fails.
   Evaluate that control on the same development contexts and apply all six gates.
6. Save the decision, configuration, correction scale, and final parameter
   fingerprints before opening test data. Any failed development gate leaves
   test data untouched. If an independent test set was supplied, evaluate the five frozen arms once on the
   fixed 10M-token test set, with 5,000 paired bootstrap replicates.

The test report includes confidence intervals. The contract says test results
must remain “directionally consistent”; this implementation fixes that meaning
before evaluation as **all five primary point NLL contrasts strictly negative**.
It does not introduce a new test-set significance threshold. A positive result
recommends drafting a separate tiny-from-scratch pilot specification; it never
launches pretraining or permits post-test tuning.

Training uses a clean frozen pass plus the differentiable frozen tail to bound
memory. `FrozenQwen.student` implements the actual full-context one-pass student:
each backbone block is visited once, and the branch only receives shallow states.
Teacher/deep tensors are not arguments to that interface. Autoregressive
generation with an incremental auxiliary KV cache is not implemented here.

## Fixed permutation details

The contract specifies the coarse buckets and seed but not the permutation pool,
quantile ties, or singleton behavior. The following choices are fixed in code
before any scientific run:

- Pool eligible correction vectors across the complete optimizer update, on CPU.
  Transient targets are recomputed from the current frozen teacher/backbone and
  current batch; no persistent deep-state memory is used.
- Relative-position bin is `floor(10 * token_index / valid_context_length)`.
  Positions are offsets in the model context, including for isolated packing.
  Norm decile edges are NumPy linear quantiles over eligible vector norms in
  that update; equal norms share a bucket (`searchsorted(..., side="right")`).
- Inside each crossed bucket, shuffle indices with NumPy
  `default_rng(SeedSequence([20260922, one_based_update]))` and rotate by one.
  Every mapped index differs from its original index and stays in its bucket.
  Buckets with fewer than two targets fail explicitly. Do not silently merge
  buckets or leave a self-mapped target.
- Permute only supervision. Student inputs, positions, masks, LM labels, and
  eligible target counts remain unchanged. No-source student/teacher corrections
  remain exactly zero; a permuted supervision vector at such a position need not
  be zero if its coarse bucket also contains small nonzero vectors.
- Save indices, mapping, bucket assignments, quantile edges, seed, and update
  for audit. Correction vectors themselves are not saved as a training cache.

Pooling is independent of microbatch boundaries. BF16 arithmetic can still vary
with batch shape; use one microbatch setting for all matched arms. Singleton
failure or any other failed attempt requires investigation, not an automatic
change to the rule after viewing results.

## Inputs and command

Use the completed sampler's `train/en` and `eval/en` directories directly as
training and development inputs. The shared loader applies the same deterministic
tokenization, packing, and order to every arm; no separate export stage is needed.
See [PCC_DATA_HANDOFF.md](PCC_DATA_HANDOFF.md) for the exact recipe and budgets.
The running B200 sampling job remains untouched.

Full training consumes 19,988,480 input tokens; development consumes 10,000,000.
Screen prefixes must match these streams exactly, with unchanged source metadata.
Keep the sampled directories at the paths recorded by the screen. Standalone
probe runs recreate the same ordered contexts; `pipeline` reuses the shared cache.
Insufficient usable tokens fail explicitly, without changing the fixed split.

`--test-data` is optional. Without it, passing development gates completes as
`validation_complete_test_not_supplied`, with no test confirmation or scaling
recommendation. A supplied independent test Dataset must support the fixed 10M
tokens and is not inspected until the dev gates pass and decisions are frozen.
This command is a scientific training run, **not a smoke test**, and has not
been run in this development session:

```bash
conda run -n train_env python -m pcc probe \
  --model-path /LOCAL/PATH/Qwen3-0.6B-Base-ddc928429ed09d9ad603fd762053d0434c15e865 \
  --screen-dir /LOCAL/PATH/completed-screen \
  --train-data /LOCAL/PATH/data/TOKENIZER_SLUG/train/en \
  --val-data /LOCAL/PATH/data/TOKENIZER_SLUG/eval/en \
  --microbatch 1 --output temp/pcc-probe-001
```

CPU is the default; all inputs are offline. There are no CLI overrides for
budgets, selected pair, seeds, or gate skipping. Output must be a fresh directory.
There is no automatic retry/resume. `test-started.json` records the single test
attempt even if it fails, and existing run directories are refused. The operator
must preserve the locked-test policy across separate output directories; creating
a different directory is not authorization for a second look at the test set.

## Artifacts and validation limits

Outputs include provenance, config, data policy, preflight, fresh initialization,
trained adapter checkpoints, teacher calibration, per-update loss/gradient logs,
per-sequence summed losses/counts, correction/gate/cosine summaries, permutation
audits when triggered, bootstrap contrasts, frozen decisions, and timing/memory.
Training timing includes all clean teacher-target passes and shallow recomputation.
Evaluation timing includes teacher diagnostics, so it is not an isolated student
inference-throughput benchmark. Optimizer state is not saved for resume.

`complete.json` is published last. A caught failure writes `failure.json` and
does not publish completion. A failed gate is a completed negative experiment,
not a software failure. Abrupt termination/storage failure may leave only partial
artifacts; absence of `complete.json` must never be interpreted as success.

Local tests cover real tiny-model teacher/student/control updates, exact
calibration, independent full-batch loss/gradient comparison, permutation,
one-pass output/gradient equivalence, frozen weights, checkpoint reproduction,
and test-lock failure/success branches. Positive state-machine tests inject
synthetic metrics; they establish control flow only. Tiny budgets are test
fixtures, not scientific CLI options.

The actual pinned pretrained snapshot and fixed data artifacts remain untested.
The research revision and existing sampling revision still differ. Resolve their
provenance and verify preprocessing equivalence before a scientific run. No
pretrained gain, production memory requirement, GPU behavior, or throughput has
been established by these local tests.
