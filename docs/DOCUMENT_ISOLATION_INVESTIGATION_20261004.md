# Document isolation on B200: implementation, experiments, and findings

Date: 2026-10-04. This report consolidates the investigation of packed-document
isolation in the Qwen3-0.6B / Deep-KV project. It covers the code, experiments,
failures, results, remaining uncertainty, and current recommendation.

**Current recommendation:** use dense SDPA isolation for the next real experiments
if minimizing uncertainty is the priority. FA4 passed the same-weight numerical
checks and was faster, but its equivalence over a training trajectory has not
been established. This is a recommendation, not an implemented production switch.

## 1. What we are trying to change

Training packs multiple documents into fixed 2,048-token sequences. The existing
pipeline appends an explicit `<|endoftext|>` token to each document, but an EOS
marker by itself does not prevent a later document from attending to earlier
documents in the same sequence.

The requested behavior is causal attention **within each document only**:

```text
Packed sequence:  [ A tokens ... EOS_A | B tokens ... EOS_B ]

Existing causal attention: B can read earlier B tokens AND all of A.
Document isolation:        B can read earlier B tokens only.
```

This is document-boundary masking of decoder self-attention, not the separate
encoder–decoder mechanism often called cross-attention. The same isolation must
also hold in custom auxiliary attention paths.

The implementation tested has these semantics:

- The appended EOS belongs to the document that precedes it.
- Positions restart at zero for each document fragment inside a packed sequence.
- Tokens cannot attend to another document, even if it appears earlier.
- The model still predicts each document's EOS. The next-token loss excludes
  predicting the first token of document B from document A's EOS.
- Auxiliary source eligibility and losses must respect the same boundaries.

Consequently, isolation changes the training objective as well as the attention
mask. Different outputs or gradients between isolated and cross-document modes
are expected; kernel correctness must be checked with matching semantics.

## 2. Existing packing versus the experimental changes

[train.py](../train.py) uses the shared two-map pipeline in
[deep_kv/packing.py](../deep_kv/packing.py): tokenize documents with an appended
`<|endoftext|>`, then concatenate and split into fixed-length chunks. Hugging Face
Datasets supplies caching. The current grouping drops an incomplete remainder
per map batch; it is not one uninterrupted corpus-wide stream that drops only
one final remainder. This investigation did not change that behavior.

Production preprocessing currently does **not** emit the per-token document IDs
needed to enable isolation. The experimental full-model benchmark preserves the
same style of packing and adds explicit source document IDs alongside tokens.
It does not infer boundaries from EOS token values: a literal EOS in source text
must not accidentally create a new document boundary.

The existing [Context](../deep_kv/model.py) supports segment-aware attention and
prediction-target masks when document IDs are supplied. This capability is not
the same as enabling it in the production data/collator path.

**Production training has not been switched to either dense document isolation
or FA4 document isolation.** EOS insertion remains enabled. The attention-backend
experiments are isolated in benchmark scripts.

## 3. Implementations investigated

| Implementation | How it works | Scope checked |
|---|---|---|
| Implicit causal SDPA | Uses causal attention without a dense document mask | Fast cross-document control; does not isolate documents |
| Explicit causal SDPA | Supplies an explicit causal mask | Previous cross-document baseline |
| Dense isolated SDPA | Supplies a same-document, causal mask | Model isolation tests, full training, and gradient comparisons |
| FA4 varlen isolation | Represents each document fragment as a sequence using `cu_seqlens` and calls `flash_attn_varlen_func` | Attention tests and full baseline training |
| FlexAttention | Expresses same-document causality through block masks | Attention-level feasibility and numerical tests; no full training comparison |

No custom CUDA kernel was written. The FA4 adapter uses the existing variable
length API, flattens Q/K/V into packed token layout, passes cumulative sequence
lengths and maximum lengths, and restores the output layout. The benchmark
requires fully packed, unpadded, zero-dropout inputs and supports full-model
**arm A only**.

Strict-past auxiliary attention was also tested at the attention-operator level:
within each document, pair `Q[1:]` with `K/V[:-1]`, use causal attention, and return
zero for the first query. Singleton documents were covered by CPU tests. This
operator check does not establish a complete FA4 implementation for every arm.

### Relevant code

| File | Responsibility |
|---|---|
| [deep_kv/packing.py](../deep_kv/packing.py) | Production EOS insertion and cached two-map packing |
| [deep_kv/model.py](../deep_kv/model.py) | Segment-aware `Context` masks, targets, and auxiliary eligibility |
| [tests/test_document_isolation.py](../tests/test_document_isolation.py) | Dense isolation across all 11 arms, packed/separate agreement, gradient leakage, checkpointing on/off |
| [scripts/benchmark_document_attention.py](../scripts/benchmark_document_attention.py) | Attention-only SDPA timing and backend inspection |
| [scripts/benchmark_packed_attention.py](../scripts/benchmark_packed_attention.py) | Varlen/Flex/FA4 operator correctness and forward/backward timing |
| [tests/test_packed_attention.py](../tests/test_packed_attention.py) | Independent reference checks for packed/strict-past attention |
| [scripts/benchmark_document_training.py](../scripts/benchmark_document_training.py) | Explicit document metadata, arm A adapter, HF Trainer/Accelerate runs, timing, held-out evaluation, stream hashes |
| [tests/test_document_training.py](../tests/test_document_training.py) | Boundary metadata, attention adapter, target weighting, real Trainer accumulation, and evaluation checks |
| [scripts/check_trained_attention.py](../scripts/check_trained_attention.py) | Strict trained-checkpoint loading, fixed-weight gradients, repeatability, FP32 references, and result collection |
| [tests/test_trained_attention.py](../tests/test_trained_attention.py) | Tied-weight checkpoint formats, conflict rejection, and preservation of failed-gate evidence |
| [envs/attention_bench.txt](../envs/attention_bench.txt) | Separate experimental dependency recipe |

The benchmark Trainer normalizes LM loss by the actual eligible target count
across all ranks and accumulation microbatches. Document boundaries make those
counts unequal even when input sequence lengths match. CPU tests check that
accumulated gradients produce the same update as the corresponding full batch.
The original training objective and custom-arm implementations were not replaced.

## 4. Environment and full-training recipe

Recorded B200 environment: eight NVIDIA B200 GPUs, driver 580.167.08,
PyTorch 2.14.1+cu130, Transformers 5.9.0, Accelerate 1.13.0,
FA4 `4.0.0b33`, and CUTLASS DSL 4.8.0. FA4 was tested in the separate
`attention_bench` environment; the original `train_env` and driver were unchanged.
These are the tested versions, not a claim about the latest available releases.

The 30- and 200-update experiments used:

| Setting | Value |
|---|---|
| Model | Full Qwen3-0.6B architecture, arm A |
| Initialization | Random weights, seed 42; **not pretrained-weight finetuning** |
| Model/tokenizer assets | Pinned Qwen3-0.6B-Base configuration and tokenizer |
| Data | English from the downloaded `nht10/cx_sampled_old` pool |
| Training pool | 8,192 packed rows / 16,777,216 input tokens, derived from the first 24,000 training documents |
| Sequence length | 2,048 |
| GPUs | Eight per training run, HF Trainer/Accelerate with DDP |
| Microbatch / accumulation | 16 sequences per GPU / 4 microbatches |
| Global input tokens per update | 1,048,576 |
| Precision | BF16 execution with FP32 master parameters |
| Activation checkpointing | Disabled for decoder/LM/auxiliary paths |
| LM chunk size | 128 |
| Optimizer | AdamW, betas 0.9/0.95, weight decay 0.1, gradient clipping at 1.0 |
| LR schedule | Peak 3e-4, 28,600 total updates, 1,430 warmup updates, cosine minimum LR ratio 0.1 |
| Stop point | 30 or 200 updates, without shortening the LR schedule |

The small training pool repeats; 200 updates correspond to 12.5 passes over it.
Every measured update remains inside LR warmup. These are diagnostic runs, not
full-data convergence experiments or evidence of final model quality.

## 5. What we tried and what happened

### A. Verify document isolation before optimizing it

Dense segment masks passed the local and B200 FP32/BF16 model tests across all
11 arms. Perturbing an earlier document caused exactly zero change in later
isolated-document outputs. Cross-document embedding gradients were zero.
Packed and separate-document losses/gradients agreed within the tested precision
tolerances. Unsegmented controls demonstrated leakage, so these were not tests
that would pass without isolation. Active auxiliary output weights and activation
checkpointing on/off were exercised.

These establish dense-mask model semantics on the fixtures tested, not full-scale
FA4 correctness for all custom arms.

### B. Measure attention-only cost and test API alternatives

An initial B200 SDPA test measured 0.2535 ms for implicit causal attention and
0.6239 ms for document-isolated dense attention: about 2.46 times the attention
forward/backward time. Explicit causal SDPA was almost identical to dense
isolation at 0.6238 ms.

A subsequent packed-API benchmark gave the following ragged-document timings:

| Attention implementation | Forward + backward |
|---|---:|
| Implicit causal SDPA, cross-document | 0.339 ms |
| Dense isolated SDPA | 0.711 ms |
| FA4 varlen isolation | 0.525 ms |
| FlexAttention with FA4 backend | 0.965 ms |

These used batch 2, sequence 2,048, GQA 16/8 heads, head dimension 128, and
20 measured iterations. Wrapper movement overhead was included; metadata setup
and initial compilation were excluded. The two timing tables came from separate
benchmarks and should not be combined into one controlled measurement.

The corrected FA4/Flex-FA4 tests passed normal-causal and strict-past output/QKV
gradient checks for single, equal, and ragged documents. An initial adapter attempt
failed because the FA4 API returned a tuple and the B200 Flex-FA4 path required
256-token blocks. Both were corrected before the passing run.

**The approximately 2.5× figure was attention-only, not a full-training slowdown.**

### C. Measure complete training updates: five modes, 30 updates each

All five runs completed. Times below use the slowest rank per update and the
median of updates 6–30. They include forward, LM loss, backward, DDP, clipping,
optimizer, scheduler, and data delivery; startup, evaluation and saving are excluded.

| Mode | Seconds/update | Peak allocated GPU memory |
|---|---:|---:|
| Previous explicit SDPA, cross-document | 2.42576 | 110.62 GiB |
| Implicit causal SDPA, cross-document | 2.06230 | 103.62 GiB |
| FA4, cross-document | 2.08819 | 103.62 GiB |
| Dense SDPA, isolated documents | 2.42524 | 110.62 GiB |
| FA4 varlen, isolated documents | 2.07173 | 103.62 GiB |

FA4 isolation took approximately 14.6% less time than dense isolation and was
within 0.5% of the implicit causal control in this short baseline test. Dense
isolation was approximately equal in time to the previous explicit-mask baseline.
Thus the apparent cost depends strongly on which existing implementation is the
comparison point. These results are one run per mode, not repeated confidence
estimates or timings for auxiliary arms.

Initial same-weight full-model checks passed: FA4/dense gradient relative L2
error was 0.5408%, with comparable gradient errors against FP32 (about 1.23%
for both BF16 implementations). Exact document isolation passed. Both isolated
training losses decreased from about 12.12 to 10.98, and the 30-update trajectories
were very close.

### D. Extend the matched isolated runs to 200 updates

Both fresh runs completed and saved final weights. The entire token/document-ID
stream matched per rank, learning rates matched, and all recorded losses and
gradient norms were finite. A separate 512-row held-out pool contained 1,048,576
input tokens and 1,046,868 eligible prediction targets.

| Measurement | Dense isolated SDPA | FA4 isolation |
|---|---:|---:|
| Final training loss | 7.1168804 | 7.1171484 |
| Held-out loss, evaluated with common dense backend | 7.1107591 | 7.1074235 |
| Median seconds/update, updates 6–200 | 2.42939 | 2.07476 |
| Peak allocated GPU memory | 110.62 GiB | 103.62 GiB |

The predeclared comparison **failed**:

| Check | Observed gap | Required limit | Result |
|---|---:|---:|---|
| Maximum absolute training-loss gap | 0.01066065 | <0.01 | Failed |
| Maximum relative gradient-norm gap | 48.098% | <3% | Failed |
| Held-out loss gap using the same evaluation backend | 0.00333556 | <0.01 | Passed |
| Same final FA4 weights evaluated through dense versus FA4 | 0.00000258 | <0.01 | Passed |

Through update 100, agreement was very close. The norm-gap limit was first
exceeded at update 137 and was exceeded at 49 updates overall. At update 196,
dense/FA4 gradient norms were 1.76583/0.91649. This produced the maximum 48.098%
norm gap, along with the maximum loss gap. At update 200 the norm gap was still
42.44%, so the issue was not just one isolated spike.

The training processes did not crash. The summarizer saved `comparison.json`
with `passed=false`, then failed its assertion as intended. Thresholds were not
relaxed. Neither the close held-out loss nor the later diagnostic changes this
historical failure into a pass.

See [the 200-update report](DOCUMENT_ATTENTION_STABILITY_20261004.md) and the local
[training curves](../artifacts/document-stability-200-20261004-a01/training_comparison.png).

### E. Separate kernel error from training drift: identical trained weights

We then loaded each final checkpoint and compared both backends at unchanged
weights. The eight cases were two checkpoints × training/validation × microbatch
2/16. Each GPU ran one case independently, without optimizer updates or DDP
reductions. The smaller batches are subsets of the larger ones, not independent
additional data samples.

Each case compared full parameter gradients, loss, final hidden states and sampled
logits, and repeated each BF16 capture. The four microbatch-2 cases also used FP32
math SDPA with TF32 disabled. All parameter fingerprints were unchanged.

| Comparison | Full-gradient relative L2 difference |
|---|---:|
| FA4 versus dense SDPA, identical trained weights | 0.181%–0.532% |
| Same comparison at microbatch 16 only | 0.181%–0.266% |
| Dense pipeline versus repeated dense capture | 0.119%–0.302% |
| FA4 pipeline versus repeated FA4 capture | 0%–0.111% |
| Dense BF16 versus FP32 math reference | 0.539%–1.209% |
| FA4 BF16 versus FP32 math reference | 0.540%–1.163% |

All eight backend comparisons and all four paired FP32-reference cases passed
the unchanged limits: 3% gradient relative L2, 2% hidden/logit relative L2, and
0.01 absolute loss. Maximum FA4/dense loss difference was 0.00012493; gradient
cosine similarity was at least 0.9999858. No nonfinite gradients or OOM occurred.
Repeated forward captures matched exactly; backward captures sometimes varied.
This variation concerns the complete model/loss/backward pipeline, not an isolated
measurement of attention-kernel nondeterminism.

An initial attempt failed before GPU captures because the diagnostic loader
rejected Trainer checkpoints containing both cloned copies of tied embedding
weights. The loader was corrected to verify alias consistency and load strictly;
regression tests cover deduplicated, cloned and conflicting copies. The retry
preloaded both actual checkpoints on CPU before GPU reclamation. All ten focused
CPU tests passed on B200. The checkpoints themselves were never modified.

See [the trained-weight report](TRAINED_ATTENTION_CHECK_20261004.md).

## 6. What the 48% means—and what remains unknown

The two percentages measure different quantities:

```text
200-update trajectory comparison:
    abs(norm(g_FA4 at weights_FA4) - norm(g_dense at weights_dense))
    / norm(g_dense at weights_dense)

Same-weight correctness comparison:
    norm(g_FA4 at weights_W - g_dense at weights_W)
    / norm(g_dense at weights_W)
```

A 48% trajectory norm gap is not a measurement of a 48% kernel gradient error.
The same-weight results argue against a large FA4 backward error in the tested
checkpoints and batches. Similar FP32-reference errors for both backends and
small repeated-backward variation are consistent with accumulated numerical drift.

However, **we have not established the cause of the large trajectory difference
or demonstrated that it is harmless over long training**. The evidence is limited
to one matched training pair, a repeated small data pool, warmup-only training,
a small held-out pool, and selected fixed-weight batches. The checks do not
establish identical long-term quality, behavior at later learning rates, or FA4
integration correctness for custom auxiliary arms.

The earlier statement that FA4 was “safe” should be read narrowly as passing the
tested baseline numerical checks. It should not be interpreted as a guarantee
of equivalent training trajectories. Likewise, dense SDPA was not shown to be
more accurate numerically; its conservative advantage here is avoiding the extra
FA4 beta dependency and custom production integration.

## 7. Current recommendation and possible next work

Given the user's concern about the unexplained 48% difference, recommend
**dense SDPA isolation for the next real experiments**. In the matched 200-update
measurement, dense isolation took approximately 17.1% more time per step than
FA4; equivalently, FA4 took 14.6% less time than dense. Dense isolation was close
to the old explicit-mask cross-document baseline in the 30-update comparison.
This recommendation has not yet been implemented or accepted as a launch request.

For a production implementation, preserve the established Trainer/Accelerate
recipe and change only the necessary data metadata, collator, position/mask,
loss-target and loss-normalization paths. Apply one consistent backend and
boundary policy to all compared arms. Recheck auxiliary attention, losses,
resume behavior and data order before real runs. A backend change does not make
old cross-document and new isolated training directly equivalent.

If FA4 is revisited, the next informative control is a fresh dense-SDPA run with
the same seed/data/recipe, compared against the existing dense trajectory. That
would measure trajectory variation without changing the backend. Multiple repeats
would be stronger than a single extra run. This control is proposed, not run;
a small same-backend backward variation alone does not show that repeated dense
training develops the observed 48% gap.

## 8. Evidence and operational status

| Stage | Submission | Local evidence or detailed record |
|---|---|---|
| Dense isolation on B200 | `e14a630` | `temp/tjx3-document-a03.log`; [project notes](PROJECT_NOTES.md) |
| Corrected packed-API benchmark | `ca21c7b` | `temp/tjx3-packed-kernels-a02.log`; [project notes](PROJECT_NOTES.md) |
| Five-mode, 30-update training | `daea6f6` | `artifacts/document-training-20261004-a01/`; 50 source-file hashes verified |
| Two-mode, 200-update training | `e06d5f9` | `artifacts/document-stability-200-20261004-a01/`; 24 source-file hashes verified |
| Corrected same-weight follow-up | `3d5070b` | `artifacts/trained-attention-check-20261004-a02/`; 21 source-file hashes verified |

Artifacts and `temp/` files are local/ignored, so these paths will not exist in a
fresh Git clone. Small results were retrieved and checked; large final model
weights remained on the B200 at the last verification. Detailed reports contain
archive checksums, remote paths and collector identities. Experiment launch
commands are preserved in the submission commits' `commands.sh` histories.

Before GPU tests, the resource Accelerate configuration was copied and verified,
`accelerate env` was checked, only identified authorized burn workers were stopped,
and all eight GPUs were required to be free. Automatic burn recovery also ran
after failures. Following the final successful diagnostic, workers 19173–19180
were restored with advancing collective progress at 13:57:24 UTC; a read-only
collector verified live identities and the released guard at 13:58:16 UTC.
These are recorded observations, not a new live status check.

At the end of this investigation, `commands.sh` is inactive (`#0`). This report
request launched no new tests or training and made no environment changes.

## 9. Addendum: per-step analysis of the 48% gap

This section re-reads the per-update logs of the two existing 200-update runs
(`benchmark/{sdpa_isolated,fa4_isolated}/trainer_state.json` under
`artifacts/document-stability-200-20261004-a01/`). No new run was launched. The
question is whether the 48% gradient-norm gap indicates a danger specific to
FA4, or ordinary behavior that dense SDPA also shows.

### 9.1 When and how the runs separated

| Interval | Observation |
|---|---|
| Updates 1–101 | Identical: gradient norms within 0.1%, losses equal to four decimals. |
| Update 102 | First gradient-norm difference above 0.1%. |
| Update 114 | First loss difference above 1e-4. |
| Updates ~120–200 | The trajectories diverge progressively. |

This is the expected pattern when two runs of the same algorithm differ only by
tiny floating-point differences: the differences grow over many updates until
the runs are no longer step-for-step comparable.

### 9.2 The largest gap came from a dense SDPA spike

Gradient norms around the maximum gap:

| Update | Dense grad norm | FA4 grad norm | Dense loss | FA4 loss |
|---:|---:|---:|---:|---:|
| 194 | 1.246 | 0.903 | 7.1842 | 7.1772 |
| 195 | 0.873 | 0.966 | 7.1513 | 7.1526 |
| **196** | **1.766** | 0.916 | **7.1833** | 7.1726 |
| 197 | 0.904 | 0.991 | 7.1775 | 7.1730 |
| 198 | 1.164 | **1.571** | 7.1445 | 7.1485 |
| 199 | 0.824 | 0.840 | 7.1437 | 7.1408 |
| 200 | 0.988 | 1.407 | 7.1169 | 7.1171 |

At update 196 the **dense** run spiked to 1.62× the median of its own previous
ten gradient norms; FA4 was unremarkable. The maximum loss gap (0.0107) occurred
at the same update, when the dense loss rose with its spike. By update 200 the
loss gap was 0.0003.

A spike here means a gradient norm above 1.5× the median of that run's own
previous ten logged norms. Both runs had such spikes:

| Run | Spike updates |
|---|---|
| Dense SDPA | 187 (1.556), 196 (1.766) |
| FA4 | 198 (1.571) |

So dense SDPA itself produced spikes within these 200 updates. The 48% figure
measures two noisy runs spiking at different updates, not an FA4-only
instability.

### 9.3 The cross-run gap is comparable to each run's own step-to-step noise

Over updates 150–200:

| Quantity | Dense SDPA | FA4 |
|---|---:|---:|
| Median change in grad norm between consecutive updates, same run | 18.6% | 15.6% |
| Maximum change between consecutive updates, same run | 102.2% | 67.5% |

| Quantity | Value |
|---|---:|
| Median dense-versus-FA4 gap at the same update | 12.8% |
| Maximum dense-versus-FA4 gap at the same update | 48.1% |
| Updates where FA4's norm exceeded dense's | 31 of 51 |

The median cross-run gap is smaller than either run's median step-to-step
change, and the sign of the gap alternates. Late in this short run, gradient norm
is a highly variable quantity in both backends. That variability is plausible
because:
- the pool is small and repeated 12.5 times;
- the learning rate is still rising in warmup;
- batch composition changes every update.

### 9.4 Effect of gradient clipping

Training clips the global gradient norm at 1.0, so reported norms above 1.0 do
not translate into proportionally larger updates. At update 196 the clipped
norms were 1.000 (dense) and 0.916 (FA4), an 8.4% difference in applied update
scale rather than 48%. Clipping does not remove all differences: over updates
150–200 the largest clipped-norm gap was 35.3%, at an update where both norms
were below 1.0.

### 9.5 Interpretation

- Neither run showed signs of training instability. Losses fell monotonically
  in trend (12.12 → 10.38 → 9.20 → 7.87 → 7.12 at updates 1/50/100/150/200,
  within 0.001 between runs at those points). Every gradient
  spike recovered on the next update. There was no sustained norm growth, loss
  blow-up or nonfinite value.
- The 48% maximum was caused by a dense SDPA spike. Dense SDPA can and did show
  the same kind of event within 200 updates.
- Together with the same-weight comparison (Section 5E), the evidence is
  consistent with ordinary divergence of numerically perturbed trajectories,
  not an FA4 correctness problem.

This remains an inference from one pair of runs. It does not prove that two
dense runs would diverge in the same way. The direct test is the control from
Section 7: rerun dense SDPA with the same seed, data and recipe, and compare it
with the existing dense trajectory. The prediction is that dense versus dense
will also separate after roughly 100 updates, with late-run gradient-norm gaps
of similar size.

### 9.6 Effect on the recommendation

The 48% gap should no longer be read as evidence against FA4. The
recommendation to use dense SDPA isolation for the next experiments still
stands, for the other reasons in Sections 6–7:
- Dense isolation costs about the same as the current explicit-mask baseline.
- It is already implemented and tested for all 11 arms.
- FA4 is integrated only for arm A, and is a beta dependency in a separate
  environment.

FA4 remains a reasonable later optimization, worth about 15% per update, after
it is integrated and tested for every arm that will use it.

## 10. Production Arm A: matched 2,500-update comparison (2026-10-05)

Read-only retrieval at commit `88bddbd` compared the completed dense and FA4
baseline runs. No GPU workload was launched or stopped. Local revision-7 P1/P3
changes were not deployed. Both runs use the same saved model/data/training
configuration, seed42, train/eval fingerprints, eight GPUs, BF16, microbatch16,
accumulation4, sequence2048, EOS boundaries and per-document position resets.
Removing the FA4 `attention_backend` field makes saved configurations identical.
Learning rates match exactly at all250 logged training points. Their full
schedule is28600 updates with1430 warmup updates, stopped at2500.

### Convergence comparison

| Update | Dense SDPA loss | FA4 loss | FA4 minus dense |
|---|---:|---:|---:|
| 0 | 12.124935 | 12.124942 | +0.000007 |
| 512 | 5.657214 | 5.651754 | -0.005460 |
| 1024 | 4.348823 | 4.356540 | +0.007717 |
| 1536 | 3.749307 | 3.758197 | +0.008890 |
| 2048 | 3.484767 | 3.485469 | +0.000702 |
| 2500 (full validation) | 3.477941 | 3.477173 | -0.000768 |

Updates0–2048 use the same128-row monitor subset; the final measurement uses
all4882 validation rows (~10M tokens). Do not interpret the last row as a change
on the same evaluation population. Final perplexities:32.39297 dense /32.36810
FA4. Final absolute loss difference0.00076804 (~0.0221% of dense loss).

Across250 matching ten-update training-loss averages, mean absolute difference
is0.00330542 and maximum0.04351387 atupdate990 (4.52586327 dense /4.56937714
FA4). Maximum relative difference is0.96145%. For logged current-update LM losses,
mean absolute difference0.00462758 and maximum0.02817057. Logs cover every tenth
update; these are not claims about every individual update. All logged losses
and gradient norms are finite.

### Same-weight numerical comparison versus trajectory differences

The production preflight compared identical initialized weights and two real
packed2048-token sequences, BF16, before any optimizer update:

| Quantity | FA4 versus dense |
|---|---:|
| Absolute loss difference | 0.00004196 |
| Hidden-state relative L2 | 0.56898% |
| Sampled-logit relative L2 | 0.63636% |
| Full-gradient relative L2 | 0.54365% |
| Full-gradient cosine similarity | 0.99998523 |
| Largest reported per-parameter gradient relative L2 | 1.30428% |
| Cross-document output/embedding-gradient leakage | Exactly zero in the tested case |

These are close numerical results, not bitwise equality. This preflight uses
microbatch2, not16. The older identical-trained-weight tests in Section5.E cover
microbatch2/16 and FP32 references, but use the earlier200-update checkpoints.
**No same-weight output/gradient comparison on the new2500-update checkpoints
was run in this read-only review.**

For the separately trained2500-update models, logged pre-clipping gradient norms
show mean relative gap13.2256% and maximum63.2908% atupdate980:
0.51419210 dense versus0.83962822 FA4. Definition:abs(FA4−dense)/dense.
Both are below the clipping threshold1 at that update, so clipping cannot erase
that particular norm difference. After warmup (logged updates>1430), the mean gap
is9.8498% and maximum44.5329%. Atupdate2500, norms are0.17089225/0.16141333.
These compare gradients at different learned weights; they are not a kernel
relative-L2 error measurement and give no information about gradient direction.
They must not be represented as all gradients agreeing within1%.

The stronger evidence now supports **similar baseline loss convergence through
2500 updates, including after warmup**, together with close same-weight numerical
checks on the tested inputs. It does not establish identical trajectories,
statistical equivalence across seeds, full-schedule equivalence, or correctness
of FA4 for P1/P3. No evidence here establishes an accuracy advantage for FA4.
A matched-weight backend swap on the new final checkpoints would extend the
numerical check to their current weights; a repeated same-backend run remains
the appropriate control for quantifying ordinary trajectory variation.

Evidence:
- `temp/fa4-dense-numerical-review-20261005-a01.log`, SHA256
  `d3c12db8c3e3d6bedddb36141567a7847a416466cd32aaf4ef057fd427b0d20a`.
- Extracted configs, full Trainer histories, result files, runtime receipts,
  numerics and computed summary:
  `temp/fa4-dense-numerical-review-20261005-a01/`.
- Raw-source SHA256 values are retained in `source-manifest.json`; extracted
  files are reformatted JSON, so their byte hashes can differ from raw sources.

## 11. Final-checkpoint same-weight checks passed (2026-10-05)

This closes the final-checkpoint numerical gap described in Section10. On the
user's request, both completed Arm A checkpoint-2500 models were strictly loaded
and each was evaluated with both dense SDPA and FA4 at unchanged weights. The
production Arm A source used for training was retained; the uncommitted
revision-7 P1/P3 implementation was not deployed.

### Method and acceptance

`scripts/check_proxy_trained_attention.py` reuses the existing strict tied-weight
loader, parameter fingerprinting, full-gradient comparison and numerical gates.
Five focused CPU tests passed locally on the committed source snapshot and on
B200. CPU preflight loaded both full checkpoints before GPU reclamation. The
first512 documents from each original English train/validation split were packed
using the shared EOS/document-segment functions; each split supplies16 sequences
of2048 tokens. These are diagnostic batches from the real sources, not a replay
of a particular optimizer update. Micro2 cases use prefixes of micro16, not
independent additional datasets. Positions reset at each document boundary.

Eight independent GPU cases: two checkpoints × train/validation × microbatch2/16.
Each takes dense and FA4 BF16 forward/backward captures, then repeats both to
measure within-backend variation. The four micro2 cases additionally use FP32
math SDPA with TF32 disabled as a reference. FP32 micro16 was deliberately not
run because its capacity has not been established. No optimizer was constructed,
no checkpoint was overwritten, and no training was resumed.

The unchanged gates are absolute loss<0.01, full-gradient relative L2<3%,
hidden/logit relative L2<2%, and matching target counts. Logits are sampled every
128th position; loss and parameter gradients use all valid targets. All gradients,
losses and captured outputs were checked finite. All eight backend comparisons
and all eight BF16-versus-FP32 comparisons (two backends × four cases) passed.

### Production microbatch16 results

| Checkpoint | Batch source | Absolute loss gap | Gradient relative L2 | Gradient cosine | Hidden relative L2 | Sampled-logit relative L2 |
|---|---|---:|---:|---:|---:|---:|
| Dense-trained | train | 0.00005198 | 1.2394% | 0.99992321 | 0.5047% | 0.2080% |
| Dense-trained | validation | 0.00004745 | 0.7323% | 0.99997326 | 0.4982% | 0.2161% |
| FA4-trained | train | 0.00006604 | 1.2730% | 0.99991900 | 0.5000% | 0.2056% |
| FA4-trained | validation | 0.00000525 | 0.7598% | 0.99997122 | 0.4946% | 0.2119% |

Across all eight cases (including micro2), backend gradient relative L2 spans
0.6966–1.2730%, maximum absolute loss gap0.000279665, and minimum gradient cosine
0.99991900. Production micro16 maximum loss gap is0.000066042.

Micro2 FP32-reference gradient relative L2 ranges:
- Dense BF16 versus FP32 math:0.9224–1.2157%.
- FA4 BF16 versus FP32 math:0.9089–1.1907%.

Both backends therefore have similar gradient accuracy against FP32 in these
cases. Repeated captures at unchanged weights had identical forward outputs and
losses. Backward gradients varied: dense-repeat relative L2=0.3120–0.5177%;
FA4-repeat=0–0.2980%. This is whole-model backward variation, not a standalone
attention-kernel nondeterminism measurement. It does not establish that FA4 is
universally deterministic or more accurate.

All in-memory parameter fingerprints were unchanged, and SHA256 of both source
checkpoint files matched before/after the complete job. Peak allocated GPU memory
was101.736GiB for micro16 and34.129GiB for micro2 including FP32 capture. No OOM or
nonfinite result occurred. GPU cases and validation finished successfully in
98.70seconds, at08:02:47UTC /16:02:47 Asia/Singapore.

### Interpretation and handoff

These results support numerical correctness of the current Arm A FA4 integration
at the new final weights, alongside the similar2500-update convergence already
measured. The separately trained models'63.3% maximum gradient-norm gap is not
reproduced as a same-weight backend gradient error; their training trajectories
have different learned weights. This test does not establish the precise cause
of that trajectory gap or full-schedule/multiple-seed equivalence. FA4 validation
for custom proxy arms remains separate. No additional baseline backend check is
required for the question posed here.

Accelerate config was copied and verified (eight GPUs, BF16). Known burn workers
46080–46087 were identity-checked and stopped; all eight GPUs were verified free
before the diagnostic. The supervisor restored the approved burn automatically
at08:03:58UTC. Fresh08:05:22UTC inspection verified workers47839–47846, one per GPU,
all100% utilization, all-rank readiness and newly advancing collective progress;
the guard was released. Environments/drivers and training checkpoints were not
modified. commands.sh was returned to#0 after retrieval.

Launch commit:`e2b5bfa`; final monitor:`eae9f2c`. Remote output root:
`/mnt/local/_outputs/deep-llms_th2/proxy-final-attention-check-20261005-a01`.
Local evidence:
- `temp/proxy-final-check-monitor-a02.log`, SHA256
  `14b5295fb3667566f60a549b38c72e49c7669ad88a755d8bf7af992160055427`.
- `temp/proxy-final-check-results-a01/`: extracted manifest, supervisor, run,
  summary and burn receipts, with exact JSON byte hashes verified against the
  remote artifact records.
- Summary SHA256 matches the run manifest:
  `d2e88f3eef4765fcc3fea379663c239d13ccec851ed34bc4022c1577f5b2ee95`.
