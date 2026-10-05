# Recommendations on proxy target normalization

Date: 2026-10-05 (Asia/Singapore).

Reviewed specification: [proxy heads, revision 6](proxy_heads_P1_P3_spec_v3.md),
especially Sections 5.5, 6, 7 and test T9. This is a recommendation, not an
implementation or training-result report. The current code still implements
the earlier static-mask normalization and P3 full-state targets.

## Recommendation

Adopt the proposed per-channel running standardization, variance floor and
target clipping in place of the static channel mask. Retain cosine as the
default auxiliary loss. Defer Smooth L1 to an ablation after a cosine arm shows
a useful signal.

Here, clipping means capping normalized target values at ±10; it does not refer
to a CLIP-style contrastive objective or to gradient clipping.

For each raw target channel, the proposed transformation is:

```text
scale²[c] = max(running_variance[c], 0.01 * median(running_variance)) + 1e-6
target[c] = clamp((raw_target[c] - running_mean[c]) / sqrt(scale²[c]), -10, 10)
```

This addresses the scale imbalance directly while retaining all channels:

- Centering removes the shared channel offset.
- Per-channel scaling reduces domination by high-variance channels.
- The variance floor limits amplification of nearly constant channels.
- Clipping limits the contribution of rare extreme values.

Each run estimates statistics for its own representations. There is no external
calibration checkpoint or list of channel indices to transfer between seeds.
A mask measured on another run is not guaranteed to identify the same outlier
channels, and a fixed mask cannot adapt as representations change.

This is a stronger operational design for the screen, not evidence that the
method will improve held-out language-model loss. The synthetic cosine results
quoted in the specification motivate the change but were not independently
reproduced in this review and do not establish a training gain.

## Clarifications before implementation

### 1. Clipping changes learning gradients

Replace the statement that clipping never affects gradients with:

> Target construction, including normalization and clipping, is detached.
> Therefore no gradient propagates through the target branch. Clipping still
> changes the regression target and consequently the prediction-side gradients.

In the block variant, those auxiliary gradients reach the estimator. In the
flow variant, they also reach the backbone through the estimator input.
Detachment does not make the choice of target irrelevant to optimization.

### 2. Define the running variance precisely

The specified update averages each optimizer step's token variance:

```text
running_variance = 0.99 * running_variance + 0.01 * step_variance
```

It is not the exact pooled variance of all historical tokens when step means
change. It omits the between-step mean variation that a pooled variance would
include. Averaging observed variances is a reasonable normalization convention;
PyTorch BatchNorm also maintains a moving average of observed variance, though
its estimator and training-time behavior differ from this proposal.
[PyTorch BatchNorm documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html)

Keep the specification's convention initially and label it accurately. Do not
silently substitute a different variance estimator. Monitor whether running
statistics lag rapidly changing representations, particularly early in training.

For T9, test the mean/variance expectations on a stationary synthetic
distribution. Floored channels need not have unit variance, and clipping can
also change the mean and variance. Exact zero mean/unit variance should not be
an acceptance requirement for arbitrary clipped targets from a changing model.
Specify tolerances for the approximate held-batch check before implementing it.

### 3. Make initialization numerically stable without retaining all targets

Initialization first needs the global mean, then squared deviations from that
mean. A single streaming pass cannot discard raw targets and later compute
those deviations without recomputation or an alternative stable algorithm.

Prefer two no-grad forwards on the same first microbatch:

1. Collect and all-reduce sums/counts to initialize each global mean.
2. Recompute the same targets with unchanged weights, collect squared deviations
   from that mean, and all-reduce them to initialize variance.

Do not advance the dataloader or optimizer between these passes, or use the
unstable unshifted `E[x²] - E[x]²` shortcut. Release completed target windows
promptly instead of retaining all P3 increment targets at full token width.
The model currently has zero attention dropout; any future stochastic target
computation would also need controlled replay.

During ordinary training, accumulate the specified shifted sums in FP32,
all-reduce sums and counts, and update once per optimizer step. Freeze statistics
within that step, during activation-checkpoint recomputation and during evaluation.
Update the same statistics in lambda-zero controls and save all buffers and
initialization state in checkpoints.

### 4. Describe the gate as learned rescaling, not exact inversion

Keep the per-channel gate. It gives the model flexibility to map standardized
predictions into useful channel scales for K/V computation.

However, a gate cannot exactly invert clipping, and RMS normalization removes
prediction magnitude. Avoid claiming that the gate reconstructs raw residual
increments exactly. A more accurate statement is:

> Per-channel gates can learn a useful rescaling of standardized predictions;
> exact recovery of the original target values is not required or guaranteed.

## P3 increments are a separate methodological change

Revision 6 additionally replaces full-state targets with:

```text
increment[proxy_layer, deep_layer] = h[deep_layer] - h[proxy_layer - 1]
```

This is well motivated: subtracting the input residual removes its direct
contribution, so the target emphasizes what subsequent blocks compute instead
of rewarding a direct copy of shallow content. This does not make the increment
independent of shallow information, nor establish that it is easier to predict.

Clarify that `h[proxy_layer - 1]` is unaffected by the current layer's proxy
operation; it can already contain effects from earlier proxy layers.

Implement the specified per-(proxy layer, deep layer) statistics, FP32 detached
subtraction, normalization before band averaging, and document-reset EMS.
Treat this as a new target definition, not merely removal of the mask. Record
the normalization/target version and hyperparameters in the run configuration,
and reject silent resume from the old target definition.

## Evidence, comparison and next steps

Target normalization has precedent: data2vec normalizes block representations
before averaging them and discusses preventing high-norm layers from dominating.
Its NLP/vision targets use layer normalization, while speech uses instance
normalization. That supports the general motivation, not this exact running
per-channel scheme, clipping threshold or P3 increment target.
[data2vec paper, Section 3.3](https://proceedings.mlr.press/v162/baevski22a/baevski22a.pdf)

Use the proposed global running statistics rather than subtracting a separate
per-document temporal mean: the latter would remove document-level components
that P3's slow timescale is intended to predict.

Before training, verify FP64-reference statistics, distributed synchronization,
unchanged buffers during recomputation/evaluation, exact save/resume, detached
targets and gradient routing. Log clipping fraction, floored-channel counts,
median variance and seconds per update. The specification's clipping thresholds
are investigation triggers, not demonstrated universal safe ranges.

The completed dense-SDPA A baseline remains usable when data, seed, schedule,
attention and execution settings match. A has no proxy auxiliary targets, so
these changes alone do not require retraining it. Report matching must recognize
that distinction while remaining strict between proxy arms and their lambda-zero
controls. The FA4 A variant remains a separate backend experiment.

No code changes, specification edits or GPU work were performed for this recommendation.
