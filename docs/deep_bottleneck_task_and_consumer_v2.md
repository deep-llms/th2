# Deep Bottleneck: Task-Aware and Consumer-Aware Variants
## Option 2: Two Extractor Objectives, Shared One-Way Code Alignment

**Status:** Proposed experimental specification, version 2. This document includes both the earlier task-aware extractor proposal and the consumer-aware proposal. Neither variant has yet been shown to improve the existing runs. First-run defaults are experimental choices, not established optima.

## 1. Objective and scope

Learn a compact property of a deep representation, then train a shallow predictor to produce that property early for use by future shallow queries.

The real computation is identical in both variants:

$$
\text{shallow state}
\longrightarrow \text{predicted code}
\longrightarrow \text{auxiliary K/V}
\longrightarrow \text{future shallow attention}.
$$

The actual deep state still appears later in the same backbone forward. It supplies the extractor's input, not an input to the real shallow computation.

**Test both versions separately.** They differ in how the extractor learns a useful code, not in the real predicted-code architecture:

| Version | Extractor objective | Aligned training objective |
|---|---|---|
| **Task-Aware-Align** — earlier, simpler proposal | Predict the next token directly from the extracted deep code | $L_{\mathrm{LM}}+L_{\mathrm{extract}}+0.3L_{\mathrm{align}}$ |
| **Consumer-Aware-Align** — newer proposal | Predict the next token from a shallow state after it reads past deep codes through the current auxiliary consumer | $L_{\mathrm{LM}}+L_{\mathrm{use}}+0.3L_{\mathrm{align}}$ |

Do not add both extractor objectives to one run. These are alternatives, not a combined fourth loss.

Both versions have three responsibilities:

- **Main LM loss:** train the real predicted-code model, including its predictor and consumer, to improve language modeling.
- **Extractor objective:** select deep information by task utility or local shallow-consumer utility, depending on the version.
- **One-way alignment loss:** teach the shallow predictor to match the extracted code without letting alignment pull the extractor toward the predictor.

In the task-aware version, alignment can transfer task-relevant information into the predictor, while the real LM loss directly teaches the predictor and consumer how to use it. A separate consumer-utility loss is not a prerequisite for that mechanism. Neither version guarantees an improvement in held-out LM loss.

**Both use a new bottleneck architecture, not merely another loss for the unchanged F/G architecture.** The old direct shallow-to-K/V projections are replaced by shallow-to-code-to-K/V mappings. Keep the surrounding attention branch unchanged.

Both use one token-parallel backbone forward, with no recurrent token-by-token execution, separately pretrained teacher, tail rerun, or calibration stage. Both add training-only extractor/readout computation. Only the consumer-aware version additionally evaluates attention over actual extracted deep codes for its utility loss.

## 2. Inherited backbone and tensor locations

Reuse the backbone, data pipeline, and auxiliary attention placement from the implemented runs. The existing document convention uses **1-based block numbering**:

| Symbol | Tensor |
|---|---|
| $h_t^s$ | Residual after block 4, before block 5 input normalization |
| $u_t^s$ | The existing block-5 normalized attention input |
| $Q_t^s$ | The native block-5 query after its normal Q normalization and RoPE |
| $h_t^d$ | Residual after block 20, before block 21 input normalization |

Use the exact hook indices recorded in the executed runs. If those differ from the written convention, record that discrepancy rather than silently moving the hooks.

The inherited configuration has 16 query heads, 8 KV heads, and head dimension 128. Preserve the existing head mapping, positional treatment, native attention, and auxiliary output injection. This version does not introduce hybrid heads or replace native self-attention.

Use the auxiliary visibility rule:

$$
\operatorname{allow}(t,j)
=
(j<t)\ \land\ \operatorname{backbone\_allows}(t,j).
$$

The predicted branch and the training-only consumer use exactly this rule, including packing isolation and padding. A query with no allowed source produces zero auxiliary output; never softmax an all-masked row.

## 3. Modules and real forward path

### 3.1 Code representation

Use a code width of **128** for the proposed first implementation. Define parameter-free, per-token RMS normalization over code coordinates:

$$
R(a)=\frac{a}{\sqrt{\operatorname{mean}_k(a_k^2)+\epsilon}},
\qquad \epsilon=10^{-6}.
$$

There is no learned scale or bias in $R$, no running statistic, and no calibration. Use the normalized code in the actual decoder as well as in alignment; do not normalize only for the loss.

The shallow predictor produces:

$$
\hat z_t=R(P(u_t^s)).
$$

The deep extractor produces:

$$
z_t^d=R(E(\operatorname{sg}(h_t^d))).
$$

Here $\operatorname{sg}$ means stop-gradient. The deep tensor's value comes from the current model and batch. It is not a clean, feedback-free counterfactual and is not a cached target from an older checkpoint.

For a simple first implementation, use separate bias-free linear maps $P$ and $E$. Their input widths are the backbone hidden width and their output widths are 128. They are not tied.

### 3.2 Code-to-K/V decoder

Replace the old direct auxiliary K/V projections with two bias-free linear decoder maps:

$$
\hat K_t^{\mathrm{content}}
=\operatorname{KNorm}_{\mathrm{aux}}(D_K\hat z_t),
\qquad
\hat V_t=D_V\hat z_t.
$$

Each decoder outputs 8 KV heads × 128 dimensions. Apply the existing RoPE and position IDs to the content keys exactly once, then use the same GQA grouping as before.

The real auxiliary branch remains:

$$
\hat m_t
=\operatorname{MHA}(Q_t^s,\hat K_{<t},\hat V_{<t}),
\qquad
\hat c_t=W_O^{\mathrm{aux}}\operatorname{ConcatHeads}(\hat m_t).
$$

Add $\hat c_t$ at the existing attention-residual location:

$$
h_t^s+\operatorname{NativeAttentionOutput}_t+\hat c_t,
$$

then run the normal remainder of the consumer block and the rest of the backbone. Preserve the zero-initialized auxiliary output projection and all other existing initialization conventions.

Only $\hat z$ is used in this real forward. Actual $z^d$, deep K/V, and either training-only readout are never fed back into it.

## 4. Shared loss — Main language-modeling objective

Keep the existing next-token loss unchanged:

$$
L_{\mathrm{LM}}
=\operatorname{CE}(\mathrm{main\ logits},\mathrm{labels}).
$$

It trains the backbone and the real predicted-code branch: $P$, $D_K$, $D_V$, auxiliary key normalization, and $W_O^{\mathrm{aux}}$.

**Validation LM loss means this objective only.** Do not compare a total containing auxiliary losses against the earlier held-out LM numbers.

## 5A. Extractor objective — Task-aware version

### 5A.1 Purpose

This is the earlier, simpler proposal. Learn a deep code that retains information useful for the next-token task, then transfer that code to the shallow predictor through alignment.

The real main LM objective still trains the predictor and decoder through their actual attention use path. The design therefore combines task-guided transfer with direct end-to-end learning of how to use the predicted code.

It does **not** claim that next-token predictiveness of the deep code automatically guarantees usefulness to a future shallow consumer.

### 5A.2 Direct code readout and loss

Use the shared extractor definition:

$$
z_t^d=R(E(\operatorname{sg}(h_t^d))).
$$

Add a separate training-only readout $C_{\mathrm{task}}$ that reads this normalized code directly:

$$
\ell_t^{\mathrm{task}}=C_{\mathrm{task}}(z_t^d).
$$

For the first implementation, use one independent, bias-free linear map from the 128-dimensional code to the vocabulary. It is not tied to the main LM head and is not used by the real forward.

Define:

$$
\boxed{
L_{\mathrm{extract}}
=\operatorname{mean}_{t\in\mathcal T}
\operatorname{CE}(\ell_t^{\mathrm{task}},x_{t+1}),
}
$$

where $\mathcal T$ contains non-padding source positions with a valid next-token label under the existing data pipeline. Reuse the existing label-shift and ignore-mask convention; do not shift labels twice.

Unlike the consumer-aware query set $\mathcal Q$, $\mathcal T$ does not require an available strict-past source. The task-aware readout predicts from the code at the current source position, so the first token may contribute when it has a valid label.

### 5A.3 Gradient routing

$L_{\mathrm{extract}}$ directly trains **$E$ and $C_{\mathrm{task}}$ only**.

- Keep the deep input detached: $E(\operatorname{sg}(h^d))$.
- Do not detach $z^d$ before passing it to $C_{\mathrm{task}}$; the extractor must receive this loss.
- Detach $z^d$ only when it is used as the alignment target in Section 6.
- No extractor-loss gradient enters the backbone, shallow predictor, decoder, auxiliary attention, or main LM head.

This version does not run the loss-only consumer from Section 5B. It does not produce attention messages from actual deep codes, even for its extractor loss.

Because the direct code readout does not pass through $W_O^{\mathrm{aux}}$, zero initialization of the real auxiliary output projection does not by itself block the task-aware extractor's initial gradient.

### 5A.4 What this loss does not guarantee

A code useful to $C_{\mathrm{task}}$ may contain information that the shallow consumer cannot exploit, or information already available in shallow states. The extractor is not explicitly optimized for shallow predictability or incremental consumer utility.

However, the shallow predictor is not trained by alignment alone: $L_{\mathrm{LM}}$ also directly trains it and the decoder through the actual use path. The hypothesis is that these two signals complement each other.

Lower task CE and lower alignment error are diagnostics, not sufficient evidence of success. Judge the method by held-out **main LM loss** against its matched no-alignment control.

## 5B. Extractor objective — Consumer-aware version

### 5B.1 Purpose

Unlike the task-aware objective in Section 5A, this objective does not read $z_t^d$ alone. It evaluates the code through a local shallow consumer. This is an alternative surrogate, not an established improvement over the task-aware version.

Instead, the local loss evaluates a shallow token **after it reads past deep codes through the same kind of consumer used by the real branch**.

It is a local surrogate for usefulness to the shallow consumer. It does not evaluate the true downstream tail after a deep-code intervention; doing that would require an additional tail computation.

### 5B.2 Reuse the consumer with detached parameters

After $h^d$ becomes available, decode $z^d$ through the current auxiliary decoder and attention consumer. Reuse the current parameter values, but detach their parameters **only for this utility computation**:

$$
K_t^E
=\operatorname{RoPE}\left(
\operatorname{KNorm}_{\mathrm{aux},\operatorname{sg}(\theta)}
\left(\operatorname{sg}(D_K)z_t^d\right)
\right),
$$

$$
V_t^E=\operatorname{sg}(D_V)z_t^d,
$$

$$
m_t^E
=\operatorname{MHA}\left(
\operatorname{sg}(Q_t^s),K_{<t}^E,V_{<t}^E
\right),
$$

$$
c_t^E
=\operatorname{sg}(W_O^{\mathrm{aux}})
\operatorname{ConcatHeads}(m_t^E).
$$

The query is detached. The decoder weights, auxiliary key-normalization parameters, and auxiliary output projection are detached. **The extracted code and the operations consuming it remain differentiable.**

Do not wrap this entire consumer computation in `no_grad()`: that would prevent the utility loss from training $E$. Do not detach $K^E$, $V^E$, or $c^E$ after computing them.

There is no persistent frozen copy of the consumer. The loss-only consumer uses the current weights at each update, while the real consumer continues learning through $L_{\mathrm{LM}}$.

### 5B.3 Local prediction objective

Use a separate training-only readout $C_\eta$ on the shallow residual plus the extracted-code message:

$$
\ell_t^{\mathrm{local}}
=C_\eta\left(\operatorname{sg}(h_t^s)+c_t^E\right).
$$

Then:

$$
\boxed{
L_{\mathrm{use}}
=\operatorname{mean}_{t\in\mathcal Q}
\operatorname{CE}(\ell_t^{\mathrm{local}},x_{t+1})
}
$$

where $\mathcal Q$ contains non-padding query positions with both a valid next-token label and at least one allowed strict-past source. Reuse the existing label-shift and ignore-mask convention; do not shift labels twice.

This local readout uses $h^s+c^E$, not a rerun of the consumer's full block or downstream tail. That simplification is deliberate and limits what the local objective can establish.

**Proposed readout default:** one independent, bias-free linear map from backbone hidden width to vocabulary size. Its exact parameterization was not fixed in the discussion; this is the simplest explicit implementation choice, not a tested optimum. It is trained only by $L_{\mathrm{use}}$ and is not tied to the main LM head.

This choice can be substantial: it adds $d_{\mathrm{model}}\times|\mathcal V|$ training-only weights plus local vocabulary logits. The task-aware readout instead has $128\times|\mathcal V|$ weights. Count this difference as well as the consumer-aware loss-only attention cost. A tied or low-rank alternative would change this default and should be recorded explicitly rather than silently substituted.

The utility loss directly trains **$E$ and $C_\eta$ only**. It does not directly update the backbone, shallow predictor, or consumer parameters.

### 5B.4 What this loss does not guarantee

The local head can rely primarily on $h^s$ and ignore the message. Joint learning of $E$ and the local head can also produce a code that helps the local readout more than the real tail. Neither CE nor code normalization guarantees a non-collapsed or useful code.

As a small diagnostic, evaluate the same local head on $\operatorname{sg}(h^s)$ without $c^E$ on the same eligible queries. A lower CE with the message is evidence of local usefulness, not proof of end-to-end usefulness. This is a metric, not an additional training objective.

## 6. Shared loss — One-way code alignment

Use:

$$
\boxed{
L_{\mathrm{align}}
=\operatorname{mean}_{t,k}
\operatorname{SmoothL1}\left(
\hat z_{t,k},\operatorname{sg}(z_{t,k}^d);\beta=1
\right).
}
$$

Average over non-padding source-token positions and code dimensions. This is a source-token loss, not an attention-query loss: the first token can have a useful code for later queries even though it has no past source of its own.

In terms of an error $e$, the per-coordinate function is:

$$
\operatorname{SmoothL1}_{\beta=1}(e)
=\begin{cases}
\tfrac12 e^2,& |e|<1,\\
|e|-\tfrac12,& |e|\ge1.
\end{cases}
$$

There are two distinct detach points:

| Detach | Meaning |
|---|---|
| $E(\operatorname{sg}(h^d))$ | Extractor-side losses do not backpropagate through the deep backbone input |
| $D(\hat z,\operatorname{sg}(z^d))$ | Alignment does not train the extractor or change its target to follow the predictor |

Alignment updates $P$ and the shallow backbone through $u^s$. It does **not** directly train $E$, either training-only readout, or the code-to-K/V decoder.

The extractor still changes online through $L_{\mathrm{extract}}$ in the task-aware version or $L_{\mathrm{use}}$ in the consumer-aware version. Stop-gradient on the alignment target does not freeze it across training steps.

**Do not use joint/two-sided alignment in either version.** Removing `stopgrad(z_deep)` would add a different pressure: making the extracted code easier to match. That is not the same objective as making it useful for the task or the shallow consumer.

## 7. Total objectives and gradient routing

Carry forward the proposed option-2 starting coefficients.

**Task-Aware-Align:**

$$
\boxed{
L_{\mathrm{TA}}=L_{\mathrm{LM}}+L_{\mathrm{extract}}+0.3L_{\mathrm{align}}.
}
$$

**Consumer-Aware-Align:**

$$
\boxed{
L_{\mathrm{CA}}=L_{\mathrm{LM}}+L_{\mathrm{use}}+0.3L_{\mathrm{align}}.
}
$$

The extractor-objective coefficient of 1 and alignment coefficient of 0.3 are **experimental defaults**, not values established by TED or the previous routing-KL runs. Equal numeric coefficients across different losses do not imply equal gradient strength. Do not tune these coefficients during a run.

| Objective | Backbone | Predictor $P$ | Decoder/auxiliary consumer | Extractor $E$ | Active training-only readout |
|---|---|---|---|---|---|
| $L_{\mathrm{LM}}$ — both versions | Yes | Yes | Yes | No | No |
| $L_{\mathrm{extract}}$ — task-aware only | No direct gradient | No | No | Yes | $C_{\mathrm{task}}$ |
| $L_{\mathrm{use}}$ — consumer-aware only | No direct gradient | No | Detached parameters; gradient passes through operations to the code | Yes | $C_\eta$ |
| $L_{\mathrm{align}}$ — both versions | Shallow path only | Yes | No | No | No |

These are direct autodiff paths. Shared global gradient clipping can still couple update magnitudes across parameter groups. Use the same optimizer/clipping setup and module set **within each aligned/no-alignment pair**, rather than treating the table as a guarantee of independent updates.

Compute loss reductions and code RMS reductions in fp32. Normalize each objective by its own valid element count across the global update, not by a sum that grows with sequence length, microbatch count, or device count. Keep the existing backbone execution precision.

No native K/V reconstruction, routing KL, G-style message imitation, covariance penalty, adaptive loss balancing, or new gate is added.

## 8. One-forward execution order

The real LM path is shared:

1. Run the normal prefix to the shallow hook and compute $\hat z$.
2. Decode $\hat z$ to auxiliary K/V, run the real strict-past branch, and continue the normal backbone. Retain shallow residuals and queries for the loss-only consumer **only in the consumer-aware version**.
3. When the deep hook is reached, retain $\operatorname{sg}(h^d)$. Finish the same backbone forward and compute the main LM loss.
4. Compute $z^d=R(E(\operatorname{sg}(h^d)))$.
5. Compute exactly one extractor objective:
   - **Task-aware:** apply $C_{\mathrm{task}}$ directly to $z^d$ and compute $L_{\mathrm{extract}}$.
   - **Consumer-aware:** pass $z^d$ through the current detached-parameter consumer, apply $C_\eta$ to $\operatorname{sg}(h^s)+c^E$, and compute $L_{\mathrm{use}}$.
6. For aligned runs, compute $L_{\mathrm{align}}$ against `z_deep.detach()`. The corresponding NoAlign run omits this term.
7. Backpropagate the combined objective and perform the ordinary optimizer update.

For the consumer-aware loss, do not globally toggle consumer parameters to `requires_grad=False` around a second module call. Use a functional application of their detached current values so the already-built real forward still has its ordinary LM gradient path.

At initialization, $W_O^{\mathrm{aux}}=0$ also makes the consumer-aware utility message zero. Consequently, the utility gradient into $E$ is initially zero through this path. That is expected; it can become nonzero once LM training opens the auxiliary output projection. Do not change the real branch initialization to bypass this behavior. This particular limitation does not apply to the task-aware direct code readout.

## 9. Experiments and matched controls

### 9.1 Two primary variants

The two proposed methods to try are **Task-Aware-Align** and **Consumer-Aware-Align**. They share the same real predictor, bottleneck width, decoder, consumer, target detach rules, alignment loss, and inference architecture.

Only the extractor-training objective and the training-only computation needed to evaluate it differ. Do not combine the two utility objectives.

### 9.2 Corresponding no-alignment controls

The new bottleneck changes the architecture relative to B/F/G, so those old runs alone cannot isolate the benefit of code alignment. Retain the existing matched-control design for each extractor objective:

| Pair | Run | Training objective |
|---|---|---|
| Task-aware | **Task-Aware-NoAlign** | $L_{\mathrm{LM}}+L_{\mathrm{extract}}$ |
| Task-aware | **Task-Aware-Align** | $L_{\mathrm{LM}}+L_{\mathrm{extract}}+0.3L_{\mathrm{align}}$ |
| Consumer-aware | **Consumer-Aware-NoAlign** | $L_{\mathrm{LM}}+L_{\mathrm{use}}$ |
| Consumer-aware | **Consumer-Aware-Align** | $L_{\mathrm{LM}}+L_{\mathrm{use}}+0.3L_{\mathrm{align}}$ |

The previous document's `Bottleneck-NoAlign` is the **Consumer-Aware-NoAlign** row above; only its label is clarified here.

Thus there are **two proposed aligned methods**, or **four run configurations when including both matched controls**. The controls are not additional proposed architectures.

Within each pair, retain the same extractor, readout, extractor-loss computation, parameter initialization, and optimizer setup. The only difference is whether alignment trains the shallow pathway. A NoAlign run still trains its loss-only extractor but does not transfer its code through an alignment objective.

Do not assume the two NoAlign runs are numerically identical merely because neither transfers code by alignment. Their training-only gradients, compute, and parameter sets differ, and shared global clipping can affect the real model's updates. Merging them into one control would require an explicitly different gradient-management setup, which this version does not introduce.

### 9.3 Training conditions

Use the actual established training recipe: approximately 1M tokens per optimizer update, the intended approximately 30k-step scheduler, and approximately 1,400 warmup steps. Do not revive the older 32k-token-batch pilot schedule. The exact saved run configuration takes precedence over these rounded descriptions.

The continuation horizon for this new architecture has not been fixed in the option-2 discussion. Choose one common horizon before launch and compare at identical completed updates. This document does not silently change that budget.

Start the new runs from matched fresh initialization. Use identical shared real-model weights and identical $P,E,D_K,D_V$ initialization across variants where shapes match; use identical training-only readout initialization within each pair. Do not silently reshape or load B/F/G predictor weights into the bottleneck. Keep data, ordering, global token batch, and evaluation sequences matched.

Training compute is **not** identical between task-aware and consumer-aware methods. The latter has an additional loss-only attention calculation and a larger default readout. Report parameter counts, throughput, peak memory, and main LM quality at the same token/update budget. Both methods have the same inference architecture.

### 9.4 Main comparisons

- **Task-Aware-Align vs Task-Aware-NoAlign:** does task-filtered deep code transfer improve the real LM beyond the same bottleneck without alignment?
- **Consumer-Aware-Align vs Consumer-Aware-NoAlign:** does locally consumer-filtered code transfer improve the real LM beyond its own matched control?
- **Task-Aware-Align vs Consumer-Aware-Align:** which of these two concrete extractor-training recipes yields better main LM quality, with their training-cost differences reported?

Compare B/F/G at matched horizons as additional context. Do not infer that the new architecture benefits from deep knowledge solely because it beats an older model with different forward parameterization.

Lower alignment error or extractor CE alone is not success. A tiny single-run LM difference is not by itself a reliable gain. Do not predeclare the consumer-aware method superior because its extractor objective is more elaborate.

## 10. Minimal correctness checks

1. **Forward isolation:** neither extracted deep code nor a training-only readout enters the actual LM forward. Disabling auxiliary loss computation leaves real logits unchanged at fixed weights and inputs. With matched real parameters, both variants produce the same real logits.
2. **Causality and masks:** the real branch and, when present, the loss-only consumer obey strict-past visibility intersected with the backbone mask; empty-source rows are finite and zero. Modifying future input tokens cannot change earlier real auxiliary outputs.
3. **Extractor-gradient isolation:** task-extractor-only backward updates $E$ and $C_{\mathrm{task}}$; consumer-utility-only backward updates $E$ and $C_\eta$. Neither directly updates the backbone, predictor, or consumer parameters. Test the consumer-aware path after setting a nonzero auxiliary output projection in a tiny test model, so zero initialization does not hide a broken gradient path.
4. **Alignment-gradient isolation:** alignment-only backward updates $P$ and the shallow path, but not $E$ or the deep target branch.
5. **Real-path learning:** LM-only backward can train the predicted-code branch and backbone. Initial zero gradients into upstream auxiliary weights with zero $W_O^{\mathrm{aux}}$ are expected, not a reason to detach the branch.
6. **Matched runs and reductions:** each pair has identical initial states, data order, training horizon, and applicable loss reductions; only the alignment coefficient changes from 0 to 0.3. Task CE uses $\mathcal T$; consumer CE uses $\mathcal Q$; alignment uses non-padding source tokens and code dimensions.

## 11. Interpretation and boundaries

This proposal distinguishes three concepts:

- **Task utility:** deep code predicts labels through a head of its own. This is what the task-aware extractor objective measures.
- **Local consumer utility:** a shallow state predicts labels after reading past deep codes through the current auxiliary consumer. This is what the consumer-aware extractor objective measures.
- **End-to-end utility:** the real model's held-out LM loss improves when it uses predicted codes. This is the success criterion for both methods.

Neither extractor objective guarantees the third property. Neither explicitly forces the extractor to choose a code that is easy for the shallow predictor to estimate. One-way alignment deliberately leaves code selection to the extractor objective and prediction to the shallow pathway.

The task-aware version is not presumed inferior: task relevance can transfer through alignment, and the real LM gradient directly trains the predictor and consumer to exploit it. The consumer-aware version tests a more usage-specific local surrogate, but its local head may ignore the message or learn a use different from the real tail. The two experiments resolve that trade-off empirically.

These designs are inspired by task-aware filtering but are not reproductions of TED. Both learn the extractor online; the consumer-aware version additionally uses a loss-only shallow consumer. No claim that TED validates these exact objectives, stop-gradient layouts, or coefficients is made here.

At inference, remove $E$, the active training-only readout, and the extractor/alignment loss computations. Keep only the shallow predictor, code-to-K/V decoder, and the real auxiliary branch. Actual deep layers remain part of the backbone as usual.

**Core comparison:** teach the early communication channel using a task-filtered deep code versus a locally consumer-filtered deep code, with the real architecture and one-way alignment held fixed. Judge both by the real LM, not by matching quality alone.

## 12. Implementation (2026-09-30)

The four arm names above are accepted by `train.py --arm`, `deep_kv make-jobs`,
`deep_kv report`, and the checkpoint staging helper. Implementation is in the
existing `deep_kv/model.py` and `deep_kv/training.py`; there is no additional
training entry point or preprocessing stage.

The bottleneck width is fixed at 128, RMS epsilon at 1e-6, extractor coefficient
at 1, and alignment coefficient at 0 or 0.3 according to the arm. Default block
coordinates remain consumer=5/deep_target=21. The new target is the residual
**entering** block 21, rather than the native block-21 K/V used by earlier arms.
All four arms start fresh through `DeepKV.from_scratch`; old B/F/G checkpoints
are incompatible. A bottleneck arm can resume its own native Trainer checkpoint.

The ordinary EOS packing, dataset cache/shuffle, optimizer, shared gradient
clipping, scheduler, and native Trainer/Accelerate loop are retained. For these
arms only, Trainer's item-count hook counts LM labels, extractor labels/queries,
and alignment source tokens separately across the accumulated distributed batch.
This avoids weighting short or masked examples as though they had equal counts.
The scalar `eval_loss` is main LM CE; `eval_objective` includes the auxiliaries.
Legacy A–G evaluation fields retain their established meanings.

Consumer evaluation also reports `eval_loss_use_no_message` and
`eval_message_ce_gain` (positive means the extracted-code message helps the local
readout). This diagnostic is not added to the training objective. Results include
unique total/inference/training-only parameter counts, actual completed-update
token throughput for that invocation, and peak allocated/reserved CUDA memory
over all ranks during `train()` (including its periodic evaluation/checkpoints).
CPU runs report null CUDA memory. These are runtime measurements, not promised
B200 performance; final full evaluation is outside that training-memory interval.

To generate a sequential queue, first choose one common `STOP_AFTER` explicitly:

```bash
python -m deep_kv make-jobs --config deep_kv.b200.json \
  --output bottleneck-jobs.json --stop-after "${STOP_AFTER:?choose a common horizon}" \
  --arms Task-Aware-NoAlign Task-Aware-Align Consumer-Aware-NoAlign Consumer-Aware-Align
```

This only writes a queue. Each arm requests all eight GPUs, followed by the
matched-LM comparison. The recipe retains 28,600 schedule steps, 1,430 warmup
steps, sequence length 2,048, and 1,048,576 tokens/update (microbatch 16,
accumulation 4, eight ranks). The recipe's historical default cutoff is not a
new budget decision. Select one shared checkpointing configuration and verify
full-model B200 memory/throughput before launching these larger training graphs.

`tests/test_bottleneck.py` covers initialization, forward isolation, single-pass
capture, causality/masks, separate gradient paths, zero-output startup, FP32
reductions under BF16, checkpointing parity, uneven-mask accumulation, native
training/save/resume, checkpoint staging, and sequential queue/report integration.
`tests/deep_kv_resume_worker.py` additionally supports all four arms for eight-rank
CPU BF16 train/resume and an uneven-mask distributed update against a single
global-batch SGD reference. These tests establish small-model correctness, not
research gains or full-model GPU capacity.
