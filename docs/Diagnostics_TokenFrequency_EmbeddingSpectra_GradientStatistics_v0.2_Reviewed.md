# Diagnostics for Capacity-Allocation Language Models

## Token-Frequency Loss, Embedding Spectra, and Gradient Statistics

**Reviewed version — 9 September 2026.** Technical definitions, normalization choices, and interpretation cautions have been tightened for the Qwen-derived/CulturaX experiments.

**Purpose.** This note defines three diagnostics for the Qwen-derived architecture experiments:

1. **Token-frequency loss** — which token-frequency groups gain or lose?
2. **Embedding spectra** — how much of each embedding space is actually used?
3. **Gradient statistics** — where does the learning signal go, and does tying create conflicting gradient pressure?

These diagnostics are intended to explain *why* an architecture behaves differently. They do not replace the primary evaluation: held-out next-token loss, loss versus training tokens/compute, and downstream evaluation.

---

## 1. Experimental conventions

Use the same conventions for every model arm.

### 1.1 Compare models at matched checkpoints

Run diagnostics at the same training-token checkpoints, for example:

```text
250M, 500M, 1B, 2B, 3B, 5B tokens
```

For expensive diagnostics, a smaller set is sufficient:

```text
0, 500M, 1B, 2B, 5B tokens
```

Always compare models trained on the same data manifest and evaluated on the same held-out examples.

### 1.2 Use the exact training tokenizer

Token frequency is defined using the tokenizer used by the model.

For the Qwen-derived experiments:

```text
tokenizer = Qwen tokenizer
dataset   = selected CulturaX training manifest
```

Do **not** use word frequencies, whitespace-token frequencies, or frequencies measured with another tokenizer.

### 1.3 Keep an immutable diagnostic manifest

Before comparing architectures, freeze:

- the training documents used to compute token frequencies;
- the validation/test documents used for loss diagnostics;
- a small set of fixed probe batches used for gradient statistics;
- the tokenizer revision;
- the model checkpoints being compared.

This prevents the diagnostic itself from changing between model arms.

### 1.4 Separate measurements from explanations

A diagnostic can show:

> rare tokens have higher loss in A128 than in B0.

It does **not** by itself prove:

> the 128-dimensional input embedding destroyed rare-token information.

Mechanistic claims require an intervention, such as increasing only the input rank from 128 to 256 and checking whether the rare-token loss changes as predicted.

---

# 2. Token-frequency loss

## 2.1 Question

The asymmetric-embedding experiment compresses the input embedding while allocating more rank to the output space.

A useful question is:

> Does a smaller input embedding hurt rare tokens more than common tokens?

This is especially important when using the large Qwen vocabulary with English CulturaX, because many vocabulary entries may be rare or absent from the selected English training sample.

## 2.2 Step 1 — count token frequencies on the training data

Count each token ID in the **exact tokenized training sample**.

Let

\[
c(v)
\]

be the number of times vocabulary token \(v\) appears in the training token stream.

Use the same document-boundary/EOS convention as training.

### Recommended exclusions

For the frequency diagnostic:

- exclude padding tokens;
- report EOS separately or exclude it from the bucket analysis;
- keep EOS in the normal overall language-model loss.

EOS is often extremely frequent and can otherwise dominate a frequency group.

### Minimal counting code

```python
import torch

V = tokenizer.vocab_size
counts = torch.zeros(V, dtype=torch.long)

for input_ids in tokenized_training_stream:
    ids = torch.as_tensor(input_ids, dtype=torch.long)
    counts += torch.bincount(ids, minlength=V)
```

For a very large streaming corpus, save the final `counts` tensor with the dataset/tokenizer manifest.

## 2.3 Step 2 — define fixed frequency buckets

Create the buckets **once from training frequencies**, then reuse them for every architecture. Make the ranking deterministic: sort seen token types by `(count descending, token_id ascending)` so equal-count ties cannot change bucket membership across runs.

For the first experiment, use non-overlapping buckets based on token-type frequency rank among tokens with `count > 0`:

| Bucket | Token types |
|---|---|
| Head | top 1% most frequent seen token types |
| Common | next 9% |
| Mid | next 40% |
| Rare | next 40% |
| Tail | bottom 10% of seen token types |
| Unseen | `training count = 0`, if such tokens occur in evaluation |

This gives:

```text
Head   = 0–1%
Common = 1–10%
Mid    = 10–50%
Rare   = 50–90%
Tail   = 90–100%
```

The percentages describe **token types**, not token mass.

### Why use a separate unseen bucket?

With a multilingual Qwen tokenizer and English-only CulturaX training, many vocabulary entries may never appear as input tokens.

Do not mix those rows with merely rare tokens.

## 2.4 Step 3 — compute per-target-token negative log-likelihood

For target token \(x_t\), define:

\[
\ell_t = -\log p_\theta(x_t \mid x_{<t}).
\]

For bucket \(b\):

\[
L_b =
\frac{
\sum_t \mathbf{1}[\operatorname{bucket}(x_t)=b]\ell_t
}{
\sum_t \mathbf{1}[\operatorname{bucket}(x_t)=b]
}.
\]

This is **token-weighted mean loss inside the bucket**.

Do not average the loss once per vocabulary type unless you explicitly want a separate type-weighted analysis.

### PyTorch sketch

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def bucketed_nll(model, batch, token_to_bucket, num_buckets):
    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask")

    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).logits

    pred = logits[:, :-1].float()
    target = input_ids[:, 1:]

    token_loss = F.cross_entropy(
        pred.reshape(-1, pred.size(-1)),
        target.reshape(-1),
        reduction="none",
    ).reshape_as(target)

    valid = torch.ones_like(target, dtype=torch.bool)
    if attention_mask is not None:
        valid &= attention_mask[:, 1:].bool()

    target_buckets = token_to_bucket[target]

    sums = torch.zeros(num_buckets, device=token_loss.device)
    counts = torch.zeros(num_buckets, device=token_loss.device)

    for b in range(num_buckets):
        mask = valid & (target_buckets == b)
        sums[b] += token_loss[mask].sum()
        counts[b] += mask.sum()

    return sums, counts
```

Aggregate `sums` and `counts` over the entire evaluation set, then compute:

```python
mean_loss = total_sums / total_counts
```

Do not average batch-level means; batches can contain different numbers of tokens from each bucket.

## 2.5 What to report

For every model, report:

| Bucket | Eval target count | Mean NLL | Perplexity |
|---|---:|---:|---:|
| Head | ... | ... | ... |
| Common | ... | ... | ... |
| Mid | ... | ... | ... |
| Rare | ... | ... | ... |
| Tail | ... | ... | ... |
| Unseen | ... | ... | ... |

More importantly, compare each candidate directly against the reference:

\[
\Delta L_b = L_b^{\text{candidate}} - L_b^{\text{reference}}.
\]

Negative is better.

Example:

```text
Δ NLL: A128 - B0

Head     -0.010
Common   -0.006
Mid      -0.001
Rare     +0.009
Tail     +0.031
```

This pattern would suggest that the average result hides a tail-token regression. It would **not yet prove the cause**.

## 2.6 Useful follow-up interventions

If A128 loses mainly on rare/tail tokens:

1. compare A128 with A256;
2. keep the output rank fixed in a special diagnostic control if the budget allows;
3. test the nonlinear input mapping;
4. measure input-embedding gradient/update statistics by frequency bucket.

A successful intervention should change the specific failure mode predicted by the hypothesis.

## 2.7 Important cautions

- Do not compare buckets across different tokenizers as if they were identical.
- Do not define frequency using validation/test data.
- Keep unseen tokens separate from merely rare tokens.
- If almost no unseen token occurs as an evaluation target, report its count and avoid overinterpreting its mean.
- Report evaluation target counts for every bucket and uncertainty for bucket differences when those differences become part of a central claim; tail buckets may have far fewer target occurrences than head buckets.

---

# 3. Embedding spectra

## 3.1 Question

The embedding experiment deliberately restricts the ranks of the input and output representations.

For example:

```text
B0:   tied 1024
A128: input 128 / output 896
A256: input 256 / output 768
A512: input 512 / output 512
```

We want to know:

> How many directions in these learned spaces carry substantial variance, and how does this change during training?

A singular-value spectrum is a simple way to inspect this.

## 3.2 Which matrices should be analyzed?

Analyze both the **raw parameter spaces** and the **effective body-facing spaces**, but treat the effective matrices as the primary comparison. Raw factor spectra can change under equivalent reparameterizations of the two factors even when their product—and therefore the model function at that interface—stays the same.

### B0 — tied baseline

\[
E \in \mathbb{R}^{V\times1024}.
\]

### A128 — separate embeddings

Raw input table:

\[
E_{\text{in}} \in \mathbb{R}^{V\times128}.
\]

Input adapter:

\[
P_{\text{in}} \in \mathbb{R}^{128\times1024}.
\]

Effective input vectors seen by the Transformer:

\[
X_{\text{in}} = E_{\text{in}}P_{\text{in}}
\in \mathbb{R}^{V\times1024}.
\]

Raw output table:

\[
E_{\text{out}} \in \mathbb{R}^{V\times896}.
\]

Output adapter:

\[
P_{\text{out}} \in \mathbb{R}^{1024\times896}.
\]

Effective output vectors in body coordinates are:

\[
X_{\text{out}}
=
E_{\text{out}}P_{\text{out}}^\top
\in \mathbb{R}^{V\times1024}.
\]

This gives a common 1024-dimensional coordinate space for body-facing input and output representations.

Their rank ceilings still differ:

\[
\operatorname{rank}(X_{\text{in}})\le128,
\qquad
\operatorname{rank}(X_{\text{out}})\le896.
\]

## 3.3 Which vocabulary rows should be included?

For English-only training with the Qwen tokenizer, do not rely on one spectrum over all vocabulary rows.

Report at least:

1. **seen-token rows:** `training_count > 0`;
2. **all vocabulary rows:** secondary diagnostic;
3. optionally, spectra for Head/Common/Mid/Rare/Tail subsets.

A separate input row that never occurs in training may stay near initialization, while output rows can still receive full-softmax gradients even when they are never positive targets.

Use the **seen-token spectrum as the primary interpretation**.

## 3.4 Center the matrix

For selected vocabulary matrix \(X\):

\[
X_c = X - \mathbf{1}\mu^\top,
\]

where

\[
\mu = \frac{1}{N}\sum_{i=1}^{N}X_i.
\]

This removes the common mean vector and focuses the spectrum on variation across token representations.

## 3.5 Compute the spectrum

Let:

\[
X_c = U\Sigma V^\top
\]

with singular values

\[
s_1 \ge s_2 \ge \dots \ge s_r.
\]

Because body-facing dimension is only 1024, compute the smaller covariance:

\[
C = \frac{1}{N}X_c^\top X_c.
\]

If \(\lambda_i\) are the eigenvalues of \(C\), then the **literal singular values** of \(X_c\) are

\[
\sigma_i = \sqrt{N\lambda_i}.
\]

For normalized spectrum shape, effective rank, stable rank, and cumulative-energy fractions, the common factor \(\sqrt N\) cancels. Therefore it is convenient to work with

\[
q_i = \sqrt{\lambda_i} = \sigma_i/\sqrt N.
\]

Do not label `q_i` as the literal singular values in saved results.

### Practical implementation

```python
import torch

@torch.no_grad()
def covariance_spectrum(X):
    X = X.float()
    X = X - X.mean(dim=0, keepdim=True)

    n = X.shape[0]
    C = (X.T @ X) / n

    eigvals = torch.linalg.eigvalsh(C)
    eigvals = eigvals.clamp_min(0).flip(0)

    q = eigvals.sqrt()                  # sigma / sqrt(n)
    sigma = q * (n ** 0.5)              # literal singular values
    return eigvals, q, sigma
```

For large matrices, accumulate first and second moments in chunks rather than storing every row on one device:

\[
\mu = \frac{1}{N}\sum_i x_i,
\qquad
C = \frac{1}{N}\sum_i x_i x_i^\top - \mu\mu^\top.
\]

Accumulate these statistics in at least float32 (float64 on CPU is preferable when practical), then eigendecompose the final small covariance matrix.

## 3.6 Normalize spectrum shape

Use explained-energy fractions. With either literal singular values \(\sigma_i\) or covariance-scaled values \(q_i=\sigma_i/\sqrt N\):

\[
p_i
=
\frac{\sigma_i^2}{\sum_j \sigma_j^2}
=
\frac{q_i^2}{\sum_j q_j^2}
=
\frac{\lambda_i}{\sum_j \lambda_j}.
\]

Plot either `p_i` or cumulative explained energy:

\[
C_k = \sum_{i=1}^{k}p_i.
\]

## 3.7 Summary metrics

### Effective rank

\[
H(p) = -\sum_i p_i\log p_i
\]

\[
r_{\text{eff}} = \exp(H(p)).
\]

### Stable rank

\[
r_{\text{stable}}
=
\frac{\|X_c\|_F^2}{\sigma_1^2}
=
\frac{\sum_i \lambda_i}{\lambda_1}.
\]

The covariance form on the right is convenient because it avoids reconstructing literal singular values.

### Dimensions needed for 90% / 95% energy

\[
k_{90}
=
\min\left\{k:\sum_{i=1}^{k}p_i\ge0.90\right\},
\]

and similarly for \(k_{95}\).

Recommended table:

| Model | Matrix | Rank ceiling | Effective rank | Stable rank | k90 | k95 |
|---|---|---:|---:|---:|---:|---:|
| B0 | E | 1024 | ... | ... | ... | ... |
| A128 | raw `E_in` | 128 | ... | ... | ... | ... |
| A128 | effective `X_in` | 128 | ... | ... | ... | ... |
| A128 | raw `E_out` | 896 | ... | ... | ... | ... |
| A128 | effective `X_out` | 896 | ... | ... | ... | ... |

## 3.8 What to look for

### Case A — input compression is not binding strongly

If the **effective input matrix** of the baseline itself has effective rank near the A128 ceiling, A128 may not sacrifice much useful input-space capacity according to this diagnostic. Compare normalized spectra as well as the scalar effective-rank summary.

### Case B — A128 saturates the rank ceiling

If A128 has effective rank close to 128 and A256 spreads meaningfully beyond 128 dimensions, the A128 representation may be capacity-constrained.

Combine this observation with the A128→A256 intervention before making a causal claim.

### Case C — output space uses additional rank

If a higher-rank output allocation uses substantially more effective rank and has better LM loss, the result is consistent with output-side rank being useful.

The spectrum is supporting evidence, not proof.

## 3.9 Plot spectra over training

Useful checkpoints:

```text
initialization
500M
1B
2B
5B tokens
```

Plot:

```text
effective rank vs. tokens
k95 vs. tokens
normalized singular-value curves
```

This can show whether a low-rank space becomes saturated early or only after long training.

---

# 4. Gradient statistics

## 4.1 Question

We want to know:

> Which components receive strong learning signals, and does the signal become unusually large, weak, sparse, or conflicting under a particular architecture?

Gradient diagnostics are especially important when comparing tied and untied embeddings.

## 4.2 When to log gradients

For ordinary logging:

- accumulate gradients normally across microbatches;
- with AMP/gradient scaling, **unscale gradients first**;
- record statistics **before gradient clipping**;
- record them before `optimizer.step()`.

Suggested frequency:

```text
every 100–500 optimizer steps
```

For expensive path-decomposition diagnostics:

```text
every 1,000–5,000 steps
```

using a fixed probe batch and **without taking an optimizer step**.

## 4.3 Do not compare raw gradient L2 norms alone

For parameter tensor \(W\) with \(n\) elements and gradient \(G\):

### Parameter RMS

\[
\operatorname{RMS}(W)
=
\frac{\|W\|_2}{\sqrt n}.
\]

### Gradient RMS

\[
\operatorname{RMS}(G)
=
\frac{\|G\|_2}{\sqrt n}.
\]

### Gradient-to-weight ratio

\[
R_{g/w}
=
\frac{\operatorname{RMS}(G)}
{\operatorname{RMS}(W)+\epsilon}.
\]

These are more comparable across matrices with different sizes.

## 4.4 Components to monitor

For asymmetric embeddings:

```text
E_in
P_in
E_out
P_out
```

For SWT/SWT-R additionally:

```text
each stage-transition matrix
attention output projections around boundaries
FFN down projections around boundaries
RMSNorm scales
```

Also log one representative attention and FFN matrix from each stage.

## 4.5 Minimal logging code

```python
import math
import torch

def tensor_stats(name, p, eps=1e-12):
    if p.grad is None:
        return None

    w = p.detach().float()
    g = p.grad.detach().float()

    n = p.numel()

    w_l2 = torch.linalg.vector_norm(w).item()
    g_l2 = torch.linalg.vector_norm(g).item()

    w_rms = w_l2 / math.sqrt(n)
    g_rms = g_l2 / math.sqrt(n)

    return {
        f"{name}/weight_l2": w_l2,
        f"{name}/grad_l2": g_l2,
        f"{name}/weight_rms": w_rms,
        f"{name}/grad_rms": g_rms,
        f"{name}/grad_to_weight": g_rms / (w_rms + eps),
    }
```

If training uses FSDP, ZeRO, tensor parallelism, or another sharded setup, do not interpret statistics from one local parameter shard as global matrix statistics. Compute globally reduced sums/counts (and maxima/quantiles with an appropriate distributed procedure), or reconstruct the selected diagnostic tensor offline. Record whether the logged gradient is pre- or post-data-parallel all-reduce.

## 4.6 Row-wise embedding gradients by token frequency

For an embedding matrix

\[
E\in\mathbb{R}^{V\times d},
\]

compute each vocabulary row's gradient norm:

\[
g_v = \|\nabla_{E_v}\mathcal{L}\|_2.
\]

Aggregate using the same frequency buckets from Section 2.

```python
@torch.no_grad()
def row_grad_norms(embedding_weight):
    g = embedding_weight.grad
    if g is None:
        return None

    if g.is_sparse:
        g = g.to_dense()

    return torch.linalg.vector_norm(g.float(), dim=1)
```

For every bucket report:

- mean and median **row gradient RMS**;
- 90th percentile of row gradient RMS;
- fraction of rows with exactly zero gradient;
- optionally the raw row L2 norm as a within-matrix debugging statistic.

For a row with dimension \(d\), define:

\[
\operatorname{rowGradRMS}(v)
=
\frac{\|\nabla_{E_v}\mathcal{L}\|_2}{\sqrt d}.
\]

This normalization matters when comparing `E_in` and `E_out`, because their row dimensions differ (for example 128 versus 896). Raw row L2 norms are acceptable for comparing frequency buckets **within the same matrix**, but not as the main cross-matrix statistic.

A direct implementation is:

```python
@torch.no_grad()
def row_grad_rms(embedding_weight):
    g = embedding_weight.grad
    if g is None:
        return None
    if g.is_sparse:
        g = g.to_dense()

    d = g.shape[1]
    return torch.linalg.vector_norm(g.float(), dim=1) / (d ** 0.5)
```

### Important difference between input and output tables

With a normal full softmax:

- input lookup rows receive direct input-path gradients only when they are accessed;
- the output table usually receives dense softmax gradient pressure across the vocabulary.

Therefore `E_in` and `E_out` naturally have different gradient-density patterns.

This is not a bug; it is part of the phenomenon being studied.

---

# 5. Special diagnostic for the tied baseline

## 5.1 Why B0 needs extra care

In B0:

\[
E_{\text{input}} = E_{\text{output}} = E.
\]

The shared gradient is:

\[
g_{\text{total}}
=
g_{\text{input-path}}
+
g_{\text{output-path}}.
\]

A normal `.grad` hides the two contributions.

## 5.2 Decompose the tied gradient on a fixed probe batch

For diagnostic-only passes using the same parameter values:

1. normal tied model → `g_total`;
2. detach the output use of `E` → `g_input`;
3. detach the input lookup use of `E` → `g_output`.

Conceptually:

```python
# Input path active, output path blocked
input_vectors = F.embedding(ids, E)
logits = h @ E.detach().T
```

```python
# Input path blocked, output path active
input_vectors = F.embedding(ids, E.detach())
logits = h @ E.T
```

Use a custom diagnostic forward. Do not alter the normal training path.

Check numerically that:

\[
g_{\text{total}}
\approx
g_{\text{input}}
+
g_{\text{output}}.
\]

## 5.3 Gradient cosine similarity

Compute:

\[
\cos(g_{\text{input}},g_{\text{output}})
=
\frac{
\langle g_{\text{input}},g_{\text{output}}\rangle
}{
\|g_{\text{input}}\|_2
\|g_{\text{output}}\|_2
}.
\]

Interpretation:

- positive: compatible pressure;
- near zero: largely orthogonal pressure;
- negative: conflicting pressure on that probe batch.

Also compute the cosine separately for each token-frequency bucket by flattening only the corresponding vocabulary rows.

**Coverage caution.** `g_input` is sparse across vocabulary rows on any finite probe batch, whereas `g_output` is usually dense. A bucket-level cosine can therefore be dominated by the small subset of input rows that happened to occur in that probe. Use a fixed probe set large enough to cover many rows, aggregate gradients over several probe batches before computing the cosine, and report the number/fraction of rows with nonzero input-path gradient in each bucket. As a secondary analysis, also compute cosine restricted to rows with nonzero `g_input`.

Recommended table:

| Bucket | Active input rows | Input grad RMS | Output grad RMS | Cosine |
|---|---:|---:|---:|---:|
| Head | ... | ... | ... | ... |
| Common | ... | ... | ... | ... |
| Mid | ... | ... | ... | ... |
| Rare | ... | ... | ... | ... |
| Tail | ... | ... | ... | ... |

A useful finding may be that conflict is concentrated in certain frequency groups. Do not assume this outcome in advance.

---

# 6. Optimizer-update statistics

Gradients are not identical to actual AdamW updates.

Periodically record:

\[
R_{\Delta/w}
=
\frac{
\operatorname{RMS}(\Delta W)
}{
\operatorname{RMS}(W)+\epsilon
},
\]

where:

\[
\Delta W = W_{\text{after step}} - W_{\text{before step}}.
\]

This incorporates learning rate, Adam moments, weight decay, and clipping.

Do it infrequently because copying large embedding matrices is expensive.

---

# 7. Recommended diagnostic schedule

## Every validation checkpoint

Run:

```text
overall validation NLL
token-frequency NLL
```

Suggested checkpoints:

```text
250M
500M
1B
2B
3B
5B tokens
```

## At larger checkpoints

Run embedding spectra:

```text
initialization
500M
1B
2B
5B
```

Analyze:

```text
raw E_in / E_out
effective X_in / X_out
seen-token rows
optionally frequency subsets
```

## During training

Every 100–500 optimizer steps:

```text
weight RMS
gradient RMS
gradient/weight ratio
```

Every 1,000–5,000 steps on fixed probe batches:

```text
row-gradient statistics by frequency
tied input/output gradient decomposition
gradient cosine similarity
```

Occasionally:

```text
optimizer update / weight ratio
```

---

# 8. Minimum figures for a paper

## Figure 1 — loss by frequency

```text
x-axis: token-frequency bucket
y-axis: candidate NLL - baseline NLL
series: A128, A256, A512
```

## Figure 2 — effective embedding spectra

Two panels:

```text
Input-facing spectrum
Output-facing spectrum
```

Plot normalized cumulative energy versus component index for B0/A128/A256/A512.

Use seen-token rows for the primary plot.

## Figure 3 — effective rank over training

```text
x-axis: training tokens
y-axis: effective rank
```

Separate input-facing and output-facing representations.

## Figure 4 — gradient pressure

For B0:

```text
input-path gradient RMS
output-path gradient RMS
input/output gradient cosine
```

split by token-frequency bucket.

For A models:

```text
E_in row-gradient RMS
E_out row-gradient RMS
```

by token-frequency bucket. Use dimension-normalized row RMS, not raw row L2 norm, for the cross-matrix comparison.

---

# 9. Interpretation examples

## Pattern 1

```text
A128 overall loss ≈ B0
A128 tail-token loss worse
A256 fixes tail loss
A128 X_in spectrum saturates near rank 128
```

Reasonable conclusion:

> The 128-dimensional input representation appears to be a binding constraint for low-frequency tokens in this training regime; increasing input rank to 256 reduces the targeted deficit.

## Pattern 2

```text
A128 beats B0
input loss does not degrade by frequency
A128 output effective rank > A512 output effective rank
```

Reasonable conclusion:

> Under the tested table budget, allocating more rank to the output representation improves language modeling without a measurable rare-token input penalty.

## Pattern 3

```text
B0 tied gradient cosine strongly negative
especially for rare tokens
untied A256 improves rare-token loss
```

Reasonable conclusion:

> The result is consistent with conflicting input/output learning signals under tying, concentrated in rare-token rows.

A stronger mechanism claim would require modifying the tying/gradient-sharing rule directly.

---

# 10. Reproducibility checklist

Save:

- model checkpoint identifier;
- exact processed/scored training-token counts;
- dataset manifest hash;
- tokenizer revision;
- token-frequency count file;
- token-to-bucket mapping;
- validation/test manifest;
- probe-batch IDs;
- model parameter names and shapes;
- mixed-precision / gradient-scaling state;
- whether gradients were recorded before clipping;
- spectrum centering/subset convention;
- code revision.

Save the token-to-bucket mapping directly rather than reconstructing it later.

---

# 11. Recommended implementation order

### Step 1

```text
training token counts
→ fixed frequency buckets
→ bucketed validation NLL
```

This is cheap and immediately useful.

### Step 2

```text
E_in / E_out extraction
→ effective matrices
→ covariance spectrum
→ effective rank / stable rank / k90 / k95
```

Run this offline from checkpoints.

### Step 3

```text
gradient RMS
→ gradient/weight ratio
→ dimension-normalized row-gradient RMS by frequency
```

Add lightweight training instrumentation.

### Step 4

```text
tied-gradient path decomposition
→ input/output gradient cosine
```

Treat this as a targeted mechanism diagnostic rather than standard training instrumentation.

---

# 12. Core principle

These diagnostics should answer increasingly specific questions:

```text
Where does the model lose?
        ↓
Token-frequency loss

What representation constraint accompanies the loss?
        ↓
Embedding spectra

What learning signal may be producing that representation?
        ↓
Gradient statistics

Does changing the suspected cause change the predicted behavior?
        ↓
Controlled architecture intervention
```

The final intervention is essential.

A spectrum or gradient plot can suggest a mechanism. The architecture intervention is what makes the explanation convincing.

---

## Implementation in this repository

See [DIAGNOSTICS.md](DIAGNOSTICS.md) for the implemented workflow, commands,
artifact definitions, tests and limitations. The offline diagnostics are optional
workers alongside PPL/benchmarks in `eval.eval_parallel`; they do not modify the
current training job. Live pre-clip/update instrumentation is a separately
attachable helper for future training, not a reconstruction of historical logs.

The first frozen frequency definition uses the complete selected packed training
pool, explicitly not the exact prefix consumed before a 5k-step stop. Counts,
seen/unseen interpretation and consumed checkpoint token budgets are labelled
separately. Effective spectra use actual input/output body widths, including the
unequal widths of C/D. Statistical uncertainty and plotting remain follow-ups
before making central mechanistic claims.
