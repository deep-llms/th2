# Candidate Architectures for the Next Capacity-Allocation Runs

**Reviewed v0.4 — 10 September 2026.** Budget matching and causal-interpretation notes were tightened after the first six-model results.

## Goal

This note defines seven architecture directions for the next experiments:

1. **Tied low-rank embeddings** — isolate the value of weight tying from full 1024-dimensional rank.
2. **Partially shared input/output embeddings** — preserve some of the benefit of tying while allowing input/output specialization.
3. **Input-heavy untied allocations** — extend the fixed `r_in + r_out = 1024` curve beyond 512/512 and test a direct input-expansion variant.
4. **Fixed-residual variable-compute blocks** — keep a 1024-dimensional residual stream while using narrower computation inside selected layers.
5. **Wide–narrow–wide width allocation** — a strong comparator for the original narrow-to-wide hypothesis.
6. **Direct width-changing blocks** — change stage width inside a boundary block instead of adding a separate `Linear(d → D)` after the block.
7. **Larger-output variants** — test an output width of 1280, and a cleaner output width of 1024.

All models remain **Qwen3-derived**: Qwen-style pre-RMSNorm blocks, Q/K RMSNorm, GQA, SiLU-gated MLP, RoPE, and bias-free linear layers are retained unless explicitly stated otherwise.

---


# 1. Tied low-rank embedding controls

## Motivation

The current experiments change two important properties at the same time:

```text
B0:
full-rank 1024
+
exactly tied input/output table

A128/A256/A512:
lower-rank input/output spaces
+
fully untied tables
```

Therefore B0's advantage does not tell us whether the important factor is:

```text
full rank
or
weight tying
or
both.
```

A tied low-rank control isolates this more directly.

## T768

Use one shared vocabulary table:

```text
E_shared: V × 768
```

for both input and output.

Input:

```text
token
  ↓
shared E: 768
  ↓
Linear 768 → 1024
  ↓
uniform Qwen body: 1024 × 6
```

Output:

```text
final hidden: 1024
  ↓
Linear 1024 → 768
  ↓
same shared Eᵀ
  ↓
logits
```

In equations:

\[
h_0 = E[x]P_{\text{in}},
\]

\[
\text{logits}
=
(h_LP_{\text{out}})E^\top.
\]

The same `E` is used on both sides, so the input and output weights remain tied.

### Budget status

Under the current six-layer Qwen-derived configuration with `V = 151,936`:

```text
B0 tied-1024: ≈249.97M parameters
T768:          ≈212.65M parameters
```

So T768 is about **15% smaller than B0**. It is **not** a total-parameter-matched isolation of tying versus rank. The saving is inherent to sharing one lower-rank table.

Interpret it asymmetrically:

- if T768 approaches B0 despite being much smaller, that is strong evidence that full 1024 rank is not necessary in this regime;
- if T768 loses, the loss cannot by itself distinguish lower rank from lower total capacity.

For a closer fixed-vocabulary-budget test of **shared versus private capacity**, the partial-sharing model in §2 is more informative.

## Optional T512

A second point is:

```text
shared E: V × 512
input adapter: 512 → 1024
body: 1024 × 6
output adapter: 1024 → 512
same shared Eᵀ
```

Under the same counting convention, T512 is about **173.23M parameters**, roughly 31% below B0. It is therefore an aggressive compression experiment, not a matched control.

Do not begin with many ranks. Run `T768` first; add `T512` only if the T768 result makes a lower tied rank scientifically useful.

## Interpretation

```text
T768 ≈ B0 despite ~15% fewer parameters
→ low-rank tying is surprisingly efficient; full 1024 rank is not required here.

T768 clearly loses
→ lower tied rank and/or the reduced total capacity is costly; this result alone
  does not isolate which one.
```

Because tying inherently saves parameters, there is no single comparison that perfectly isolates “tying” while keeping per-side rank, body, and total parameter count all unchanged. Treat T768 as a **diagnostic efficiency/control architecture**, not a pure causal isolation.

---

# 2. Partial-sharing embedding architecture

## Motivation

The current results favor the exactly tied B0 model for language modeling.

The fully separated models:

```text
A128: input 128 / output 896
A256: input 256 / output 768
A512: input 512 / output 512
```

all lose to B0 on held-out language-model loss.

A natural next question is:

> Can we preserve a shared lexical space while giving the input and output some private capacity?

## Simple parameter-budget example

Use three vocabulary tables:

```text
Shared          : V × 512
Input-private   : V × 128
Output-private  : V × 384
```

The total vocabulary-table width is:

```text
512 + 128 + 384 = 1024
```

so the vocabulary-table parameter budget is the same as one `V × 1024` tied table.

With a uniform 1024×6 body, the input/output adapters add parameters. Under the current Qwen-derived counting convention this example is approximately:

```text
Partial-share 512/128/384: ≈251.54M
B0:                         ≈249.97M
```

or about **0.63% above B0**. It is therefore close-budget, not exact-total-budget matched.

Also note the per-side dimensional ceilings:

```text
input representation before body adapter: 512 + 128 = 640
output scoring components:                512 + 384 = 896
```

Thus this experiment changes both **sharing structure** and **per-side rank allocation**. Its advantage over the fully untied A-family is that part of the vocabulary representation is explicitly shared, not that it isolates sharing as the only changed variable.

## Input side

For token `x`:

```text
shared[x]         512
input_private[x]  128
        ↓
concatenate
        ↓
640-dimensional token representation
        ↓
input adapter → first body width
```

In equations:

\[
e_{\text{in}}(x)
=
[E_{\text{shared}}(x);E_{\text{in-private}}(x)].
\]

The body receives:

\[
h_0=e_{\text{in}}P_{\text{in}}.
\]

## Output side

The final hidden state is projected into two spaces:

```text
final hidden
   ├─→ shared-output projection  → 512
   └─→ private-output projection → 384
```

The vocabulary logits are the sum:

\[
\text{logits}
=
z_s E_{\text{shared}}^\top
+
z_o E_{\text{out-private}}^\top.
\]

The same `E_shared` table therefore receives both input-side and output-side learning signals, while the private tables can specialize.

## What this tests

This architecture separates two possible explanations for B0's advantage:

```text
B0 may win because:
1. full rank 1024 is important
or
2. sharing/tie regularization is important
or
3. both
```

Partial sharing asks whether we can keep benefit (2) without forcing the entire input and output representation to be identical.

## First allocation to try

```text
shared = 512
input-private = 128
output-private = 384
```

Later, only if needed, sweep the shared fraction:

```text
shared 768 / private 64+192
shared 512 / private 128+384
shared 256 / private 192+576
```

Do not begin with a large sweep.

---


# 3. Input-heavy untied allocation variants

## Motivation

The current fixed-table-budget allocation curve is:

```text
A128: 128 input / 896 output
A256: 256 input / 768 output
A512: 512 input / 512 output
```

All three satisfy:

\[
r_{\text{in}}+r_{\text{out}}=1024.
\]

The tested points stop at the symmetric 512/512 allocation. Two additional input-heavy points show whether performance continues to deteriorate as capacity is moved from the output side to the input side.

These are **curve-extension controls**, not assumed improvements.

## A640 — moderately input-heavy

```text
Input embedding:  640
Output embedding: 384
```

with:

```text
640 + 384 = 1024.
```

Architecture:

```text
E_in: V × 640
    ↓
Linear 640 → 1024, bias=False
    ↓
Qwen body: 1024 × 6
    ↓
final RMSNorm(1024)
    ↓
Linear 1024 → 384, bias=False
    ↓
E_out: V × 384
    ↓
logits
```

This tests a moderate shift beyond A512 toward input-side capacity.

## A768 — strongly input-heavy

```text
Input embedding:  768
Output embedding: 256
```

with:

```text
768 + 256 = 1024.
```

Architecture:

```text
E_in: V × 768
    ↓
Linear 768 → 1024, bias=False
    ↓
Qwen body: 1024 × 6
    ↓
final RMSNorm(1024)
    ↓
Linear 1024 → 256, bias=False
    ↓
E_out: V × 256
    ↓
logits
```

This is a stronger stress test of a small output prediction space.

Its final linear vocabulary interface has numerical rank at most 256, so a large loss increase would be consistent with the output side becoming too constrained. That interpretation still requires comparison with the other allocation points; rank alone is not a causal explanation.

## A768-Direct — direct input expansion

Use the **same vocabulary allocation** as A768:

```text
Input embedding:  768
Output embedding: 256
```

but remove the standalone input adapter:

```text
A768:
E_in 768
    ↓
Linear 768 → 1024
    ↓
Block_1024 × 6
```

and replace it with a first boundary block that expands directly:

```text
A768-Direct:
E_in 768
    ↓
BoundaryBlock_768→1024
    ↓
Block_1024 × 5
```

One direct widening form is:

\[
u
=
x+\operatorname{Attention}_{768}
(\operatorname{RMSNorm}_{768}(x)),
\]

\[
y
=
\operatorname{Pad}(u,1024)
+
\operatorname{MLP}_{768\rightarrow1024}
(\operatorname{RMSNorm}_{768}(u)).
\]

Here:

- attention in the first block operates at width 768;
- the MLP `down_proj` outputs 1024 directly;
- the old 768-dimensional residual is zero-padded to 1024;
- there is **no standalone `Linear(768 → 1024)` after the block**;
- the remaining five blocks are ordinary Qwen-style 1024-wide blocks.

The output side stays unchanged:

```text
final hidden 1024
    ↓
Linear 1024 → 256
    ↓
E_out: V × 256
```

## What the comparison tells us

The useful comparisons are:

```text
A512 → A640 → A768
```

which tests increasingly input-heavy vocabulary allocation, and:

```text
A768
vs.
A768-Direct
```

which tests the **bundled direct-expansion design** against the adapter-based design.

This is **not** a clean isolation of the standalone adapter. A768-Direct also changes:

- the first attention computation from width 1024 to width 768;
- the first block's MLP output geometry;
- the number of ordinary 1024-wide blocks from six to five;
- the complete parameter/FLOP budget.

Therefore a win or loss should be attributed to the direct first-block interface as a whole, not specifically to “removing one linear layer.”

Do not reuse A768's parameter count for A768-Direct. Recount the complete instantiated model and report its compute separately.

If A768 is already clearly worse than A512/A640, do not automatically extend to an even more extreme `896/128` allocation.

---

# 4. Current stage transition: separate linear map

The C/D experiments use this transition:

```text
QwenBlock_d
    ↓
Linear(d → D, bias=False)
    ↓
QwenBlock_D
```

The next Qwen block begins with its normal Qwen input RMSNorm.

Example:

```text
Block_512
Block_512
    ↓
Linear 512 → 1024
    ↓
Block_1024
Block_1024
```

This is the **T1 reference transition**.

Advantages:

- simple;
- easy to debug;
- leaves each Qwen block unchanged;
- current C/D results already use it.

The separate resize projection is custom; standard Qwen does not change residual width between layers.

---

# 5. Direct width-changing block

## Idea

Instead of:

```text
QwenBlock_d
    ↓
extra Linear(d → D)
    ↓
QwenBlock_D
```

make the **boundary block itself output width `D`**.

For a widening boundary:

```text
input width d
    ↓
Qwen attention at width d
    ↓
attention residual at width d
    ↓
Qwen-style gated MLP
but its down projection outputs D
    ↓
output width D
    ↓
next QwenBlock_D
```

A simple widening form is:

\[
u
=
x+\operatorname{Attention}_d(\operatorname{RMSNorm}_d(x)),
\]

\[
y
=
\operatorname{Pad}(u,D)
+
\operatorname{MLP}_{d\rightarrow D}
(\operatorname{RMSNorm}_d(u)),
\qquad D>d.
\]

`Pad(u,D)` keeps the original `d` residual coordinates and fills the newly introduced coordinates with zeros.

The MLP's final `down_proj` changes from:

```text
intermediate → d
```

to:

```text
intermediate → D
```

so **there is no standalone `Linear(d → D)` transition**.

## Example

Instead of:

```text
Block_512
    ↓
Linear 512 → 768
    ↓
Block_768
```

use:

```text
BoundaryBlock_512→768
    - attention operates at 512
    - residual stays 512 through attention
    - MLP down_proj outputs 768
    - old residual is zero-padded to 768
        ↓
Block_768
```

## Important caution

This is **not standard Qwen**.

It changes the residual/MLP structure of the boundary block and may have different parameter count and optimization behavior from T1.

Therefore compare:

```text
same stage widths + T1 transition
vs.
same stage widths + direct boundary transition
```

and keep the number of logical blocks fixed. Even then, the MLP output matrix, shortcut rule, parameter count, and FLOPs differ, so this comparison tests **transition parameterization as a bundle**, not the presence/absence of one matrix in isolation.

The parameter counts from the existing C/D T1 experiments **must not be reused** for this direct version; recount the instantiated model.

---

# 6. Output-1280 architecture

## Goal

Test a model whose final hidden representation and vocabulary-output representation are both 1280:

```text
last hidden = 1280
output dim  = 1280
```

This removes the current D interface:

```text
1280 hidden → 896 output
```

## Near-250M T1 version

One feasible schedule under the current Qwen-derived parameterization is:

```text
Input embedding: 64

Body:
256
256
512
512
768
1280

Output embedding: 1280
```

Architecture:

```text
E_in: V × 64
    ↓
Linear 64 → 256

Block_256
Block_256
    ↓
Linear 256 → 512

Block_512
Block_512
    ↓
Linear 512 → 768

Block_768
    ↓
Linear 768 → 1280

Block_1280
    ↓
final RMSNorm(1280)

E_out: V × 1280
    ↓
logits
```

With the current T1/Qwen parameter-count convention and `V = 151,936`, this is approximately:

\[
250.63\text{M parameters}.
\]

That is close to B0's approximately 249.97M parameters.

## Main concern

The output is no longer compressed, but the **input embedding is only 64-dimensional**.

So this model trades one possible bottleneck for another:

```text
current D:
large late representation → smaller output

O1280:
very small input representation → large output
```

Use it as a capacity-allocation experiment, not as an obviously superior architecture.

---

# 7. Output-1024 architecture

## Motivation

A more conservative design is:

```text
last hidden = 1024
output dim  = 1024
```

This avoids both:

```text
1280 → 896 compression
```

and the very expensive `V × 1280` output table.

## Clean progressive-width version

A simple version is:

```text
Input embedding: 256

Body:
512
512
768
768
1024
1024

Output embedding: 1024
```

Architecture:

```text
E_in: V × 256
    ↓
Linear 256 → 512

Block_512
Block_512
    ↓
Linear 512 → 768

Block_768
Block_768
    ↓
Linear 768 → 1024

Block_1024
Block_1024
    ↓
final RMSNorm(1024)

E_out: V × 1024
    ↓
logits
```

This has no shrinking stage:

```text
512 → 768 → 1024
```

and:

```text
final body width = output width = 1024.
```

Under the current T1/Qwen counting convention, this version is approximately:

\[
253.87\text{M parameters},
\]

about 1.6% above B0.

## Closer-budget option

If an approximately 250M total is preferred while keeping the same body:

```text
Input embedding: 232

Body:
512
512
768
768
1024
1024

Output embedding: 1024
```

is approximately:

\[
250.21\text{M parameters}.
\]

`232` is chosen for budget matching, not because it has a theoretical special meaning.

A more hardware-friendly round input width such as `224` is also possible, but will land slightly below the target budget.

---


---

# 8. Fixed-residual variable-compute architecture

## Motivation

The original C/D models change the **full residual width**:

```text
C:
256 → 256 → 512 → 512 → 1024 → 1024

D:
512 → 512 → 1024 → 1024 → 1280 → 1280
```

A weaker early result could therefore come from either:

```text
1. narrow early computation
or
2. compressing the entire residual representation.
```

This variant separates those two effects.

## Core idea

Keep the residual stream at width 1024 for all six layers:

```text
Residual representation:
1024 → 1024 → 1024 → 1024 → 1024 → 1024
```

but use a narrower internal computation width in selected layers.

Example active widths:

```text
512
512
768
768
1024
1024
```

Conceptually, for a layer with active width `d < 1024`:

```text
full residual x: 1024
        │
        ├──────────────────────── identity residual ────────────────┐
        │                                                          │
        ↓                                                          │
Linear 1024 → d                                                    │
        ↓                                                          │
Qwen-style computation at width d                                  │
        ↓                                                          │
Linear d → 1024                                                    │
        ↓                                                          │
        └────────────────────────── add ────────────────────────────┘
        ↓
full residual: 1024
```

A more precise residual-delta form is:

\[
z_\ell=P_{\text{down},\ell}x_\ell,
\]

\[
z'_\ell=\operatorname{QwenBlock}_{d_\ell}(z_\ell),
\]

\[
x_{\ell+1}
=
x_\ell
+
P_{\text{up},\ell}(z'_\ell-z_\ell).
\]

This is important: `QwenBlock_d` already contains its own attention and MLP residual additions. We therefore lift **only the block's computed residual delta** `(z' - z)` back into the 1024-dimensional outer residual stream.

When `d = 1024` and both projections are identities, the equation reduces exactly to an ordinary Qwen block:

\[
x_{\ell+1}=\operatorname{QwenBlock}_{1024}(x_\ell).
\]

This makes the fixed-residual interpretation much cleaner than adding the entire narrow block output on top of `x`.

## Important implementation choice

This is **not** a stock Qwen decoder block.

For the first version, keep the outer residual width fixed at 1024 and treat the narrow computation as a residual update. Do not simultaneously add stagewise residual resizing.

A concrete initial schedule is:

```text
residual width: 1024 throughout

active compute:
512, 512, 768, 768, 1024, 1024
```

For the first implementation, let the inner `QwenBlock_d` keep the normal Qwen pre-RMSNorm, Q/K RMSNorm, GQA and gated MLP rules for width `d`; do not add another standalone norm around the outer projections unless it becomes a separate ablation. The projection initialization, exact parameter count and FLOPs must be specified before training.

## What this tests

If this model performs much better than C at a similar compute budget, it would suggest:

> preserving a wide residual representation matters more than keeping every early computation wide.

If it performs similarly to C, narrow computation itself may be the larger problem.

This is one of the highest-value body-architecture diagnostics because it separates **representation width** from **computation width**.

---

# 9. Wide–narrow–wide comparator

## Motivation

The original SWT hypothesis is approximately:

```text
narrow early
→ medium
→ wide late
```

A strong alternative hypothesis is:

```text
wide early
→ narrow middle
→ wide late
```

This tests whether early capacity is especially important and whether capacity can be removed more safely from middle layers.

## Simple comparator

A conservative profile is:

```text
1024
1024
768
768
1024
1024
```

using the same transition rule as the SWT comparison.

A more aggressive budget-reallocation profile could eventually use widths above 1024 at the ends, but only after solving the complete matched budget.

## Why this is a comparator

Wide–narrow–wide allocation has close prior work and should be treated as a **strong baseline/comparator**, not automatically as our architectural contribution.

Its purpose is to answer:

> Is monotonic widening actually the right direction, or is early capacity more valuable than we assumed?

## Interpretation

```text
narrow→wide loses
wide→narrow→wide wins
```

would support the finding:

> capacity can be removed more safely from middle layers than from the earliest layers.

If both variable-width profiles lose to the uniform model, the value of nonuniform residual width itself becomes questionable in this regime.

---

# 10. Recommended experiment order

Do not change everything at once.

## Experiment 1 — T768 tied low-rank control

Keep the body uniform at 1024:

```text
B0: tied 1024
vs.
T768: tied 768 + input/output adapters
```

This is a useful low-rank tying diagnostic, but remember that T768 is ~15% smaller than B0. It is not a total-budget-matched tying-vs-rank isolation.

## Experiment 2 — partial sharing

Keep the uniform 1024 body:

```text
B0 tied 1024
vs.
partial-shared 512 + input-private 128 + output-private 384
```

Run this after T768 so that a partial-sharing result can be interpreted against a simpler tied-low-rank control.

## Experiment 3 — input-heavy allocation curve

Extend the uniform-body allocation family:

```text
A512: 512 / 512   existing
A640: 640 / 384   new
A768: 768 / 256   new
```

Use the same uniform `1024 × 6` body and the same training setup.

This tests whether the degradation continues as output rank is reduced below 512.

If A768 is informative, compare:

```text
A768
vs.
A768-Direct
```

to compare adapter-based expansion with the bundled direct first-block expansion. Do not interpret it as an adapter-only ablation.

## Experiment 4 — fixed-residual variable-compute

Test:

```text
residual = 1024 throughout

active compute:
512,512,768,768,1024,1024
```

against the uniform B0-style body with the same vocabulary interface.

This isolates cheap early computation from full residual compression.

## Experiment 5 — wide–narrow–wide comparator

Test a profile such as:

```text
1024,1024,768,768,1024,1024
```

with the same transition rule and vocabulary interface used for the variable-width comparison.

Treat it as a strong baseline, not as the main proposed method.

## Experiment 6 — output-rank mechanism check

Keep D's body fixed and change only the output dimension:

```text
D-896   existing
D-1024  new
```

This directly tests whether the `1280 → 896` interface is costly.

Do not require equal total parameters for this mechanism ablation.

## Experiment 7 — clean output-1024 architecture

Test:

```text
input ≈232–256
body 512,512,768,768,1024,1024
output 1024
```

against a similar-budget uniform reference.

## Experiment 8 — direct transitions

After a promising width schedule is identified, compare:

```text
T1: block → separate Linear(d→D)
vs.
Direct: boundary block itself outputs D
```

Keep all stage widths and embedding interfaces fixed.

## Experiment 9 — output-1280 architecture

Only if larger output rank looks useful:

```text
input 64
body 256,256,512,512,768,1280
output 1280
```

This is the most aggressive allocation and should not be the first follow-up.

---

# 11. Summary

The next architecture set should answer specific mechanism questions rather than simply expand the grid.

| Direction | Main question | Priority |
|---|---|---:|
| **Partial sharing** | Can we preserve a tied subspace while allowing specialization at nearly the same total budget? | **Highest** |
| **T768 tied low-rank** | How efficient is exact tying when the shared rank is reduced? | **High diagnostic** |
| **A640 / A768 input-heavy** | What happens when the fixed table budget moves beyond the 512/512 symmetric point? | Medium-high control |
| **A768-Direct** | Does the bundled direct first-block expansion behave differently from adapter-based expansion? | After A768 |
| **Fixed residual / narrow compute** | Is residual compression, rather than cheap computation, what hurts C/D? | **High** |
| **Wide–narrow–wide** | Is early capacity more valuable than monotonic widening assumes? | **High comparator** |
| **Output 1024** | Does a cleaner full-width prediction interface improve stagewise models? | Medium |
| **Direct transition** | Is the separate resize projection itself suboptimal? | After a good width profile |
| **Output 1280** | Does very large output rank justify severe input/body compression? | Low / aggressive |

Recommended first mechanism runs:

```text
1. Partial-share 512/128/384
2. T768
3. Fixed-residual variable-compute
```

The input-heavy curve can be run in parallel as a compact allocation study:

```text
A640
A768
then A768-Direct only if A768 is informative
```

These runs answer different questions: the first group isolates **why B0/C/D behaved as they did**, while the second completes the **input-versus-output allocation curve**.


---

## Review notes for implementation

Before launching any new variant:

1. instantiate the exact model and record **unique parameter count**;
2. estimate or measure training FLOPs/token rather than inferring efficiency from parameters;
3. keep the Qwen3 block conventions fixed unless the variant explicitly changes them;
4. distinguish **mechanism ablations** from **budget-matched architecture comparisons**;
5. for any direct/boundary block, verify the residual equation with unit tests at matching dimensions;
6. do not interpret a lower-rank model losing as evidence against tying unless total-capacity differences are accounted for.

---

## Implementation mapping

The reviewed variants are implemented in `capacity_allocation/modeling.py` and
use the existing `train.py --arm ...` path. This records availability, not a
decision to train the entire set.

| Design name | CLI arm | Exact parameters |
|---|---|---:|
| Tied rank 768 | `T768` | 212,646,400 |
| Optional tied rank 512 | `T512` | 173,226,496 |
| Partial shared 512/128/384 | `P512-128-384` | 251,542,016 |
| Input-heavy 640/384 | `A640` | 251,017,728 |
| Input-heavy 768/256 | `A768` | 251,017,728 |
| Direct input expansion | `A768-Direct` | 243,939,328 |
| Fixed residual / variable compute | `FixedResidual` | 217,853,440 |
| Wide–narrow–wide comparator | `WNW` | 238,827,008 |
| D with output rank 1024 | `D-1024` | 266,729,216 |
| Output 1024, input 256 | `O1024-I256` | 253,865,472 |
| Output 1024, input 232 | `O1024-I232` | 250,206,720 |
| Output 1280 | `O1280` | 250,627,840 |
| C with direct boundaries | `C-Direct` | 198,813,184 |
| D with direct boundaries | `D-Direct` | 246,855,424 |

The tied and partially shared arms use parameter identity, not copied values.
Because a tied table cannot receive the independent output-table initialization
used by the A-family, its output projection receives the equivalent initial
gain `sqrt(reference_width / output_rank)`. This preserves the intended initial
logit scale while leaving the forward equation unchanged and introducing no
new parameter.

For `FixedResidual`, “same vocabulary interface” is resolved as B0's exactly
tied dense 1024 table with no input/output vocabulary adapters. This makes its
comparison with B0 a body-compute intervention rather than another simultaneous
embedding intervention. `C-Direct` and `D-Direct` provide concrete same-width-
schedule comparisons for the reusable direct-widening block; unsupported direct
shrinking fails configuration validation.
