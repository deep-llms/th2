# RankLift — Nonlinear Rank Expansion for an Exactly Tied Token Interface

Status: 2026-09-01. **Implemented and CPU-tested; production GPU smoke and
throughput gates remain before a full training run.** The method must pass the
systems and short-training gates in §8 before a checkpoint-10k run is
justified.

---

## 1. Summary

RankLift compresses the tied input embedding and output classifier of a causal
language model while retaining all of the following:

- the original fixed tokenizer and vocabulary;
- one exact flat-vocabulary softmax;
- the same effective token table on the input and output sides;
- end-to-end training from random initialization;
- ordinary dense matrix multiplications rather than routing or expert kernels.

Each token owns a small private code. A shared nonlinear function expands that
code into a wider feature vector, and a shared linear projection maps the
features to the model hidden dimension. The private code controls
vocabulary-scaled parameter storage; the expanded feature width controls the
maximum discriminative rank of the centered token table and the cost of the
softmax.

The proposed contribution is therefore **not** merely “use a gated MLP for an
embedding.” The hypothesis is:

> Per-token storage dimension and exactly tied classifier rank can be
> decoupled by a shared nonlinear feature lift, recovering output capacity
> without storing a wide private vector for every vocabulary item.

At the Qwen3-0.6B interface dimensions (`V=151,936`, `d=1024`), the reference
configuration uses 19,396,128 trainable interface parameters: 8.02× smaller
than the 155,582,464-parameter dense tied table and slightly smaller than the
19,579,904-parameter global-LR128 control.

This is a falsifiable candidate, not a predetermined paper result. It should
be abandoned if it cannot beat the matched GroupReduce, Slim, and nonlinear
factorization controls on a quality–efficiency Pareto frontier.

---

## 2. Motivation

The project's uniform-capacity compressed interfaces have converged to a
similar quality range: global low-rank tied R128, SharedLocal, and PureLocal
remain near mean PPL 33.9, while frequency-tiered GroupReduce improves to
31.33 but remains far behind the dense tied model at about 24.6. The detailed
results are recorded in `docs/PROJECT_NOTES.md`.

For a global factorization

```text
E = Z P^T,       Z in R^(V×r), P in R^(d×r),
```

the effective tied table has matrix rank at most `r`. With `r=128`, every
token classifier is restricted to the same 128-dimensional linear subspace.
Increasing `r` directly also increases the dominant vocabulary-scaled term
`V*r`, so an approximately 8× parameter budget does not permit a much wider
private factor.

RankLift keeps only `c` private values per token but transforms them through a
shared nonlinear map into `m>c` features. Across the vocabulary, nonlinear
features can be linearly independent of the original code columns, so the
effective matrix can have rank as high as `m`, rather than `c`.

The nonlinearity is essential. If the lift were linear,

```text
F = [Z, Z A] = Z [I, A],
```

then `rank(F) <= c`; the wider representation would only be a
reparameterization of global low rank.

---

## 3. Method

### 3.1 Notation

- `V`: vocabulary size (`151,936`).
- `d`: Transformer hidden size (`1024`).
- `c`: private code width (`124`).
- `q`: nonlinear lift width (`336`).
- `m = c + q`: complete feature width (`460`).
- `Z in R^(V×c)`: learned private token codes.
- `A, B in R^(c×q)` and `a, b_lift in R^q`: shared gated-lift parameters.
- `P in R^(d×m)`: shared projection to the Transformer hidden dimension.
- `b_global in R^d`: global input bias, following the repository's existing
  compressed-embedding convention.

`RMSNorm` below is parameter-free and is applied independently to each token
code. It must use the same explicit epsilon in every path.

### 3.2 Nonlinear feature lift

For token `i`:

```text
u_i = RMSNorm(z_i)
g_i = SiLU(u_i A + a) * (u_i B + b_lift)
f_i = concat(z_i, g_i)                         in R^m
e_i = f_i P^T + b_global                       in R^d
```

`*` is element-wise multiplication. Stacking all token rows gives

```text
F = [Z, G(Z)]                                  in R^(V×m)
E = F P^T + 1 b_global^T                       in R^(V×d).
```

`E` is the single effective token table. No separate output parameters are
allowed.

The gated lift is the reference implementation because it supplies
multiplicative nonlinear features with two small shared projections. It is
not, by itself, the novelty claim and must be compared with simpler ReLU and
SiLU lifts (§9).

Initialization follows Qwen's 0.02 normal scale for `Z`, both lift matrices,
and `P`; all three affine biases start at zero. Xavier initialization is not
used here: RMS-normalized codes have unit-scale coordinates, so Xavier lift
branches would make the multiplicative features dominate the private codes
and produce an excessively large effective table at step zero.

### 3.3 Input lookup

Only requested rows need to be constructed:

```python
z = Z[input_ids]
u = rms_norm(z)
g = silu(u @ A + a) * (u @ B + b_lift)
f = cat([z, g], dim=-1)
embedding = f @ P.T + b_global
```

This path never materializes a `V×d` table.

### 3.4 Exact tied output head

For contextual hidden states `H in R^(N×d)`, exact logits are

```text
logits = H E^T
       = (H P) F^T + (H b_global) 1^T.
```

The implementation must compute this factorized expression directly:

```python
F = embed.materialize_features()               # (V, m), not (V, d)
projected_hidden = hidden @ P                  # (..., m)
logits = projected_hidden @ F.T                # (..., V)
logits += (hidden @ b_global).unsqueeze(-1)
```

This is algebraically identical to multiplying by the effective input table.
It produces one `N×V` logits tensor and uses normal GEMMs. It must not build
the full `V×d` table, construct multiple expert logits tensors, or use a
straight-through estimator.

The global-bias term is identical for every vocabulary item and therefore
cancels in softmax, but it is retained for exact consistency with the input
embedding and the repository's other tied compressed interfaces.

### 3.5 Exact tying invariant

The implementation must satisfy, in both forward values and gradients,

```text
head(H) == H @ materialize_effective_table().T
```

up to the expected floating-point tolerance. “Uses the same factorization”
is not sufficient: there must be one shared set of `Z`, lift, projection, and
bias parameters, with no independently registered output copy.

---

## 4. Parameter and storage accounting

Reference configuration:

| Component | Shape | Parameters |
|---|---:|---:|
| Private codes `Z` | `151,936 × 124` | 18,840,064 |
| Lift matrices `A`, `B` | `2 × 124 × 336` | 83,328 |
| Lift biases `a`, `b_lift` | `2 × 336` | 672 |
| Projection `P` | `1024 × 460` | 471,040 |
| Global bias | `1024` | 1,024 |
| **Total** | | **19,396,128** |

Comparators:

| Interface | Parameters | Compression vs dense tied |
|---|---:|---:|
| Dense tied `V×d` | 155,582,464 | 1.00× |
| Global tied LR128 | 19,579,904 | 7.95× |
| RankLift `c=124, q=336` | 19,396,128 | 8.02× |

The trainable checkpoint is therefore genuinely approximately 8× compressed.
Runtime working memory must be reported separately:

- compact private-code table in BF16: about 35.9 MiB;
- materialized `F` in BF16: about 133.3 MiB;
- a materialized dense `E` in BF16 would be about 296.8 MiB and is forbidden
  in the normal head path.

During training, `F` cannot be cached across optimizer steps because both the
codes and lift parameters change. During inference, where parameters are
frozen, `F` may be generated once and cached. Such a cache improves latency
but reduces runtime-memory compression; checkpoint size, peak training
memory, and serving working memory must never be conflated.

---

## 5. Computational properties

The dominant vocabulary projection has width `m=460`, or 44.9% of the dense
head width `d=1024`. The additional full-vocabulary lift computes two
`V×c` by `c×q` GEMMs, totaling approximately 12.66 billion MACs each time all
features are materialized. For a large training micro-batch this should be
small relative to the `N×m` by `m×V` output GEMM, but it must be measured.

Expected advantages:

- one standard vocabulary GEMM rather than several expert-specific logits;
- no learned routing, dispatch, sorting, or token-to-expert communication;
- no reconstruction of a `V×d` table;
- output arithmetic below the dense head and simpler kernels than a direct
  Tensor-Train contraction.

Expected limitations:

- Slim or P-VQ may require less output arithmetic because they reuse much
  smaller codebooks;
- materializing and backpropagating through `F` adds working memory;
- the whole model may improve much less than the nominal 55% reduction in
  head width because the Transformer body is unchanged;
- a poor implementation can erase the algebraic advantage through repeated
  feature materialization or unnecessary contiguous copies.

Consequently, no speed claim is permitted from FLOP counting alone.

---

## 6. Capacity and limitations

Ignoring the shared row offset `b_global`, which contributes the same scalar
to every output logit, nonlinear feature columns permit

```text
rank(F P^T) <= min(rank(F), rank(P)) <= m = 460,
```

instead of the LR128 limit of 128. This is a rank ceiling, not a guarantee
that training will use all 460 directions. The implementation must log the
effective numerical spectrum of centered sampled or fully materialized tables
at evaluation checkpoints. The uncentered table
`E = F P^T + 1 b_global^T` has a formal ceiling of `m+1`, but that shared
rank-one offset cannot distinguish vocabulary items and cancels in softmax.

RankLift still gives each token only `c=124` independent trainable values.
Each row lies on a shared 124-dimensional nonlinear manifold embedded in the
wider feature space. It is therefore more expressive than a 124-dimensional
linear subspace but not equivalent to giving every token 460 independent
coordinates. This distinction must be explicit in any paper claim.

Rare tokens are the main risk: their private codes receive few input-side
updates, although exact tying provides dense output-side gradients. The
existing frequency-binned perplexity evaluation is mandatory.

---

## 7. Nearest prior work and novelty boundary

The equation/code/protocol audit for every implemented comparator is recorded
in `docs/comparator_fidelity_audit.md`. Experiment names and paper claims must
follow that audit.

RankLift sits near several established methods:

- **Distilled Embedding / Funneling decomposition** applies a nonlinearity to
  a low-dimensional token factor and uses the same reconstructed matrix for
  input and output. Its nonlinear representation remains width `r`, so the
  final table remains rank at most `r`; its published protocol also starts
  from a dense pretrained embedding and uses embedding distillation.
- **DeFINE** maps a low-dimensional token embedding through a deep
  expand/reduce network, but its language-model classifier remains in a low
  output dimension. It is the closest “map–expand” architectural precedent.
- **Slim Embeddings** uses combinatorial parameter sharing and includes a
  tied experiment, but does not give every token a private continuous code.
- **TT-Embedding** can avoid ordinary matrix low rank, but the published
  language-model experiments use separate TT decompositions for embedding
  and softmax. An exactly tied version in this repository is an adaptation.
- **GroupReduce** assigns different low-rank spaces to frequency groups. The
  published method is post-hoc and compresses input and output separately;
  the repository's matched exactly tied from-scratch version is an
  adaptation and is currently the strongest measured compressed control.
- **P-VQ** reuses the embedding representation in softmax and accelerates the
  output, but uses vector quantization and a staged clustering curriculum.
- **Adaptive Input/Softmax, MultiHashFormer, and T-FREE** escape parts of the
  same bottleneck by changing the output distribution, prediction target, or
  tokenizer. They belong in a separate-interface comparison rather than the
  strictly controlled flat-softmax table.

The defensible novelty claim, if supported, is narrow:

> A shared nonlinear expansion raises the rank of the same effective table
> used for fixed-tokenizer input lookup and a flat tied softmax, while
> vocabulary-scaled trainable storage remains at the small private-code
> width and execution remains GEMM-based.

The gated activation alone is not novel. Exact tying alone is not novel.
From-scratch training alone is not novel. A paper must demonstrate that the
decoupling principle produces a better measured frontier than the methods
above.

Primary references:

- Lioutas et al., 2020, Distilled Embedding:
  <https://aclanthology.org/2020.findings-emnlp.250/>
- Mehta et al., 2020, DeFINE: <https://arxiv.org/abs/1911.12385>
- Li et al., 2018, Slim Embeddings: <https://arxiv.org/abs/1711.09873>
- Khrulkov et al., 2020, Tensorized Embedding Layers:
  <https://aclanthology.org/2020.findings-emnlp.436/>
- Chen et al., 2018, GroupReduce: <https://arxiv.org/abs/1806.06950>
- Zhang et al., 2021, P-VQ:
  <https://ojs.aaai.org/index.php/AAAI/article/view/17688>
- Baevski and Auli, 2019, Adaptive Input Representations:
  <https://arxiv.org/abs/1809.10853>
- Xue et al., 2026, MultiHashFormer:
  <https://arxiv.org/abs/2606.28057>
- Deiseroth et al., 2024, T-FREE:
  <https://aclanthology.org/2024.emnlp-main.1217/>

---

## 8. Pre-registered evaluation and stop gates

### 8.1 Fair protocol

The controlled table must hold constant:

- Qwen3-0.6B Transformer body and tokenizer;
- training corpus, sample artifacts, order, and number of tokens;
- optimizer, schedule, BF16 policy, global batch, seed policy, and checkpoint
  steps;
- approximately 19.4M trainable tied-interface parameters;
- exact flat-vocabulary cross-entropy.

Every result must label whether it is a faithful published protocol or a
from-scratch/exact-tied adaptation.

### 8.2 Mandatory controls

1. Dense tied Qwen baseline.
2. Global tied LR128.
3. Matched end-to-end tied GroupReduce.
4. Parameter-matched tied Slim.
5. Same-width nonlinear/Funneling factorization.
6. DeFINE-style input expansion with a low-dimensional tied classifier.
7. Exact-tied TT adaptation, if a production-safe execution path is
   available.
8. RankLift.

P-VQ's faithful curriculum is a separately labelled post-training comparison.
Adaptive Softmax, MultiHashFormer, and T-FREE are a separate-interface table.

### 8.3 Systems gate: 100 steps

Before any long run, verify:

- forward and backward complete under the production DDP configuration;
- no unused parameters and no rank-dependent numerical divergence;
- exact-head value and gradient tests pass;
- resume and configless checkpoint loading are exact;
- peak allocated/reserved memory is below the dense tied run;
- end-to-end tokens/s is at least 90% of dense, unless RankLift already
  supplies a clearly superior quality point at the same wall-clock compute;
- no full `V×d` reconstruction occurs in the normal head path.

Failure of correctness, OOM, or a material slowdown caused by the interface
stops the experiment immediately.

### 8.4 Quality gate: 500 steps

Run at least three controlled seeds for RankLift, GroupReduce, Slim, and the
nearest nonlinear control. Compare validation loss/PPL versus both tokens and
wall-clock time. RankLift proceeds only if it:

- clearly beats global LR128;
- beats or is trending better than matched GroupReduce and the nonlinear
  controls beyond seed variation; and
- does not obtain the apparent gain by consuming more wall-clock compute.

### 8.5 Full result: checkpoint 10k

A 10k run is justified only after both gates pass. Report:

- multilingual validation PPL and per-language PPL;
- zero-shot and finetuning benchmarks;
- frequency-binned and token-level PPL diagnostics;
- interface parameters, checkpoint bytes, training peak memory;
- training tokens/s and time to checkpoint;
- output-head latency, prefill latency, decode latency, and serving working
  memory with and without a frozen `F` cache;
- at least two long-run seeds for differences small enough to plausibly be
  random.

The current matched-GroupReduce result, mean PPL 31.33, is the minimum
quality bar. Merely beating LR128 is not a publishable success.

---

## 9. Required ablations

1. **Linear lift:** replace `G(Z)` with `Z A`. This should not raise rank
   beyond `c` and validates the proposed mechanism.
2. **Nonlinearity:** ReLU, SiLU, and the reference gated lift at matched
   parameter count.
3. **Expansion width:** sweep `m` while adjusting `c` to preserve the total
   interface budget. This traces quality versus softmax cost.
4. **Input-only expansion:** a DeFINE-style control in which the Transformer
   sees an expanded embedding but the tied classifier stays at width `c`.
5. **Tying:** exactly tied versus independent compressed output, labelled as
   a capacity diagnostic rather than the primary method.
6. **Normalization and biases:** parameter-free RMSNorm on/off and lift
   biases on/off. Custom linear layers should include bias in the reference
   configuration.
7. **Rank diagnostics:** singular-value spectrum and effective numerical rank
   at multiple checkpoints.
8. **Frequency diagnostics:** head, torso, and tail PPL using the project's
   existing fixed bins.

---

## 10. Implementation requirements and tests

Implemented code structure:

```text
compositional/nonlinear_factorizations.py RankLift, Funneling, and DeFINE
compositional/tied_head.py                TiedRankLiftHead
compositional/test_nonlinear_factorizations.py correctness tests
scripts/train_ranklift_tied.sh            production launcher
```

The same wiring exposes the parameter-matched Funneling control and the
fixed-tokenizer DeFINE adaptation. Slim and TT live in
`compositional/compressed_baselines.py`; TT table materialization is chunked
and activation-checkpointed so the production configuration does not form its
former full-vocabulary 123 GB intermediate.

The current regression count is recorded in `docs/PROJECT_NOTES.md` after each
implementation audit. The remaining production gates are the multi-GPU
BF16 smoke and measured B200 memory/throughput checks. This session's
container cannot communicate with its NVIDIA driver, so those results are not
claimed here.

Mandatory tests:

1. Exact parameter count for the reference configuration.
2. Input lookup equals rows selected from a materialized reference table.
3. Factorized head equals `hidden @ E.T` in FP32 and BF16 tolerances.
4. Gradients for every parameter match the materialized reference on a tiny
   model.
5. A linear-lift fixture has rank no greater than `c`; a nonlinear fixture
   can exceed `c`.
6. The head does not register a second copy of the embedding module.
7. State-dict round trip, strict loading, configless loading, and resume are
   exact.
8. Four-rank DDP smoke test with `find_unused_parameters=False`.
9. Gradient checkpointing and mixed-precision training produce finite loss
   and gradients.
10. Production-shape memory and throughput benchmark records allocations and
    confirms that the normal path does not materialize `V×d`.

The feature materialization must be recomputed after every optimizer update.
Any within-step reuse must preserve autograd correctness and must not retain a
graph across micro-batches. Optimization should begin only after the simple
reference implementation passes all value and gradient tests.

---

## 11. Paper decision rule

RankLift is not sufficient for a main-paper claim merely because it trains or
beats global low rank. Continue this research direction only if it produces a
Pareto improvement over matched GroupReduce, Slim, TT, and the closest
nonlinear controls.

A credible main-paper package would require:

- consistent wins at multiple compression ratios, not only one 8× point;
- validation at more than one model scale or dataset regime;
- a demonstrated relationship among private-code width, effective rank,
  quality, and real execution cost;
- exact protocol separation for from-scratch, post-hoc, and changed-output
  methods;
- reproducible systems measurements on normal kernels.

If RankLift wins quality but is slower, or wins speed but not quality, it may
still form one useful Pareto point. If it only beats LR128, or if its gain
disappears across seeds, the method should be dropped rather than promoted as
a positive result.
