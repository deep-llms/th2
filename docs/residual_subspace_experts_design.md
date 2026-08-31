# Tied Residual Subspace Experts

## Status

Implemented as the `residual_subspace_experts` arm. The first intended screen
is `residual_subspace_experts_tied_g12_r120_q80`; it has not yet been trained.
The method is a research candidate, not an established improvement.

## Effective embedding

For token factor $z_i\in\mathbb{R}^{r}$, the exactly tied vector is

\[
e_i = B_0(z_i) +
\sum_{g\in\operatorname{TopK}(i)}
\pi_{ig} V_g(U_g(z_i)).
\]

The router is context independent:

\[
s_{ig}=\frac{
\operatorname{normalize}(W_qz_i)^⊤
\operatorname{normalize}(k_g)}{\tau},
\qquad
\pi_i=\operatorname{softmax}(\operatorname{TopK}(s_i)).
\]

Consequently, a token has the same effective vector at the input and output
for the entire forward pass. There is no copied classifier table and no stale
hard assignment artifact.

The production configuration is:

- vocabulary $V=151{,}936$, hidden width $d=1024$;
- global/token-factor rank $r=120$;
- 12 residual experts, each with bottleneck rank $q=80$;
- router dimension 32 and top-2 routing;
- router temperature 1.0.

## Parameter budget

Including all custom Linear biases and router parameters:

| Part | Parameters |
|---|---:|
| Token factors, $V\times120$ | 18,232,320 |
| Global projection and bias | 123,904 |
| 12 biased $120\to80\to1024$ experts | 1,111,488 |
| Router projection, bias, and keys | 4,256 |
| **Total** | **19,471,968** |

The tied global LR128 implementation has 19,579,904 parameters. The candidate
is 107,936 parameters (0.55%) smaller, so it is a close matched-budget screen.

Although the global path has rank 120, the table is not globally restricted to
that rank: token-dependent diagonal gates multiply the expert terms. Ignoring
bias contributions, its algebraic upper bound is
$120+12\times80=1080$, sufficient to reach hidden rank 1024.

## Initialization and optimization

- Token factors, global projection, router, expert keys, and expert up
  projections use normal initialization with standard deviation 0.02.
- Expert down weights and both residual biases start at zero.
- Therefore step zero is exactly the global rank-120 embedding.
- Expert up weights are nonzero, so expert down weights receive useful
  gradients on the first step. Initializing both expert factors to zero would
  incorrectly block this learning signal.
- The training launcher uses `lambda_div=0.01`.
- Load balancing uses hard top-2 load together with the full pre-top-k router
  softmax. This is important: a loss computed only from sparse top-k weights
  cannot give an unselected/dead expert a recovery gradient.
- Normalized expert keys are excluded from AdamW decay. Their scale is removed
  exactly by normalization, so decay cannot regularize the function and would
  only push their norm toward the router epsilon.

The auxiliary loss is consumed once per trainer forward and its graph reference
is cleared even when the configured weight is zero.

## Exact tied output

The global logits are

\[
(HB_0^\top)Z^\top + Hb_0.
\]

For each expert, the head gathers only the tokens selecting that expert and
computes

\[
(HV_g^\top)U_g(Z_{I_g})^\top + Hb_g,
\]

then multiplies each vocabulary column by its route weight and accumulates it
into the single flat-vocabulary logit tensor. This is algebraically equal to
`hidden @ embed.materialize().T`; it does not construct a persistent
$V\times d$ parameter or change the flat softmax.

Each model forward computes the full-vocabulary route once. Input token lookup
gathers from that route, and the tied head consumes the exact same top-k index
and weight tensors. This is stronger than merely recomputing the same formula:
in BF16, small-batch and full-vocabulary GEMMs may use different kernels and
round a borderline top-k decision differently.

During training, routing and expert token latents are rebuilt with a complete
autograd graph. During no-gradient evaluation, the vocabulary routing and
down-projected expert token latents are cached after their first use. The cache
is invalidated automatically when returning to training, loading a state dict,
or moving/changing the module's device or dtype. This avoids a
vocabulary-wide router calculation at every autoregressive decoding step.

## Experiment entry points

- Training launcher:
  `scripts/train_residual_subspace_experts_tied.sh`
- Sequential runner name:
  `residual_subspace_experts_tied_g12_r120_q80`
- Output directory:
  `/mnt/local/_outputs/sparse_embedding/residual_subspace_experts_tied_g12_r120_q80`

The launcher retains the B200 comparison protocol: BF16, batch size 16 per
device, gradient accumulation 4, seed 42, cosine schedule, 500 warmup steps,
and `ddp_find_unused_parameters=false`. Empty expert selections are deliberately
kept in the output-head autograd graph, so disabling unused-parameter discovery
is valid.

## Correctness coverage

`compositional/test_residual_subspace_experts.py` verifies:

- exact production parameter count;
- exact rank-120 behavior at initialization;
- factorized logits against explicit table materialization in float64;
- matching hidden and parameter gradients for the two computations;
- top-2 normalization and token-static routing;
- finite-precision sharing of one route between input lookup and output head;
- useful first-step residual gradients;
- all experts remain in autograd even when some receive zero tokens;
- full-probability load-balancing gradients and one-time graph consumption;
- strict state schema and configless reconstruction;
- eval-cache reuse and every invalidation path;
- trainer integration and finite tiny-Qwen loss/backward;
- ordinary tied-head save/load ownership and round trips.

This coverage establishes implementation correctness, not empirical superiority.
