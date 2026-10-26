# Efficient LLM direction: selective heads and shared KV

Background review copied from `/disk/thuat/DOT/docs/EFFICIENT_LLM_RESEARCH_20261026.md`.
References to our previous model/results below describe historical DOT work, not
experiments or required architecture in this new project.

Research reviewed 26 October 2026 (SGT). Research and planning only; no new
training, remote commands, environment changes or GPU work.

## Recommendation

The objective is credible: preserve useful model quality while reducing actual
inference cost. There is strong prior work supporting both selective attention
and cross-layer KV sharing. However, **“MoE + selective heads + shared KV” is not
by itself a sufficient research contribution**. Much of this combination, and
several apparently natural extensions, already exists.

I recommend investigating **structured attention/KV compression of a pretrained
GQA model**, with one explicit serving objective. Start by reproducing a simple
baseline and measuring the available savings. Do not start another architecture
sweep or add STOP, an MLP MoE and learned cross-layer reconstruction together.

For lower decode latency, the most useful starting reference is Polar Sparsity.
For smaller persistent cache, use GQA and CLA as the starting references. These
are different objectives and need different baselines. I have not established a
novel, launch-ready method from this review.

## What we would actually save

| Objective | Mechanism | What does not follow automatically |
| --- | --- | --- |
| Smaller persistent KV cache | Fewer independent KV heads, fewer cache-producing layers, bounded histories | Faster execution of each attention layer |
| Less attention computation | Fewer active queries/heads or fewer attended positions | Fewer KV entries written and retained |
| Fewer cache reads during decode | Avoid reading whole KV groups or token blocks | Equivalent quality or low routing overhead |
| Faster end-to-end generation | Savings exceed dispatch, indexing, projection and other overhead | A kernel speedup implies the same model speedup |
| Higher capacity-limited throughput | Fit more requests into the same memory | Lower latency at an unchanged batch size |

These distinctions should determine the experiment before choosing a router.

## Strongest relevant prior work

Results in this section are the authors' measurements, not replications on our
B200. Different model sizes, runtimes, batch sizes and contexts prevent ranking
the reported speedups directly.

### Selective heads and attention experts

| Work | Mechanism and evidence | Consequence for us |
| --- | --- | --- |
| [MoA, EMNLP 2022](https://aclanthology.org/2022.emnlp-main.278.pdf) | Selects query/output head experts while sharing K/V; evaluated on translation and masked language modeling. | Head routing itself is established; this is not direct evidence about modern GQA decoding. |
| [MoH, ICML 2025](https://proceedings.mlr.press/v267/jin25l.html) | Combines always-active shared heads with routed heads, using only part of the head pool per token. | A natural shared-head/routed-head design already exists. |
| [SwitchHead, NeurIPS 2024](https://arxiv.org/html/2312.07987v3) | Mixtures in value/output projections allow fewer attention matrices; SwitchAll also combines this with feed-forward MoE. The 262M C4 example uses 44% of baseline compute and 27% of activation memory with comparable PPL. | “Attention experts + MLP experts” is already studied. These are compute/memory ratios, not equivalent end-to-end speed ratios. |
| [Polar Sparsity](https://arxiv.org/html/2505.14884v3) | Predicts active attention heads/groups and executes selective attention with custom kernels. Reports up to 2.2× end-to-end speedup, especially in batched decoding; preserves dense QKV projections for cache consistency. | Particularly relevant systems baseline. It already supports GQA group sparsity. |
| [DuoAttention, ICLR 2025](https://arxiv.org/html/2410.10819v1) | Separates full-history retrieval heads from streaming heads with short histories. Reports up to 1.67× memory reduction and 1.50× decode speedup for its GQA setting. | Another way to reduce reads and storage while keeping every head; must compare against it for long-context claims. |

MoH's Llama-3-8B experiment is not evidence of a cheap conversion: the paper
describes 300B tokens of domain adaptation followed by 100B tokens of MoH
continued training. Its throughput comparison also does not establish a win
against our best dense FA4 execution. We should not transfer its headline
quality improvement to a small adaptation budget.
[MoH training details and efficiency section](https://arxiv.org/html/2410.11842v3).

### Sharing cache across layers

| Work | Mechanism and evidence | Consequence for us |
| --- | --- | --- |
| [CLA](https://arxiv.org/html/2405.12981v1) | Adjacent layers reuse KV activations. Experiments at 1B/3B find approximately 2× cache reduction with near-baseline accuracy for selected configurations, especially MQA plus pair sharing. | Mandatory simple baseline; it does not directly reduce the repeated KV reads in each consuming layer. |
| [MLKV, NAACL Findings 2025](https://aclanthology.org/2025.findings-naacl.305.pdf) | Shares KV across both heads and layers, including uptraining experiments. | Joint head/layer sharing is already established. |
| [LCKV, ACL 2024](https://arxiv.org/html/2405.10637v2) | Uses deep-layer KV at shallower layers, with previous-token dependence and iterative parallel training/prefill. | Relevant to our old deep-to-shallow idea, but introduces substantial training/prefill complexity. |
| [YOCO, NeurIPS 2024](https://arxiv.org/abs/2405.05254) | A self-decoder produces shared global KV for a cross-decoder. Its architecture permits skipping much of the cross-decoder's historical prefill. | Prefill savings require the actual architectural dependency structure; ordinary CLA does not inherit them. |
| [SkipV1Former](https://arxiv.org/html/2510.16807v1) | Reuses a subset of first-layer value heads in later layers. Around 25% total KV reduction is possible; reported throughput remains within roughly ±2% of baseline at tested batch sizes. | “Reuse shallow V only” already exists and can be primarily a memory improvement. |
| [FusedKV](https://arxiv.org/html/2512.03870v1) | Reconstructs later caches from earlier sources; Lite uses fixed sources. The full method's memory-bound example has approximately 1.5× higher time per output token; Lite stays close to baseline. | Learned sharing can lose latency despite saving storage. Extra reconstruction must earn its cost. |

The previously discussed **26× LCKV result** needs its original setting attached:
Table 1 compares a 30B model with CPU offloading on an RTX 3090, changing batch
size from 4 to 83 as cache capacity improves; throughput changes from 0.23 to
5.99 tokens/s. This is not a 26× single-request, unchanged-batch, all-GPU speedup.
[LCKV, generation-throughput experiment](https://arxiv.org/html/2405.10637v2).

## Newer work removes several obvious novelty claims

The following sources deserve checking before proposing a method. The 2026
arXiv entries below are treated as preprints, not independently replicated facts.

| Apparent new idea | Existing overlap |
| --- | --- |
| Choose one query head within each GQA group | [Grouped Query Experts, June 2026](https://arxiv.org/html/2606.20945v1) does this while retaining dense KV. Its 250M/30B-token study is small; the highlighted design also includes a shared head, so it activates 9 of 16 heads, not 8 total. |
| Randomly reuse earlier KV during training; choose sharing depth at deployment | [R-CLA, April 2026](https://arxiv.org/html/2604.22782v1) trains exactly this robustness, including pretrained-model adaptation. |
| Learn a nonuniform number of KV groups across layers | [DHA, NeurIPS 2024](https://arxiv.org/html/2406.06567v2) already learns head fusion and layer-dependent allocation. This is distinct from sharing identical KV activations between layers. |
| Route tokens among attention experts with different KV grouping | [mixSGA, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1166/) studies heterogeneous grouped-attention experts. |
| Reuse sparse-attention choices across layers | [HShare, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/de7dc701a2882088f3136139949e1d05-Abstract-Conference.html) and [You Only Index Once, June 2026](https://arxiv.org/html/2606.06467v1) share selected token indices; the latter combines this with YOCO. These are token choices, not head choices. |
| Share an indexer's cache while retaining per-layer choices | [LatentIndex, October 2026](https://arxiv.org/html/2610.04635v1) does this. Its 2.30–2.72× latency figure concerns the indexer, not whole-model decoding. |
| Train with richer head transformations, merge them away at inference | [Virtual Head Attention, October 2026](https://arxiv.org/html/2610.17959v1) explores such reparameterization. Its export does not mean half as many attention matrices; physical projections and virtual attention heads are different counts. |

Also retain a strong simple alternative: change query width and KV-group count
without adding a router. [Cost-Optimal GQA, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.272.pdf)
studies this tradeoff. A complex method needs to outperform the appropriate
smaller-attention baseline, not just the original architecture.

For much longer contexts, token/block sparsity becomes a separate strong
competitor: [NSA](https://arxiv.org/abs/2502.11089) and
[MoBA](https://arxiv.org/abs/2502.13189) reduce attended token blocks. Head
selection cannot be evaluated as if dense attention were the only alternative.

## Causality and implementation constraints

### Skipping a read does not permit skipping a write

Suppose head/group `g` is unused by the query at token `t`. At token `t+1`, the
router may select `g` and need its `K[t,g]` and `V[t,g]`. Omitting those entries
changes the model unless we explicitly define missing history, reconstruct it,
or commit to a persistent selection pattern.

A clean initial design writes every KV group and selectively reads groups for
the current query. This can save attention work but does not reduce the number
of persistent cache entries. A joint storage/read method needs an additional
cache lifecycle rule; the two savings cannot simply be multiplied.

### GQA changes what a skipped head saves

Our [12-layer D configuration](/disk/thuat/DOT/configs/bdot/D.json) has **16 query heads,
8 KV heads and head dimension 128**. Selecting one query from every group leaves
all eight KV groups active. It reduces query-side attention arithmetic, but
does not halve unique KV storage or necessarily halve cache-read traffic.

Selecting whole groups offers a clearer read-saving opportunity, but removes
both query heads in each omitted group. The quality tradeoff must be measured.
This is a design constraint derived from GQA, not a newly proposed method.

### Sharing storage does not merge sequential layer computation

Each later layer has a new query derived from its own hidden state. Even if two
layers point to the same KV buffer, they generally execute separate attention
operations and read that buffer again. Small saved projections, memory-system
caching and increased batch capacity can still help; a 2× capacity saving alone
does not establish a 2× decoding speedup.

Sharing `W_K/W_V` also does not share the resulting cache when the two layers
project different hidden states. A cache-saving implementation must reuse the
activations and storage, not merely tie projection parameters.

### Preserve the actual causal computation

Earlier-layer KV for the current token is available without a cycle. A deep
layer's current-token KV is unavailable to an earlier layer; using previous
tokens' deep KV makes decoding causal but introduces a recurrence that ordinary
parallel teacher forcing does not reproduce automatically.

Likewise, selecting heads from one token's current representation is causal;
selecting the best tokens across a complete sequence can use future information.
[Mixture-of-Depths](https://arxiv.org/html/2404.02258v1) explicitly deals with this
training/inference routing issue. We should not import its sequence-wide top-k
rule into autoregressive inference unchanged.

Qwen's Q/K normalization and RoPE also constrain transformations: a linear head
mixture cannot generally be moved across nonlinear normalization, or arbitrary
position rotations, without changing the operation. Any claimed “merge away at
inference” optimization needs an equivalence test in the exact Qwen placement.

## What the cache numbers mean for our model

Ideal BF16 storage, excluding allocator and workspace overhead:

`bytes = 2 × layers × KV_heads × head_dim × batch × cached_tokens × 2`.

The current 12-layer layout therefore needs **48 KiB per cached token per
sequence**. At batch 1 that is 48 MiB at 1,024 tokens, 384 MiB at 8,192, and
1.5 GiB at 32,768. At batch 16 and 32,768 it is 24 GiB.

The longer lengths are storage extrapolations, **not evidence that the current
1,024-context pilot model works at those lengths**.

| Modification to this layout | Ideal KV-storage reduction |
| --- | --- |
| 16 → 8 query heads, still 8 KV heads | 0% |
| 8 → 4 KV heads, still 16 query heads | 50% |
| Pairwise KV sharing across all 12 layers | 50% |
| Pairwise sharing only within the 8 middle layers; retain 2 stem + 2 coda caches | 33.3% |

Memory capacity is unlikely to be the central motivation for this small model
at batch 1 and context 1,024 on B200. A realistic longer-context or batched
serving workload is needed to establish relevance.

For latency, use the measured bottleneck. If attention is 30% of elapsed time,
halving its cost gives at most `1 / (0.70 + 0.15) = 1.176×` speedup before routing
overhead. This is an illustrative Amdahl calculation, not a measured profile.

## Which direction is worth pursuing?

**First choice: structured compression with measured inference savings.** Use a
pretrained model with useful downstream ability and a supported context length.
Choose either batched-decode latency or cache capacity as the primary objective.
This avoids spending most of the budget learning basic language modeling before
we can tell whether the compression is useful. It is still an empirical choice:
adaptation can fail even where training a modified architecture from scratch
would succeed.

The research question I would investigate is:

> At a fixed serving cost, where must attention retain independent KV
> representations, and where can we remove query or KV work without losing
> useful predictions?

This could lead to a structured, nonuniform layout selected by calibration and
then adapted with ordinary LM training. Such a layout should use regular dense
kernels wherever possible. **This question is not yet a novelty claim:** DHA,
MLKV, R-CLA, mixSGA and cost-optimal GQA overlap substantially. A publishable
method would need a specific missing allocation rule, causal constraint or
measured capability that those baselines do not provide.

If selecting groups dynamically yields a genuine advantage over a static
layout, introduce a cheap router and an execution path that skips work. The
router must justify itself through quality at the same measured cost. Balanced
loads and changing decisions are insufficient evidence.

**Do not prioritize another deep-feedback proxy or an all-at-once MoE/MoH/STOP
architecture.** The former repeats our difficult approximation/training-cost
problem; the latter obscures which mechanism helps. Conditional depth remains
credible, but [LayerSkip](https://aclanthology.org/2024.acl-long.681/) is a stronger
reference than arbitrary exit: it combines early-exit training with
self-speculative verification. Exact verification preserves its full adapted
model's distribution, not the original pretrained model before adaptation.

Our [full-validation B-DOT diagnostics](/disk/thuat/DOT/docs/BDOT_PARTIAL_ROUTER_DIAGNOSTICS_20261025.md)
show nondefault routing without an own-route NLL advantage. Our
[STOP study](/disk/thuat/DOT/docs/BDOT_RESULTS_20261025.md) saved too little depth to overcome overhead.
Neither result proves selective attention cannot work. They do explain why the
next study needs a concrete cost reduction and a simpler matched alternative.

## Smallest useful next investigation — proposed, not launched

1. **Establish the serving bottleneck without training.** On one suitable
   pretrained model, measure prefill and cached decode separately at short and
   long supported contexts, with small and batched requests. Record whole-model
   latency, peak memory and KV bytes. Compare against its best validated dense
   backend; do not time sparse code against an unnecessarily slow dense path.
2. **Check whether useful structure survives removal.** Calibrate fixed head/group
   selection or adjacent-layer KV reuse on data separate from evaluation. Compare
   simple importance selection and random removal. Run the altered model on its
   own hidden-state trajectory. Dense head outputs may help diagnose an oracle
   upper bound, but their computation cannot be omitted from a deployed router's
   cost. A failed unadapted intervention is not proof that adaptation cannot help.
3. **Only if the opportunity survives, use a small matched adaptation study.**
   Start with dense continuation, the relevant simple reduction (fewer KV groups
   for a memory target, fewer query heads for a compute target), one published
   sharing/selection method and at most one new candidate. Match data, adaptation
   budget and evaluation; compare equal memory and equal measured latency as
   well as equal training tokens.

Quality should include held-out LM loss, a fixed downstream suite and suitable
long-context retrieval/comprehension tests if claiming long-context efficiency.
Use an agreed acceptable quality loss before looking at candidate results.
Report same-batch latency separately from maximum-throughput-at-fixed-memory
results, and include adaptation cost. The existing 12-layer/1,024-context setup
is useful for correctness checks, but insufficient for a broad serving claim.

If static reduction matches the router, prefer the static method. If the saved
operations do not reduce whole-model cost, stop before a long training sweep.

## Useful official code entry points

- [Polar Sparsity](https://github.com/susavlsh10/Polar-Sparsity): head/group
  prediction and selective-attention kernels.
- [MoH](https://github.com/SkyworkAI/MoH): shared/routed-head architecture.
- [SwitchHead](https://github.com/robertcsordas/switchhead): attention experts.
- [DuoAttention](https://github.com/mit-han-lab/duo-attention): retrieval/streaming
  head calibration and inference.
- [FusedKV](https://github.com/LivingFutureLab/FusedKV): implementation references;
  its Megatron modifications are not a complete standalone training project.
- [Cost-Optimal GQA](https://github.com/thunlp/cost-optimal-gqa): simpler attention
  allocation baselines.

Reading these implementations does not establish compatibility with our Qwen,
FA4 or B200 environment. No dependencies have been installed for this review.
