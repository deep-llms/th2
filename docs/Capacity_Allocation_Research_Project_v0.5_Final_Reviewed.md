# Where Should Language Models Be Wide?

> Historical design draft. The 2026-09-08 user-approved Qwen3 six-layer setup
> supersedes this document's GPT-2/Llama execution details and parameter counts.
> Follow [CAPACITY_EXPERIMENTS.md](CAPACITY_EXPERIMENTS.md) for current code/data/commands.

Asymmetric embeddings and stagewise width reallocation

**Working draft v0.5 — final setup review, 8 September 2026**

Status: research proposal with a selected pilot setup. The parameter calculations have been checked; model quality, training stability, and novelty beyond the cited comparisons remain unestablished. No training run is reported here.

> **Central question.** At a fixed model budget, how should capacity be distributed among input token representations, contextual computation across depth, and output prediction?

<a id="overview"></a>

## 1. Project overview

### Two connected experiments

**Embedding allocation.** Compare one tied, 1024-dimensional vocabulary table with separate 128-dimensional input and 896-dimensional output tables. Also test 256/768 and 512/512. The vocabulary-table budgets match; projection parameters and computation do not automatically match.

**Width allocation.** Reduce the widths of early Transformer layers and either retain the savings or reinvest them in late layers wider than the original 1024-dimensional body. The reinvestment version is the main architectural hypothesis, not merely a smaller model.

**These are separate budget decisions.** Splitting a tied table into `128 + 896` does not free vocabulary-table parameters for the body: the sum still equals 1024, and adapters add parameters. In the reinvestment experiment, the budget for wider late layers comes from narrower early layers.

### Proposed architecture family

Use the descriptive name **Stagewise Widening Transformer (SWT)**. Call the reinvestment variant **SWT-R**. These are working names, not verified unused names or claims of novelty. Input/output rank allocation is an independent axis, not an inseparable part of SWT.

### Starting implementation

Use bias-free linear input/output adapters and ordinary pre-normalized Transformer blocks. First compare embeddings with a uniform body; then hold the embedding design fixed and compare body-width schedules. Nonlinear input mappings and alternative transitions come later, unless a transition fails basic stability checks. The selected pilot uses **CulturaX English**, the **Hugging Face Llama implementation trained from scratch**, and **B0 versus A128 first**; see [the concrete setup](#setup).

**Reading guide:** [starting model/data/run setup](#setup), [embedding architectures](#embeddings), [body architectures](#bodies), [transitions](#transitions), [parameter accounting](#budgets), [experiment sequence](#experiments), and [publication paths](#publication).

### What this project should deliver

The goal is a controlled account of where capacity is useful. A successful architecture could support an efficiency or modeling paper. A failed architecture could still support a substantive boundary or mechanism study, but a single losing configuration is not a publication contribution.

### Basis and limits of this draft

The supplied working document introduces embedding-rank allocation in Part III, §5d [1](#ref-1). The width schedules and architectural choices below consolidate the subsequent discussion. External evidence is cited separately. Equations and numerical budgets labeled as calculations are derived for the stated assumptions. They are not training results or evidence that one design is better.

This draft narrows the earlier memory-versus-depth agenda to embeddings and body width. Memory modules, partial tying, teacher models, and test-time adaptation are out of scope. It does not carry forward the earlier claim that equal table budgets imply equal complete-model parameters or FLOPs.


---

<a id="prior-work"></a>

## 2. Research questions and prior-work boundary

| Question | Hypothesis to test | What would not establish it |
| --- | --- | --- |
| Q1: Shared or separate? | At a fixed **vocabulary-table** budget, separate, narrower input/output spaces can outperform one shared full-width space; complete-model budget differences must then be controlled explicitly. | A win caused by undisclosed adapter overhead, a changed body, or an unmatched total budget. |
| Q2: Where should width live? | Moving capacity from early layers to later layers can improve quality per resource. | A smaller model losing less accuracy than expected, without a cost comparison. |
| Q3: Does the interface matter? | Stage transitions and embedding rank can change which width schedule works. | Changing several components at once and assigning the effect to one of them. |

### What the referenced coupling paper already does

Chung et al. [2](#ref-2) use “coupled” to mean shared input/output weights. Their tied 768-width baseline has 177M pretraining parameters; the untied 128/768 and 768/128 configurations each have 192M. They also reinvest input-embedding savings into a uniformly wider body (768→1024) or more layers (12→23), emphasizing fine-tuning budgets after discarding the output table.

Their reported experiments do not implement our 128/896-versus-tied-1024 table-budget comparison or the proposed narrow-early/wider-late schedule. Their wider-body model has 260M pretraining and 168M fine-tuning parameters, versus 177M/177M for the tied baseline. Our causal language model retains its output head. This is a distinction from that paper, **not evidence that the exact experiment is absent from all literature** [2](#ref-2).

### Close architectural precedents

**Variable-Width Transformers** compares several width profiles, favors wide early and late layers with a narrower middle in its studied setting, and preserves inactive coordinates in a global residual stream. It reports instability for a tested full-representation resizing projection [3](#ref-3). These results motivate both a strong comparator and a transition-control experiment.

**StepsNet** already uses increasing-width stages, introducing additional slices of an initial representation at later stages [4](#ref-4). **DeLighT is an especially important conceptual precedent:** its block-wise scaling explicitly places shallower, narrower computation near the input and deeper, wider computation near the output [12](#ref-12). Its custom DeLighT blocks and scaling mechanism are not the same as changing the full residual width of otherwise standard decoder-only Transformer blocks, but the broad “move capacity from early to late layers” principle is therefore established. OpenELM likewise uses layer-wise nonuniform allocation, though it scales attention/FFN capacity rather than implementing the same residual-width schedule [13](#ref-13). ALBERT and DeFINE provide factorized-input and learned embedding-mapping precedents [5](#ref-5), [6](#ref-6). Neither “small input, large output” nor “progressively wider layers” is a sufficient standalone novelty claim.

> **Contribution to aim for:** a new, reproducible result about capacity allocation, its interaction with the embedding interface, or its resource-dependent limits. Validate the novelty boundary again before positioning a submission.


---

<a id="embeddings"></a>

## 3. Architecture A: asymmetric embeddings

### Terminology and notation

| Term | Meaning in this document |
| --- | --- |
| `V` | Vocabulary size; unchanged across a controlled comparison |
| `L` | Number of Transformer blocks |
| `d_l` | Hidden/residual width processed by block `l` in the T1 reference |
| `m_l` | FFN intermediate width; distinct from the block’s hidden width |
| `r_in`, `r_out` | Input and output embedding dimensions |
| Tied / coupled | The input lookup and vocabulary head use the same learned table |
| Untied / separate | Input and output tables are independently learned |
| Linear adapter | One matrix multiplication that changes dimensions |
| FFN | A token-wise nonlinear network; SwiGLU is the chosen form here |

`128 + 896` describes a table-parameter budget, **not concatenation of the input and output vectors**. All diagrams retain sequence length and use causal attention. `B0` names the baseline. The separate narrow-first body candidate is called `N128` below to avoid confusing it with `B0`.

### B0 — tied uniform reference

```text
Token IDs -> shared table E: V x 1024
          -> L standard blocks, width 1024
          -> RMSNorm(1024) -> logits using E transpose
```

### A — separate compact input and larger output

```text
Token IDs -> E_in: V x 128
          -> Linear 128 -> 1024
          -> L standard blocks, width 1024
          -> RMSNorm(1024)
          -> Linear 1024 -> 896
          -> logits using E_out: V x 896
```

The initial mapping and output adapter are bias-free linear projections. `E_in` and `E_out` are independently learned. `E_out` is the vocabulary-head weight, not an additional table beside that head. Keep the body and final-normalization placement unchanged in the first embedding comparison.

```text
h₀ = E_in[x] P_in;     z = RMSNorm(h_L) P_out;     logits = z E_outᵀ
```

| Arm | Input / output | Role |
| --- | --- | --- |
| B0 | 1024 / 1024, shared | Original tied reference |
| A128 | 128 / 896, separate | Primary input-light candidate |
| A256 | 256 / 768, separate | Less aggressive input compression |
| A512 | 512 / 512, separate | Symmetric allocation control |

For all three untied arms, `r_in + r_out = 1024`. With the same uniform 1024-wide body, their table and linear-adapter parameter counts are equal. Compute differs because the vocabulary projection depends on `r_out`.

**Interpretation limit:** B0 versus A changes both weight sharing and embedding rank/parameterization. It tests the combined design choice, not the isolated effect of untying. The A128/A256/A512 comparison studies allocation within the untied family. A full-width untied control could help diagnosis later, but would exceed this table budget.

### Nonlinear input variation — after the linear reference

Replace only the input mapping with a bias-free SwiGLU network: `128 → intermediate 512 → 1024`. This is a proposed use of the SwiGLU construction [7](#ref-7), not a published result for this architecture. It adds token-local nonlinear computation, not context. Its 655,360 weights replace a 131,072-weight linear input adapter; count the 524,288-weight increase. A linear shortcut is a later variation, not a default.

**Rank interpretation, derived from the stated matrices.** A linear input mapping gives an effective vocabulary matrix of rank at most 128. The specified nonlinear network can exceed that bound, but its final `512 → 1024` linear map still limits the effective matrix rank to at most 512. Higher attainable rank does not guarantee a better model.

A bias-free 896-dimensional vocabulary head bounds the rank of the **raw logit matrix across contexts** by 896. This is not the same statement about the probability matrix after softmax. Extra body width can still improve computation, but does not remove the final linear-head bound [9](#ref-9).


---

<a id="bodies"></a>

## 4. Body architectures: where to widen

### N128 — contextualize at 128, then expand

```text
Embedding(V, 128)
 -> 1 block at width 128 -> Linear(128, 1024)
 -> L-1 blocks at width 1024
 -> RMSNorm(1024) -> Linear(1024, 896)
 -> vocabulary logits using E_out: V x 896
```

This is a valid, aggressive control. Its first attention operation is 128-wide even though later states are 1024-wide. Specify its FFN intermediate width separately. Defer it until Architecture A trains correctly; it combines severe early compression with an abrupt expansion.

### C — SWT with savings retained

```text
Embedding(V, 128) -> Linear(128, 256)
 -> 2 blocks at 256 -> Linear(256, 512)
 -> 2 blocks at 512 -> Linear(512, 1024)
 -> 2 blocks at 1024
 -> RMSNorm(1024) -> Linear(1024, 896)
 -> vocabulary logits using E_out: V x 896
```

This six-block example uses less body capacity than six 1024-wide blocks. It asks whether savings are worthwhile at an acceptable quality loss. It is not an equal-capacity comparison by default.

### D — SWT-R with savings reinvested

```text
Embedding(V, 128) -> Linear(128, 512)
 -> 2 blocks at 512 -> Linear(512, 1024)
 -> 2 blocks at 1024 -> Linear(1024, 1280)
 -> 2 blocks at 1280
 -> RMSNorm(1280) -> Linear(1280, 896)
 -> vocabulary logits using E_out: V x 896
```

Relative to the uniform 1024-wide body, early compression pays for final layers wider than 1024. This example reinvests some savings, not necessarily all of them. Final hidden width 1280 and output embedding width 896 are independent choices.

| Body profile (illustrative) | Main question |
| --- | --- |
| 1024 × 6 | Uniform body reference |
| 256, 256, 512, 512, 1024, 1024 | Can we retain quality while keeping the savings? |
| 512, 512, 1024, 1024, 1280, 1280 | Is late capacity a better use of the saved budget? |

> **These schedules are prototypes, not chosen optima or exactly matched models.** Cap the maximum width, retain the same block count for the primary width-allocation study, and solve the complete parameter budget before confirmation runs. Do not repeatedly double widths without accounting for quadratic cost.


---

<a id="transitions"></a>

## 5. Stage transitions and final-layer output

Use normal blocks within a stage. At a boundary, distinguish resizing from adding nonlinear computation. In the equations below, states are row vectors; `d` is the attention/input width and `D` is the next width. The attention operation includes its output projection back to `d`.

A bias-free SwiGLU FFN with intermediate width `m` is [7](#ref-7):

```text
FFN_d→D(v) = (SiLU(v W_g) ⊙ (v W_u)) W_down
W_g, W_u: d x m;    W_down: m x D
```

`⊙` denotes elementwise multiplication. Both RMSNorms in a block have independent learned gains. Begin with the attention residual:

```text
u = x + Attention_d(RMSNorm_d(x))
```

### T1 — ordinary block, then linear resizing

```text
h = u + FFN_d→d(RMSNorm_d(u));     y = h P_d→D
```

This is the transparent reference for C and D. Moving the projection inside the block class does not change the architecture. A full-row-rank widening projection is injective, so widening is not inherently information-destroying. Optimization and conditioning are separate issues.

### T2 — the existing FFN returns the next width

```text
y = u P_d→D + FFN_d→D(RMSNorm_d(u))
```

Both the shortcut and the FFN must output `D`; attention still runs at `d`. With the same intermediate width `m`, T2 minus T1 has `m(D-d)` additional parameters: widening costs more; shrinking costs less. This calculation assumes the same learned `d × D` shortcut/adapter matrix in both designs. It does not compare either design with an extra full FFN after a normal block.

### T3 — padded shortcut for widening only

```text
y = [u; 0] + FFN_d→D(RMSNorm_d(u)),     D > d
```

Old coordinates retain an identity **shortcut**, not guaranteed preservation in the complete block. For the same `d`, `D`, and `m`, its parameter difference from T1 is `m(D-d)-dD`; it is not necessarily cheaper. It cannot handle shrinking by the same rule.

Padding can also affect the next normalization. For the isolated vector `[u; 0]`, unit gains and negligible epsilon imply that RMSNorm rescales the old coordinates by `sqrt(D/d)` relative to normalization at width `d`. This is a calculation from RMSNorm’s definition [8](#ref-8), **not the exact scale of T3 after its nonzero FFN contribution**. Monitor old/new coordinate scales rather than assuming this is an upgrade.

### Output decision

Initially use a final norm at the body width, then a **linear** adapter to 896. Do not add a full output FFN merely to resize. A later alternative is T2 in the final block, producing 896 directly, followed by RMSNorm(896). This changes the normalization interface and must be evaluated as such.

```text
(u + f(u)W)P = uP + f(u)(WP)
```

The folding identity above holds without an intervening normalization or activation. It does not make the initial output design and the direct-output design equivalent after moving the final norm.

> **Decision:** T1 is the reference to validate, not a proven stability winner. T2 and T3 are alternatives, not assumed upgrades. If T1 fails, test the interface before rejecting the width schedule.

A global residual with inactive coordinates carried forward is a separate comparator [3](#ref-3). Distinguish its **stored residual width** from each block’s active computational width. In a purely growing schedule, newly added coordinates may have no earlier wide state to recover; an initialization rule is still necessary. Do not describe it as restoring information that never existed.


---

<a id="budgets"></a>

## 6. Complete parameter and resource accounting

Count each shared tensor once. Include vocabulary tables, adapters, attention, FFNs and normalization gains. The untied output table **is** the vocabulary-head weight; do not count it twice. Define “body parameters” and “total parameters” explicitly.

### Embedding-only comparison

```text
P_tied_table = V × 1024
P_untied_tables = V(r_in + r_out) = V × 1024
```

```text
P_linear_adapters = r_in d₁ + d_L r_out
```

For a uniform 1024-wide body and `r_in+r_out=1024`, adapters add 1,048,576 weights. At `V=156,000`, the tables alone use 159.744M parameters, so this configuration cannot be a 125M **total** model. The unchanged-body comparison is table-matched, not total-parameter-matched. Report it first, then add a total-budget control.

The selected English pilot instead uses `V=50,257`; its smaller totals are provided in [§12.5](#setup). The large-vocabulary calculation here is retained to show how budgets change with `V`.

### A calculable block specification

For standard multi-head attention with all Q/K/V/output widths equal to `d`, bias-free SwiGLU and two gain-only RMSNorms, the exact count under this specification is:

```text
P_block(d, m) = 4d² + 3dm + 2d
m(d) = 128 × ceil((8d/3)/128)
```

The rounding rule is a proposed convention, not a universal optimum. SwiGLU uses three matrices [7](#ref-7); each chosen RMSNorm contributes one length-`d` gain vector [8](#ref-8), [14](#ref-14). There are no biases, learned positional tables, per-head norms or extra gates in this example. Grouped-query attention requires a different count.

| Block width d | 128 | 256 | 512 | 1024 | 1280 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Intermediate width m(d) | 384 | 768 | 1408 | 2816 | 3456 |

```text
P_total = P_tables + r_in d₁ + d_L r_out
          + Σₗ(4dₗ² + 3dₗmₗ + 2dₗ)
          + Σ_boundaries(d_before d_after) + d_L
```

This formula is for T1 with untied tables and learned linear input/output adapters. Sum boundary matrices only where a transition is actually present. Omit an adapter only if it is absent or a parameter-free identity—not merely because its input/output dimensions match. For B0, use one shared table and no input/output adapters. Recalculate for T2/T3.

| Six-block example | Tables | Other parameters | Total |
| --- | --- | --- | --- |
| B0: tied, uniform | 159.744M | 77.084M | 236.828M |
| A128: split, uniform | 159.744M | 78.132M | 237.876M |
| C: split, savings retained | 159.744M | 35.430M | 195.174M |
| D: split, reinvested | 159.744M | 74.822M | 234.566M |

Calculated examples, rounded to three decimals; `M = 1,000,000`. The exact totals are B0 **236,827,648**, A128 **237,876,224**, C **195,174,400**, and D **234,565,888**. “Other” includes blocks, transitions, adapters and all norms.

**A128, not B0, is the fixed-embedding reference for C and D.** D is 3,310,336 parameters below A128. Adjust a declared architectural dimension or compare nearby budget curves before claiming equality. Adding unused parameters would not create a meaningful control.

For intuition, proportional FFNs give main block-parameter cost approximately proportional to `Σ d_l²`. D’s squared-width sum is 93.75% of six 1024-wide layers, before rounding and boundary effects. This is not a total-model or runtime ratio. Vocabulary-head compute, attention over positions, transitions and hardware utilization also matter; measure them separately.


---

<a id="experiments"></a>

## 7. Controlled experiment sequence

The staged plan below is a proposal, not a fixed compute commitment. [§12](#setup) selects the initial data, tokenizer, model configuration and proposed token horizons; hardware and final optimization budgets still need to be registered. Change one architectural axis at a time.

| Stage | Runs and controls | Decision enabled |
| --- | --- | --- |
| 0. Implementation | Check B0 and A128 first. Then validate T1 in one small widening model. | Are shapes, causal masking, parameter counts and optimization correct? |
| 1. Embedding allocation | B0, A128, A256 and A512 with one uniform body. Add the total-budget control; test nonlinear input afterward. | Does the combined separation/rank-allocation design help? |
| 2. Body allocation | Fix one embedding design. Start with uniform, C and D; add final-stage-only and wide–narrow–wide controls for promising or informative results. | Are the savings useful? Does reinvestment improve the trade-off? |
| 3. Interface diagnosis | Hold widths and embeddings fixed; compare selected transitions, accounting for their capacity and normalization differences. Run this earlier if instability appears. | Is the interface a plausible cause of failure? |
| 4. Interaction and confirmation | Cross embedding design with uniform/nonuniform body; confirm selected comparisons at another scale or horizon. | Does the effect of a body profile depend on embedding design? |

### Budget matching is not one experiment

**Equal training tokens** tests learning on the same amount of data, while recording differences in compute. **Equal total training compute** tests resource efficiency and may entail different token counts. **Equal quality** compares the measured resources needed to reach a target. Keep these analyses separate.

For the tied reference, hold the body fixed first and disclose the adapter overhead. A secondary total-budget control can adjust a declared FFN dimension or use nearby-budget curves. That secondary control no longer isolates embeddings alone, so report both rather than hide the change.

### A meaningful body comparison

Use a tuned uniform baseline at each relevant target budget. Comparing C only with a larger uniform model does not establish efficiency: add a uniform model near C’s budget. Compare D with a matched-budget uniform reference.

The final-stage-only control retains a narrow body until a wide final stage. Keep its block count, final hidden width and output head aligned with D where feasible, then solve the remaining budget. This tests gradual expansion beyond final-stage capacity. The wide–narrow–wide profile is a relevant alternative [3](#ref-3).

**Separate shape from interface.** A width-profile comparison should use the same transition rule. Reimplementing the published wide–narrow–wide model with its own carry-forward residual is additionally useful, but changes both shape and interface; label that comparison accordingly.

In the interaction grid, a tied 1024-dimensional table can connect to a varying-width body using adapters; count them too. Different body widths do not force untying. Treat this as an **embedding-design × body-profile** interaction, not pure tying, because B0 and A also differ in rank. If endpoint widths differ, the adapter cost `r_in*d_1 + d_L*r_out` changes with the allocation, even when `r_in+r_out` is fixed.

> **Stop expanding the grid when a comparison is not interpretable.** First fix instability or a budget mismatch. Additional architectures do not compensate for a weak control.


---

<a id="measurement"></a>

## 8. Measurement, uncertainty and reproducibility

### Primary evidence

Measure validation next-token negative log-likelihood in nats per scored token throughout training; perplexity is `exp(loss)`. Keep tokenizer, evaluation text and scored-token masking fixed. Plot loss against tokens and cumulative compute. Reserve a separate test set for confirmation, and define which components are included in reported body/non-table counts.

Define rare-target-token bins using training-corpus frequency counts, then keep those bins fixed for evaluation. Evaluate a second relevant domain only if the training/evaluation design supports it. Do not quietly add multilingual claims to a monolingual study.

### Systems evidence

Measure training throughput, peak allocated memory, prompt-processing (prefill) latency, per-token decoding latency and key/value-cache use. Record hardware, precision, kernels, batch size, context/output lengths, warm-up and timing procedures. Smaller matrix counts alone do not establish a faster implementation.

### Optimization controls

Use comparable tuning opportunities, not necessarily identical best hyperparameters. Record the initialization, residual scaling, learning-rate search, optimizer groups, clipping, normalization placement and all diverged runs. A width-changing model may need different scaling; an unfairly untuned model is not evidence of an architectural limit.

### Practical significance and uncertainty

Before confirmation, define a useful loss improvement `δ_gain > 0` and, for efficiency claims, an acceptable loss increase `δ_loss > 0`. Let `Δ = candidate loss - baseline loss`; negative is better. With a prespecified uncertainty interval `[lo, hi]`:

| Interval condition | Supported interpretation within the tested regime |
| --- | --- |
| `hi < 0` | Evidence of lower loss |
| `hi < -δ_gain` | Evidence of at least the specified useful improvement |
| `lo > -δ_gain` | The interval excludes an improvement of that size |
| `hi < δ_loss` | Evidence that the loss increase is below the allowed margin; resource savings must be established separately |

A point estimate near zero or a wide interval is not proof of equivalence. Choose the interval method and confidence level before examining confirmation results.

Use independent training seeds for the central claims; three is a proposed starting floor, not a universal power guarantee. Quantify seed variation separately from evaluation-sample uncertainty. Resampling held-out documents cannot replace independent training runs. Confirm architecture selection on held-out runs rather than report the best pilot as an unbiased estimate.

### Diagnostics support interventions; they do not replace them

Track activation/gradient scales, transition-matrix conditioning and representation spectra on the same sampled contexts. Compare spectra in a way that accounts for differing widths and sample sizes. Plots alone do not establish information loss; intervene on the suspected component and check a predicted behavioral change.

> **Reproducibility package:** configuration manifests, exact trainable-parameter counts, tokenizer/data versions, seeds, selection rules, training curves, timing scripts and failed-run logs. Keep the final test set outside architecture selection.


---

<a id="failure-analysis"></a>

## 9. If an architecture fails: diagnose, do not relabel

“Not working” can mean worse quality, unstable training, slow convergence, or a poor systems trade-off. These outcomes require different follow-up tests. The experiments below are hypotheses for investigation, not promised explanations.

| Observed outcome | Discriminating follow-up | Potential conclusion if supported |
| --- | --- | --- |
| A128 loses to tied B0 | Compare A256/A512; vary only input mapping; examine output-rank controls with honest budgets. | A boundary on input compression, output capacity, or the value of sharing—not a blanket failure of untying. |
| C saves resources but loses quality | Compare tuned models across the same quality–cost curve. | A useful trade-off, or evidence that another model is at least as accurate and no more costly. |
| D loses despite reinvestment | Reallocate capacity toward early layers; include final-stage-only and alternate profiles with explicit budgets. | A boundary on late reallocation or on the value of gradual widening—not a universal statement about depth. |
| Only full-projection transitions fail | Keep widths fixed; compare interfaces, normalize budget differences and provide comparable tuning. | Evidence implicating the interface, not proof that all nonuniform widths are ineffective. |
| Narrow input + narrow body fails jointly | Run the small embedding × body factorial with complete accounting. | An interaction between two compression choices, not simply either main effect. |
| Results reverse with training duration | Run prespecified longer-horizon confirmations and compare cost. | A convergence or data-budget boundary rather than an unconditional ranking. |

### Three worthwhile negative-result directions

**Boundary study.** Identify how much early width can be removed, at which training budgets, without losing a practically important amount of quality. Test a prediction at a held-out scale or duration.

**Mechanism study.** Distinguish limited early contextual computation from residual-transition problems and input compression. Require interventions, not only attractive representation plots.

**Interaction study.** Determine whether the preferred body profile changes with embedding allocation. A precise result consistent with no practically important interaction can be informative; a nonsignificant test alone does not establish absence.

> A full-rank widening map need not lose information; a nonlinear block can still alter an identity shortcut; and a small dimension alone does not prove that token identity is destroyed. State only the limitation that the evidence actually identifies.


---

<a id="publication"></a>

## 10. Main-conference and Findings publication paths

The target is a top-tier main-conference paper or, where applicable, **Findings of the ACL family**. Findings is a specific publication outlet, not a generic fallback track at every conference. “A*” expresses the project’s ambition, not a verified ranking or an acceptance prediction.

### What the policies support

ARR’s review form emphasizes soundness and reproducibility for Findings and additionally considers novelty and impact for main-conference recommendations [10](#ref-10). NeurIPS 2026 recognizes negative results, but expects deeper analysis and substantial significance/originality; a successful fix is not required [11](#ref-11). These examples do not guarantee acceptance, and policies must be rechecked for the chosen submission cycle.

| Result pattern | Potential paper contribution | Evidence needed |
| --- | --- | --- |
| A or D wins | Modeling or efficiency method | A useful gain against strong budget-matched controls, reproduction and an explanation of scope. |
| Architecture loses, boundary is new | Controlled empirical / Findings-oriented study | Precise uncertainty, fair tuning, meaningful coverage and guidance beyond one configuration. |
| Architecture loses, explanation changes a design decision | Main-conference-oriented negative or mechanism study | A consequential, previously unresolved explanation, causal interventions and confirmation. |
| Quality loss buys real deployment savings | Efficiency frontier study | Measured trade-offs at relevant operating points; no claim that lower FLOPs automatically mean lower latency. |
| One configuration loses; no new insight | Pilot or technical report | Insufficient by itself for the main/Findings goal. Do not manufacture a general claim. |

### A failed preferred method can still answer a good question

The paper should be organized around capacity allocation, not around defending SWT. A strong hypothetical conclusion might identify when later width cannot replace earlier computation, or show that a residual interface—not the width profile—causes a loss. Such conclusions require new evidence beyond existing variable-width work.

A weaker report says only that 256→512→1024 performed worse and speculates about a bottleneck. A stronger report defines the tested regime, excludes plausible alternatives, measures uncertainty, and provides a result that changes a useful design choice.

### Proposed paper structure

Question and prior-work boundary; architecture family and budgets; controlled results; targeted mechanism/boundary experiments; systems trade-offs; limitations and reproducibility. Use a result-neutral working title until the evidence determines the claim.


---

<a id="implementation"></a>

## 11. Implementation decisions and first milestone

| Component | Initial decision | Later variation |
| --- | --- | --- |
| Body | Dense causal pre-RMSNorm blocks; fixed sequence length | Nonuniform widths, not sequence downsampling |
| Attention | Standard causal multi-head attention; head dimension 64 | Grouped-query attention only as a separate design |
| FFN | SwiGLU; record each intermediate width explicitly | Small nonlinear input network after the linear reference |
| Transitions | T1: bias-free linear maps at stage boundaries | T2 direct FFN output; T3 padding; residual-preserving comparator |
| Output | Final gain-only RMSNorm at body width; linear adapter to r_out; vocabulary head | Direct final-block output, recording the changed normalization interface |
| Bias | False in new linear/FFN/head matrices | Preserve a different reference convention if one is selected |
| Budget matching | Count the instantiated complete model | Declared tolerance plus nearby-budget sensitivity; no dummy parameters |

RMSNorm [8](#ref-8) and SwiGLU [7](#ref-7) are choices, not proven optima. Gain-only RMSNorm and bias-free projections follow an established implementation convention [14](#ref-14); RMSNorm does not make an arbitrary bias redundant. At head dimension 64, widths 128/256/512/1024/1280 use 2/4/8/16/20 heads. Thus changing width also changes head count under this convention. Output dimension 896 does not require 896-wide attention.

### Still to choose before training

The initial dataset, tokenizer, six-block baseline, 2048-token sequence length, RoPE, normalization and proposed token horizons are now selected in [§12](#setup). Before training, freeze the dataset/tokenizer revisions and split manifest, software versions, hardware/precision, optimizer and tuning budget, effective batch size, seeds, effect margins and parameter-matching tolerance. Validate initialization and residual scales for the implemented variants. The `V=156,000` counts in §6 remain illustrative; the English pilot uses `V=50,257`.

### Unit tests and smoke-run gate

Verify every residual shape, true tensor sharing in B0, separate storage in A, causal masking, cache shapes and exact unique-parameter counts. In evaluation mode with matching masks/positions, check full-sequence versus incremental logits within a declared tolerance. Check finite forward/backward values, intended gradient flow and stable boundary scales. Training correctness is the first gate; beating the baseline is not required to proceed.

### First research milestone

Establish B0 versus the three embedding allocations with one body, then compare uniform versus savings-retained C and reinvested D under fixed embeddings. Keep a nonlinear input network and direct 896-output block out of the first body comparison. Advance only when a result is stable enough to motivate either a stronger architecture or a discriminating explanation.

> **Success is not restricted to SWT winning.** Continue for a useful quality–cost improvement or a new, well-supported finding. Stop expanding an uninformative grid that merely repeats known behavior or remains dominated by unresolved implementation issues.


---

<a id="setup"></a>

## 12. Starting setup: Hugging Face model, CulturaX, and first runs

**Decision:** start with **CulturaX English** and a **randomly initialized, six-layer Llama-family model implemented in Hugging Face Transformers**. Test **B0 versus A128 first**, then A256/A512, and only then the width schedules. This section selects an initial experimental setup; it does not change the hypotheses or turn the earlier illustrative schedules into established optima.

### 12.1 Setup at a glance

| Item | Selected starting point |
| --- | --- |
| Primary dataset | `uonlp/CulturaX`, language configuration `en`, text field `text` |
| Dataset policy | One pinned public release; one frozen document-selection/split manifest with stable document IDs, shared by all arms |
| HF implementation | `transformers.LlamaForCausalLM` with a custom `LlamaConfig` |
| Existing HF family reference | `HuggingFaceTB/SmolLM2-135M` uses `LlamaForCausalLM`; it is a reference, not the exact experimental configuration |
| Weight initialization | Random initialization for every arm; no pretrained Transformer or embedding weights |
| Tokenizer | `openai-community/gpt2`, tokenizer only; vocabulary size `V = 50,257` |
| Uniform body | 6 blocks; hidden width 1024; 16 query heads and 16 key/value heads; head dimension 64 |
| FFN / normalization | SwiGLU with intermediate width 2816; pre-RMSNorm; final RMSNorm at body width |
| Position / sequence | Standard RoPE, base 10,000; sequences of 2048 tokens |
| Bias / dropout | Bias-free matrices; zero dropout for the initial reference |
| First comparison | B0: tied 1024 versus A128: separate 128/896, both with the same uniform body |
| Proposed horizons | 1B-token screening; 5B-token confirmation for selected comparisons, subject to hardware and precision requirements |

The repository, language configurations and text field are documented by CulturaX [15](#ref-15), [16](#ref-16). HF supplies the Llama model/configuration classes [17](#ref-17); the SmolLM2 reference uses that class but has different dimensions [18](#ref-18). The GPT-2 vocabulary size is documented in its released configuration [19](#ref-19). The remaining settings are **project choices**, not claims that these sources recommend this exact combination.

**Pilot scope.** With `V=50,257`, the tied B0 vocabulary table alone has **51,463,168 parameters**, about **40.0%** of the 128.55M-parameter six-layer model. That makes this pilot deliberately sensitive to embedding-allocation choices, which is useful for Q1 but may exaggerate effects relative to deeper models where the table is a smaller fraction of the total. In the proposed 12-layer confirmation, the same table is about **25.0%** of total B0 parameters. Therefore a six-layer result is a screening result for the allocation hypothesis; broad architectural claims require the deeper confirmation.

### 12.2 Why CulturaX is an appropriate choice

CulturaX is the project author's chosen primary corpus. Its language-specific subsets let this study begin with English and later use the same dataset family for multilingual confirmation. The dataset paper describes cleaning and deduplication for multilingual language-model training [15](#ref-15). This is a practical reason to use it, not evidence that the architecture will work better on it than on another corpus.

**Use English only for the first experiment.** Do not simultaneously change architecture, language sampling and tokenizer. Initial conclusions should be scoped to the selected CulturaX English distribution. A later independent English domain or corpus can test generality; it is not required for the first pilot. Using the author's dataset is compatible with a controlled architecture study: apply the same selection and preprocessing to every arm and do not treat the study as an unbiased comparison of CulturaX against other datasets.

**Access and provenance.** The HF dataset currently requires accepting its access conditions and authentication. Its card points to the underlying mC4/OSCAR terms and warns about residual sensitive content [16](#ref-16). Record the applicable release and conditions; do not assume authorship removes third-party restrictions. An existing local author copy is usable if its version, selected files and preprocessing can be mapped to a reproducible manifest. Do not silently mix private or revised data with the public release.

The repository metadata maps configuration `en` to English Parquet files [16](#ref-16). With access granted, the documented configuration can be opened using the current Datasets API [20](#ref-20):

```python
import os
from datasets import load_dataset

# Set CULTURAX_REVISION to the selected repository commit in the run manifest.
# Authenticate with an authorized HF account; do not put access tokens in code.
english = load_dataset(
    "uonlp/CulturaX",
    "en",
    split="train",
    streaming=True,
    token=True,
    revision=os.environ["CULTURAX_REVISION"],
)
```

This opens the source stream, **not a ready-made evaluation split or a randomized training sample**. Validate access in the training environment, then freeze the preprocessing below. The API example is not a tested end-to-end training pipeline.

**Sampling and splits.** Select a deterministic sample across the English release rather than simply taking the first file or stream prefix. Build and freeze a manifest before architecture training. A practical stable document key is a hash of the available provenance plus the content digest, for example `SHA256(source || url || timestamp || SHA256(text))`; store the source shard/row locator as well. If the original CulturaX deduplication pipeline exposes duplicate-cluster identifiers, prefer assigning whole clusters to a split. Otherwise assign by the stable document key and then audit cross-split exact and near duplicates. Preserve source provenance in the manifest; if only a subset of shards is used, document the selection rule and inspect its source/domain composition.

Assign documents or duplicate clusters to disjoint train/validation/test partitions **before** token packing. Freeze the resulting IDs and reuse exactly the same documents, order policy and packed-token files for every architecture arm. A proposed starting reservation is approximately 20M tokenizer-measured tokens each for validation and final test, with at least 5B training tokens available. These are experiment targets, not existing CulturaX split sizes. The precise hash recipe, split thresholds and any dedup-cluster metadata used must be part of the released manifest.

**Tokenization and packing.** Reuse the pinned GPT-2 tokenizer without adding vocabulary entries. Tokenize document text without automatic special tokens, append one existing EOS token per document, and pack within each split into 2048-token sequences. For the simple pilot, use ordinary causal masking across each packed sequence: EOS marks a boundary but does not prevent attention across it. Preserve the same packing and scoring convention across arms. Do not reset positions or add document-isolated masks for only one architecture.

Mask padded positions by position/attention mask, not by removing every occurrence of an EOS token that also serves as a padding ID. Prefer no padding in the full training packs and `pad_token_id=None` in the initial model config. Record tokens processed and targets actually scored separately. Corpus headline token counts are not a substitute for counts under the experiment's tokenizer. Materialize and reuse the tokenized sample for comparable training and throughput measurements; do not let live data-download latency become an architecture result.

### 12.3 Which Hugging Face model to use

**Use HF's built-in `LlamaForCausalLM` implementation for B0, with the explicit research configuration above—not the pretrained weights of an existing checkpoint.** HF supports constructing this family directly from a configuration [17](#ref-17). This preserves the pre-RMSNorm/SwiGLU/RoPE design already specified in the plan.

For a concrete Hub reference, `HuggingFaceTB/SmolLM2-135M` is an available Llama-family model. Its published configuration uses hidden width **576**, **30** layers and grouped-query attention [18](#ref-18). Therefore, it is **not** our six-layer, 1024-wide model. Do not label the custom B0 as a reproduction of SmolLM2, and do not claim that the modified A/SWT models already exist on the Hub. An unmodified SmolLM2 configuration initialized from scratch can be a later standard-shape comparator; it is not the primary test of `128 + 896 = 1024`.

The GPT-2 repository supplies **only the tokenizer** here [19](#ref-19). We are not using GPT-2's learned positional embeddings, LayerNorm/GELU blocks, or pretrained weights. The tokenizer's original model-context setting is not the new model's context limit: tokenize documents without truncation, then let the common packer form 2048-token inputs.

A minimal B0 constructor, using the documented HF configuration interface [17](#ref-17), is:

```python
from transformers import LlamaConfig, LlamaForCausalLM, set_seed

set_seed(0)  # Example seed; use the registered seed list for actual comparisons.
config = LlamaConfig(
    vocab_size=50257,
    hidden_size=1024,
    intermediate_size=2816,
    num_hidden_layers=6,
    num_attention_heads=16,
    num_key_value_heads=16,  # MHA, not grouped-query attention.
    head_dim=64,
    hidden_act="silu",
    max_position_embeddings=2048,
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    rms_norm_eps=1e-6,
    initializer_range=0.02,
    attention_bias=False,
    mlp_bias=False,
    attention_dropout=0.0,
    tie_word_embeddings=True,
    bos_token_id=50256,
    eos_token_id=50256,
    pad_token_id=None,
    use_cache=False,  # Enable caching separately for decode correctness/timing.
)
model = LlamaForCausalLM(config)  # Random initialization; no checkpoint loaded.
assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
assert sum(p.numel() for p in model.parameters()) == 128_546_816
```

**Implementation boundary.** The constructor covers only B0. A128/A256/A512 require replacing the input lookup and vocabulary head with the adapters/tables in §3. Remove unused full-width tables; merely supplying `inputs_embeds` while retaining an unused original embedding would distort the count. Disable automatic weight tying in the separate-table arms and verify unique storage after loading/saving. SWT requires custom per-stage blocks, transitions and cache handling; a single scalar `hidden_size` in stock `LlamaConfig` does not implement the width schedule. Save the custom configuration and code with the experiments.

The code is a setup sketch, not a complete trainer. Pin Transformers, PyTorch, Datasets and tokenizer revisions; check the exact constructor against the installed version. `rope_parameters` follows the current HF interface [17](#ref-17); older pinned versions may use `rope_theta`/`rope_scaling` instead. The stock Llama matrices use the selected reference initialization, but the new dimension-changing adapters require an explicit scale policy; see the next subsection.

### 12.4 Initialization and scale control

**Do not initialize every custom adapter with the same `0.02` standard deviation.** That would create a dimension-dependent scale confound. For example, with independent `0.02`-std input embeddings and a `128 → 1024` projection also initialized at `0.02`, the projected per-coordinate standard deviation is only about `0.02 × sqrt(128) × 0.02 ≈ 0.0045`, much smaller than the B0 embedding scale of about `0.02`. A256 and A512 would start at different scales again. The same issue occurs at T1 stage transitions.

For the initial comparison, use a **fan-in/per-coordinate variance-preserving initialization for every dimension-changing projection**:

```text
P_in  : r_in -> d_1      initialize Var(P_in)  ≈ 1 / r_in
P_T1  : d -> D           initialize Var(P_T1)  ≈ 1 / d
P_out : d_L -> r_out     initialize Var(P_out) ≈ 1 / d_L
```

This keeps the RMS of the projected representation approximately unchanged when the input coordinates have comparable variance. A semi-orthogonal initialization with the same RMS target is an acceptable alternative; pick one rule and use it for all arms.

The untied output head needs a separate scale check because its dot-product dimension changes with `r_out`. With `P_out` preserving per-coordinate variance and `E_out` initialized with the same `0.02` std for every arm, initial logit variance decreases with `r_out`. To avoid making the allocation sweep partly a softmax-temperature experiment, either:

1. initialize `E_out` with `std = 0.02 × sqrt(1024 / r_out)`, while keeping B0's tied table at `0.02`; or
2. keep `E_out` at `0.02` and apply a fixed head scale `sqrt(1024 / r_out)` to match the same **initial** logit variance. This is not optimization-equivalent to rescaling the initialization, so record which policy is used.

Use one policy for A128/A256/A512 and record it. The scale is an optimization control, not a claimed architectural contribution. Report initial embedding RMS, pre/post-transition RMS, final-feature RMS and logit standard deviation on a fixed smoke batch. If conclusions depend strongly on this scale policy, treat initialization as part of the result rather than hiding it.

For SWT/SWT-R, apply the same variance-preserving rule to every T1 boundary. A negative result from a width-changing model whose transition projection collapses or explodes activation scale is not evidence against nonuniform width.

### 12.5 Architecture versions and budgets to test first

The following counts use **`V = 50,257`**, the six-block specification in §6, and linear T1 transitions. They replace the vocabulary size only; the `V = 156,000` examples in §6 remain illustrative larger-vocabulary budgets. These are calculations checked against shape-only parameter definitions, not trained model results.

| Priority | Arm | Body widths | Input/output tables | Exact total parameters |
| --- | --- | --- | --- | ---: |
| First | B0 | 1024 × 6 | Tied 1024 | 128,546,816 |
| First | A128 | 1024 × 6 | Separate 128/896 | 129,595,392 |
| Next embedding control | A256 | 1024 × 6 | Separate 256/768 | 129,595,392 |
| Next embedding control | A512 | 1024 × 6 | Separate 512/512 | 129,595,392 |
| Later body test | C / SWT | 256, 256, 512, 512, 1024, 1024 | Separate 128/896 | 86,893,568 |
| Later reinvestment test | D / SWT-R | 512, 512, 1024, 1024, 1280, 1280 | Separate 128/896 | 126,285,056 |

**Run order:** validate B0 and A128; screen the four uniform-body embedding arms; then test C and D against an A-family uniform reference with the **same embedding allocation**. A128 is the predefined body reference. If pilot evidence motivates using A256 instead, change every body arm to A256, recount the adapters, and label that as a separate comparison. Embedding gains are not a prerequisite for testing width allocation; correct, stable training is.

Do not claim equal total budgets: A adds 1,048,576 parameters over B0, and D is 3,310,336 below the A128 uniform reference. Add the declared budget controls in §7. In particular, compare C with a tuned uniform model near **its own** budget, not only with the larger 129.60M reference. The six-layer baseline is a tractable hypothesis test, not a claimed optimal model at that size; add a conventional deeper/narrower control before broad efficiency claims.

Keep the **linear input mapping, linear T1 boundaries and linear output adapter** for the first relevant comparison. Defer N128, nonlinear input FFNs, direct-output final blocks and padded transitions unless an interface needs diagnosis.

### 12.6 Training horizons and confirmation

| Phase | Proposed allocation | What it can establish |
| --- | --- | --- |
| Smoke tests | Small batches and a short run; B0/A128 first | Shape/count correctness, causal loss, gradient flow and gross stability |
| Screening | 1B processed tokens per selected arm; one seed initially | Large differences and convergence behavior; not a conclusive small gain or null result |
| Confirmation | 5B processed tokens per selected arm; at least three independent seeds | Selected central claims, subject to the uncertainty and effect-margin requirements in §8 |
| Depth confirmation | 12 blocks at baseline width 1024; same tokenizer and dataset protocol | Whether a result survives a less shallow model, before changing backbone or language setup |

These are proposed **token budgets**, not runtime estimates, compute-optimal prescriptions, or a requirement to run every arm at every horizon. Freeze validation/test documents and match data exposure within each same-token comparison. Register the learning-rate schedule for each horizon; an early checkpoint from a 5B-token decay schedule is not equivalent to a run whose schedule finishes at 1B. Fix effective batch size in tokens across arms within a comparison using accumulation where necessary.

**Interpretation of the six-layer pilot.** Because only two blocks occupy each SWT stage, a negative C/D result at six layers is not sufficient for a general claim that stagewise width reallocation fails. Before a negative architectural conclusion or publication claim, confirm the relevant comparison in the deeper model (or another prespecified depth) with a correspondingly re-solved width budget. A large, stable positive pilot may justify advancing sooner, but still requires depth confirmation for broad scope.

For the 12-block uniform models, the corresponding calculated totals are **205,629,440** (B0) and **206,678,016** (A128). Recalculate and budget-match the nonuniform schedules at that depth rather than assuming that repeating a six-block pattern is optimal. Final optimizer settings, learning-rate search, effective batch, precision, hardware, effect margins and matching tolerance must be recorded before confirmation; they are not established by this document.

### 12.7 Multilingual and Qwen follow-up

Stay within CulturaX for the initial language-extension study. One proposed compact mixture is **English, Spanish, Chinese and Arabic** (`en/es/zh/ar`), which are available configurations [16](#ref-16). The mixture is a proposed coverage choice, not an optimal allocation. Freeze language sampling weights and record **token** exposure per language; language filtering and sampling must be identical across architecture arms.

Before changing the backbone, choose one multilingual tokenizer and run an English-only control and the multilingual mixture **with that same tokenizer** on the selected small model family. Compare reference versus candidate in both data settings. Recount the vocabulary and adapters. This separates tokenizer/vocabulary changes from language-mixture changes; it does not eliminate the fact that a fixed total multilingual budget gives each language a different exposure. Report per-language outcomes and tokenizer-consistent losses rather than comparing raw perplexity across different tokenizers.

For a later HF backbone confirmation, use the architecture configuration of **`Qwen/Qwen3-0.6B-Base`**. Its released config has hidden width 1024, 28 layers, vocabulary size 151,936, tied embeddings, 16 query heads, 8 key/value heads and head dimension 128 [21](#ref-21). In particular, its query projection width is not simply the residual width; do not reuse the pilot's `4d²` attention count. Recount the complete implementation. Keep this as a base-model architecture comparison trained from scratch on matched CulturaX data, not a comparison between our short-trained model and the released pretrained checkpoint. Adapting existing pretrained weights would be a separate project with conversion and continued-training controls.

> **First action:** implement the HF-backed B0 and A128 models, freeze the CulturaX English sample and GPT-2 tokenizer, and run the correctness/screening gate. Do not begin with all languages, a pretrained Qwen conversion, or the full architecture grid.


---

<a id="sources"></a>

## 13. Sources and evidence notes

Numbered references distinguish external evidence from the supplied project source. Architecture choices and experimental plans are proposals; algebra and numerical budgets are this document’s calculations. The v0.3 review checked sources on 8 September 2026, including the coupling paper’s tables. The v0.4 update additionally checks CulturaX metadata and HF configuration/API sources for the setup. This is not an exhaustive novelty search or independent reproduction of the cited results.

**Changes in v0.5:** retained the v0.4 CulturaX/HF setup; added an explicit stable-document manifest rule; quantified the unusually high embedding-table share of the six-layer pilot; added mandatory variance/scale controls for asymmetric embeddings and T1 width transitions; and clarified that a six-layer negative SWT result requires deeper confirmation before broad claims. The code snippets document setup APIs; no end-to-end data access or training run is claimed.

<a id="ref-1"></a>

[1] Supplied working document. Embedding Parameters vs. Depth in Small Language Models: Design Review, Literature Map, and Research Plan. August 2026.  
Part III, §5d supplies the rank-allocation proposal; subsequent conversation supplies the widening and reinvestment variants.

<a id="ref-2"></a>

[2] [Chung, H. W., et al. Rethinking Embedding Coupling in Pre-trained Language Models. arXiv:2010.12821v1, 2020.](https://arxiv.org/pdf/2010.12821v1)  
Consulted §3–5 and Tables 2, 3 and 6, including the pretraining/fine-tuning budget distinction.

<a id="ref-3"></a>

[3] [Wu, Z., et al. Variable-Width Transformers. arXiv:2606.18246v1, 2026.](https://arxiv.org/html/2606.18246v1)  
Close width-profile and residual-interface comparator; findings apply to its tested construction.

<a id="ref-4"></a>

[4] [Han, D., et al. Step by Step Network. arXiv:2511.14329v1, 2025.](https://arxiv.org/html/2511.14329v1)  
Precedent for increasing-width stages with progressive introduction of input-feature slices.

<a id="ref-5"></a>

[5] [Lan, Z., et al. ALBERT: A Lite BERT for Self-supervised Learning of Language Representations. arXiv:1909.11942.](https://arxiv.org/abs/1909.11942)  
Factorized embedding precedent; not evidence for this causal-LM allocation.

<a id="ref-6"></a>

[6] [Mehta, S., et al. DeFINE: DEep Factorized INput Token Embeddings for Neural Sequence Modeling. arXiv:1911.12385.](https://arxiv.org/abs/1911.12385)  
Learned compact-to-large embedding mappings; distinct from the proposed small input SwiGLU.

<a id="ref-7"></a>

[7] [Shazeer, N. GLU Variants Improve Transformer. arXiv:2002.05202.](https://arxiv.org/abs/2002.05202)  
SwiGLU construction and its three-matrix parameterization.

<a id="ref-8"></a>

[8] [Zhang, B., and Sennrich, R. Root Mean Square Layer Normalization. arXiv:1910.07467.](https://arxiv.org/abs/1910.07467)  
Normalization definition; padding-related scaling arguments are deductions.

<a id="ref-9"></a>

[9] [Yang, Z., et al. Breaking the Softmax Bottleneck: A High-Rank RNN Language Model. arXiv:1711.03953.](https://arxiv.org/abs/1711.03953)  
Background for distinguishing output features from final linear-head rank.

<a id="ref-10"></a>

[10] [ACL Rolling Review. Review Form.](https://aclrollingreview.org/reviewform)  
Main-conference versus Findings assessment criteria; access date above.

<a id="ref-11"></a>

[11] [NeurIPS 2026. Reviewer Guidelines: Negative Results.](https://neurips.cc/Conferences/2026/ReviewerGuidelines)  
Publication-policy example, not an acceptance prediction or deadline recommendation.


<a id="ref-12"></a>

[12] [Mehta, S., et al. DeLighT: Deep and Light-weight Transformer. arXiv:2008.00623.](https://arxiv.org/abs/2008.00623)  
Earlier block-wise depth/width allocation; not a reproduction target for the proposed residual resizing.

<a id="ref-13"></a>

[13] [Mehta, S., et al. OpenELM: An Efficient Language Model Family with Open Training and Inference Framework. arXiv:2404.14619.](https://arxiv.org/abs/2404.14619)  
Earlier layer-wise allocation in language models; distinguishes the broad principle from this exact architecture.

<a id="ref-14"></a>

[14] [Meta. LLaMA reference model implementation, `llama/model.py`.](https://raw.githubusercontent.com/meta-llama/llama/main/llama/model.py)  
Bias-free projections, gain-only RMSNorm and final normalization before vocabulary projection. This is a moving code reference; pin a commit if implementation decisions depend on it.


<a id="ref-15"></a>

[15] [Nguyen, T., et al. CulturaX: A Cleaned, Enormous, and Multilingual Dataset for Large Language Models in 167 Languages. LREC-COLING 2024.](https://aclanthology.org/2024.lrec-main.377/)  
Primary dataset paper; used to motivate the selected corpus family, not to claim that this architecture will improve on it.

<a id="ref-16"></a>

[16] [UONLP. CulturaX Hugging Face dataset card.](https://huggingface.co/datasets/uonlp/CulturaX) [Repository configuration metadata.](https://huggingface.co/datasets/uonlp/CulturaX/blob/main/README.md)  
Verified language configurations, English Parquet mapping, text/provenance fields and stated access conditions. Pin the actual data revision used; do not rely on a moving `main` reference in the experiment manifest.

<a id="ref-17"></a>

[17] [Hugging Face Transformers. Llama model and configuration documentation.](https://huggingface.co/docs/transformers/en/model_doc/llama)  
Built-in model/configuration support, random initialization from config, attention/head fields and current RoPE parameter interface. Documentation is version-sensitive; pin the installed implementation.

<a id="ref-18"></a>

[18] [HuggingFaceTB. SmolLM2-135M configuration.](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/main/config.json)  
Available Hub reference using `LlamaForCausalLM`, with hidden width 576 and 30 layers. It is not the custom six-layer B0 or a source of pretrained weights for the pilot.

<a id="ref-19"></a>

[19] [OpenAI community. GPT-2 configuration and tokenizer repository.](https://huggingface.co/openai-community/gpt2/blob/main/config.json)  
Vocabulary size 50,257 and EOS ID 50,256. Only tokenizer assets are reused, not the GPT-2 body or pretrained weights.

<a id="ref-20"></a>

[20] [Hugging Face Datasets. Loading methods.](https://huggingface.co/docs/datasets/en/package_reference/loading_methods)  
Current `load_dataset` arguments for language configuration, pinned revision, authentication and streaming. The CulturaX access gate must be satisfied independently.

<a id="ref-21"></a>

[21] [Qwen. Qwen3-0.6B-Base configuration.](https://huggingface.co/Qwen/Qwen3-0.6B-Base/blob/main/config.json)  
Later backbone reference; its GQA and head dimensions differ from the pilot. Recompute parameters rather than importing the standard-MHA formula.
