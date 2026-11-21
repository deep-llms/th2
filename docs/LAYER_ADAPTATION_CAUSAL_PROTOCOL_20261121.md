# Why does changing layer order fail, and what repairs it?

Status (21 November, completed and locally replayed): all four paired2000-update continuations and both seed studies COMPLETE, plus the frozen path-background pilot. Small exports13affe6f resolved and both archive/member/backup hashes verified. Loss scores replayed locally; raw geometry moments and weights/bases remain node-only. The [results and interpretation](LAYER_ADAPTATION_RESULTS_20261121.md) distinguish adaptation/rollback evidence from descriptive geometry. B recovers about0.30nat but retains0.082/0.084nat gap; the null-input SVD removal harms both seeds. DCLM-only/context512, exploratory/historical overlap unknown. Historical diary entries below retain their original status.

## The question and the correction

The user proposed continuing a sufficiently trained model along two branches, one unchanged and one reordered, and inspecting how representations adapt. This is the main experiment. The earlier swap-loss inventory established a phenomenon already known to the user. The provenance work and CPU fixtures did not establish its mechanism. Native-versus-skip is a supporting usefulness control, not the new project's main result or a reason to postpone this experiment indefinitely.

We distinguish three questions:

1. What computation changes immediately when order changes with weights fixed?
2. During subsequent learning, which parameter changes are responsible for recovery, conditional on the other changes?
3. Does recovery restore the old computation, or learn another way to solve the task?

An activation heatmap answers none of these alone. The deliverable is a measured chain from changed input, through a specified operation, to downstream loss, with a controlled intervention and a failed alternative explanation. Neither a positive patch effect nor successful retraining identifies a unique, universal cause.

## Reading that changes the experiment

### Mathematical decision brief: two vectors are not a mechanism (21 November clarification)

The immediate endpoint is order-sensitive computation and its repair, not a
claimed extraction of a layer's intrinsic knowledge. A transferable useful
transformation is the downstream objective. Two activation vectors alone do
not uniquely decompose content, coordination and positional dependence.

For a matched example/token and a declared site, let a,b be the two column
vectors. Start with the exact identity
`||a-b||^2 = (||a||-||b||)^2 + 2||a||||b||(1-cos(a,b))`.
It separates radial and angular distance, not causes of task loss. Zero norms
make angle undefined. PCA on just two centered observations has rank at most
one, so it cannot discover a general semantic subspace.

Across matched examples construct X,Y with observations in rows and channels
in columns. Keep sequence/position/block/slot alignment explicit and separate
seeds. With D=Y-X, separate mean delta mu and centered variation Dc=D-1 mu^T.
SVD `Dc=U S V^T` gives feature-space directions in V; a small-rank spectrum
would mean observed changes concentrate in a few directions, not that these
directions encode position or cause most of the damage. Separate PCA plots
have arbitrary signs/rotations and are not coordinate comparisons. Keep the
original shared coordinates for functional interventions; do not fit an
alignment that hides a fixed consumer's incompatibility.

The existing paired moments support the algebraic difference scatter
`Cdd=Cxx+Cyy-Cxy-Cxy^T` and mean `my-mx`. The DEPLOYED reporter currently
returns X/Y spectra, CKA, means, norms and distances, not a difference-SVD
or tested low-rank repair. Its moments come from confirmation observations;
they MUST NOT be used to select a new subspace and then call effects on those
same observations independent confirmation.

The targeted follow-up already contemplated in Section3 must derive a basis
from the separate calibration rows at one explicitly registered site. Retain
the prespecified rank8 (or actual available rank); do not search ranks for a
favorable result. Handle mean-shift and centered directions separately. A
donor correction to a recipient vector is
`b' = b + P(a-b)`, where `P=V8 V8^T`; apply it at a selected component/update
while preserving the recipient residual and recomputing downstream operations
with fixed recipient weights. Score actual NLL with unchanged token masks.
Use identity/self-patch, complementary projection, seeded random rank8,
separately norm-matched perturbations and reverse-direction controls. Full
state replacement that simply reconstructs canonical computation is an
identity check, not the scientific result. Freeze site/basis/control choices
before evaluating a genuinely uninspected confirmation split, or label any
reuse exploratory. A failure of the top-variance subspace does not exclude
important low-variance directions or nonlinear/context-specific mechanisms.

Conditional effect is `L(recipient)-L(patched recipient)` on matched examples,
in nat/token, with per-sequence signs and per-seed results. It is neither a
percentage of the cause nor proof that a deployable translator can produce
the donor correction without the reference run. The purpose is to distinguish
scale sensitivity, task-relevant coordinate changes and distributed/contextual
effects; current attention pattern/value and FFN gate/up interventions provide
the computational localization before adding a subspace experiment.

After adaptation, the proposed fixed-input four-cell analysis uses inputs
h0,h1 from parent/adapted prefixes and the same target block with weights
F0,F1: `F0(h0), F0(h1), F1(h0), F1(h1)`. Evaluate all with one fixed suffix
and matched masks; a second suffix is a separately labelled panel. This
distinguishes changed input recruitment from changed block function within
that context, unlike comparing only F0(h0) with F1(h1). It remains a targeted
follow-up, not an already deployed measurement. The deployed pair/rest weight
rollback supplies a coarser first localization with ordinary-training controls.

Operational meaning of a reusable layer contribution: its residual update
improves task loss relative to skipping that update in several declared
contexts, and a proposed retained component preserves that benefit on new
contexts while removing it loses benefit. Agreement between A/B, a high CKA,
or the complement of difference-PCs alone cannot establish this. The current
single-swap study cannot establish arbitrary-path portability or uniquely
extract context-free knowledge. No training/measurement expansion is implied
by this clarification; preserve the running frozen study and select the next
bounded intervention from its evidence.

All five sources below were read through their main text and available appendices in the pinned versions, not just abstracts. Exact PDF hashes/coverage are in `results/layer_dependency_20261120/reading_ledger.json`. These are methodological evidence, not experiments on our checkpoints.

| Primary source | Relevant finding and scope | Design consequence |
| --- | --- | --- |
| [Kornblith et al., CKA, ICML 2019](https://proceedings.mlr.press/v97/kornblith19a.html), including separate supplement | Similarity metrics have different invariances. Linear CKA is invariant to orthogonal feature transformations and isotropic scaling; its Transformer appendix does not establish a universally best metric. | Measure raw, same-coordinate changes as well as centered CKA. A fixed downstream reader need not tolerate a rotation that CKA ignores. Separate residual state from newly written updates. |
| [Ding et al., Grounding Representation Similarity, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/0c0bf917c7942b5a08df71f9da626f97-Paper.pdf), main plus supplemental appendices A–E | Similarity can miss changes that affect probe performance; no tested metric reliably tracks all functional differences. Their PC-removal test retrains a classifier, not a fixed LM suffix. | Geometry nominates hypotheses. Validate with the actual unchanged downstream computation and NLL. Do not interpret large-variance directions as necessarily task-important. |
| [Heimersheim & Nanda, How to use and interpret activation patching, v1](https://arxiv.org/abs/2404.15255v1), all 13 pages | Patching conclusions depend on the source, destination, metric and direction; redundant/joint computation complicates necessity and sufficiency. | Define every intervention's donor/recipient and fixed context. Include both rescue and damage directions, self-patch parity and joint-effect checks. |
| [Hase et al., Does Localization Inform Editing?, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/3927bbdcf0e8d1fa8aa23c26f358a281-Abstract-Conference.html), main plus appendices A–C | In their factual-editing settings, localization does not reliably select where editing works best. This is not a theorem that localization is useless. | Separate activation mediation from parameter adaptation. Use explicit weight rollback/transfer, and compare against normal continuation. Do not choose a training repair solely from the largest patching peak. |
| [Zhang & Nanda, Towards Best Practices of Activation Patching, v1](https://arxiv.org/abs/2309.16042v1), main plus appendices A–I | Corruption, metric and simultaneous patch width change localization results; joint effects need not equal sums of individual effects. | Use the real order change as the intervention, not arbitrary Gaussian noise. Report unnormalized paired NLL changes and separate KL fidelity. Start with individual computational components, then measure a prespecified interaction. |

The previous full readings on FFN memories/concept promotion, residual computation, and stitching remain relevant. Their interpretations do not establish that an FFN's gate is exclusively “coordination” or its values exclusively “knowledge.” Our SwiGLU projections jointly determine a context-dependent transformation.

### Additional full readings: comparison, decomposition and functional tests

Read main text plus available appendices/supplements for the four versions below.
Coverage and byte hashes are in the reading ledger. Figure captions/equations
were read; rendered plots were not individually inspected in this pass. No
paper experiment was reproduced and no new model fitting was launched.

| Method and source | What it actually identifies | Decision for this study |
| --- | --- | --- |
| [SVCCA, Raghu et al. 2017](https://proceedings.neurips.cc/paper/2017/hash/dc6a7e655d7e5840e66733e9ee67cc69-Abstract.html), main10pp + supplement7pp | SVD reduces sampled activation spaces; CCA finds maximally correlated linear projections across matched observations. Paper also projects activations and tests network accuracy. | Optional shared-subspace description; do not equate aligned correlation with compatibility for a fixed suffix. Fit on calibration and test function separately. Main says variance threshold but AppendixA sums singular values rather than squared values; specify an implementation instead of silently treating the two as identical. |
| [PWCCA, Morcos et al. 2018](https://papers.nips.cc/paper_files/paper/2018/hash/a7a3d70c6d17a73140918996d03c014f-Abstract.html), main10pp + supplement7pp | Weights canonical correlations by original-representation projections to reduce influence of unstable directions in their settings. The pseudo-distance is asymmetric. | Do not label stable shared directions knowledge or changing directions noise. Report direction, sample/rank conditions and matched input histories. It is not the main causal endpoint. |
| [LEACE, Belrose et al. 2023](https://arxiv.org/abs/2306.03819), all20pp inclA-J | Minimum mean-square affine edit subject to zero cross-covariance with an explicit target concept; linear prediction guarantees under stated losses. | Useful only after a defensible concept is specified. A/B identity is not a clean label for path dependence versus useful content. No immediate eraser fit; measure actual task loss if later used. Oblique projections can grow norms, and sequential erasure changes later fitting distributions. |
| [Sparse Autoencoders Find Highly Interpretable Features in Language Models, Cunningham et al. v3](https://arxiv.org/abs/2309.08600v3), all20pp inclA-G | Sparse dictionary learning can expose features obscured in neuron coordinates; selected feature edits affect IOI behavior. Dictionary size/sparsity/reconstruction and layer type matter. | Candidate semantic follow-up at one localized site, not an automatic knowledge extractor. Use a shared frozen dictionary across A/B, preserve reconstruction residual for feature edits, and test reconstruction-only damage and feature-specific functional effects on held-out examples. No SAE fit in current study. |

For CCA, the canonical correlations are singular values of the whitened
cross-covariance `Cxx^(-1/2) Cxy Cyy^(-1/2)` on its supported ranks. Learned
linear alignments must not conceal an incompatibility of the actual unchanged
consumer. CKA and CCA answer different invariance questions; neither is a
semantic identification procedure.

For LEACE, let `W=(Cxx^(1/2))^+` and let Q project orthogonally onto the column
space of `W Cxz`. The affine eraser is
`r(x)=x-W^+ Q W (x-mu)`. The minimal-change criterion is in representation
space, not task NLL. With a binary A/B label the mean cross-covariance has rank
at most one: removing it is not removing every difference between paths.
The guarantee is not arbitrary nonlinear unreadability or a universal
zero-one accuracy claim.

A sparse dictionary expresses an activation as `h=sum_j c_j(h) d_j + e(h)`.
Unlike PCA, directions need not be orthogonal and may outnumber dimensions;
only a small subset is active per example. The dictionary must be learned
from many activations before analyzing one vector. An interpretable name,
reconstruction accuracy and a causal contribution are three separate checks.
For a shared dictionary, a targeted donor edit is
`b'=b+sum_{j in S}(c_j(a)-c_j(b)) d_j`, retaining the recipient reconstruction
error e(b). Replacing b by its reconstruction alone can itself damage the
model and needs its own control. Separate dictionaries for A/B introduce a
feature-alignment problem; matching decoder cosines alone is insufficient.
AppendixD.4 of the SAE paper itself reports failed weight-based layer-to-layer
feature matching. Its selected IOI edits do not establish arbitrary-context
portability. AppendixB also provides no unique preferred sparsity decomposition.

Execution priority remains: finish existing component interventions and
pair/rest rollback; use that evidence to nominate a bounded computational
site; then test calibration-derived difference directions functionally using
the decision brief above. A shared SAE is a later option if the question
requires semantic feature resolution rather than simply the geometry of the
order-induced discrepancy. LEACE requires an explicit, defensible attribute;
SVCCA/PWCCA are optional summaries, not prerequisites delaying the mechanism
experiment. DAS/ROME/ICA tutorial sources found during this pass were triaged,
not fully read or counted as new evidence. No claim of an exhaustive survey.

### Method reassessment: an INSID3-like path-background test (21 November)

This is an analytical proposal answering the user's INSID3 question, not a
change to the frozen running study or a claim that an eraser already works.
The full INSID3 v1 main/supplement A-F was previously read; this pass reread
Section3.1 and SupplementA.2-A.4/B. Paper count is unchanged.
Source: https://arxiv.org/html/2603.28480v1

The transferable idea is a low-content calibration stimulus exposing a
repeatable nuisance component, followed by a separately evaluated functional
removal. INSID3 uses uncentered SVD of noise-image features and projects real
features onto the orthogonal complement. Its supplement checks black/white,
gradient and noise inputs. It debiases cross-image matching while retaining
original features for intra-image grouping. It does not delete every such
component inside every Transformer block. Its rank500 is not a rank prescription
for our model. Reduced matching bias is not proof of universally useless content.

Reassessment of the previous plan: paired continuation and component/weight
interventions are useful controlled localization, but delta-SVD of real-text
activations alone does not separate harmful path bias from different useful
computation. SAE/CCA/LEACE are tools, not competing explanations. Give the
bounded INSID3 hypothesis a falsifiable test before expanding decomposition
methods. Keep the ongoing study and use its existing common-prefix/common-suffix
central-pair boundary to avoid a new path sweep.

Hypothesis, not an established decomposition:
`h_p(x) = s(x) + b_p + i_p(x)` at a declared common boundary. Here b_p is a
repeatable path offset, i_p is content-dependent path interaction, and s is a
shared component under this working model. Their semantic meanings are not
identified by algebra; centering constraints only choose an operational
parameterization. With two paths, an additive path-only contrast is just one
mean vector; a higher-rank space of paired contrasts necessarily includes
variation across content/token contexts. Do not relabel that entire space
pure position bias.

Proposed bounded diagnostic, with parent weights frozen:
1. Use the same two paths, tokens, masks, length and token positions. Capture
   immediately after the swapped pair: common preceding computation, same
   executed inventory and common remaining suffix. This measures pair-order
   effects, not isolated absolute-depth effects. Component sites from the
   existing panel can be a later, explicitly selected refinement.
2. Use several repeated-token anchors, short periodic anchors and independently
   shuffled natural-token sequences from calibration only. These reduce some
   content/structure but are not content-free; retain families separately.
   Keep BOS/boundary treatment matched and exclude control tokens from scored
   natural-text evaluation. Uniform random embedding vectors are not the default.
3. For each anchor q and matched position compute d(q)=h_B(q)-h_A(q). Estimate
   mean mu separately and SVD the centered contrast matrix; do not SVD pooled
   raw h_A/h_B and mistake shared content for path dependence. Prespecify a
   small rank ceiling and refuse degenerate ranks. Test mean transfer and
   subspace overlap on held-out anchor families; then test overlap with paired
   natural-text differences. Overlap is geometry, not a causal fraction.
4. A failed transfer from anchors to real text rejects this calibration as a
   useful path-background estimate. Do not respond by automatically increasing
   rank or fitting a large translator. An anchor failure does not rule out a
   content-dependent mechanism; return to component interactions already measured.
5. If stable, test a fixed mean correction h_B-mu and, as a separately labelled
   stronger hypothesis, a shared projection Q=I-UU^T at the one boundary.
   Evaluate A/original, A/Q, B/original, B/Q with the original fixed suffix and
   real DCLM targets; rank/energy controls, reverse/add-back sensitivity, and a
   norm-matched control distinguish dimension loss and scale effects. Freeze
   basis/rank/site before uninspected confirmation; current studied rows reused
   for discovery are exploratory. The correction itself needs no paired native
   donor at inference, unlike b+P(a-b).
6. Require actual B NLL improvement, canonical cost within a predeclared small
   tolerance, advantage over matched random controls, and consistency reported
   separately by seed. Gap reduction caused only by damaging A is failure.
   A/Q benefit alone is not order-specific improvement. A projection learned
   on one fixed checkpoint is not automatically valid for adapted checkpoints.

Two important null-input traps follow from this actual model's code, not a
new experiment. Attention/MLP linears have no bias; RMSNorm and RoPE map zero
to zero. Injecting all-zero embeddings therefore gives a trivial zero state,
not a useful analogue of the vision probe. For a perfectly repeated nonzero
token without a distinct boundary token, V can be identical at all causal
positions: different attention patterns still sum to the same value. Absence
of position variation on that anchor is not absence of positional computation.
Use multiple anchors and inspect degeneracy rather than treating an empty
prompt or a single repeated token as a definitive null stimulus.

Interpretation ladder: stable low-content contrast -> transfer to real text ->
beneficial fixed intervention -> replication on another declared context.
Only the first two establish a candidate shared direction; the third supplies
conditional functional evidence. None alone extracts a layer's intrinsic
knowledge. If the removed component is useful elsewhere, preserve it as a
separate channel/conditioning signal rather than declaring it universally
nuisance; that is a future architecture hypothesis, not implemented here.

## 1. Paired continuation, not a new architecture sweep

Candidate parent: the existing Horizon dense D seed0/1 terminal checkpoints (80k updates). Reuse requires actual checkpoint/fullstate hashes, source/schema parity and validation-curve inspection. “80k” does not establish convergence. Report the late validation slope and the normal continuation's improvement; if the parent is still rapidly improving, describe the study as adaptation at that maturity, or extend a common trunk before splitting. Do not manufacture a convergence claim.

For each authenticated parent, create only two training branches:

- **A, normal:** middle path `[0,1,2,3]`.
- **B, changed:** middle path `[0,2,1,3]`, the central adjacent pair. This pair is fixed before new results; no search for a favorable swap.

Both retain stem/coda, model size, readout, objective, token masks and all parameters. No router or TG. Clone full Adam state, step counters and RNG. Optimizer state follows **parameter identity**, not execution slot; change the execution list, not module registration or tensor names. Use exactly the same recorded future minibatches and LR policy. Any stochastic masks must be matched by module identity, or dropout must be absent in both. No reset-Adam/high-LR confound.

Initial bounded proposal: 2,000 matched updates per branch; snapshots at 0, 10, 50, 200, 500, 1,000 and 2,000. With the historical batch32 and length512 this is 32,768,000 input tokens per branch, not necessarily that many scored targets. This is a proposed budget, not a claim of sufficient recovery. Record actual runtime and recovery slope; an unfinished recovery is not evidence of impossibility. Any extension uses the same declared pairing, not an unreported method sweep.

At each snapshot, report each branch under its training order. At 0/200/2000 also cross-evaluate both weight sets under both orders. This gives `L(A,canonical)`, `L(A,swap)`, `L(B,canonical)`, `L(B,swap)`. Keep per-seed values, token-weighted paired differences and per-sequence distributions. `L(B,swap)-L(A,canonical)` is the remaining performance gap; improvement relative to B at time0 alone conflates adaptation with ordinary learning.

## 2. What to record in the representations

Use exactly the same token sequences/positions, in evaluation mode, without generation or KV-cache reuse. Report two distinct alignments: **same execution slot** and **same parameterized block**. The latter changes preceding computation/depth, so it is not a controlled depth effect.

Record only bounded samples at embedding, block input, post-attention and block output:

- Residual RMS, update RMS and update/residual ratio; token-wise angles and signed paired differences.
- Pre/post-RMSNorm state, attention update and FFN update separately. A large shared residual can conceal a changed update in whole-state cosine/CKA.
- Feature-centered covariance spectrum and CKA on fixed rows; explicitly use observations-by-features and center each feature across observations. Preserve raw means/norms separately. Do not align away the very change a fixed consumer must handle.
- For the moved pair: Q/K after their actual normalization and RoPE, attention probabilities on valid causal positions, V, projected attention output, SwiGLU gate/up/product/down output. Attention maps are descriptions, not importance scores.

The existing model has Q/K RMSNorm, grouped-query attention, RoPE, pre-norm residual blocks and a tied embedding/readout. An instrument that silently substitutes a generic Transformer is invalid. Explicit attention reconstruction must reproduce the native branch output and NLL within a registered precision tolerance before use. All large activations stay on-node; keep per-sequence losses, summary moments and pinned sample IDs in small exports.

Geometry first looks at the immediate perturbation and its evolution **relative to branch A's drift**. High similarity with loss damage or low similarity with recovered loss are both informative outcomes, not metric failures to conceal. No fitted probe/SAE is needed for this first experiment.

## 3. Distinguishing mechanisms with interventions

The following are our proposed experiments, not claims established by the cited papers. Start at fixed parent weights (time0), then repeat only informative contrasts at the terminal snapshots. Discovery rows and confirmation rows are separated before inspecting outcomes.

| Hypothesis | Intervention, with the remaining recipient computation fixed | Evidence it can supply; limitation |
| --- | --- | --- |
| Residual/update scale mismatch is important | At the moved block's input, replace only its per-token RMS with the same-example native-context RMS: `h' = h_swap * rms(h_native)/max(rms(h_swap),eps)`. Preserve direction. Also compare normalized directions and subsequent update/residual ratios. | Rescue supports sensitivity to this scale intervention. Failure argues against this simple correction, not against every normalization mechanism. Norm values can carry content; this is not proof of a purely numerical cause. Pre-norm does not normalize the residual bypass. |
| Attention selects different preceding tokens rather than merely carrying different values | For the **same fixed block**, obtain native-context and changed-context attention pattern P and value V. Evaluate all four `O(P_i V_j)` combinations, with identical GQA expansion, mask and fixed output projection, inserted into the same recipient residual/suffix. | Separates conditional pattern and value effects plus their interaction. Requires actual Q/K normalization/RoPE. Does not prove an interpretable semantic feature or that either hybrid is naturally reachable. |
| FFN input changes which computation is recruited | At the same fixed block, cross native/changed `g=SiLU(W_gate x)` and `u=W_up x`: four `W_down(g_i*u_j)` outputs in the same recipient state/suffix. | Tests gate-side versus up-side dependence and interaction in this concrete SwiGLU. Neither side is synonymous with knowledge; a small global cosine change can conceal important coordinate changes. |
| A limited representation subspace mediates damage | Only if geometry nominates one boundary, use a prespecified rank8 basis from calibration-row activation differences; patch projected donor-minus-recipient deltas, plus complementary and seeded random-rank controls. Report perturbation norms; add norm-matched controls separately. Freeze the basis before confirmation. | Functional effects can support that intervention subspace. PCA variance is not semantic content, and rank or coordinates are not a count of concepts. A basis fitted on outcomes would invalidate confirmation. |
| Recovery occurs in consumers instead of restoring the old producer | Evaluate the weight crossover below; inspect whether loss recovers while old-coordinate activations remain different. | Can establish conditional importance of parameter groups for recovery. It cannot turn geometry into a unique decomposition of learned knowledge. |

All hybrid activation cells use one declared recipient residual, suffix weights and token mask. Native component tensors are cached from the **same example**, never its label or future-token states. Donor and recipient full runs remain separate controls; a donor-donor component inside the recipient is not labelled the entire native model. Repeat rescue and reverse/damage directions where numerically valid. Expose negative effects rather than truncating them to zero.

**Avoid a tautology:** replacing the entire state after the swapped pair with the canonical state, followed by the unchanged canonical suffix, reconstructs the canonical computation. This is a useful identity check, not discovery of why the pair failed. Component/subspace interventions retain recipient computation and require stronger controls.

**Input versus function:** after training, at a fixed cut in the swapped graph, cross cached inputs from parent and adapted prefixes with parent and adapted versions of the same target block; keep the suffix fixed. Four outputs distinguish input recruitment and weight change conditional on that suffix. Repeat with the other suffix only as a separate panel. Do not call a panel that changes prefix and suffix together an input-only test.

## 4. Where did adaptation actually occur?

Let Q be the swapped pair's parameter tensors (including norms), R all remaining tensors, 0 the parent and 1 a terminal branch. In the **same swapped graph**, evaluate:

| Cell | Pair Q | Rest R |
| --- | --- | --- |
| L00 | Parent | Parent |
| L10 | Adapted | Parent |
| L01 | Parent | Adapted |
| L11 | Adapted | Adapted |

Report the four raw losses. Pair benefit with parent context is `L00-L10`; with adapted context it is `L01-L11`. Their difference is an interaction, not a percentage of the root cause. Repeat the same construction using branch A's ordinary-training updates, so generic improvement is not called order-specific repair.

If L10 recovers while L01 does not, pair updates are sufficient relative to this parent context. If reverting Q in L11 damages performance, they also have conditional necessity. The converse pattern supports surrounding-context compensation. If only L11 recovers, this supports joint adaptation at this partition; hybrids may themselves be off-distribution. None selects the cheapest possible training repair without a further controlled comparison.

If R matters, split it only in a targeted follow-up into upstream blocks, downstream blocks and shared embedding/readout. **Embedding and LM head share a tensor in this code:** they cannot be independently rolled back without changing the architecture. Treat them as one group. Keep norm parameters with their declared group. Weight mixtures are evaluation-only; optimizer states are not spliced or used for training.

The resulting mechanism report must connect an immediate intervention result with the recovery trajectory/rollback result. A large weight delta without functional effect is not evidence that the changed weights caused recovery. A successfully repaired readout does not prove layer portability.

## 5. Data, uncertainty and admission without another audit detour

Use DCLM only. Preserve the current conditional-suffix objective (prefix32, length512), rather than changing the loss while changing order. Exact new continuation row arrays, hashes, parent/source pins and calibration/confirmation membership must be registered before execution. Training branches share the same arrays. Confirmation rows must be excluded from these new training and calibration arrays.

Existing authenticated evaluation rows may be used for an explicitly **reused, exploratory, paired checkpoint study**. Missing historical original-document joins do not prevent every exploratory comparison; they do prevent a fresh/unseen-document or complete historical non-overlap claim. Recover original-document IDs for a generalization confirmation; do not call reset E2 row boundaries document IDs. Until then report sequence-level finite-sample contrasts and no document-independent confidence claim. Report each seed separately; two seeds are replication checks, not a precise population variance estimate.

Functional readout: token-weighted NLL is primary. Full-vocabulary directional KL to the selected reference is separate fidelity evidence, not quality. No normalization by a near-zero damage gap, no clipped rescue percentages, no automatic “bypass” from a near-zero mean. For directions/subspaces chosen on calibration data, report every registered confirmation contrast, including failures.

Before launch: resolve execution HOLD appropriately; reconcile any pending receipts; fresh exact ownership/runtime; verify actual parents and data; test native forward/instrument parity, causal masks, weight-group identity, tied tensors, self-patches, both fullstate resumes and name-mapped optimizer state. Use the existing tested trainer/checkpoint infrastructure where possible. Keep gates and outputs separate from historical experiments. No unknown process stops, old-sweep resumption, bulk weight exports or silent change of tolerance.

## Deliverable and immediate next work

The experiment should produce: paired recovery curves; activation/update trajectories; a small matrix of component intervention effects; the four-cell parameter crossover; and a conclusion stating which mechanism survived which attempted falsification. It may conclude “not distinguished by these interventions.” It must not end at “swap hurts loss.”

Current execution supersedes the initial implementation plan below: at the verified node snapshot16:22UTC, four paired continuations have completed and two studies are active. Native parity, fork/resume and component qualifications preceded launch. Terminal study results and a mechanism conclusion remain unverified. Historical implementation receipts below describe their original dates; they are not current blockers. Fresh read-only studyhealth904b6a6f/a02 is pending as of the latest CURRENT_TASK entry.

### Implementation receipt, tick120

`efficient_llm/warm_order/adaptation_analysis.py` now captures the original forward with temporary hooks (block input, normalized inputs, true attention/FFN outputs, post-attention and block output), labels slot and block identity, and creates the four independent pair/rest weight combinations. It reuses the existing model and validation; no model/trainer changes. CPU-only qualification is explicit. Four tests verify native hidden/NLL parity, causality, hook cleanup on failures, unchanged input weights/RNG, exact parameter partitions/endpoint losses, finite-state rejection and joint-only recovery arithmetic. [Receipt](../results/layer_dependency_20261120/adaptation_instrument_cpu.json).

These are random-weight fixtures, not measurements of layer dependence. The paired fullstate continuation, Q/K/V and SwiGLU crossover instrumentation, actual parent/data admission and GPU qualification remain unfinished. The existing 50-case generic sweep is not being expanded.

### Paired-fork qualification, tick121

`adaptation_train.py` now loads five pinned terminal Horizon files, checks the original dense schema/source, carries Adam moments by the pinned model parameter order and saves an explicit parameter-name map. Branches differ only in the declared execution path; common recipe/row arrays, inherited terminal LR, step0/intermediate fullstates and strict resume are implemented. LR policy is explicitly constant at the inherited terminal value in both branches, not a cosine restart.

Four CPU tests use the actual Horizon trainer to create a tiny synthetic80-update parent, then run both4-update branches. Each uninterrupted run equals its step2/resumed counterpart including fullstate and traces. Step0 model/Adam/CPU/NumPy/Python RNG equal the parent; an independent first Adam update matches branchA. Parent hashes remain unchanged; mismatched pins/Adam order/counters, overlapping confirmation rows, changed branch on resume and CUDA execution are rejected. [Receipt](../results/layer_dependency_20261120/adaptation_train_cpu.json). The library emitted a local driver-discovery warning during CPU work; it is not a B200 failure or GPU qualification.

Component crossover instruments, actual parent/runtime admission, production evaluation/report pipeline and GPU fullstate/numerical checks remain. No claim of actual model recovery or identified cause follows from these fixtures.

### Component intervention qualification, tick122

`adaptation_components.py` implements same-block attention-pattern/value and SwiGLU gate/up four-cell interventions. Bit0 is recipient, bit1 donor. All four cells retain the recipient residual and suffix weights; attention interventions naturally recompute the recipient FFN. Native Q/K normalization, RoPE, GQA and causal masking are reproduced and checked against actual branch outputs before intervention. Precision is CPU FP32 only, with preregistered absolute2e-6/relative1e-5 parity tolerances.

The existing paired scorer supplies per-sequence NLL sums, exact token denominators and separate directional KL. Conditional effects, joint effect and interaction are unnormalized loss sums; donor whole-run scores remain separate from donor-donor components in the recipient graph. Four CPU tests establish native/self-patch parity, independent explicit recipient-suffix computation, causality in both directions, no weights/RNG mutation, hook cleanup on error, independent full-logit NLL/count parity and rejection guards. [Receipt](../results/layer_dependency_20261120/adaptation_components_cpu.json).

This completes local primitive implementations for the two targeted component mechanisms. It does not establish their importance on trained models. Actual parent/data manifests, production result/replay integration, current ownership/runtime and GPU qualification are still required. Poller HOLD was not cleared and no remote workload was submitted.

### Local panel/replay and historical parent registration, tick123

`adaptation_eval.py` now integrates a single-snapshot30-case CPU panel: six parent/A/B order evaluations, eight pair/rest weight cells and sixteen parent component interventions (two kinds, both directions, four cells, fixed target middle2). This is the registered targeted analysis, not another permutation search. Small NPZ score arrays preserve per-sequence sums/counts, with JSON report and independent numeric replay after file-hash checks. An integration fixture verifies endpoint/gap arithmetic and rejects altered report values even after its file hash is updated. [CPU receipt](../results/layer_dependency_20261120/adaptation_eval_cpu.json). KL references stay within the same parameter/readout model; no cross-model KL is claimed.

`python scripts/register_adaptation_parents.py` authenticates the historical local archive against prior inventory/backup pins, compares the required extracted metadata with archive bytes, verifies current model/trainer source hashes, and reconciles checkpoint/terminal/backup identities. [Registration](../results/layer_dependency_20261120/adaptation_parent_registration.json) contains five file pins for each dense seed and historical data pins. It does not verify present node availability or locally load weight/fullstate bytes.

Parent maturity correction: the recorded128-sequence canonical validation NLL changes from3.76069466 to3.74035635 (seed0) and3.77409583 to3.74465486 (seed1) between70k and80k updates. These parents are still improving on that reused monitor, not demonstrated converged. Their terminal LR is6e-5. Keep the matched ordinary-continuation control; do not call its improvement order-specific recovery. No fresh/generalization claim follows from this monitor.

Production integration remains incomplete: actual checkpoint loading into the panel, bounded batch streaming/time-course aggregation, new continuation/calibration/confirmation array registration and node/GPU gates. No remote request, new training or actual checkpoint diagnostic occurred in this turn.

### Snapshot-loader and streaming qualification, tick124

`adaptation_pipeline.py` checks trusted caller-supplied parent/phase file hashes, exact parent/source/schema/parameter-name identities, Adam counters and moment shapes, common A/B recipe, same update, and path accounting. It preserves caller CPU Torch RNG while constructing models. The loader does not authenticate the authority of externally supplied pins or establish current node availability.

Evaluation is now split into batch collection and final score writing. The streaming wrapper verifies dataset/boundary/selection-file hashes, preserves selection order, retains only score arrays across batches, and replays the final report. Whole-batch versus2+1 streaming was checked on a three-sequence fixture with fixed2e-6abs/1e-6rel score tolerance and exact row/count identity. The end-to-end fixture uses actual synthetic Horizon→A/B checkpoint files; unmatched steps, modified name maps and input hashes are rejected. All15 local adaptation CPU tests passed after the refactor. [Receipt](../results/layer_dependency_20261120/adaptation_pipeline_cpu.json).

This does not load the historical B200 weights or admit GPU execution. Time-course/activation-geometry aggregation, actual new row arrays and current-node/GPU gates remain. Score buffers scale with rows×cases; activation buffers scale with evaluation batch. CPU checkpoint/model-copy memory and runtime are not production-qualified.

### Activation geometry and matched timecourse, tick125

`adaptation_geometry.py` captures fixed, explicitly selected token positions separately by execution slot and block identity. It accumulates float64 centered cross-moments without retaining observations across batches and reports raw RMS, squared coordinate distance, mean cosine, mean shift, centered linear CKA and scatter spectra. Zero norms/zero scatter produce explicit undefined metrics, not apparent similarity. A rotation/scale/translation fixture has CKA1 while raw coordinates differ: CKA alone cannot establish compatibility with a fixed downstream reader. These descriptors do not attribute loss or identify the mechanism; retain targeted interventions and ordinary-continuation controls.

`adaptation_timecourse.summarize` replays each existing score panel, then requires a step-zero baseline, increasing updates, identical parent/common-recipe/data/config/cases/batch, exact row IDs and denominators. Parent reference and fork scores are checked. It exposes ordinary gain, swapped gain, their difference and remaining gap, alongside all intervention contrasts. Excess gain equals initial minus current paired gap algebraically; it is not itself evidence of a particular mechanism. The common-recipe hash now accompanies pinned snapshot provenance.

Nineteen CPU tests pass, including direct-vs-streamed geometry, alignment/degenerate guards, and an actual *synthetic* Horizon→A/B checkpoint-chain join at0/4 updates. [Receipt](../results/layer_dependency_20261120/adaptation_geometry_timecourse_cpu.json). No historical B200 weights were loaded. Geometry still needs a batch driver, snapshot/output integration and raw-moment replay. Forty-eight d768 scatter triplets alone consume about648MiB; registered selections and memory/runtime qualification must precede production use. New continuation/calibration/confirmation arrays, current-node/runtime ownership and GPU gates remain. Poller HOLD is unchanged; no remote request was sent.

### Geometry batch output and replay, tick126

`adaptation_geometry_pipeline.py` adds four explicit comparisons: parent-canonical vs parent-swap (immediate), parent-canonical vs A-canonical (ordinary drift), parent-swap vs B-swap (swapped drift), and A-canonical vs B-swap (remaining difference). These remain different questions, not interchangeable causes. The caller registers slot/block fields and token positions before measurement; selected dataset/boundary/row files are hash-checked, and row order is retained. Matching model configurations and frozen CPU execution are required.

The driver accumulates selected moments across batches, writes lossless sufficient statistics plus scalar reports, then replays the reports from those moments. Raw moments remain node-side and are explicitly excluded from small exports; they are not per-token activation exports. The default64MiB budget guards persistent moment tensors only, not model copies, native traces, temporary buffers or eigensolver memory. It is not production RSS qualification. Caller checkpoint provenance is recorded, not independently authenticated by this driver; actual use must obtain it from the pinned snapshot loader.

Six relevant CPU tests passed, including all four comparison definitions, batch2+1 versus direct3 within2e-6, exact row order, immutable weights/RNG, report hash and rehashed-numeric tamper rejection, buffer budget and moment-schema guards. [Receipt](../results/layer_dependency_20261120/adaptation_geometry_pipeline_cpu.json). Replay verifies sufficient statistics against scalar output; it does not rerun the original model. Actual new rows, full snapshot orchestration/geometry timecourse joining, node/runtime/ownership checks and GPU precision/fullstate gates remain. No actual checkpoint measurement, causal finding or remote action occurred.

### Matched snapshot-study assembly, tick127

`adaptation_study.py` now consumes an ordered step-zero-plus-later list of existing pinned paired snapshots. It prechecks checkpoint/data hashes, loads one parent/A/B panel at a time, runs the30-case loss panel and four separately labelled geometry comparisons, and joins their trajectories. Geometry must match the loss panel's exact checkpoint provenance, dataset hashes, model configuration and selected row order. Geometry keys/positions/alignment remain fixed across snapshots. No train command, automatic retry/resume or GPU path is added.

The study manifest pins every child manifest. Replay first checks each child's score/moment data, then checks registration against the study config and rebuilds the joined report. Relocating the output directory preserves replay; original checkpoint paths remain provenance metadata. Large raw moments stay node-side. A fresh output is required and partial evidence is preserved on failure.

Three integration tests passed through synthetic Horizon80→A/B0,4 checkpoints, including zero drift at the cloned fork, matched loss/geometry output, portable replay, cross-snapshot geometry mixup rejection, rehashed report/config tampering and preflight input guards. [Receipt](../results/layer_dependency_20261120/adaptation_study_cpu.json). These tests do not measure the historical models or identify a cause. Actual new row arrays, production resource/precision/runtime/ownership admission and controlled execution remain; poller HOLD was left intact.

### Concrete row plan and actual-byte reservation gate, tick128

`resources/layer_adaptation_reservation_20261121.json` pins the common historical E2 stream and partial exposure inventory. It registers2,000×32 training rows per seed (same array/order for A/B within a seed),256 calibration and512 confirmation rows, selected with seed20261121. The four sets are row-disjoint; two seeds have disjoint new training selections. These are intended counts, not arrays already created. Counts support a bounded exploratory paired study, not a precision or generalization guarantee.

`adaptation_reservation.py` requires actual token/boundary file hashes and the authenticated known-exposure inventory, checks stream extent, filters row eligibility using actual interior starts, and selects without replacement. It writes arrays/hashes and verifies pairwise/known-inventory row exclusion. It never uses reset row-start flags as original-document identity and never claims complete historical exclusion. Replay checks saved arrays, hashes and overlap; production admission must reverify source bytes and boundaries. A/B must consume the same per-seed array.

Three CPU tests passed on synthetic data: deterministic selection, rejection of interior-boundary rows, disjoint splits and known exclusions, hash/rehash-overlap guards and invalid row-space/capacity rejection. [Receipt](../results/layer_dependency_20261120/adaptation_reservation_cpu.json). Real parent stream hashes agree across both seeds and the known exposure proof; the reconstructed E2 order and exposure NPZ hashes were checked locally. Actual E2 token/boundary files are absent at configured local paths. The small exclusion NPZ is local but its node binding is not yet established. Therefore no actual new row arrays or admission READY exist; no remote request was made. This limitation does not require another full historical-document audit before exploratory paired analysis.

### Reservation-to-trainer binding, tick129

The previous trainer checked train/confirmation overlap but did not bind calibration or a reservation manifest. `adaptation_reservation.bind` now explicitly constructs row bindings from a caller-pinned manifest and seed, while retaining the registered training policy. `validate_binding` rejects substituted paths/hashes and mismatched policy. The trainer validates the binding before parent loading, hashes calibration as well as train/confirmation, checks all supplied splits for overlap and boundary eligibility, and verifies reservation seed/length against the actual loaded parent before any update. Reservation source code joins the saved source hashes; binding and calibration pins survive exact resume.

Thirty-one CPU tests passed after the integration, including three new tests covering bound A/B uninterrupted-versus-resumed fullstates, exact minibatch order, wrong-seed/substituted-calibration/manifest tampering and overlap rejection in legacy CPU mode. [Receipt](../results/layer_dependency_20261120/adaptation_binding_cpu.json). Legacy unbound CPU fixture configs remain supported; this is not a production bypass because CUDA execution is still refused. Any production preparation must use the pinned binding and independently pass node/runtime/GPU gates. No actual row reservation or model finding exists yet; poller HOLD is unchanged.

### CPU preparation entrypoint, tick130

`adaptation_prepare.prepare` takes explicit parent directories, a pinned reservation plan, a pinned parent registration, snapshot schedule and fresh output. It loads/releases one parent/fullstate/optimizer at a time, preserving caller Torch RNG. It verifies actual five-file parent identities, original source/schema/Adam, seed, batch, model/length, terminal LR and matching continuation-stream hashes. Only then does it reserve actual eligible rows and write four reservation-bound A/B configs. A/B configurations within each seed differ only in branch. The prepared manifest pins inputs/configs/source and explicitly reports `CPU_PREPARED_NOT_GPU_ADMITTED`; it does not launch training or establish ownership.

Two CPU integration tests use two synthetic80-step Horizon parents and actual fixture token/boundary files, exercise preparation through a generated config's first continuation update (inherited Adam counter81), verify immutable parents/caller RNG, and reject altered pins/schedules/wrong parent seed before reservation. [Receipt](../results/layer_dependency_20261120/adaptation_prepare_cpu.json). This is not historical B200 input admission, actual E2 reservation or a model finding. Runtime/ownership and GPU precision/fullstate qualification remain separate, and the trainer still refuses CUDA. No remote action or poller restart occurred.

### Runtime integration after monitor recovery,21November

Read-only node preflight now verifies actual two-parent file hashes and corpus hashes, runtime and eight native GPU workers; this is input/ownership evidence, not mechanism evidence. Actual CPU preparation is submitted separately; see CURRENT_TASK for its receipt.

The development trainer now contains an explicit single-visible-CUDA path with BF16 autocast, FP32 parameters/Adam, TF32 disabled, optimizer reconstruction after moving parameters, and restoration of the inherited CUDA RNG before work. A qualification call requires a bound reservation and stops no later than phase step40. Full GPU training requires a hash-pinned receipt matching code/runtime/reservation/parent and all four branch40-step finite/exact-resume/step0 checks plus native-component parity. Ownership remains an external mandatory gate. Five CPU regression/entry-guard tests pass; **no CUDA run or GPU admission is established**. The already-submitted CPU package remains frozen at its earlier source version; do not overwrite its manifest to match newer development code. GPU numerical gate generation, evaluation/geometry CUDA integration and owned execution orchestration remain to be completed.

### Registered CUDA measurement and bounded qualification

Measurement is FP32 with math SDPA and TF32 disabled on CUDA; training retains BF16 autocast with FP32 Adam/parameters and TF32 disabled. CUDA component/self-patch and CPU-versus-CUDA hidden/NLL tolerances are fixed at absolute1e-4/relative1e-4 before observing node results. CPU tolerances remain2e-6/1e-5. Runtime policy is saved with score and geometry manifests and matched across trajectory panels. Selected GPU activations alone transfer to CPU double-precision moment aggregation; large moments stay node-side.

For each A/B×seed, the qualification job loads the actual pinned parent, checks one prespecified calibration sequence in both orders, both directions of attention-pattern/value and SwiGLU gate/up interventions, all30zero-step score cells and raw replay, and selected slot/block geometry with sufficient-statistic replay. This single-sequence check establishes numerical execution only, not a mechanism/generalization finding. It then runs40updates uninterrupted and20+resume40 from identical inherited parent states, comparing model, Adam, counters, CPU/CUDA/NumPy/Python RNG and exact first40row/loss traces, plus step0 versus parent. Each branch executes80qualification updates; four branches execute320total. They are separate outputs and are not reused as the scientific continuation.

The dedicated owned queue performs only these four gates, retains logs/fullstates, and returns GPUs to the verified native workload. No automatic2000-step launch follows. Production requires all four pinned receipts and separately verified native handoff. Launcher checks fresh process identities, source/prepared hashes and free disk before any handoff. Local36true-frozen CPU tests plus one aggregation test passed; actual GPU qualification is still pending submission/results. Source package manifest resources/layer_adaptation_gpu_source_20261121.json is separate from immutable CPU preparation package.


### Actual GPU qualification a01 and correction, 21 November 15:32 UTC

Actual Horizon D parents, data reservation and all four CUDA jobs executed. FP32
CPU/CUDA forward, native component reconstruction, 30-cell scorer and geometry
replay passed the registered tolerances on one calibration sequence per seed.
This is instrument qualification, not a representative mechanism result.
All four 40-step whole-versus-20+20 resume comparisons failed exact tensor
identity; their last losses also differed. Parent step0 identity passed before
that failure. All original artifacts are retained and native8-worker/guard
handoff was verified from the subsequent health receipt.

The initial trainer left CUDA training SDPA and deterministic-algorithm policy
implicit. The a02 correction makes both branches use deterministic algorithms
with errors enabled, math SDPA, BF16 training, TF32 off, and CUBLAS workspace
`:4096:8` configured before CUDA initialization. This is a new recorded runtime
policy, not a relaxed equality tolerance or a claimed isolated diagnosis of
the a01 cause. Parent Adam/RNG/LR, rows, schedules, paths and exact40 gate remain
unchanged. Full continuation requires fresh successful gates with this source
and policy. Forward measurement remains FP32. No 2,000-step run yet.


### Registered continuation orchestration (local, pending GPU gate)

The continuation entry, owned queue and small-artifact finalizer are separate
from the gate. Four2,000-update fits use the prepared A/B seed0/1 configs and
require an exact hash-pinned gate with matching source/runtime/parent/rows.
The two seed studies retain snapshots0/10/50/200/500/1000/2000, the existing30
loss cases at each snapshot, and four geometry comparisons. Geometry selection
is fixed to slot2/3 inputs and block2/3 attention/FFN updates at token positions
31/127/255/511, using512 confirmation rows/seed, batch1 and128MiB moment buffers.
No data-dependent case selection is added. Parent component cells are repeated
controls; they do not measure changed components in the adapted checkpoints.
Weight rollbacks and normal-training controls supply adaptation evidence.

Before an execution push, require actual gate pass/current native ownership and
100GiB free disk. The queue preserves32 fullstate copies on-node (seven snapshots
plus latest for each branch), replays studies on-node, packages only score arrays,
row IDs and summary/identity JSON under25MB, and restores verified native work.
Raw moments and model states are excluded from small exports. Five relevant
CPU tests passed for delivery binding/export selection plus synthetic study
assembly; these do not qualify a new actual GPU run. No full continuation has
been launched as of this entry.

### Authorized bounded path-background pilot (21 November)

User explicitly authorized implementing the proposed methods. First implementation
is frozen-parent, same two orders, one post-pair boundary; not an SAE/LEACE/router
sweep and no additional training. For each original parent seed: fit on16 repeat
anchors from calibration rows0:16 plus16 periodic anchors from16:32; holdout16
shuffled anchors32:48 and16 natural rows48:64. Positions31/127/255/510, centered
rank8 plus separate mean. Freeze three random controls seeds4101/4102/4103.
Score first128 existing confirmation rows, explicitly reused/exploratory. Both
orders receive identity, plus/minus mean, projection, norm-restored projection,
and three random projections with separate removed-energy-matched controls
(22 cases). Report family-transfer and all prespecified functional contrasts
together; this pilot does not use an arbitrary overlap threshold to silently
select a successful result. Instability or null results are retained. Future
independent confirmation or deployment requires evidence from this pilot.

CPU qualification covers exact native suffix/NLL, causal masking, no mutation,
known-rank/degenerate examples and a complete synthetic checkpoint/input/replay
chain. Actual parent CPU reference plus new CUDA boundary/score parity is required
before GPU pilot. Local basis and model states stay on node. No new optimizer
updates, no modification/restart of the existing adaptation study.

### Follow-up: fixed-context input/block/reader factorial (21 November)

Motivation is the actual pair/rest rollback result, not a new generic sweep.
Adapted pair and adapted rest each help, but rest conflates upstream recruitment
and downstream consumption. The failed null-anchor removal does not resolve
this ambiguity. The following freezes existing checkpoints; no further training
or fitted alignment is required.

The first target is **middle.2**, identity fixed in the swap path `(0,2,1,3)`,
matching the previous computational intervention target. Compare parent P with
B2000, and separately parent P with A2000 as the ordinary-training control.
Both panels execute the same swap path; A2000 is deliberately tested away from
its training order. Do not silently compare different inventories or depths.

For each panel let H_i be embedding→stem→middle.0 from weights source i, F_b
be middle.2 from source b, and R_r be middle.1→middle.3→coda→final norm→readout
from source r, with bits0=parent and1=that panel's adapted model. On identical
tokens/masks compute all eight cells `L_ibr = NLL(R_r(F_b(H_i(x))))`.
There are four matched skips `L_iSr = NLL(R_r(H_i(x)))`. Twelve cells per
panel,24 across A/B per seed; no outcome-based target selection.

- Input effect at fixed block and reader: `L_0br − L_1br`.
- Block-weight effect at fixed input and reader: `L_i0r − L_i1r`.
- Reader effect at fixed input and block output: `L_ib0 − L_ib1`.
- Useful block contribution: `L_iSr − L_ibr` in that exact context.
- Input/block interaction for reader r:
  `(L_10r − L_11r) − (L_00r − L_01r)`.

All signs use lower loss as improvement. Report each reader panel and sequence
effects; a change in marginal block benefit is not total-model improvement.
The telescoping total from000 to111 depends on the chosen traversal, so its
terms are not unique percentages of cause. Source models all remain frozen.

**Readout caveat:** R includes its own embedding-tied output matrix. When input
source differs from reader source, the forward uses different embedding and
readout values. This is an explicit hybrid intervention, not a deployable tied
model or proof of a downstream-layer-only effect. It also changes logit
calibration; same-reader native references alone are used for KL. If reader
effects warrant finer localization, separate suffix weights from output-head
weights in a later bounded control. No such inference is made from this stage.

New `adaptation_context.py` implements native000/111 parity, shared input reuse,
fixed block identity, controlled suffix/readout and matched skips.
`adaptation_context_pipeline.py` adds exact-schema parent/A/B snapshot loading,
row-hash binding, streaming24cells, paired effect distributions and small raw
replay, with config/checkpoint/row/runtime joins. Six CPU fixture tests cover
independent hybrid/NLL reconstruction, actual head source, causality, weight/RNG
immutability, identity-block skips, a joint-only interaction counterexample,
streaming parity, and data/config/output tamper rejection. **Not actual-model
evidence or production admission.** Concrete terminal2000 parent/phase/row pins
come from the verified previous export and are registered in
`resources/layer_adaptation_context_20261121.json`; no new row reserve or train.
Remaining: frozen-source node package and actual-parent CPU references,
CPU/GPU parity and fresh owned node admission before the24-cell measurement.
Existing scientific results stay unchanged. The local CPU fixture's driver
discovery warning is retained in its log, not evidence of a B200 fault.


### Actual CPU qualification,21Nov18:44UTC

Node CPU receipt2bb543f8 verifies two real parent/A2000/B2000 panels and
24 full hidden/NLL/count reference cells per seed on a calibration row excluded
from confirmation. Prepared SHA6d45f830 and297 frozen source pins bind the
subsequent inference run. Seven actual-node CPU tests passed. GPU worker must
compare every reference tensor under unchanged abs/rel1e-4 policy, then evaluate
the registered512 rows. This CPU qualification alone is not a crossover result.

Local export verifier replays all score-derived effects and joins prepared,
config, gate, checkpoint, row and runtime hashes. It also compares native000/111
endpoints with the previous study's parent/swap, A/swap and B/swap scores on the
same row IDs and token counts. These endpoints test cross-study consistency;
they do not establish model-independent generalization.

Actual follow-up complete21Nov19:05UTC local verification: see results document
section8. Two24-cell panels/seed,512rows, CPU/GPU parity and19member replay
passed;8 native endpoint comparisons agree with the prior study. The stage
above is now measured, not merely proposed. Future attention/FFN localization
is a separate unimplemented stage and must retain fixedH0/R0 and A controls.


### Follow-up: attention/FFN weight localization at fixed H0/R0

Registered before measurement,21November. Use the same parent/A2000/B2000,
512 DCLM rows, two seeds and target middle.2 in path[0,2,1,3]. H0 and R0
including output head are ALWAYS parent. For each A/B panel compute four
cells L_af and whole-block skip LS. The attention group includes input RMSNorm
and all attention parameters (Q/K/V/O and Q/K norms); FFN group includes
post-attention RMSNorm and SwiGLU gate/up/down. These two disjoint groups
exhaust the target block's state. No weights are fit or altered.

Let z_a=H0+Attn_a(Norm_a(H0)), y_af=z_a+MLP_f(Norm_f(z_a)).
L_af=NLL(R0(y_af)). Attention-only gain=L00-L10; FFN-only=L00-L01;
joint=L00-L11; interaction=joint-attention_only-ffn_only. Report conditional
FFN gain L10-L11 versus L00-L01, and matched usefulness LS-Laf. Each FFN
weight comparison uses the SAME z_a; changing attention changes the FFN input,
so the attention effect includes downstream FFN response, not attention-only
information. Group norms are part of each intervention; this does not isolate
attention patterns/values or FFN gates/up directions. A controls ordinary
training changes, not equal-norm or random-update effects.

Five cases per panel/ten perseed; parent and joint endpoints must reproduce
prior context000/context010 and skip0S0 respectively on exact rows. Full
hidden/NLL/count CPU references on excluded calibration row, unchanged GPU
parity, source/runtime/ownership, score replay and native handoff remain
required. Raw activations and model state stay node-only. All conclusions
remain conditional/two-seed/reused exploratory, not additive causal shares.
