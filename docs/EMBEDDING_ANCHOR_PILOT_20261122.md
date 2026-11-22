# One-block warm-up and embedding-anchor pilot

22 November 2026. Concrete design draft; core forward/schedules now implemented and CPU fixture-tested.
No actual-data registration, production numerical admission or remote launch yet. Source idea:
`docs/EMBEDDING_ANCHOR_AND_TIED_TRANSLATOR_20261120.md`, section1 and the user's
clarification that the assembled stack WILL train further. This follows the
completed bounded correction trial, not an extension of its bias search.

## Question and scope

Does an embedding-mixing curriculum followed by repeated embedding anchoring
improve task quality and useful block composition after joint stack training,
relative to direct hidden-state composition from the SAME one-block warm-up?
Section12's constant correction is a path-specific baseline, not a portable
solution. Its negative transfer motivates testing a shared-input training idea;
it does not establish that anchoring will work.

Two arms identify the effect of the curriculum-plus-anchor PACKAGE conditional
on one-block warm-up. They do not isolate curriculum from anchoring or establish
that warm-up itself beats ordinary dense training. Do not claim otherwise or
add factorial arms automatically. No TG, router, STOP, domain split or new rank
search in this pilot.

## Exact proposed computation

Use four independently initialized dense causal blocks B0..B3, shared token
embedding E and shared tokenwise head H. No Transformer stem or coda. Reuse
the native dense Block, RMSNorm, SwiGLU MLP and suffix-loss conventions; the old
warm_order.Model cannot be used unchanged because it inserts stem/coda.

Proposed dimensions match the existing registered baseline: d768, FF2048,
12 query heads/4 KV heads, head_dim64, vocab50304, length512, scored tokens
32..511. New model initialized from scratch for seeds0/1; no pretrained dense
embedding or hidden body silently imported. H(h)=E.weight times
RMSNorm(MLP(RMSNorm(h))), using a shared tokenwise SwiGLU FFN of width2048.
This is an explicit choice for the user's final FFN, not another attention block.
Initialization and all architecture fields must be pinned before execution.

Phase1, shared warm-up: e=E(x), logits=H(B_i(e)); rotate i=step mod4.
Only the selected block plus shared E/H receive updates. Blocks are distinct
weights; optimizer state for inactive parameters must not advance. Save a full
warm-up checkpoint and fork weights/Adam/RNG/row cursor identically into A/C.

Phase2, donor exposure: schedule all16 ordered (recipient i, donor j) pairs
equally, including self. For both arms compute d=stopgrad(B_j(e)) on the SAME
causal sequence. This is a detached current-model donor, not an adapted external
model or cached target. Then recipient input is (1-alpha)*e+alpha*d.
Anchor arm A linearly increases alpha from0.05 to0.5 over this phase; control C
uses alpha=1 throughout. E/H and the recipient train; the donor invocation is
no-grad even for i=j. Both arms pay for the donor forward.

Phase3, joint stack adaptation: h1=B_p1(e), then
h_k=B_pk((1-alpha)*e+alpha*h_(k-1)) for k2..4.
A uses alpha0.5, C uses alpha1. All blocks and E/H train jointly. Preserve BOTH
native attention and FFN residuals inside B; do not add an extra local carry.
No embedding blend after the final block before H. No additional normalization
at the mixing boundary. Warm-up sees only one-step donors; stack adaptation is
required to expose multi-step states and cannot be replaced by frozen testing.

## Budget and paths — proposed fixed choices, not tuned winners

Per seed: shared phase1 768updates, then each arm phase2 384updates and phase3
768updates. Batch32 throughout. Global LR0.0006 with linear ramp over first
128 shared warm-up updates, then constant; AdamW(.9,.95), eps1e-8, wd0,
gradient clip1. These are feasibility-pilot choices, not an optimal recipe.
No extension or best-checkpoint selection from final measurement.

Each arm's full lineage uses61440 ordered training rows, fitting within the
existing64000-row reservation: phase1 first24576, phase2 next12288, phase3
next24576. A/C share exact rows and route schedules. Actual arrays, whole-stream
hashes, ancestry and splits MUST be bound before running; this design does not
assert current availability. Existing256 calibration/512 measurement rows may
be reused with explicit exploratory/history-overlap flags. No fresh-test claim.

Shared warm-up costs768updates per seed, not twice. Physical training totals
3072updates/seed and6144 across two seeds,196608 sequence exposures,100663296
input-token exposures. Also report donor forwards, attention/FFN/readout work,
qualification cost and measured runtime; token count alone is not compute.
No estimate of wall-clock completion is available before measurement.

At depth4 train lexicographically ordered EVEN permutations of four blocks,
cycling all12. Evaluate all24, reporting the12 odd paths separately. A local
combinatorial check found each partition has each block in each slot3 times,
and every directed unequal neighbor pair3 times. Thus held-out odd paths have
unseen complete order but no unseen adjacency or slot assignment. This does
not isolate every higher-order interaction or prove arbitrary portability.
Phase2 all-pair exposure also means pairwise novelty is not the claim.

## Decision evidence

Record shared warm-up, post-donor, stack0/192/384/768 snapshots. Calibration
curves can diagnose optimization, but primary measurement uses terminal768.
Evaluate fixed path lists with no label-based routing or best-path selection.

Primary: paired row NLL difference A-C averaged separately over even and odd
paths, per seed. Also report absolute NLL, worst-path mean, per-row spread and
train-to-held-out path gap. A smaller path gap is not success if all paths get
worse. Short training from scratch may be undertrained: a negative pilot is
limited to this budget, not a refutation of the user's architecture.

Useful composition: on two predeclared paths (0,1,2,3) and (0,2,1,3), replace
each block by identity at its input, preserving all remaining blending steps
and the suffix. Compare each skip with its own unmodified path. Report signed
per-row loss changes rather than declaring usefulness from a mean near zero.
This is conditional contribution, not intrinsic knowledge or additive causes.
Compare full depth4 and standalone-block losses descriptively; differences
also change compute/context, so they are not pure depth effects.

An anchor that reduces path variability by making blocks irrelevant fails the
intended objective. Read quality, held-out composition and skip contributions
together. Two seeds and reused measurement rows support exploratory evidence
only. Do not invent a significance/quality threshold after seeing results.

## Remaining implementation and admission

Implement the no-stem/no-coda model and shared head in a separate module;
preserve old models/results. Test independent reference arithmetic, alpha1
direct-chain parity, causal suffix masking, donor detach including self-donor,
active/inactive Adam steps, fork and resume including shared E/H, route/row
schedules and streaming score replay. Numerical tolerances remain unchanged.
Actual CPU/GPU admission must use this architecture and actual bound data;
old correction gates do not admit it. Inspect fresh ownership/runtime/source,
retain native infrastructure and all old backups, and export small reports only.
Weights/optimizer/RNG stay node-side. No launch follows from this document alone.

Implementation update22Nov: `embedding_anchor.py` and six unittest fixture cases pass on CPU. No-stem/coda forward, tied readout, explicit skip, donor detach and inactive-Adam behavior checked. Full trainer fork/resume, data binding and streaming evaluation remain. CUDA refused by this prototype; no actual-model evidence.

CPU trainer update22Nov: `embedding_anchor_train.py` provides a tiny-fixture-only trainer with fixed1920-step lineage, immutable hash-pinned fullstate snapshots, shared768-step fork, named optimizer identity/per-parameter counters and RNG restore. Nine combined tests pass, including both donor and stack resume for A/C. Production streaming data bindings and24path evaluation/replay remain. Initialization in current model uses normal(std0.02) for all Linear/Embedding weights and native unit RMSNorm weights; no pretrained weights imported. No actual-model result or remote launch.

Data/evaluation update22Nov: `embedding_anchor_data.py` authenticates the existing recipe, input hashes and row arrays before read-only streaming;61440 selected training rows retain the historical exposure caveat. `embedding_anchor_eval.py` implements24paths+8identity skips+4single-block cases perarm, signed tokenweighted paired scores and portable raw-score replay. Twelve combined CPU tests pass on synthetic files/models only. Evaluation remains fixture-only; actual terminal-checkpoint loading, trainer streaming integration and actual CPU/GPU admission are pending. No corpus or production model has been run for this pilot.

Streaming/checkpoint update22Nov: `CPUStreamTrainer` reuses the tested update/fork engine over BoundRows; nonfixture CPU restricted to40 qualification steps. `evaluate_checkpoints` authenticates common warm-up768 ancestry, same recipe/data and A/C identities; primary nonfixture measurement requires globalstep1920/all512rows.14combined synthetic CPU tests pass. Concrete recipe/pins are in `resources/embedding_anchor_pilot_20261122.json` (SHA d8fe493abf07508df5c3f922b92887516c14c26b07198e05d8385972463297d8). Data pins refer to previous verified artifacts, not current node availability. Actual CPU preparation, GPU admission and execution orchestration remain; no new remote job or model evidence.
