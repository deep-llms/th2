# Nested Ladder Embedding — Design & Implementation Specification

Status: 2026-08-28, design approved for implementation. Phase 1 (static
ladder) is implementable now; Phase 2 (learned reallocation) is specified but
gated behind Phase-1 results. This document is written so an agent can
implement Phase 1 without reading prior discussion.

Implementation status (2026-08-29): **Phase 1 implemented and locally
validated.** The model/head live in `compositional/nested_ladder.py` and
`compositional/tied_head.py`; CLI, strict loading/resume, diagnostics, matched
GroupReduce populations, runner entries, and both B200 launch scripts are
wired. `compositional/test_nested_ladder.py` covers the §4 invariants. Phase 2
remains intentionally unimplemented pending the Phase-1 result gate.

---

## 1. One-paragraph summary

The nested ladder replaces the vocabulary embedding table (and, via exact
tying, the output head) of a from-scratch Qwen3-0.6B with a **stack of shared
bases ("tiers")**: tier 1 is used by every token; each later tier is used only
by a progressively smaller, more frequent subset. A token's embedding is the
sum of its contributions from every tier it belongs to, so per-token capacity
varies with **depth** — from rank-64 for the tail up to effectively dense for
the top ~2k tokens — while all tokens share the coarse directions. This is
the "variable per-token capacity" escape from the measured dead end where all
uniform-capacity tied structures (global LR128 / SharedLocal / PureLocal)
landed within 0.12 mean PPL of each other.

Design rationale in two sentences: partitioned schemes (GroupReduce-style
blocks) make every block re-learn the same dominant directions in a private
basis — and our PureLocal-vs-global result showed independent block bases buy
nothing — while nesting learns the shared directions once and spends the
savings on refinement tiers for high-traffic tokens. Tail tokens additionally
train on a coarse basis that receives the entire corpus's gradient, targeting
the documented tail-degradation failure of compressed arms.

---

## 2. Method definition

### 2.1 Notation and parameters

- `V = 151,936` (vocab), `d = 1024` (hidden), `T` tiers.
- Tier ranks `r_1 … r_T`; tier populations `N_1 … N_T` with
  `N_1 = V > N_2 > … > N_T`, each tier's member set being a **subset** of the
  previous tier's (nested memberships).
- `depth(i)` = deepest tier containing token `i`.

Learned parameters:

- Coefficients `Z_t ∈ R^{N_t × r_t}` for each tier (row = one member token's
  mixing weights on tier t).
- Bases `W_t ∈ R^{d × r_t}` (stored `(d, r_t)`, matching the GroupReduce
  `right_factors` convention).
- One global bias `b ∈ R^d` (matching LowRank/PureLocal: a single
  d-dimensional input-side bias, no per-tier or per-token bias).

Definition:

```
e_i = b + Σ_{t=1}^{depth(i)} Z_t[slot_t(i)] @ W_t.T          # (d,)
```

where `slot_t(i)` is token i's row index inside tier t's member list.

Equivalent global view (useful for analysis, not for implementation): the
effective table is `E = Z̃ · W̃ + b` with `W̃ = [W_1|…|W_T]` stacked
(`d × Σr_t`) and `Z̃` a `V × Σr_t` coefficient matrix whose support pattern is
**nested** (tail rows use only the leading `r_1` columns). GroupReduce is the
same object with a block-diagonal support pattern and independent per-block
bases.

### 2.2 Reference configuration (Phase 1)

| Tier | rank r_t | members N_t (most-frequent prefix) | cumulative rank |
|---|---:|---:|---:|
| 1 | 64 | 151,936 (all) | 64 |
| 2 | 128 | 32,768 | 192 |
| 3 | 320 | 8,192 | 512 |
| 4 | 512 | 2,048 | 1024 |

Deliberate property: the deepest tokens own `64+128+320+512 = 1024`
coefficients over a (generically) full-rank stacked basis — the ladder
interpolates from **effectively dense** (top 2,048 tokens) down to rank-64
(tail), with no hard capability cliff between adjacent tiers.

Exact parameter count (verify in a unit test with this closed form):

```
coefficients: Σ_t N_t·r_t = 151936·64 + 32768·128 + 8192·320 + 2048·512
            = 9,723,904 + 4,194,304 + 2,621,440 + 1,048,576 = 17,588,224
bases:        d·Σ_t r_t   = 1024·1024                        =  1,048,576
bias:                                                            1,024
total interface params:                                       18,637,824
```

That is **95.2% of the global-LR128 budget** (19,579,904) → 8.35×
compression of the dense 155,582,464 interface. Running strictly *under* the
reference budget is intentional (comparisons can only be challenged in our
favor); the tier-2 population is the knob if a closer match is ever needed.
⚠️ Report the exact count logged at startup, never the budget target.

### 2.3 Depth assignment (Phase 1: static, frequency-based)

- Load counts from `resources/token_freq_sample10.npz` (key `counts`,
  int64[V]) — the same artifact the GroupReduce arm uses.
- Rank tokens by `(-count, token_id)` (the id tiebreak makes assignment fully
  deterministic, including among zero-count tail tokens).
- Tier t's member set = the first `N_t` tokens in this ranking. Nesting is
  automatic because the `N_t` are decreasing prefixes of one ranking.
- Within each tier, store member token ids **sorted ascending by token id**.

No RNG is consumed for structure (matches the repo rule that structural
mappings must not advance the global init RNG).

### 2.4 Structural buffers (all `persistent=True`, saved in `embedding.pt`)

For each tier `t ≥ 2`:

- `member_ids_t`: int64 `(N_t,)` — ascending token ids of tier members.
- `member_slot_t`: int64 `(V,)` — token id → row in `Z_t`, **−1 for
  non-members**.

Tier 1 needs no buffers (`slot_1(i) = i`). Buffers are registered under the
literal names `member_ids_{t}` / `member_slot_{t}` (t = 2…T); the module
exposes convenience properties (`self.member_ids`, `self.member_slots` —
lists indexed by `t−2`) that the pseudocode below uses. Log the counts-file
SHA256 once at startup for provenance; the saved buffers themselves are the
authoritative structure (a changed counts file cannot silently alter a
resumed run, because `load_state_dict(strict=True)` restores the original
buffers). A `validate_structure()` method should verify `member_slot_t` is
the exact inverse of `member_ids_t` and that memberships are nested —
mirroring `GroupReduceEmbed`'s consistency checks.

### 2.5 Input forward (returns `(e, None)` like every no-router arm)

```python
def forward(self, input_ids, doc_mask=None):
    flat = input_ids.reshape(-1)                          # (M,)
    e = self.Z[0][flat] @ self.W[0].T                     # (M, d)  tier 1
    for t in range(1, self.T):                            # tiers 2..T
        slot = self.member_slots[t - 1][flat]             # (M,) −1 = non-member
        mask = slot >= 0
        # Empty-selection matmuls still connect Z[t]/W[t] to autograd —
        # required with DDP find_unused_parameters=False (GroupReduce pattern).
        contrib = self.Z[t][slot[mask]] @ self.W[t].T     # (M_t, d)
        e = e.index_add(0, mask.nonzero(as_tuple=True)[0], contrib)
    e = e + self.bias
    return e.view(*input_ids.shape, self.d), None
```

(Out-of-place `index_add`/masked scatter, or an equivalent masked in-place
write into a fresh tensor as `SharedLocalEmbed` does — either is acceptable;
what is mandatory is the empty-selection graph connection for every tier on
every microbatch.)

### 2.6 Exact tied head (`TiedNestedLadderHead`, added to `tied_head.py`)

Subclass `_TiedHeadBase` (non-registering embed reference — do NOT register
the embed module again under `lm_head`).

```python
def forward(self, hidden_states):
    flat_h = hidden_states.reshape(-1, self.embed.d)      # (M, d)
    logits = (flat_h @ self.embed.W[0]) @ self.embed.Z[0].T   # (M, V)
    for t in range(1, self.embed.T):
        partial = (flat_h @ self.embed.W[t]) @ self.embed.Z[t].T  # (M, N_t)
        logits = logits.index_add(-1, self.embed.member_ids[t - 1], partial)
    logits = logits + (flat_h @ self.embed.bias).unsqueeze(-1)
    return logits.view(*hidden_states.shape[:-1], self.embed.V)
```

This is exact tying: `logits ≡ hidden @ (effective table).T`, including the
`h·b` scalar that the single input bias contributes equally to every class
(same convention as `TiedPureLocalHead`). FLOP count per position is
`Σ_t (d·r_t + r_t·N_t)` ≈ 18.6M multiply-adds — slightly *cheaper* than the
LR128 tied head (~19.6M).

Register in `make_tied_head` under arm name `nested_ladder`, and extend the
error message listing supported arms.

### 2.7 Initialization

All learned tensors `N(0, 0.02)`, bias zeros (repo-wide Qwen convention).
Known, accepted property: embedding norm at init grows ~`√(cum_rank)` with
depth (depth-4 ≈ 4× depth-1, still below dense's init norm). Log per-depth
mean embedding norms during training; an optional per-tier init scale is an
ablation knob, **not** part of the reference configuration.

---

## 3. Integration checklist (Phase 1)

1. **Module**: `NestedLadderEmbed` in a new `compositional/nested_ladder.py`
   (or appended to `compressed_baselines.py` — either, but keep the
   `(embedding, None)` forward interface and the docstring conventions).
   Export from `compositional/__init__.py`.
2. **Head**: `TiedNestedLadderHead` in `compositional/tied_head.py` +
   `make_tied_head` dispatch for `"nested_ladder"`.
3. **Args** (`CompositionalArguments` in `train_compositional.py`):
   - `--arm nested_ladder` added to choices;
   - `nested_tier_ranks` (str, default `"64,128,320,512"`);
   - `nested_tier_populations` (str, default `"151936,32768,8192,2048"`;
     validate: first entry equals vocab size, strictly decreasing, all > 0);
   - `nested_frequency_path` (default `resources/token_freq_sample10.npz`),
     `nested_frequency_key` (default `counts`).
4. **Output-mode validation**: add `nested_ladder` to the set of arms that
   **require `--tie_output`** in `validate_output_configuration` (same
   rationale as pure_local/pvq/slim/groupreduce/tt: keeping a dense lm_head
   would test a different architecture).
5. **`build_arm`**: construct from the args; when `initial_state` is given
   (resume path), structural buffers still come from the checkpoint via
   `load_state_dict(strict=True)` — build with the same flags and let the
   loader overwrite.
6. **Loading** (`compositional/loading.py`):
   - `_build_arm_from_config`: `nested_ladder` case;
   - `_infer_comp_config_from_state`: recognize the state-dict signature
     (keys `Z.0…`, `W.0…`, `member_ids_*`, `member_slot_*`, `bias`); infer
     ranks from `W.t` shapes and populations from `Z.t` shapes.
7. **Scripts/runner**: `scripts/train_nested_ladder_tied.sh` cloned from
   `train_groupreduce_e2e_tied.sh` (identical hyperparameters, output dir
   `nested_ladder_tied_t4`), plus a `run_experiments.py` entry with the
   standard `required_checkpoint_files` list.
8. **No changes needed** to `EmbeddingShim`, `SaveEmbeddingCallback`
   (`embedding.pt` = full module state incl. buffers), or
   `validate_resume_compatibility` (schema comparison covers the new keys
   automatically).
9. **Claim-1 control support**: add `--groupreduce_populations`
   (comma-separated block sizes summing to V; assignment by the §2.3
   frequency ranking; mutually exclusive with the equal-size
   `frequency_group_ids` path; from-scratch random init, no init artifact)
   so the matched partitioned control in §5 is launchable. Include a test:
   explicit populations produce blocks that are exactly the depth level-sets
   of the corresponding nested configuration.

---

## 4. Tests (write before the training harness; tiny dims, CPU)

Mirror the style of `compositional/test_compressed_baselines.py`:

1. **Tying exactness** — materialize `E` by running the input forward over
   `arange(V)` and assert `TiedNestedLadderHead(h) == h @ E.T` (fp64,
   tolerance ~1e-10).
2. **Budget closed form** — `sum(p.numel())` equals the §2.2 formula for the
   reference config and for a second, irregular config.
3. **Depth assignment** — deterministic under `(-count, id)` ordering;
   memberships nested; rebuilding from the same counts file yields identical
   buffers; a permuted counts vector yields the expected different buffers.
4. **DDP safety** — a batch containing only tail tokens still produces
   gradients (possibly zero-valued but present) for every `Z_t`, `W_t`.
5. **Reduction to LowRank** — with `T=1, r_1=128`, the module has exactly the
   LowRankEmbed parameter count, and after copying weights its forward and
   tied logits match `LowRankEmbed` + `TiedLowRankHead` numerically.
6. **Configless inference round-trip** — `_infer_comp_config_from_state` on
   the saved state reconstructs a module that loads `strict=True` and
   reproduces the forward.
7. **Grow no-op (Phase-2 precursor)** — reassigning a vacated `Z_t` row to a
   new member token *after zeroing it* (tensor shapes never change; see §6)
   leaves `e` and all logits exactly unchanged (fp32-exact: the new member's
   contribution is a product with an exact zero vector).

---

## 5. Experiments and claims (Phase 1)

All arms: matched B200 protocol (data manifest, tokenizer, seed 42, block
2048, global batch 512, checkpoint-10k screen, standard eval + finetune
battery + frequency-binned PPL).

| Arm | Role |
|---|---|
| Dense tied (retraining in progress) | ceiling / reference |
| Global LR128 tied (retrain) | uniform-capacity floor |
| **nested_ladder (this doc, static depths)** | **the method** |
| GroupReduce e2e, **matched profile** | claim-1 control (below) |
| GroupReduce e2e g20 (proportional ranks) | published-allocation baseline |

**Claim 1 (core): nesting > partitioning at matched per-token capacity.**
The matched control is GroupReduce with **explicit blocks equal to the depth
level-sets and ranks equal to the cumulative ranks**: blocks
(2,048 @ 1024) / (6,144 @ 512) / (24,576 @ 192) / (119,168 @ 64), with
`group_ids` derived from the same §2.3 ranking. ⚠️ **The current CLI cannot
launch this from scratch**: `frequency_group_ids(counts, G)` only builds
near-equal-size groups, and the `--embedding_init_path` route would also
inject converted weights (making it post-hoc, not from-scratch). Required
glue (add to the §3 checklist): a `--groupreduce_populations` argument
(comma-separated block sizes, frequency-ranked assignment reusing the §2.3
ordering, mutually exclusive with equal-size frequency grouping) feeding
explicit `group_ids` into `GroupReduceEmbed` with random init.

With that control, every token has the *same coefficient count* in both arms
(17,588,224 total); the only differences are stacked-shared vs independent
bases — the control's bases cost 1,835,008 vs our 1,048,576, and the control
has no global bias (`GroupReduceEmbed` is bias-free), so the **control is
heavier by exactly 785,408 parameters**. Report this; winning against a
slightly heavier control is the stronger result.

**Claim 2 (staged, Phase 2): learned depths > frequency depths.** Not part of
Phase 1; see §6.

Go / no-go: nested_ladder must beat global LR128 by clearly more than the
established ~0.12-PPL single-seed noise band, and beat the matched
partitioned control. Frequency-binned PPL should show the mechanism's
signature (head bins recovering toward dense at little tail cost — the tail
sharing argument predicts the tail should *not* regress vs LR128). Any
claimed win eventually needs a second pretraining seed.

---

## 6. Phase 2 — learned, function-preserving depth reallocation (gated)

Do **not** implement with Phase 1. Specified here so Phase-1 code leaves room
(the grow no-op test, per-tier index buffers rather than prefix-contiguous
storage).

- Tensor shapes are **fixed** (`N_t` never changes); a reallocation is a
  one-for-one row swap inside a tier.
- **Shrink** token j out of tier t: its `Z_t` row is vacated. Near-exact when
  the row norm ≈ 0 (weight decay 0.1 drives unused refinements there);
  optionally least-squares-absorb the removed contribution into token j's
  lower-tier coefficients to tighten the jump.
- **Grow** token i into tier t: zero the vacated row and its Adam moments,
  then update `member_ids_t` / `member_slot_t` to point it at token i. Exact
  function no-op at the move (the new member contributes an exact zero).
- **Scoring — two signals, never one** (⚠️ a single norm-based score is
  self-defeating: freshly grown tokens have zero norm and would be reclaimed
  immediately): grow candidates ranked by accumulated *demand* (EMA of
  out-of-subspace tied-row gradient energy, captured by a sampled hook on the
  logits GEMM); shrink candidates ranked by *disuse* (deepest-tier row norm);
  a **minimum-residency period** (e.g. ≥1k steps) before any grown token is
  shrink-eligible.
- **Budget exactness**: reallocation is one-for-one swaps per tier (fixed
  `N_t`), executed every M steps (e.g. 1k), rank-0 decides, broadcast to all
  ranks, buffers and optimizer state updated identically everywhere.
- Known hazards to monitor: oscillation (mitigated by residency + hysteresis
  margins), the 10k screen under-showing late-grown refinements (judge the
  Phase-2 arm accordingly or extend its run), coarse-tier gradient domination
  (log per-tier gradient norms; per-tier LR scale is the fallback lever).

---

## 7. Prior-art positioning (cite; do not overclaim)

GroupReduce (NeurIPS'18: post-hoc frequency-blocked weighted SVD, independent
bases); Adaptive Softmax / Adaptive Input (frequency bands with separate
projections); ALBERT (uniform global factorization); Matryoshka
Representation Learning (nested dims for inference truncation — the nearest
naming neighbor, different problem); RecSys mixed-dimension embeddings and
AutoEmb/ESAPN (learned per-entity embedding sizes — nearest mechanism
neighbors, without nested shared bases, function-preserving budget-exact
moves, or a tied LM setting). The claimed contribution is the combination:
**nested shared bases + exact input-output tying + from-scratch LM training
(+ Phase 2: function-preserving learned reallocation)**, evidenced against
matched controls with frequency-stratified and multilingual analysis.
