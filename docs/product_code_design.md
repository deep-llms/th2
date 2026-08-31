# Product-Code Tail Interface — Design & Implementation Specification

Status: 2026-08-31. **This is the pre-registered last experiment of the
compressed-interface line.**

Implementation status (2026-08-31): **implemented, reviewed, and locally
validated — not launched.** See §12 for the code map. Two review passes (own
line-by-line read + seven independent finder angles) were applied; the notable
fix is that gates are stored as zero-initialized **offsets** (`gate_offsets`,
effective gate = 1 + offset) because compressed-embedding parameters train in
bfloat16, where a value stored as 1.0 cannot absorb a 3e-4 step.

⚠️ Before pushing to th2: `resources/token_importance_langbalanced.npz` (+ its
`.json`) is a new, untracked artifact that both launch scripts require — it
must be committed alongside the code, like `token_freq_sample10.npz`.

---

## 1. Summary and evidence

**Diagnosis (from `docs/PROJECT_NOTES.md` §"Frequency-binned PPL diagnostic").**
At the ~19.5M tied-interface budget, the best structure so far (frequency-
tiered independent low-rank blocks, `groupreduce_matched_nested_tied_t4`,
mean PPL 31.33 vs dense 24.63) has its remaining gap concentrated in the
low-rank blocks: the rank-64 tail block (119,168 tokens, 10.5% of eval
targets) is +108% PPL vs dense and the rank-192 torso block (24,576 tokens,
22.2% of targets) is +30%; together 78% of the gap. The dense-row head block
is +4%. Rows in those blocks are confined to a 64- or 192-dimensional
**linear subspace** and, via exact tying, serve as softmax classifier rows:
119k tail tokens must be discriminated from each other along 64 directions.

**What the literature says works instead** (`docs/related_work.md`;
MultiHashFormer arXiv:2606.28057; LightRNN 2016; HashEmbed 2017;
Shu & Nakayama 2018; DPQ 2020). Input-side compression is cheap
(Kronecker Embeddings, fixed binary codes both drop the input table with no
loss — but keep a full untied output head). The one family that compresses
*both* sides and still matches or beats dense in from-scratch generative LMs
represents each token by a **short code into a few full-dimensional shared
tables** — H hash IDs (MultiHashFormer: H=3–4, B=4k–16k buckets, tied output,
beats standard transformers at 100M–3B; single-hash collapses) or a
row/column code (LightRNN, 40–100× on an 800k vocab). The effective table is
`S·C` with `S` a fixed sparse binary selection (H ones per row) and `C` a
full-rank codebook stack: **full rank, combinatorial identity, no learned
routing, no convex-weight constraint** — i.e. the sparse-composition idea
this project began with, in the form that avoids the training pathologies
our learned-routing arms (ANT/V2) exhibited.

**The idea.** Keep what was proven (private dense rows for the highest-
importance tokens); replace *every* low-rank block with product-code
composition over H shared full-dimensional tables; keep exact input–output
tying with an exact V-way softmax so PPL stays comparable to every number we
have.

Honest framing: this is not low-rank, but it *is* additive in the codebook
rows — the escape from the diagnosed failure is full rank + combinatorial
identity, not nonlinearity (§9 lists the nonlinear variant and its cost).
Novelty is moderate (a capacity-allocated hybrid of Adaptive-Input-style
tiering and MultiHashFormer-style codes, with an exact tied softmax at a
152k vocab); the result, if it passes §7, is the first ~8× tied interface
within ~20% of dense on this setup, and Phase 2 (§10) is where a mechanism
contribution would live.

---

## 2. Method definition

### 2.1 Notation

- `V = 151,936`, `d = 1024`.
- **Head set** `𝓗`: the `N_h` highest-importance tokens (importance =
  language-balanced training frequency, §2.4). Private rows `E_h ∈ R^{N_h×d}`.
- **Tail set** `𝓣 = V \ 𝓗`, `N_t = V − N_h` tokens. Each tail token `w` has a
  fixed code `c(w) = (c_1(w), …, c_H(w))`, `c_i(w) ∈ [B]`, stored as a
  persistent buffer. Codebooks `C^(i) ∈ R^{B×d}`, i = 1…H. Per-token gates
  `g(w) = 1 + δ(w)`, `δ(w) ∈ R^H` stored as the parameter `gate_offsets`
  (init 0; see §2.7 for why offsets).
- One global input bias `b ∈ R^d` (repo convention; contributes `h·b` to
  every logit under exact tying).

Definition:

```
head token:  e_w = b + E_h[row(w)]
tail token:  e_w = b + Σ_{i=1..H} g_i(w) · C^(i)[c_i(w)]
```

Equivalent global view: `E_tail = (G ⊙ S) · [C^(1); …; C^(H)]` with `S` the
fixed `N_t × HB` binary selection matrix (one 1 per hash block per row).

### 2.2 Reference configuration (primary arm)

| Component | Shape | Params |
|---|---|---:|
| Head rows | 2,048 × 1024 | 2,097,152 |
| Codebooks | H=4 × B=4,096 × 1024 | 16,777,216 |
| Gates | 149,888 × 4 | 599,552 |
| Bias | 1024 | 1,024 |
| **Total** | | **19,474,944** (99.5% of the 19,579,904 LR128 budget; 7.99×) |

Sharing density: 149,888 / 4,096 ≈ 36.6 tail tokens per bucket per hash
(MultiHashFormer's best 32k-vocab configs ran ≈2–8; LightRNN ran ≈900 with
learned assignment — we sit between). Signature space `4096⁴ ≈ 2.8e14 ≫ N_t`.

**Fallback configuration** (if §7's block-2 non-regression condition fails
or the head appears under-provisioned): head 8,192 rows (8,388,608),
H=4 × B=2,560 (10,485,760), gates 143,744×4 (574,976), bias →
**19,450,368** (99.3%). ⚠️ Log and report the exact count at startup, never
the budget target.

### 2.3 Code assignment (two variants — same module, different buffer)

**(a) Random hashed codes — the primary arm (pure from-scratch).**
`c_i(w) = MurmurHash3(token_id, seed_i) mod B` (or `blake2b` with per-hash
salt — any deterministic keyed hash), i = 1…H. Enforce **unique full
signatures**: iterate over tail tokens in id order; if a token's signature
collides with an already-assigned one, rehash its last coordinate with an
incremented salt until unique (MultiHashFormer's iterative rehashing).
Implemented as a keyed BLAKE2b of the token id; on collision the *whole*
signature is re-hashed with an incremented salt (≤64 tries), then a
deterministic linear probe over the mixed-radix signature space guarantees
termination. Deterministic, no RNG consumed, provenance = (hash name, seed),
logged at startup; the codes buffer in the checkpoint is authoritative. Store `codes ∈ int64^{N_t×H}` and `tail_ids ∈ int64^{N_t}`
(ascending), `head_ids ∈ int64^{N_h}`, and the inverse maps as persistent
buffers so checkpoints are self-describing (`loading.py` configless path).

**(b) Teacher product-quantization codes — the second arm (post-hoc-informed).**
Split the retrained dense table's rows (tail tokens only) into H contiguous
256-dim sub-vectors; run k-means with K=B on each sub-vector space (balanced
or capacity-limited so no bucket exceeds ~2× the mean); `c_i(w)` = cluster of
sub-vector i. Semantically similar tokens then share rows (the DPQ /
Shu & Nakayama lineage). ⚠️ Fidelity boundary: the code assignment uses a
dense teacher, so this arm is *post-hoc-informed from-scratch training*,
reported as such; it must never be described as pure from-scratch. Codebooks
are still randomly initialized and trained from scratch (an init-from-
centroids ablation is a separate, clearly labelled variant).

### 2.4 Head selection

Importance = language-balanced training frequency: for each language ℓ with
per-language counts `counts_ℓ` (keys `counts_<lang>` in
`resources/token_freq_sample10.npz`), `imp(w) = Σ_ℓ counts_ℓ(w) / Σ_w counts_ℓ(w)`.
Head = top `N_h` by `(−imp, token_id)`, stable. Rationale: eval and the
mean-PPL objective weight the six languages equally while training is 85.7%
English, and raw-frequency heads are 85% English-dominant types
(`PROJECT_NOTES.md` binned analysis). Uses training counts only. Precompute
`resources/token_importance_langbalanced.npz` (key `counts`, float64) with a
provenance JSON; `load_frequency_counts` already accepts float vectors.

### 2.5 Input forward

```python
def forward(self, input_ids, doc_mask=None):
    flat = input_ids.reshape(-1)                                   # (M,)
    e = self.bias.expand(flat.numel(), d).clone()
    hrow = self.head_row[flat]                                     # (M,) −1 for tail
    hmask = hrow >= 0
    e[hmask] += self.E_h[hrow[hmask]]
    trow = self.tail_row[flat]                                     # (M,) −1 for head
    tmask = trow >= 0
    codes = self.codes[trow[tmask]]                                # (M_t, H)
    gates = self.gates[trow[tmask]]                                # (M_t, H)
    tail = sum(gates[:, i:i+1] * self.C[i][codes[:, i]] for i in range(H))
    e[tmask] += tail
    return e.view(*input_ids.shape, d), None
```

DDP with `find_unused_parameters=False`: every codebook and `E_h` must be in
the graph every microbatch. Codebooks are (each `C[i]` is touched by every
tail token); `E_h` may be untouched in a tail-only microbatch — use the
empty-selection matmul pattern (`GroupReduceEmbed`) or add
`0 * E_h.sum()`-free equivalents; test 4 in §5 covers it.

### 2.6 Exact tied head (`TiedProductCodeHead`)

Subclass `_TiedHeadBase` (non-registering embed reference). **Materialize the
effective table once per micro-batch, then one dense GEMM**:

```python
def forward(self, hidden):                                  # (…, d)
    table = self.embed.materialize()                        # (V, d): head rows + gated code sums + bias
    return hidden @ table.T                                 # (…, V)
```

Why this and not a gather-based head: the table is only V×d (311 MB bf16),
built from H gathers and a masked select, and the GEMM is exactly the dense
baseline's own output cost. A gather-based alternative (per-codebook (N, B)
bucket GEMMs + `index_select` over N_t columns) was implemented first and
measured **~13× slower and ~45% higher peak memory** at N=4096, because the
(N, N_t) gathers are memory-bound and autograd saves one such operand per
codebook (~50 GB at the production micro-batch). The GEMM form saves the
bf16 (V, d) table for backward; the gated codebook sum inside `materialize`
is a custom autograd function (`_GatedCodebookSum`) that saves only the small
code/gate tensors and recomputes its four (V, d) gathers in backward, so no
fp32 (V, d) gather tensors are retained either (an earlier version retained
four, ~2.3 GB — caught by the independent review, §14.2). Exactness: `logits ≡ h·E_effᵀ` by
construction (tests 1/1b), gradients equal a naive per-token reference
(test 2b). `materialize` is sync-free (clamped gathers + masked select; no
`nonzero`), and it touches every parameter on every micro-batch, which is
what DDP with `find_unused_parameters=False` requires.

### 2.7 Initialization and optimizer

- `E_h ~ N(0, 0.02)`; codebooks `C^(i) ~ N(0, 0.02/√H)` so a composed tail
  row has the same expected norm as a head row at init; gates = 1; bias = 0.
- **Gates are offsets, in bf16-safe form.** The harness casts compressed
  embedding modules to bfloat16 and Adam updates them in bfloat16; the bf16
  spacing at 1.0 is 2⁻⁷ ≈ 0.0078 ≫ the 3e-4 step, so a gate stored as 1.0
  would be frozen for the whole run. `gates = 1 + gate_offsets` with
  zero-initialized offsets keeps full relative precision where it matters
  (`test_gate_offsets_can_move_in_bfloat16_where_unit_gates_cannot`).
- **Offsets are exempt from weight decay** through a module-declared
  protocol: `ProductCodeEmbed.no_decay_parameters()` returns
  `("gate_offsets",)` and `CompositionalTrainer.get_decay_parameter_names`
  honors it by suffix (wrapper prefixes such as DDP's `module.` are
  irrelevant) — the same duck-typed pattern as `pop_step_metrics`.

---

## 3. Integration checklist

1. `compositional/product_code.py`: `ProductCodeEmbed` (+ code generation
   helpers: `hashed_codes(tail_ids, H, B, seeds)`, `pq_codes(table, H, B)`),
   `(embedding, None)` forward interface; export from `compositional/__init__.py`.
2. `compositional/tied_head.py`: `TiedProductCodeHead` + `make_tied_head`
   dispatch for `"product_code"`; extend the supported-arms message.
3. `train_compositional.py` `CompositionalArguments`: `--arm product_code`,
   `product_code_head_size` (2048), `product_code_num_hashes` (4),
   `product_code_num_buckets` (4096), `product_code_assignment`
   (`hashed` | `pq`), `product_code_codes_path` (artifact from
   `scripts/make_pq_codes.py`, pq only; its provenance dict is required and
   cross-checked against head_size/num_hashes/num_buckets — code generation is an offline step on
   the node that holds the dense checkpoint, not part of the launch),
   `product_code_importance_path`
   (`resources/token_importance_langbalanced.npz`), `product_code_seed`
   (hash seed base, default 0). Require `--tie_output`
   (`validate_output_configuration`). `create_optimizer` override for gates.
4. `build_arm`: construct; on resume, structural buffers come from
   `embedding.pt` via `load_state_dict(strict=True)` (never re-derive codes
   on resume — the buffers are authoritative, as for `NestedLadderEmbed`).
5. `compositional/loading.py`: `_build_arm_from_config` case (checkpoint
   state only — the partition and codes are never re-derived from
   `train_config.json`) + `_infer_comp_config_from_state` signature (`E_h`,
   `C.0…`, `gate_offsets`, `codes`, `head_ids`, `tail_ids`, `head_row`,
   `tail_row`, `bias`), via one
   `ProductCodeEmbed.structure_from_state(state)` classmethod used by every
   site (lesson from `review_code.md` D2).
6. `scripts/make_token_importance.py` (language-balanced importance file +
   provenance JSON) and `scripts/train_product_code_tied.sh` (clone of
   `train_groupreduce_matched_nested_tied.sh` hyperparameters; two runner
   entries appended at the **end** of `EXPERIMENT_COMMANDS` so existing
   experiment indices are unchanged: `product_code_hashed_h2048`,
   `product_code_pq_h2048`; both declare `required_input_files`, which the
   runner now checks *before* `ensure_gpus_free`, so a missing artifact
   refuses the launch instead of killing a live job).
7. No changes needed to `EmbeddingShim`, `SaveEmbeddingCallback`,
   `validate_resume_compatibility`, or `eval/ppl_bytoken.py`/`ppl_bins.py`.

---

## 4. Tests (before the harness; tiny dims, CPU; style of `test_nested_ladder.py`)

1. **Tying exactness** — materialize `E_eff` from the input forward over
   `arange(V)`; assert `TiedProductCodeHead(h) == h @ E_eff.T` (fp64, 1e-12),
   with random gates ≠ 1 and a head/tail mix.
2. **Budget closed form** — parameter count equals
   `N_h·d + H·B·d + N_t·H + d` for the reference and an irregular config.
3. **Code validity** — hashed codes are deterministic under (seeds, salt
   rule), every tail signature unique, all entries in `[0,B)`; pq codes cover
   only tail tokens and respect the capacity limit.
4. **DDP safety** — a tail-only batch and a head-only batch each produce
   gradients (possibly zero) for `E_h`, every `C[i]`, `gates`, `bias`.
5. **Head selection** — language-balanced importance ordering is
   `(−imp, id)`-stable and reproducible from the counts file.
6. **Configless round-trip** — `structure_from_state` rebuilds a module that
   loads `strict=True` and reproduces logits bit-for-bit.
7. **Optimizer groups** — gates are in the no-decay parameter group.

---

## 5. Experiments (matched B200 protocol, checkpoint-10k screen)

| Arm | Role |
|---|---|
| `product_code_hashed_h2048` (§2.2 primary, random codes) | **the method, pure from-scratch** |
| `product_code_pq_h2048` (teacher-PQ codes) | code-assignment ablation (post-hoc-informed) |
| `groupreduce_matched_nested_tied_t4` (31.33, exists) | best linear structure at the budget |
| dense tied (24.63, exists) | ceiling |
| (optional) `product_code_hashed_h0` — no dense head | isolates the allocation contribution (pure MHF-exact) |

Battery: six-language PPL, 26-task zero-shot, 9-job finetune, and
`eval/ppl_bytoken.py` + `eval/ppl_bins.py` with `--populations
2048,6144,24576,119168` so the bins map onto the same blocks as the existing
diagnostic (the head block equals the dense-head set only under raw-frequency
ordering; report both orderings' bins).

---

## 6. Interpretation guide

- Tail and torso bins drop sharply, block 2 flat → the linear-subspace
  diagnosis was right; codes are the escape.
- Tail improves but block 2 regresses → private rows matter down to ~8k
  tokens: run the fallback config (§2.2).
- pq ≫ hashed → code *assignment* is the lever → Phase 2 (§10) is learned
  assignment; hashed ≈ pq → random codes suffice, sharing density is the knob.
- Neither arm moves the tail bin → the failure is not representational rank
  at all (tying/backbone interaction); stop (§7).

---

## 7. Pre-registered go / no-go (fixed before launch)

Measured on the primary arm, single seed, vs the DDP-default dense run,
same bins as the existing diagnostic:

- **Pass**: tail block ≤ **+50%** (from +108%), torso block ≤ **+15%** (from
  +30%), block 2 not worse than **+12%** (from +10%), and mean PPL ≤ **29.5**
  (from 31.33). These are one set of numbers: applying the block targets to
  the measured bin shares gives ≈ 24.0 pooled / ≈ 29.4 mean PPL.
- **Strong pass**: mean PPL ≤ 28.5 → seed replication, then Phase 2.
- **Fail**: any pass condition missed by both the primary and the fallback
  config → **stop the compressed-interface line**; write the analysis paper
  from the existing findings. No further mechanism proposals.

---

## 8. Risks (stated before the run)

- Sharing density 37/bucket exceeds MultiHashFormer's; fallback is
  B=8,192/H=3 (25.2M codebooks — over budget) or the §2.2 fallback config.
- Hashing block-2 tokens may cost more than the tail gains (monitored, §7).
- MultiHashFormer trained 20k–200k steps; our 10k screen may under-show a
  code-based interface that needs longer to organize its buckets.
- Exact V-way softmax pushes dense gradient into shared rows from every
  position (interference); MHF's factorized softmax avoids this and is the
  efficiency/regularization variant if training is unstable.
- Single seed, as always.

## 9. Variants explicitly not in the base run

- Linear adapter `W_s ∈ R^{d×d}` after the sum (MHF has one): preserves the
  exact gather-sum head (`h·W_s Σ = (W_sᵀh)·Σ`); +1.05M params; try only if
  the base passes.
- Nonlinear adapter (MLP after the sum): breaks the decomposition — the tied
  head must materialize the tail table (`N_t × d` per step, ≈ +10% step time).
  Only if the base passes and the residual gap is clearly in-bucket.
- MHF-style factorized product softmax: changes the output distribution
  family; PPL then needs the renormalize-over-assigned-signatures convention.

## 10. Phase 2 (only after a pass): learned code assignment

LightRNN-style periodic reassignment of tail tokens to buckets (bipartite
matching against reconstruction/loss signals) or CCE-style clustered
sketches — the mechanism contribution, sitting on top of a working base,
with the pq arm as its static upper-bound reference. Not to be built before
§7 passes.

## 11. Prior art to cite

MultiHashFormer (2026), HashFormers (2022), Hash Embeddings (2017), LightRNN
(2016), Shu & Nakayama (2018), DPQ (2020), Slim Embeddings (2018), Adaptive
Input/Softmax, GroupReduce/DiscBlock (allocation), Kronecker Embeddings and
fixed binary codes (input-side-only evidence), P-VQ (shared+exclusive split —
the single-codebook ancestor already implemented as `PVQEmbed`).

## 12. Code map (implemented 2026-08-31)

| Component | Path |
|---|---|
| Embedding module, code generation (`hashed_codes`, `pq_codes`), partition rule (`head_tail_partition`), `structure_from_state`, `no_decay_parameters`, `pop_step_metrics` | `compositional/product_code.py` |
| Exact tied head `TiedProductCodeHead` (materialize once + one dense GEMM) + `make_tied_head` dispatch (`"product_code"`) | `compositional/tied_head.py` |
| Package export | `compositional/__init__.py` |
| CLI flags (`--arm product_code`, `--product_code_*`), `build_arm` branch (fresh / resume / pq with artifact cross-checks), tie-output requirement, weight-decay protocol | `train_compositional.py` |
| Strict checkpoint loading + configless inference | `compositional/loading.py` |
| Language-balanced importance artifact builder | `scripts/make_token_importance.py` → `resources/token_importance_langbalanced.npz` (+ `.json` provenance) |
| Teacher-PQ codes builder (run on the node holding the dense checkpoint) | `scripts/make_pq_codes.py` → `resources/pq_codes_h2048_4x4096.pt` (+ `.json`) |
| Launch scripts | `scripts/train_product_code_tied.sh` (hashed, primary), `scripts/train_product_code_pq_tied.sh` (pq; `PRODUCT_CODE_CODES_PATH` env override) |
| Runner entries (last two in the list) | `run_experiments.py`: `product_code_hashed_h2048`, `product_code_pq_h2048` |
| Tests (§4 invariants + naive-reference table with non-zero bias, head-vs-materialized gradient equality, bf16 gate movability, CUDA-resident `load_state_dict`, saved-tensor size bound, artifact/provenance rejection, partition-helper contract) | `compositional/test_product_code.py` |
| Binned evaluation after training | `eval/ppl_bytoken.py` → `eval/ppl_bins.py --populations 2048,6144,24576,119168` |
| Local end-to-end smoke through the real entry points (single-GPU or `--gpus N` DDP; never the runner) | `scripts/smoke_product_code_e2e.py` → logs + `summary.json` under `--scratch` |

## 13. Verification brief for an independent reviewer

Read this document top to bottom first; everything below assumes it. You do
not need the conversation history. The implementation was written by one
agent and reviewed by that agent plus seven independent review angles; your
job is to find what they missed, not to re-derive the design.

### 13.1 Scope (exactly these paths; everything else is pre-existing)

`compositional/product_code.py`, `compositional/tied_head.py`
(`TiedProductCodeHead` + dispatch), `compositional/__init__.py` (export),
`train_compositional.py` (flags, `build_arm` product_code branch,
`validate_output_configuration`, `get_decay_parameter_names` /
`_matches_declared_no_decay`), `compositional/loading.py` (product_code
branch + inference signature + `supported_tied_arms`), `run_experiments.py`
(two last entries + `required_input_files` pre-flight), `scripts/
make_token_importance.py`, `scripts/make_pq_codes.py`,
`scripts/train_product_code_tied.sh`, `scripts/train_product_code_pq_tied.sh`,
`resources/token_importance_langbalanced.{npz,json}`,
`compositional/test_product_code.py`.

### 13.2 Invariants that must hold (each has a test; check the tests are honest)

1. **Exact tying**: `TiedProductCodeHead(h) == h @ E_effᵀ` where `E_eff` is
   the §2.1 table — verified against a *naive per-token loop* with non-zero
   bias and non-unit gates in fp64 (not just against `materialize`).
2. **Gradient equality** between the tied head and `h @ materialize().T`.
3. **Budget**: `sum(p.numel())` equals `N_h·d + H·B·d + N_t·H + d`;
   reference config = **19,474,944**; fallback = 19,450,368.
4. **Structure**: head/tail ids partition the vocabulary, inverse maps are
   exact, every tail signature is unique, all codes in `[0, B)`; corrupt
   state must be rejected on `load_state_dict`, **including on a CUDA-resident
   module** (this was a real bug: Trainer resume loads onto the GPU model).
5. **Determinism / no RNG consumption** for structure (same seed ⇒ identical
   parameters regardless of importance vector).
6. **DDP safety**: head-only and tail-only batches produce a gradient for
   every parameter (`find_unused_parameters=False` in the launch script).
7. **bf16 gates**: parameters train in bfloat16; gates are `1 + gate_offsets`
   and offsets must actually move under a 3e-4 step (a stored 1.0 cannot).
   Also confirm the effective gate/tail sum is formed in fp32 (`work_dtype`).
8. **Weight decay**: `gate_offsets` excluded via the module's
   `no_decay_parameters()`; check the trainer honors it on a real `Trainer`.
9. **Artifacts**: PQ artifact provenance is required and cross-checked
   (head_size / num_hashes / num_buckets / tail partition); the importance
   ordering used by the module, the trainer, and both scripts is the single
   `head_tail_partition` helper.
10. **Runner**: new entries are the **last two** in `EXPERIMENT_COMMANDS`
    (existing indices unchanged) and `required_input_files` is checked
    *before* `ensure_gpus_free`.

### 13.3 How to run the checks

```bash
PY=/home/users/thien/miniconda3/envs/sparse_emb/bin/python   # torch 2.7.1, transformers 5.9
$PY -m pytest compositional/test_product_code.py compositional/test_tied_head.py \
    compositional/test_nested_ladder.py compositional/test_compressed_baselines.py \
    compositional/tests.py -q          # expected: all pass (117 at hand-off)
```

End-to-end through the real entry point (what the unit tests cannot show):
build a tiny `Qwen3Config(vocab_size=151936, hidden_size=128,
num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
head_dim=64, tie_word_embeddings=False)` + the Qwen3-0.6B tokenizer saved to
one dir; save two small `datasets.Dataset({"text": [...]})` folders under
`<data>/en` and `<data>/zh`; then run `train_compositional.py` with the flags
from `scripts/train_product_code_tied.sh` (override `--config_name/
--tokenizer_name/--data_dir/--output_dir`, add `--block_size 64
--per_device_train_batch_size 2 --gradient_accumulation_steps 2
--max_steps 6 --save_steps 3 --report_to none --dataloader_num_workers 0`).
Then rerun with `--max_steps 9` to exercise **automatic resume**; then
`eval/ppl_bytoken.py` and `eval/eval_checkpoint.py --ppl-only` on the last
checkpoint. Multi-GPU: prefix with
`$PY -m accelerate.commands.launch --num_processes 4 --multi_gpu`. Expected:
all exit 0, `gate_offsets` in `embedding.pt` non-zero, verification gap in
`ppl_bytoken` ≤ 1e-3. For the PQ arm, fabricate a dense `model.safetensors`
holding `model.embed_tokens.weight (151936, 128)` and run
`scripts/make_pq_codes.py` → then train with `--product_code_assignment pq
--product_code_codes_path <artifact>`.

⚠️ **Do not run `run_experiments.py` on a shared machine**: its
`ensure_gpus_free` SIGKILLs every GPU process on the host.

### 13.4 Already found and fixed — do not re-report unless you find it broken

Device-mismatch in `validate_structure` on CUDA (resume crash); gather-based
tied head replaced by materialize + GEMM (13× faster, ~50 GB fewer saved
activations); bf16 gate freezing and the `1 + offset` bf16 dead zone; PQ
artifact provenance/partition cross-checks; resume dimension check; runner
index shift and pre-flight; triplicated partition rule; duplicated
`file_sha256`; stale `--tie_output` help; hard-coded no-decay name → module
protocol; dead fresh-build path in `loading.py`; `pq_codes` coverage check,
capacity-aware duplicate resolution, `argsort`-free capacity repair;
sync-free `materialize`/metrics; start-up cost. From the independent
addendum (§14): guaranteed PQ duplicate repair, recomputed gathers in the
gated sum (no saved fp32 (V, d) tensors), env-resolved non-empty runner
pre-flight for the PQ artifact.

Deliberately kept (not bugs): derived partition buffers are persisted and
cross-validated (self-describing checkpoints, as `NestedLadderEmbed` does);
`hashed_codes` keeps a two-stage collision strategy (MHF-style salted rehash,
then a guaranteed linear probe); the 14-file `required_checkpoint_files`
list is duplicated per runner entry like every existing entry.

### 13.5 Out of scope

`commands.sh` (a pre-existing runner job, not part of this change), the TT
baseline OOM (`review_code.md` C1), the PVQ curriculum repair loop, and
`scripts/dropbox_downloader.py`.

### 13.6 What "correct" means for the pre-registered run

If everything in 13.2 holds, the arm is fit to launch; the *experiment's*
verdict is then §7 (tail ≤ +50%, torso ≤ +15%, block 2 ≤ +12%, mean PPL
≤ 29.5), measured with `eval/ppl_bytoken.py` + `eval/ppl_bins.py
--populations 2048,6144,24576,119168`. Before pushing: the importance
artifact is untracked and must be committed with the code.

## 14. Verification addendum — unresolved issues (2026-08-31)

An additional adversarial review found that the **primary hashed arm is
correct**, but the complete implementation must not yet be described as fully
verified because the optional teacher-PQ path has a real correctness failure.

### 14.1 Teacher-PQ duplicate repair can fail

`pq_codes` first assigns one cluster per sub-vector and then calls
`_resolve_duplicate_signatures`. The current repair searches only signatures
that differ from the duplicate's original signature in one coordinate, using
that coordinate's ranked centroid candidates. It does not search combinations
of changes across multiple coordinates or fall back to an exhaustive
mixed-radix probe. It can therefore raise
`RuntimeError("could not make signature unique ...")` even when
`B^H >= N_t` and many valid signatures remain unused.

This was reproduced on valid small inputs: duplicate-heavy and fully identical
tables failed, and 27 of 60 adversarial randomized configurations failed. The
existing PQ unit test covers only one favorable configuration and therefore
does not establish the promised general uniqueness guarantee.

**Consequence:** do not generate or train `product_code_pq_h2048` until the
repair has a deterministic guaranteed-termination fallback that also preserves
the documented occupancy bound. This issue does **not** affect
`product_code_hashed_h2048`: `hashed_codes` uses a separate salted rehash plus
guaranteed mixed-radix linear probe, and the production configuration produced
149,888 unique tail signatures.

### 14.2 The tied-head saved-memory claim is too strong

The statement in §2.6 that backward "saves only the `(V, d)` table" is not
literally true. To differentiate the learned per-token gates, autograd also
saves one gathered FP32 codebook-row tensor of shape approximately `(V, d)`
for each of the four hashes. Saved-tensor inspection confirmed four such FP32
tensors. At the reference dimensions these tensors occupy about 2.29 GiB,
while the final BF16 table occupies about 0.29 GiB, for a lower bound of about
2.58 GiB before the input-side composition and smaller tensors.

The materialized tied head is still much smaller than saving four
`(N, N_tail)` operands at the production micro-batch, and this finding does
not change its numerical correctness. It does mean that the prose and
`test_tied_head_saves_only_table_sized_tensors` must be revised: that test only
proves the absence of an `(N, N_tail)` saved operand, not that the effective
table is the only saved activation. Measure actual peak memory on the B200
before launch rather than budgeting only 311 MB for this path.

### 14.3 PQ runner preflight does not honor the artifact override

`scripts/train_product_code_pq_tied.sh` accepts
`PRODUCT_CODE_CODES_PATH`, but the corresponding `required_input_files` entry
in `run_experiments.py` always checks
`resources/pq_codes_h2048_4x4096.pt`. A valid override can consequently be
refused before launch; conversely, the preflight checks only `isfile`, whereas
the launcher later requires a non-empty file with `test -s`. Align the runner
with the resolved artifact path and require a non-empty file before any GPU
cleanup.

### 14.4 What the additional verification did establish

- 116 tests passed and one CUDA-only test was skipped in the review sandbox.
- The production hashed module has exactly 19,474,944 parameters, the expected
  2,048/149,888 head/tail partition, unique signatures, and finite BF16 logits.
- Twenty-four randomized float64 cases matched an independent naive
  per-token implementation exactly in both forward values and every parameter
  gradient.
- Real BF16 AdamW updates at learning rate `3e-4` moved `gate_offsets` in ten
  tested seeds; the no-decay integration also passed.
- The language-balanced importance artifact reproduced bit-for-bit from the
  six per-language source count vectors and its recorded hashes match.

Current launch verdict: the **hashed primary arm may proceed after a CUDA/DDP
smoke test**, but the **teacher-PQ arm is blocked** by §14.1. No result from the
hashed arm should be used to imply that the PQ implementation was validated.

### 14.5 Resolution (same day)

All three addendum items were fixed and covered by tests:

- **14.1** `_resolve_duplicate_signatures` now guarantees uniqueness whenever
  `B^H >= n`: single-coordinate nearest candidates → single-coordinate over
  all buckets → deterministic mixed-radix linear probe (capacity-respecting
  pass first, then unconstrained). The occupancy bound is best-effort and the
  realized `max_bucket_occupancy` is recorded in the artifact provenance.
  `test_pq_duplicate_repair_is_guaranteed_on_adversarial_tables` covers a
  fully identical table, a 75%-duplicate table, and 40 randomized small-space
  configurations. The PQ arm is no longer blocked.
- **14.2** The gated codebook sum is a custom autograd function that
  recomputes its gathers in backward; saved-tensor inspection on the
  production shape now shows the bf16 (V, d) table as the only vocab-sized
  saved activation. §2.6 prose and the test
  (`test_tied_head_saves_at_most_one_table_sized_tensor`) were corrected.
- **14.3** `run_experiments.py` resolves `${VAR:-default}` specs in
  `required_input_files` (the PQ entry now uses
  `${PRODUCT_CODE_CODES_PATH:-resources/pq_codes_h2048_4x4096.pt}`) and
  requires a non-empty file, matching the launcher's `test -s`.
  `test_runner_resolves_required_inputs_and_requires_nonempty` covers it.
- The CUDA/DDP smoke the addendum asked for was run on a 4×A100 host through
  `accelerate launch --num_processes 4 --multi_gpu` with the launch script's
  flags on a tiny backbone and the real vocabulary/importance file: fresh
  training, checkpoint with `rng_state_0..3`, automatic resume, and
  `eval_checkpoint --ppl-only` all exit 0; gate offsets move in bf16; tail
  signatures stay unique across the round trip.

### 14.6 Independent re-verification of the fixes (2026-08-31)

A second review inspected and reran the fixes from §14.5. Code-level verdict:
the three reported implementation defects are resolved.

- **PQ uniqueness:** the exact corpus that exposed the old repair failure now
  passes, including duplicate-heavy, fully identical, and near-saturated
  signature spaces. A repeat of the earlier 60-configuration sweep produced
  **0 uniqueness failures** (the old implementation failed 27). The
  mixed-radix fallback therefore supplies the promised termination guarantee
  whenever `B^H >= n`.
- **Custom-autograd correctness:** 32 randomized float64 configurations were
  compared with an independent explicit per-token implementation. Forward
  values and every parameter gradient matched; the worst observed gradient
  difference was `1.07e-14`. A production-shape BF16 forward/backward
  (`V=151936`, `d=1024`, head 2048, `H=4`, `B=4096`) produced finite logits,
  finite gradients for every parameter, and non-zero gradients for all
  599,552 gate offsets.
- **Saved activations:** saved-tensor inspection found exactly one
  vocabulary-sized activation, the BF16 effective table (observed in
  transposed `(d, V)` form as saved by the GEMM), and no FP32
  vocabulary-sized gathered codebook tensors. `_GatedCodebookSum` correctly
  recomputes those gathers during backward.
- **Runner preflight:** environment/default expansion, non-empty-file
  validation, and placement before `ensure_gpus_free` all passed inspection
  and tests. The runner and PQ launcher now resolve the same artifact.
- **Regression and static checks:** the focused suite completed with
  **118 passed, 1 CUDA-only skipped**; Python compilation, shell syntax, and
  `git diff --check` also passed.

The occupancy limit after PQ duplicate repair is intentionally **best-effort**,
not a strict invariant. In an identical 300-row adversarial case, maximum
occupancy was 101 against the nominal limit 100; this remains approximately
2× mean occupancy and is consistent with the revised method description. The
artifact provenance must continue to record and report the realized maximum.

One non-blocking test-quality improvement remains: the repository test named
`test_tied_head_gradients_match_materialized_head` compares two paths that
both pass through `_GatedCodebookSum`. The independent naive-gradient check
described above passed, but it should be made a permanent unit test so a future
custom-backward regression cannot satisfy both sides of the same comparison.

Evidence boundary: this second review could not independently rerun CUDA/DDP
because its execution sandbox had no NVIDIA driver access, and it found no
local log for the 4×A100 run asserted in §14.5. That assertion is therefore
retained as externally reported evidence rather than independently reproduced
evidence. The CPU/code-level checks do not reveal a remaining launch blocker
for either the hashed or teacher-PQ arm.

### 14.7 Follow-ups to §14.6 (same day)

- The requested permanent test exists:
  `test_tied_head_gradients_match_explicit_per_token_reference` compares the
  tied head's parameter gradients against `_naive_table`, an explicit
  per-token loop that does not go through `_GatedCodebookSum` (atol 1e-12,
  fp64, non-zero bias, non-unit gates).
- The CUDA/DDP evidence boundary is closed with a durable, rerunnable
  artifact: `scripts/smoke_product_code_e2e.py --gpus 4` builds the fixtures
  and runs fresh training → checkpoint (`rng_state_0..3`) → automatic resume
  → `ppl_bytoken` → `eval_checkpoint` → `make_pq_codes` → PQ launch under
  `accelerate launch --multi_gpu`. Its logs and `summary.json` from the 4×A100
  run are under `temp/product_code_smoke_20260831_ddp4/` (all seven stages
  PASS; gate offsets moved in 100% of entries; codes unique after the round
  trip). Any host with GPUs can reproduce it in ~4 minutes.
- Total test count after these additions: 128 across all suites (20 for
  this arm).
