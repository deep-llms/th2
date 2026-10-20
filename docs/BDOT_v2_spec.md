# B-DOT v2: Implementation and Experiment Specification

Version 1.4, 2026-10-20. Audience: an engineering agent who will implement the model, test it and run the experiments.

What changed in each version (Appendix B lists the details):
- **Version 1.2** answers the implementation review of version 1.1.
- **Version 1.3** switches every layer to the Qwen3 architecture: the design of the released Qwen3-0.6B, trained from scratch, with 12 layers in the main experiments.
- **Version 1.4** answers the curriculum-horizon review:
  - every run has one complete schedule, shared by the LR and the routing curriculum;
  - Stages 0a and 0b train 2.5B tokens;
  - the data is the prepared English pool.

---

## 0. How to use this document

1. Read sections 1 to 3 to understand the model. Section 2 defines the terms; a few of them (step, gate) also have common meanings elsewhere in ML, so read it first. Section 3.4 includes a worked example of one token's path.
2. Build in the phases of section 6. Phase A is only what the first pilot (Stage 0a) needs; build the later phases only after the pilot gate (Gate 0a) passes. Pass the acceptance tests in section 7 at the points it names.
3. Run the experiments in the order of section 9. Each stage ends with a gate; record the gate decision before starting the next stage. A failed gate is never rescued by adding STOP, re-entry or a larger model.
4. Every option maps to a config key in section 8. Use those exact names in configs and logs.
5. If something is ambiguous, pick the simplest choice that keeps the `r = 0` model identical to its baseline, and record the choice in the run log.
6. Appendix A is a one-page checklist of the whole job.

Implementation target: PyTorch, bf16 training, `scaled_dot_product_attention` (or FlashAttention / FlexAttention) for attention. The Qwen3 layer may be taken from Hugging Face `transformers` (`Qwen3DecoderLayer` and its parts) or reimplemented; T1 checks it against `Qwen3ForCausalLM` either way.

---

## 1. Summary

B-DOT v2 is a decoder-only transformer in which **each token chooses which layer ("module") to run at each middle step**, within a limited window around the layer a normal model would run at that step. (The name comes from v1, a *bucketed dynamic-order transformer*; in v2 buckets are only an ablation.)

**Architecture.** Every layer is a Qwen3 decoder layer, as in the released Qwen3-0.6B (section 3.1). Every model is trained from scratch; no pretrained weights are used. The main experiments use 12 layers instead of Qwen3-0.6B's 28, and Stage 5 returns to the full depth.

**Why this design.** Looped transformers and layer-skipping models show that a fixed layer order is not always the best use of a network's layers. The two public fully dynamic designs (both preprints) leave the question open:
- Library of Layers routes whole chunks of tokens, because it found per-token routing unstable.
- DNA's result depends on the variant. Its top-1 model trails its dense baseline (validation loss 2.754 vs 2.720), while its top-2 model, which has more active parameters, beats it (2.674).

B-DOT tests a *bounded* amount of freedom per token. Its key/value design is meant to keep per-token routing stable (section 3.5), and every variant is compared with a baseline built from the same code.

**What is new and what is not.**
- **Prior art:**
  - Sharing K/V while giving each expert its own Q/O is the split used by Mixture of Attention Heads.
  - Reusing FFN experts from neighbouring layers, with a candidate window that widens during training, is ReXMoE.
- **B-DOT's distinct element:** per-token, bounded, cross-step selection of a whole module (Q/O projections plus MLP), with K/V owned by the step so that every query reads keys in one shared space.
- **What must be shown:** that this improves a stated quality–cost trade-off. Combining known parts does not establish that on its own.
- **Control:** grid row 18 (MLP-only routing) checks whether routing the attention projections adds anything over ReXMoE-style routing.

**The central hypothesis.** Step-owned K/V makes keys compatible across modules, but it does not make a module's inputs the same at every depth: earlier routing still changes them. Whether one module can work well at several nearby depths is the hypothesis under test, not something the K/V design guarantees.

- The first `n_stem` layers and the last `n_coda` layers are ordinary, fixed layers.
- The middle runs `S` steps. At each step, every active token runs exactly one module, chosen by a small router from its candidate window.
- With the window closed (`r = 0`), the model reduces to a known baseline built from the same code:
  - a standard transformer (`mode = compute_matched`);
  - a looped transformer (`mode = depth`);
  - a mixture of whole-layer experts (`mode = capacity`, where the router still picks one of two experts per step).
- Every routed variant is therefore compared against its own `r = 0` model.
- Optional features:
  - **STOP:** a token skips its remaining middle steps and goes straight to the coda.
  - **Re-entry:** a token goes back once to an earlier step and runs forward again.

**First question (Stage 0a, before anything else is built):** does bounded learned routing train stably, get used, beat random routing in the same window, match the dense model's loss and stay within the runtime budget? The rest of the plan is justified only if it does.

Research questions:

1. Does per-token choice of layer (dynamic order) improve quality on top of:
   - (a) a dense model at equal compute;
   - (b) extra parameters per FLOP (mixture of experts);
   - (c) extra depth per parameter (looping)?
2. Can a learned STOP save measured compute at equal quality?
3. Does re-entry help multi-hop composition?

---

## 2. Glossary and notation

| Term | Meaning |
|---|---|
| `d` | Model width (`d_model`) |
| `H`, `H_kv`, `head_dim` | Query heads, key/value heads, head dimension (128 in Qwen3; not `d / H`) |
| `q_dim`, `d_kv` | `q_dim = H · head_dim` (2048 in Qwen3-0.6B, twice `d`); `d_kv = H_kv · head_dim` (1024) |
| `d_ff` | SwiGLU hidden width |
| Stem / coda | The fixed first `n_stem` layers (stem) and last `n_coda` layers (coda), as in Library of Layers' prelude/core/coda |
| Step `s` | Position in the middle, `s = 0 .. S-1`. All tokens move through the steps in lockstep. An optimizer update is always called a *training step*. |
| Module `m` | A routable unit, `m = 0 .. M-1`. What it owns depends on `kv_mode` (section 3.5). |
| `home(s)` | The module(s) a fixed-order model runs at step `s` |
| `pos(m)` | The position of module `m` in the default order (section 3.2) |
| Reach `r` | How far from the home position a token may pick a module |
| Candidate set `C_t(s)` | The modules token `t` may choose at step `s` |
| Reuse | Whether a token may run a module more often than the default schedule does |
| Default path | Running the home module at every step; this is the `r = 0` model |
| Off-default share | Fraction of (token, step) choices that are not a home module |
| Layer-equivalent (LE) | Weight FLOPs of one ordinary layer for one token (section 3.11) |
| Pass | One run through middle steps `0 .. S-1`. Re-entry adds a partial second pass (section 3.10). |
| Active / stopped / waiting token | At a given step, an *active* token runs a module. A *stopped* token has used STOP. A *waiting* token has decided to re-enter and sits out the rest of the first pass. Stopped and waiting tokens still write K/V (section 3.5.2). |
| Re-entry point | A step after which a token may decide to re-enter (section 3.10) |
| Router gate `g` | The multiplier on a module's update (section 3.7). Not the same as **Gates 0a to 4**, the go/no-go decisions that end each experiment stage (section 9). |
| FiLM | A learned per-step scale and shift on a module's MLP input (section 3.8) |
| ID / OOD | In-distribution / out-of-distribution test splits of the synthetic tasks (section 10) |

---

## 3. Architecture

### 3.1 Layout

```
tokens
  -> embedding (tied with the output layer)
  -> stem: n_stem ordinary pre-norm layers, full causal attention
  -> middle: S steps
       at each step, each active token runs ONE module from its candidate set
       optional STOP: a token leaves the middle early and waits for the coda
       optional re-entry: a token goes back once to an earlier step
  -> coda: n_coda ordinary pre-norm layers, full causal attention
  -> final RMSNorm -> output layer -> logits
```

**Every layer is a Qwen3 decoder layer,** exactly as in the released Qwen3-0.6B (Hugging Face `Qwen3ForCausalLM`), but trained from scratch:

```
x = RMSNorm_attn(h)
q = RoPE(q_norm(W_Q · x))       # q_norm and k_norm: RMSNorm over head_dim, per head
k = RoPE(k_norm(W_K · x))
v = W_V · x
h = h + W_O · Attn(q, k, v)     # causal; grouped-query: 16 query heads share 8 K/V heads
h = h + MLP(RMSNorm_mlp(h))     # MLP(u) = W_down(silu(W_gate u) * W_up u)
```

**Details used everywhere** (the Qwen3-0.6B config):
- `d = 1024`;
- 16 query heads and 8 K/V heads, all with `head_dim = 128`. So `W_Q` maps 1024 → 2048, `W_K` and `W_V` map 1024 → 1024, and `W_O` maps 2048 → 1024;
- `d_ff = 3072`;
- RMSNorm eps 1e-6; QK-norm as above; RoPE base 1,000,000 over the full head dimension;
- no bias in any linear layer; no dropout;
- vocabulary 151,936, with the output layer tied to the embedding;
- initialisation as in the Hugging Face implementation: every linear and embedding weight is normal with std 0.02, and norm weights start at 1.

**Depth.** The released model has 28 layers. The main experiments use 12 (stem 2, middle 8, coda 2) to keep runs cheap, and Stage 5 returns to 28. At 12 layers the dense model has 344M parameters, of which 156M are the embedding.

### 3.2 Modes: how modules map to steps (`mode`)

| `mode` | Modules `M` | `home(s)` | `pos(m)` | With `r = 0` it is |
|---|---|---|---|---|
| `compute_matched` | `S` | `{s}` | `m` | The Qwen3 model with `n_stem + S + n_coda` layers |
| `capacity` | `2S` | `{2s, 2s+1}` | `floor(m / 2)` | A mixture of whole-layer experts: two experts per step, top-1 |
| `depth` | `S / 2` (`S` must be even) | `{s mod M}` | `m` | A looped transformer with schedule `0,1,...,M-1,0,1,...,M-1` |

With the default `kv_mode = step`, K/V projections belong to steps, not modules (section 3.5 explains why). This changes two of the reductions:

- **`capacity`:** the two experts of a step share that step's K/V and differ only in Q/O/MLP.
- **`depth`:** the looped modules share Q/O/MLP across loops, but each step has its own K/V. With `kv_mode = module_band`, each module owns its K/V too, so the loop shares every weight: a standard looped transformer.

In `capacity` mode with `r = 0` and `routing = learned`, the router still chooses between the two home experts of each step. With `routing = fixed`, only expert `2s` runs, so the model is a dense transformer built from those experts.

### 3.3 Candidate sets (`window`)

Every window type builds a boolean mask over modules for each (token, step). Modules outside the mask get logit `-inf`.

**`absolute` (default).** Anchored to the step index:

```
C(s) = union of home(s') for s' in [s - r, s + r], clipped to [0, S - 1]
```

Example with `compute_matched`, `S = 5`, `r = 2`, where ✓ means allowed. Each row (step) picks exactly one ✓.

```
           M0  M1  M2  M3  M4
step 0      ✓   ✓   ✓   ·   ·
step 1      ✓   ✓   ✓   ✓   ·
step 2      ✓   ✓   ✓   ✓   ✓
step 3      ·   ✓   ✓   ✓   ✓
step 4      ·   ·   ✓   ✓   ✓
```

**`relative`.** Anchored to the module the same token used at the previous step. Use it only in `compute_matched` and `capacity` modes, and only with `r >= 1`.

```
C_t(0) = all modules        (with a drift cap: { m : pos(m) <= drift_cap })
C_t(s) = { m : |pos(m) - pos(prev_t)| <= r }
         and, if drift_cap is set, |pos(m) - s| <= drift_cap
```

- Without a drift cap, a token can wander far from its default depth. For example, `M4, M3, M2, M1, M0` is allowed with `r = 1`.
- At `r = 0` a relative window is degenerate: the token would repeat its first module at every step.

**`bucket` (ablation).** Fixed disjoint groups of `b` consecutive steps. The config key `r` holds the bucket size `b`.

```
C(s) = union of home(s') for all s' in the same bucket as s      (bucket of s = floor(s / b))
```

A bucket of size 2 is the counterpart of reach 1, and a bucket of size 4 the counterpart of reach 2.

**`free`.** Every module is a candidate at every step.

### 3.4 Reuse rule (`reuse`)

**`reuse = true`.** Any candidate may be chosen, including a module the token has already used. Choosing AA instead of AB means the token skips B.

**`reuse = false`.** Each module has a quota, and a choice is allowed only if the remaining steps can still meet every quota.

| Mode | Quota per module | Meaning |
|---|---|---|
| `compute_matched` | exactly 1 | The path is a permutation of the modules |
| `depth` | exactly 2 | Each module runs twice, in any allowed order |
| `capacity` | (not used with `reuse = false` in this plan) | — |

In this plan, `reuse = false` is used only with `absolute` (and `bucket`, `free`) windows. With relative windows and no reuse, `r = 1` leaves only the default path (with a drift cap; without one, also its reverse). It is also never combined with STOP or re-entry: a token that stops early or re-runs steps cannot meet exact quotas.

**Feasibility table.** A greedy deadline rule (force a module once its remaining available steps equal its remaining quota) is not enough. In `depth` mode with absolute `r = 1` and `S = 8`, it reaches a dead end in about 1.5% of random paths. Use an exact table instead, built on the CPU for each config and reach.

```
# state = (s, q): step index and remaining quota per module (q starts at every module's quota)
# C(s) is the window's candidate set; e_m is the one-hot vector of module m
def feasible(s, q):                                        # memoised
    if s == S: return all(q == 0)
    return any(feasible(s + 1, q - e_m) for m in C(s) if q[m] > 0)

enumerate every state reachable from (0, q_initial) through feasible moves;
give each state an integer id and store:
    allowed[id]    = bool[M]   # m in C(s), q[m] > 0 and feasible(s + 1, q - e_m)
    next_id[id, m] = id of the state (s + 1, q - e_m)
```

- **At run time:** each token carries `state_id` (an int tensor `[B, T]`). The step's candidate mask is `allowed[state_id]`. After choosing `m*`, set `state_id = next_id[state_id, m*]`.
- **Size:** the tables are tiny. At the main depth (`S = 8`) the largest has 58 reachable states (`depth`, absolute `r = 1`). The largest in the whole plan is the same setting at Stage 5's full depth (`S = 22`), with about 31,000.
- **One table per reach:** curriculum A changes the reach during training, so build a table for every reach the run uses.
- **With `reuse = true`:** no table is needed; compute the candidate mask directly from the window rule.

**Number of distinct paths at `S = 8` with `reuse = false`** (acceptance test T4 checks these):

| Setting | Paths |
|---|---|
| `compute_matched`, absolute `r = 1` | 34 |
| `compute_matched`, absolute `r = 2` | 400 |
| `compute_matched`, bucket 2 | 16 |
| `compute_matched`, free | 40,320 |
| `depth`, absolute `r = 1` | 143 |
| `depth`, absolute `r = 2` | 1,440 |

**Worked example** (`compute_matched`, `S = 8`, absolute `r = 1`):

- Default path: `M0 M1 M2 M3 M4 M5 M6 M7`.
- With `reuse = false`, a token might take `M0 M2 M1 M3 M4 M6 M5 M7`: two neighbouring swaps, and still every module once.
- With `reuse = true`, it might take `M0 M1 M1 M3 M4 M5 M6 M7`: it runs M1 twice and skips M2. Its off-default share is 1/8.

### 3.5 Key/value ownership (`kv_mode`)

This is the most important design choice. Different tokens run different modules at the same step, yet attention needs keys and values that every token's query can read; `kv_mode` decides who owns the K/V projections.

Why it matters: if each token wrote its key with its own module's `W_K`, a query from module A would be compared with keys made by module B, in a different "language". Library of Layers reports that this mismatch makes per-token routing unstable. The default (`step`) avoids it: all keys and values at a step use that step's projection, while the module still decides what to look for (`W_Q`), what to write back (`W_O`) and how to transform the token (MLP).

```
One middle step s with kv_mode = step (three tokens; token 2 chose a different module):

               token 1 -> M3        token 2 -> M4        token 3 -> M3
keys/values    K = W_K[s]·x, V = W_V[s]·x for every token (one shared set per step)
query          W_Q[M3]·x1           W_Q[M4]·x2           W_Q[M3]·x3
then           attention over the shared K/V -> W_O and MLP of the token's own module
```

| `kv_mode` | Module owns | Step owns | How attention works at step `s` | Notes |
|---|---|---|---|---|
| `step` (default, variant A) | `W_Q`, `q_norm`, `W_O`, MLP norm, MLP | Attention-input norm, `W_K`, `k_norm`, `W_V` | Every token writes K/V with the **step's** projection from its current state. Each token's query uses its chosen module's `W_Q`. | One K/V set per step, as in a normal model. Same split as Mixture of Attention Heads. |
| `module_band` (variant B) | Everything (attention norm, `W_Q`, `q_norm`, `W_K`, `k_norm`, `W_V`, `W_O`, MLP norm, MLP) | Nothing | For every module `m` that someone may use at step `s`, every token's state is projected with **module `m`'s** `W_K`, `W_V`. A token running `m` reads only module `m`'s K/V. | The module keeps its full identity. K/V cost grows with the window: about `2r+1` sets per step for an absolute window in `compute_matched` mode, twice that in `capacity`. |
| `module_mixed` (diagnostic) | Everything | Nothing | Each token writes K/V **once**, with its own chosen module's projection; queries read whatever each token wrote. | The per-token setting that Library of Layers reports as unstable. Stage 0b only. |
| `lol_chunk` (variant C) | Everything | Nothing | Routing per chunk of 128 tokens, not per token; attention inside the chunk (section 3.5.1) | Our version of Library of Layers |
| `exclude` (variant D) | Everything | Nothing | A token attends only to positions up to and including its own whose token chose the **same** module at this step | DNA-style. Stage 0b only. |

With `r = 0`, both `step` and `module_band` are identical to the corresponding standard model (section 3.2).

**What moves between steps is not an intact decoder block.** With `kv_mode = step`, a module is its `W_Q`, `q_norm`, `W_O`, MLP norm and MLP. The attention-input norm, `W_K`, `k_norm` and `W_V` stay with the step.

**Route scope (`route_scope`).**
- `qo_mlp` (default): a module owns `W_Q`, `q_norm`, `W_O` and the MLP, as above.
- `mlp`: the step also owns `W_Q`, `q_norm` and `W_O`, so a module is only its MLP norm and MLP. This is a ReXMoE-style control (grid row 18).
- `mlp` is used only with `kv_mode = step`.

#### 3.5.1 `lol_chunk` details (after the Library of Layers paper)

The paper calls these token groups "windows"; this spec calls them **chunks**, to avoid confusion with candidate windows (section 3.3).

**Source check.** This section was checked against the paper's text: Algorithm 1, sections 3.2 and 3.3, and the training details in its appendix. The PDF is at https://openreview.net/pdf?id=O4y4PcNmYD.
- **The paper's reason for chunks:** with per-token routing, "a vector at one token position extracted by a selected attention layer might attend to vectors at previous token positions extracted by different selected attention layers. We observe that this mismatch makes training unstable and harms performance."
- **Document blocking:** the paper also uses document-blocked attention, as this spec does (section 4.1).
- **Optimizer:** the paper uses Muon; this spec keeps AdamW for every variant.

- **Chunks:** size `chunk_size = 128`, stride `chunk_stride = 64`. At step `s`, position `p` belongs to chunk `floor((p + (s · chunk_stride mod chunk_size)) / chunk_size)`. Here `p` is the position within the document (positions restart at each document, section 4.1), so chunks never span two documents.
  - On even steps the boundaries are at 128, 256, ...
  - On odd steps they are at 64, 192, ..., so the first chunk has 64 tokens.
- **Shared choice:** all tokens of a chunk run the same module at that step, with the same router gate value.
- **Router input (causal):**
  - Chunk `k >= 1`: the mean of the per-chunk mean states of all earlier chunks `0 .. k-1`. Use the states entering step `s`.
  - Chunk 0: `0.5 · (e_s + h_first)`, where `e_s` is a learned vector per step and `h_first` is the state of the document's first token.
  - The router itself (norm, logits, balancing, exploration) is the one of section 3.7, applied to this input. Candidates: all modules (free window).
- **Attention:** a token attends only to positions up to and including its own, inside its own chunk, with its module's own Q/K/V/O. The stem and coda use full causal attention.
- **Not used here:**
  - The visit-count router input (paths differ inside a chunk, because chunks shift).
  - Quotas (reuse is always on).
  - STOP and re-entry.
- **Differences from the paper (deliberate):** we use the same router, gate (`st`), FiLM and balancing as the other variants, so that only the routing unit and the attention scope differ. The paper scales the update by the router probability (our `gate = scale`) and applies FiLM after every norm.

#### 3.5.2 Stopped and waiting tokens

At every executed step, tokens that are stopped or waiting still write K/V from their current (held) state:

- `step`: with the step's `W_K`, `W_V`;
- `module_band`: with each needed module's `W_K`, `W_V`;
- `module_mixed`: with the home module's projection;
- `exclude`: nothing (nobody attends to them at this step).

They do not query. This follows CALM and FlexiDepth and keeps every token visible to later tokens.

### 3.6 One middle step: exact computation

Batch shapes:
- `h`: `[B, T, d]`;
- `active`: `[B, T]` bool;
- `choice`: `[B, T]` module ids;
- `gate`: `[B, T]`, equal to 1 in value (section 3.7).

`kv_mode = step`:

```
def middle_step_step_kv(h, s, choice, gate, active):
    x = attn_norm[s](h)                          # step-owned norm, all tokens
    k = rope(k_norm[s](W_K[s] @ x))              # all tokens, including stopped/waiting
    v = W_V[s] @ x                               # (k, v: 8 K/V heads of head_dim 128)
    q = zeros(B, T, H*head_dim)                  # 16 query heads
    for m in unique(choice[active]):             # implement as MoE dispatch (sort by module)
        idx = active & (choice == m)
        q[idx] = q_norm[m](W_Q[m] @ x[idx])      # q_norm, k_norm act per head over head_dim
    a = causal_sdpa(rope(q), k, v)               # one GQA call for the whole batch, causal and
                                                 # blocked by document (section 4.1)
    delta = zeros_like(h)
    for m in unique(choice[active]):
        idx = active & (choice == m)
        h1 = h[idx] + W_O[m] @ a[idx]
        u  = film(s, mlp_norm[m](h1))            # step FiLM, section 3.8
        delta[idx] = (h1 + MLP[m](u)) - h[idx]   # the module's full update
    return where(active, h + gate * delta, h)
```

`kv_mode = module_band`:

```
def middle_step_band(h, s, choice, gate, active):
    delta = zeros_like(h)
    for m in unique(choice[active]):             # training: only modules someone chose
        x_m = attn_norm[m](h)                    # module-owned norm, ALL tokens
        k_m = rope(k_norm[m](W_K[m] @ x_m))      # all tokens' K/V in module m's projection
        v_m = W_V[m] @ x_m
        idx = active & (choice == m)
        q = zeros(B, T, H*head_dim); q[idx] = q_norm[m](W_Q[m] @ x_m[idx])
        a = causal_sdpa(rope(q), k_m, v_m)       # only rows idx are used
        h1 = h[idx] + W_O[m] @ a[idx]
        u  = film(s, mlp_norm[m](h1))
        delta[idx] = (h1 + MLP[m](u)) - h[idx]
    return where(active, h + gate * delta, h)
```

**Two cases the code above does not show:**
- **Soft phase (curriculum B):** run the step once for each of the two candidates, and mix the two deltas as in section 3.7.
  - Both candidates start from the same input `h` and the same visit vector.
  - K/V writing, and `exclude` membership, follow the top-1 choice. The second candidate's query reads the same keys and values (with `exclude`: the keys whose top-1 is that candidate, plus the token's own position).
  - Exceptions: with `module_band` and `lol_chunk`, each candidate reads its own module's K/V (for `lol_chunk`, within the chunk). With `lol_chunk`, the top two are chosen and mixed per chunk.
- **Second pass (re-entry):** the same code, with second-pass router biases and FiLM vectors. Keys and values come from the stored first-pass cache (section 3.10).

**Decoding with a KV cache, per `kv_mode`** (test T10 checks that it matches the full forward pass):
- `step`: one cache per step. Every token appends its K/V at every first-pass step, stopped and waiting tokens included.
- `module_band`: one cache per (step, module), filled eagerly. At step `s`, every token's state is projected with every module in `P(s)`, the set of modules that any token could choose at step `s`:
  - `C(s)` for absolute and bucket windows;
  - `{m : |pos(m) - s| <= drift_cap}` for relative windows (all modules without a drift cap);
  - all modules for the free window.

  A later token may choose a module that no earlier token needed, so caching only the chosen modules is not enough. Memory per step is `|P(s)|` times one K/V set.
- `module_mixed`: one cache per step; each token appends the K/V of the module it ran (stopped tokens: the home module's).
- `exclude`: one cache per step that also stores each entry's module id; a query reads only the entries with its own module id.
- `lol_chunk`: per step, keep a running sum of the completed chunks' mean states. A chunk's module is chosen when its first token arrives, and the cache only needs the current chunk's K/V.
- Re-entry: the second pass reads the first-pass caches and appends nothing. The `reentry.attention = recompute` ablation is training-only and has no cached decoding.

**Efficiency:**
- Group tokens by module (argsort by `choice`), run each module on a contiguous slice, then scatter back. Grouped GEMM is preferred.
- Never compute every module for every token.
- `lol_chunk` and `exclude` need block-structured masks on top of the document mask; FlexAttention is the simplest way to build them.
- **Stopped tokens in training:** stopped and waiting tokens still occupy attention rows; their queries are simply unused. STOP saves training compute only through module dispatch, and its real savings appear in decoding.
- **Router latency:** router FLOPs are tiny, but dispatch, small matrix multiplies and synchronisation can dominate latency. Measure; do not infer speed from FLOPs.

### 3.7 Router

There is one router per step. It costs under 0.1% of compute. Compute logits and softmax in fp32.

**Inputs for token `t` at step `s`:**
- `x = RMSNorm_router_s(h_t)`: the token's current state;
- `b[s, :]`: a learned bias per step and module;
- `c_t`: how many times token `t` has used each module so far, counted across both passes (a vector of length `M`).

**Logits:**

```
z[m] = w[s, m] · x + b[s, m] + (U[s] @ c_t)[m]      for m in C_t(s)
z[m] = -inf                                           otherwise
p    = softmax(z)
```

Shapes: `w` is `[S, M, d]`, `b` is `[S, M]`, `U` is `[S, M, M]`, and the balance bias `beta` is `[M]`.

- **Selection (top-1):** `m* = argmax(z + beta)`. Here `beta` is the per-module balance bias. It is used for selection only and never enters `p`. Ties go to the lowest module id, including ties between the two home experts in `capacity` mode.
- **Router gate (`gate = st`, straight-through, default):** `g = exp(log p[m*] - stopgrad(log p[m*]))`. This equals `p[m*] / stopgrad(p[m*])` but avoids 0/0 when exploration picks a candidate whose probability has underflowed.
  - In the forward pass `g = 1`, so the token runs exactly module `m*`.
  - In the backward pass the router receives the signal "would a larger update from `m*` have lowered the loss?".
  - This is a surrogate gradient. It does not tell the router whether another module would have done better, so exploration and the soft phase are real optimisation choices, not details.
  - The alternative `gate = scale` uses `h + p[m*] · delta`.
- **Default prior and initialisation:**
  - `b[s, home(s)] = default_prior` (+1.0); in `capacity` mode, both home experts get it.
  - `w` is initialised with std `0.1 / sqrt(d)`, and `U` with zeros.
  - At initialisation the model therefore follows the default path.
- **Balancing (loss-free, after Wang et al.):**
  - **Counts:** for each first-pass step `s`, count the active tokens and, for each module `m`, the active tokens whose top-1 choice is `m`. Sum both over all micro-batches and all ranks of the training step (all-reduce) before using them. Second-pass (re-entry) choices are not counted.
  - **Shares:** `share_s(m)` is the second count divided by the first. Skip any step with no active token.
  - **Load and target:** `load[m]` is the mean of `share_s(m)` over the non-skipped steps. `target[m]` is the same mean taken for the default path, where `default_share_s(m)` is 1 for the home module of step `s` (0.5 for each of the two home experts in `capacity` mode) and 0 otherwise.
  - **Update:** once per training step, after the backward pass: `beta[m] += balance_rate · sign(target[m] - load[m])`. A model on the default path therefore never moves `beta`, with or without STOP.
  - **Timing:** `beta` stays fixed within a training step and during evaluation, so no sequence's routing depends on its own or later tokens. It is saved in checkpoints.
- **z-loss:** add `z_loss_coef · mean(logsumexp(z over candidates)^2)` to the loss. The mean is over every router decision of an active token in both passes, with the sum and the count reduced over micro-batches and ranks. Reported loss and perplexity exclude router terms.
- **Exploration:** in training only, with probability `explore_eps`, replace `m*` with a uniformly random allowed candidate.
- **Soft phase (curriculum B):**
  - Choose the top two allowed candidates by `z + beta`, as for top-1, and weight them by `p` renormalised over the pair.
  - Both candidates start from the same `h` and the same visit vector: `h_out = h + sum_c w_c · delta_c`.
  - In `step` mode this costs about `2 - kv_frac` times the hard middle compute, during the soft phase only (section 3.11).
  - Visit counts, quotas and balancing count only the top-1 choice. If only one candidate is allowed, use top-1.
  - Soft training is a relaxation of the hard model. With `reuse = false` a soft path is not an exact permutation; only the hard path is.
- **Causality:** every router input is the token's own state, so routing is causal.
  - Do **not** use expert-choice routing (top-k tokens per module over the sequence). It reads future tokens.
  - Never drop tokens (no capacity factor), here or in the FFN-MoE baseline. Every token runs what it chose; otherwise a token's computation would depend on later tokens.

**Control routings (`routing`):**

| `routing` | Choice | Purpose |
|---|---|---|
| `learned` | Router as above | Main |
| `fixed` | The home module (in `capacity` mode, expert `2s`) | Equals `r = 0` |
| `random` | Uniform over the allowed candidates, fresh each forward pass | Chance control |
| `hash` | The k-th allowed module in increasing id order, with `k = mix32(token_id, s) mod n_allowed`. `token_id` is the token's vocabulary id; `mix32` is in section 12. | Fixed-assignment control (as in Hash Layers) |

### 3.8 Step FiLM and visit scaling

- **Step FiLM (`step_film = true`):**
  - Each step, and each second-pass step (section 3.10), has zero-initialised vectors `film_gain_s, film_bias_s ∈ R^d`.
  - The module's MLP input becomes `u · (1 + film_gain_s) + film_bias_s`.
  - This lets a module used at several steps behave differently at each.
- **Visit scaling (`visit_scale`):**
  - `none` (default): no scaling.
  - `inv`: the update from a token's v-th use of the same module (v ≥ 2) is multiplied by `1/v`.
  - Test `inv` only in `depth` mode (Stage 0b, P4).
  - The equivalence test T1 uses `none`.

### 3.9 STOP (`stop.policy`)

- **When:** after step `s`, for `s + 1 >= stop.min_steps` and `s < S - 1`.
- **Decision (`stop.policy`):**
  - `none`: never stop.
  - `random` (trunk training): `stop ~ Bernoulli(stop.train_random_prob)` per active token. Training only; at evaluation use `none`, `learned` or `constant`.
  - `learned` (after the STOP head is trained): stop if `sigmoid(stop_head(RMSNorm(h_t)) + b_stop[s]) > stop.threshold`.
  - `constant` (Stage 3 baseline): `stop ~ Bernoulli(stop.constant_prob)`, also at evaluation.
- **Effect:**
  - The token becomes inactive and its state is held.
  - It still writes K/V at the remaining steps (section 3.5.2).
  - It goes through the coda layers with everyone else.
- **STOP head:** `Linear(d, 1)` plus a per-step bias, all initialised to zero. Since `sigmoid(0) = 0.5`, nothing stops at threshold 0.5 at initialisation.
- **Scope:** STOP requires `reuse = true`. It is not combined with re-entry, `lol_chunk` or `exclude` in this plan.

### 3.10 Re-entry (`reentry.policy`)

Build the re-entry mechanics before Stage 1, which uses forced re-entry. Only the learned re-entry head (section 4.4) waits until Stage 4. Re-entry requires `reuse = true`.

- **Re-entry points:** after steps `c = reentry.min_steps - 1 + k · point_every`, for `k = 0, 1, ...` and `c < S - 1`.
  - With the defaults (`S = 8`, `reentry.min_steps = 2`, `point_every = 2`), these are after steps 1, 3 and 5.
- **Choices at a re-entry point:** an active token may continue, or re-enter at a target step `j`. Each token re-enters at most once.
  - Allowed targets are block starts `j ∈ {0, point_every, 2·point_every, ...}` with `j <= c`.
  - They must also fit the budget of extra runs: `(c + 1) - j <= reentry.max_extra_steps`. A re-entering token therefore runs at most `S + max_extra_steps` modules.
  - If no target is allowed at a re-entry point, re-entry is not possible there.
  - With the defaults (`S = 8`, `max_extra_steps = 4`):

    | Re-entry point (after step) | Allowed targets |
    |---|---|
    | 1 | 0 |
    | 3 | 0, 2 |
    | 5 | 2, 4 |

- **What happens on re-entry:**
  1. **Rest of the first pass:** the token behaves like a stopped token. It holds its state and writes first-pass K/V at steps `c+1 .. S-1`.
  2. **Re-injection:** `h ← h + U_r @ (D_r @ RMSNorm(stem_out_t))`.
     - `stem_out_t` is the token's state entering the middle.
     - `D_r` maps `d → reinject_rank`; `U_r` maps back and is zero-initialised.
  3. **Second pass:** the token runs steps `j .. S-1`.
     - At each second-pass step `s`, its query reads the **first-pass K/V cache of step `s`** for all positions up to and including its own. Its own current state is carried by the residual stream.
     - It uses second-pass router biases and FiLM vectors (initialised like the first-pass ones), and the same router weights and modules.
     - Windows follow the same rules as in the first pass. For `relative` windows, `prev` at step `j` is the module the token ran at step `j - 1` of the first pass (none if `j = 0`).
  4. **Coda:** the final state goes to the coda.
- **Training implementation:**
  - First run the full first pass for all tokens, storing each step's K/V and the re-entry decisions.
  - Then run the second pass for the re-entering tokens only, reading the stored caches.
  - This is exact, because no token reads second-pass states inside the middle.
- **Policy (`reentry.policy`):**
  - `none`: never re-enter.
  - `random` (trunk training, and a Stage 4 control): each token, with probability `reentry.random_prob`, re-enters at a (re-entry point, target) pair drawn uniformly from all allowed pairs.
  - `repeat_only` (Stage 4 control): like `random`, but the target is always `j = c + 1 - point_every`, i.e. redo the block just finished.
  - `forced`: tokens flagged by the data loader (`force_mask`) re-enter at `reentry.forced_point` with target `reentry.forced_target`. The defaults are the last re-entry point and the smallest allowed target.
  - `learned`: `re_head(RMSNorm(h))` at each re-entry point, with classes `none` plus each target allowed there; the decision is the argmax.
    - Use one linear layer per re-entry point.
    - Initialise the weights to zero and the `none` bias to +1, so that nobody re-enters before the head is trained.
- **Supported `kv_mode`s:** `step` and `module_band`. For `module_band`, store each step's input states during the first pass, and compute a module's first-pass K/V from them when a second-pass token needs it.
- **Ablation (`reentry.attention = recompute`):** at second-pass steps, recompute every token's K/V from current states (the v1 behaviour).

### 3.11 Parameters and compute

Weights per Qwen3 layer: `P_layer = d·q_dim (W_Q) + 2·d·d_kv (W_K, W_V) + q_dim·d (W_O) + 3·d·d_ff`. Norms add `2·d` (two RMSNorms) plus `2·head_dim` (`q_norm`, `k_norm`). For Qwen3-0.6B, `P_layer` = 15.73M.

| Middle parameters | `kv_mode = step` | `kv_mode = module_band` |
|---|---|---|
| Per module | `2·d·q_dim + 3·d·d_ff` (13.63M) | `P_layer` (15.73M) |
| Per step | `2·d·d_kv` (2.10M) | 0 |

Reference sizes at the defaults: Qwen3-0.6B dimensions (`d = 1024`, `q_dim = 2048`, `d_kv = 1024`, `d_ff = 3072`, vocabulary 151,936, tied embedding), stem 2, coda 2, `S = 8`.

| Mode | `M` | Middle, `step` | Total, `step` | Middle, `module_band` | Total, `module_band` |
|---|---|---|---|---|---|
| `compute_matched` | 8 | 125.8M | 344.4M | 125.8M | 344.4M |
| `capacity` | 16 | 234.9M | 453.6M | 251.7M | 470.4M |
| `depth` | 4 | 71.3M | 289.9M | 62.9M | 281.5M |

Totals include the embedding (155.6M), stem and coda (62.9M), and routers and FiLM (under 0.2M).

- **With `kv_mode = step`:** all three modes have the same FLOPs per token as the 12-layer dense model.
- **With `module_band`:** every needed module's K/V adds cost. For example, `capacity` at `r = 0` computes both experts' K/V, which is about 1.07 extra LE per token.
- **The LM head is large.** With a 151,936-token vocabulary it costs about as much as 10 layers per token: 45% of the 12-layer model's weight FLOPs. `LE` does not count it; measured speed does.

**Compute accounting per token, in layer-equivalents:**

```
LE_t = n_stem + n_coda + runs1_t + (1 - kv_frac) * runs2_t + kv_frac * (kv_only_t + extra_kv_t)
  runs1_t    = module runs by token t in the first pass
  runs2_t    = module runs in the second pass (re-entry); they read stored K/V and write none
  kv_only_t  = first-pass steps at which t only wrote K/V (stopped or waiting)
  extra_kv_t = module_band only: K/V sets written for t beyond the one its own module needs,
               counting every module in the step's candidate union, as decoding must
kv_frac = 2·d·d_kv / P_layer     # 2/15 ≈ 0.133 for Qwen3-0.6B (8 K/V heads of head_dim 128)
```

`LE` counts weight FLOPs only; it ignores attention-score FLOPs and the router. Always report measured tokens per second next to it.

**Three compute measures.** Report all three, and never call two runs "equal compute" without saying which measure is meant.
1. **Nominal hard-path LE** (the formula above): inference weight FLOPs.
2. **Executed training FLOPs.** These include:
   - the soft phase: about `2 - kv_frac` times the hard middle FLOPs in `step` mode. Over a run with a 15% soft phase, that is roughly +9% of layer weight FLOPs at the defaults, or about +5% once the LM head is included;
   - exploration;
   - re-entry second passes;
   - the extra coda runs for STOP labels.
3. **Measured wall-clock time.**

Grid comparisons are at equal tokens, not at equal total training compute.

---

## 4. Training

### 4.1 Data and optimiser (12-layer Qwen3, 344M parameters)

| Setting | Value |
|---|---|
| Data | The English pool `qwen3_0.6b_base_en_30B` of the Hugging Face dataset `nht10/cx_sampled` (about 30B tokens, Qwen3 tokenizer), at a pinned revision recorded in every `config.json`. It covers every stage, Stage 5 included. |
| Validation | The pool's existing validation split, packed like the training data and shuffled once at the sequence level with seed 0. The periodic set is the first 1,024 of these sequences (1,048,576 tokens). The final set is the first 10,240 (10,485,760 tokens), or all of them if the split is smaller. The validation manifest stores the sequence indices of both sets, which serve every run, budget and stage. Before any training, check that no document appears in both splits, by document id or, where ids are missing, by content hash. Drop any validation sequence that contains a training document, and record the check. |
| Training stream | The pool's training split, packed (next row) and then shuffled once at the sequence level with seed 0 (`packed_shuffle`). Exactly one shuffle defines this order: make sure nothing else reshuffles it (the Hugging Face Trainer's default sampler does). Every run reads the order from the start, with two exceptions: Stages 4 and 5 interleave the two-hop sequences of section 10.3, and a run that continues from a trained trunk (head training and labelling, Stage 3's fine-tuned pruned baselines) reads on from the first sequence after the trunk's last one. A run with a budget of N tokens uses the first `floor(N / batch_tokens)` training steps' worth of sequences; record the rounding. |
| Packing | Documents are separated by the end-of-text token `<|endoftext|>` and packed into 1,024-token sequences. Attention is blocked by document: no token attends across a document boundary, and RoPE positions restart at 0 at each document start. Every `kv_mode` applies its own mask on top of the document mask. |
| Random streams | Separate generators for data order (fixed, seed 0), model initialisation (`seeds.model`) and routing randomness (`seeds.routing`: exploration, random routing, random STOP and re-entry, label sampling). Evaluation uses its own fixed generator and never advances the training generators. |
| Tokenizer | The Qwen3 tokenizer, from `Qwen/Qwen3-0.6B-Base`. The embedding has 151,936 rows, as in the released model; this is more than the tokenizer uses, and the unused rows are never trained toward. |
| Context | 1,024 |
| Batch | 512 sequences × 1,024 = 0.5M tokens (use gradient accumulation as needed) |
| Optimiser | AdamW, betas (0.9, 0.95), weight decay 0.1 (none on norms, biases, router biases, FiLM vectors) |
| LR | Peak 3e-4, linear warm-up over `ceil(0.02 · max_steps)` training steps, then cosine decay to 10% of peak at `max_steps` |
| Schedule | Every run has one complete schedule: `max_steps = floor(tokens / batch_tokens)`. The LR schedule and the routing curriculum (section 4.2) share this horizon. A run used by a gate or a comparison trains all `max_steps` steps; it is never a longer schedule stopped early. An interrupted run is resumed, not shortened, and a larger budget is a new run with its own schedule. Epoch-based runs (section 10) use `max_steps = epochs × steps per epoch`, rounded down; section 10.1 gives the one allowed extension. |
| Early stop | `train.stop_after = N` ends a run after N completed steps without changing `max_steps`, so the LR and the curriculum still follow the full schedule. Use it only in tests (T12, T14) and in runs labelled exploratory. Gate and grid runs leave it unset. |
| Precision | bf16 (router in fp32), grad clip 1.0 |
| Init | As in the Hugging Face Qwen3 implementation: every linear and embedding weight normal with std 0.02, norm weights 1. Routers, FiLM and the re-injection layer use their own initialisation (sections 3.7, 3.8 and 3.10). |
| Tokens | 2.5B (Stages 0a and 0b, and the Stage 2 grid: 4,768 training steps), 5B (finalists, Stages 3 and 4: 9,536 training steps). Stage 5 has its own table. |

### 4.2 Routing curriculum (`router.curriculum`)

**Horizon.**
- Every fraction below is a fraction of the run's own schedule: `progress = completed training steps / max_steps`, with the `max_steps` of section 4.1.
- A new phase starts with the first training step taken after `progress` reaches its fraction.
- Every transition comes after the LR warm-up. The config loader rejects a run whose number of warm-up steps (`warmup_steps` if set, otherwise `ceil(warmup_frac · max_steps)`) is not smaller than its first boundary: `ceil(widen_every_frac · max_steps)` for A, `ceil(soft_phase_frac · max_steps)` for B (477 and 716 when `max_steps = 4,768`).

- **A, narrow to wide:**
  - For the first 10% of the schedule, allow only the home module(s) of each step: the default path, or in `capacity` mode a choice between the two home experts. This holds for every window type.
  - Then the reach grows by 1 every 10% of the schedule until it reaches the target. For bucket windows, switch from size 1 straight to the target size.
  - At each widening, `explore_eps` jumps to `explore_eps_max` and decays linearly to 0 over the next 10%.
- **B, soft then hard:**
  - The target window from the start.
  - Soft top-2 mixing for the first 15% of the schedule, then hard top-1. No exploration during the soft phase.
  - `explore_eps` decays linearly from `explore_eps_max` to 0 between `soft_phase_frac` (15%) and `explore_end_frac` (50%) of the schedule.
- **none:** the target window and hard top-1 from the start, with no exploration. Use only for `r = 0` and for the `random` and `hash` controls.

For a 2.5B-token run (4,768 training steps) with target `r = 1`, the boundaries fall at these counts of completed training steps (the next step uses the new phase):

| Event | Completed steps |
|---|---:|
| LR warm-up ends | 96 |
| A: the window opens to `r = 1` (10%) | 477 |
| A: exploration reaches 0 (20%) | 954 |
| B: soft top-2 ends, hard top-1 starts (15%) | 716 |
| B: exploration reaches 0 (50%) | 2,384 |

Stage 0a decides between A and B. The `free` window always uses B.

### 4.3 STOP training

1. **Trunk.** Train with `stop.policy = random` and `stop.train_random_prob = 0.05` per allowed step. This teaches the coda layers to read states from every depth.
2. **STOP head.** Freeze the trunk.
   1. For each batch, run the middle without stopping. After each step `s` from `stop.min_steps - 1` to `S - 1`, run the coda on all tokens and record each token's loss `L[s]`.
      - `L[s]` at position `t` is the loss of predicting token `t + 1`. The STOP head sees only the state at `t` after step `s`; losses are supervision only.
      - This treats all tokens as exiting together. Mixed exits change other tokens' attention context, so the approximation can err in either direction. Treat the labels as a heuristic and judge the head by the measured quality and speed of the real mixed policy (step 3).
   2. Labels for `s <= S - 2`:
      - `label_rule = next`: `y[s] = 1` (stop) if `L[s] - L[s+1] < stop.label_margin`.
      - `label_rule = any_later`: `y[s] = 1` if `L[s] - min(L[s+1:]) < stop.label_margin`.
   3. Train the STOP head with binary cross-entropy on `(h after step s, y[s])`, using the `heads` settings of section 8 (AdamW, LR 1e-3, 500 training steps, batch 64 × 1,024 tokens).
3. **Evaluation.** With `stop.policy = learned`, sweep `stop.threshold ∈ {0.3, 0.5, 0.7, 0.9, never}`. At each threshold, report loss, benchmark accuracy, mean steps, `LE` and measured speed.

Do not train STOP jointly with ACT- or PonderNet-style objectives. Unsupervised halting of this kind can collapse to the minimum depth: RecurTrace's ACT and PonderNet baselines did, and Closing the Loop saw it at a low entropy weight without warm-up. The two-stage, label-based recipe above avoids the problem.

### 4.4 Re-entry training

1. **Trunk.** Train with `reentry.policy = random` and `reentry.random_prob = 0.05`.
2. **Re-entry head.** Freeze the trunk.
   1. Fix a set of 200 training batches of 64 × 1,024 tokens (`heads.label_batches`). In each batch, pick a random 10% of tokens (`heads.label_frac`).
   2. For each allowed (re-entry point, target) pair, run the batch with the picked tokens forced to re-enter at that pair; the other tokens use policy `none`. Record each picked token's gain: its loss without re-entry minus its loss with re-entry.
   3. Labels, for each picked token and each re-entry point `c`:
      - let `g(c)` be the largest gain over the targets allowed at `c`;
      - the label at `c` is that best target if `g(c) > reentry.label_margin` and no later re-entry point has a larger gain; otherwise it is `none`;
      - in words: re-enter now if it helps and waiting would not help more.
      - Forcing 10% of tokens at once measures a joint intervention, not independent per-token gains. Treat these labels as a heuristic too, and judge the head by the deployed policy's measured quality (Stage 4).
   4. Train `re_head` with cross-entropy over the allowed classes at each re-entry point, on the picked tokens only, cycling through the 200 labelled batches. Use the `heads` settings of section 8 (AdamW, LR 1e-3, 500 training steps).

---

## 5. Evaluation and metrics

**Log every 100 training steps:**
- train loss and router z-loss;
- per-module load (max/min ratio);
- off-default share per step;
- reuse rate: the fraction of tokens whose path uses some module more often than its default schedule does;
- router entropy;
- for STOP and re-entry runs: the histogram of exit steps or re-entry pairs;
- tokens per second and peak memory.

**Validation:** loss on the periodic validation set (1M tokens, section 4.1), every 100 training steps in Stages 0a and 0b and every 250 otherwise. Use the final validation set (about 10M tokens) at the end of each run. Reported loss and perplexity never include router terms.
- **Routing during periodic validation:** the routing the model is currently trained with, without exploration.
  - Learned routing: soft top-2 during curriculum B's soft phase, and hard argmax over the current window otherwise.
  - `fixed`, `hash` and `random` routing: that routing. `random` draws from the fixed evaluation generator.
  - STOP and re-entry trunks trained with random policies: policy `none`.
- **During the soft phase:** also log the validation loss with hard argmax routing. Its gap to the soft loss shows how far soft training is from the deployed policy.
- **At each curriculum transition** (A's widening, B's switch to hard routing):
  - validate the weights at the transition step twice, once with the old phase's routing and once with the new one's;
  - log the jump: new minus old.

**At the end of each run** (routing as trained, without exploration: argmax for learned routing; the controls as in periodic validation):
- Validation loss.
- Zero-shot accuracy with lm-eval-harness on `hellaswag`, `arc_easy`, `arc_challenge`, `piqa`, `winogrande` and `lambada_openai`. Use `acc_norm` where the harness provides it. Report each task and the average. All six tasks are scored by log-likelihood, so the model only needs the harness's log-likelihood interface. (Stage 5's generation tasks also need `generate_until`; see section 9.)
- Route diagnostics on the first 4,096 sequences of the final validation set:
  - off-default share;
  - reuse rate;
  - distinct paths per 10,000 tokens;
  - mean path length (with STOP).
- **Route interventions** on the trained model, compared with its own routes on the same validation tokens:
  - **default-path test:** validation loss with `routing = fixed` (home modules only). Its rise shows how much the learned deviations from the default order are worth. Gate 0a uses it.
  - **random-route test:** validation loss with `routing = random` in the same windows, using a fixed evaluation seed. Its rise shows how sensitive the model is to its routes. It can be large even when the learned deviations are worthless, so it does not gate anything.
- **Cross-seed route agreement:** the fraction of (token, step) pairs where two seeds choose the same module on the same validation tokens. In `capacity` mode, compare `pos(m)` instead of `m`, because a step's two experts are interchangeable.
- **Speed:**
  - training tokens/s;
  - decode tokens/s at batch 1 and batch 32 with a KV cache;
  - prefill tokens/s.

**Files per run:** `config.json`, `metrics.jsonl`, `final_eval.json`, and `routes_sample.npz` (the chosen module per step for the first 256 sequences of the final validation set).

---

## 6. Build order

Build in phases, so that no compute or engineering goes into later variants before the first pilot shows a signal.

**Reuse existing infrastructure** (trainer, Accelerate or DDP, checkpointing) where its contracts match this spec. Check in particular:
- document masking and position resets (section 4.1);
- gradient accumulation and the global normalisation of the loss, z-loss and balancing counts (section 3.7);
- resume of all non-optimizer state (T12).

Implement the B-DOT model on its own, not by modifying an existing example model.

**Phase A, before Stage 0a (the minimal pilot):**

0. **Infrastructure:** the data pipeline and validation manifest (section 4.1), the training loop with logging (section 5), saving and resuming of checkpoints, and the lm-eval-harness wrapper.
1. **Native dense reference:** the Qwen3 architecture of section 3.1 with `n_stem + S + n_coda` layers, from random initialisation. Hugging Face `Qwen3ForCausalLM` with the Qwen3-0.6B config and `num_hidden_layers = 12` can serve directly. It is the reference for T1 and T14. It is **not** run D of Stage 0a: D is the B-DOT model at `r = 0` with `routing = fixed` and the routed runs' FiLM option.
2. **B-DOT core, minimal:**
   - `mode = compute_matched`, `window = absolute`, `kv_mode = step`;
   - `routing = fixed`, `learned` and `random`;
   - `reuse = true` and `false` (with the feasibility tables);
   - balancing, z-loss, exploration, soft top-2, curricula A and B;
   - KV-cache decoding for `step`.
3. **Pass:** T1 to T3, plus T4, T5, T9 and T10 for this subset; then T8 and T11 to T14.

**Phase B, only after Gate 0a passes (before Stage 0b):**

4. **Windows:** `relative`, `bucket` and `free`.
5. **Modes:** `capacity` and `depth`.
6. **K/V variants:** `module_band`, `module_mixed`, `lol_chunk`, `exclude`; `route_scope = mlp`; `hash` routing.
7. **Decoding:** KV-cache decoding for every `kv_mode`.
8. **Pass:** T1, T2, T4, T5, T9 and T10 for everything added.

**Phase C, before Stage 1:**

9. **Re-entry mechanics:** first-pass cache, second pass, re-injection, and the `none`, `random`, `repeat_only` and `forced` policies. Pass T7, then T2 and T10 with re-entry on.

**Phase D, before Stage 2:**

10. **FFN-MoE baseline** (grid row 15). The standard looped baseline (row 16) needs no new code.

**Phase E, before Stage 3:**

11. **STOP:** the `random`, `learned` and `constant` policies, exit losses, STOP head, threshold sweep. Pass T6, then T2 and T10 with STOP on.

**Phase F, before Stage 4:**

12. **Learned re-entry head** (section 4.4).

---

## 7. Acceptance tests

When they must pass:
- before Stage 0a: T1 to T5 and T8 to T14, for the Phase A subset;
- before Stage 0b: T1, T2, T4, T5, T9 and T10 for everything Phase B adds (section 6);
- T7 before Stage 1;
- T6 before Stage 3.

Rerun T2 and T10 whenever STOP or re-entry is added.

| ID | Test | Pass condition |
|---|---|---|
| T1 | **Equivalence.** `routing = fixed`, `step_film = false`, `visit_scale = none`, fp32 with TF32 disabled. Compare against Hugging Face `Qwen3ForCausalLM` (Qwen3-0.6B config, `num_hidden_layers = n_stem + S + n_coda`), with its weights copied from the B-DOT model (section 3.2). Layer `s` of the reference gets the weights of step `s`'s module, plus the step's attention norm, `W_K`, `k_norm` and `W_V` in `step` mode; for `capacity`, that is expert `2s`. Run all three modes, for `kv_mode = step` and `module_band`. Where a B-DOT weight appears in several reference layers (`depth` mode), compare its gradient with the sum over those layers. Compare logits, all parameter gradients (mapped to the reference's parameters), and the parameters after one AdamW update. | Max abs difference < 1e-4 for logits, gradients and updated parameters |
| T2 | **Causality.** Change the tokens after position `t` and compare logits at positions ≤ `t`. Use learned routing with random router weights, in eval mode. Run every `kv_mode`. Then add STOP where supported (`step`, `module_band`, `module_mixed`; `stop.policy = learned` with a random STOP head). Then add re-entry where supported (`step`, `module_band`; `reentry.policy = forced` on fixed positions, and later `learned` with a random head). | Identical up to 1e-4 |
| T3 | **Router gradients.** One backward pass with `gate = st`. | Router weights and step biases get non-zero gradients |
| T4 | **Masks.** Run 10,000 random routes per (mode, window, reuse) setting; enumerate all paths for the settings in section 3.4. | Every choice is inside `C_t(s)`; with `reuse = false` the quotas are met exactly; path counts match section 3.4 |
| T5 | **Controls.** | `hash` gives the same routes on repeated calls; `random` differs between calls; both stay inside the windows |
| T6 | **STOP bookkeeping.** `stop.policy = random`, training mode. | A token stopped after step `s` has `s + 1` runs; its state does not change after stopping; its K/V are still written |
| T7 | **Re-entry bookkeeping**, with the `random` and `forced` policies. | Total runs ≤ `S + max_extra_steps`; the second pass reads first-pass caches; at most one re-entry per token |
| T8 | **Speed.** Benchmark the native dense reference, D, and `compute_matched` absolute `r = 1`, with the same micro-batch layout, compile settings and logging. Time training steps only, without validation or checkpoint saves. | Report all three. Gate 0a needs routed training throughput of at least 60% of the native dense reference's; below that, improve the dispatch before judging the pilot. |
| T9 | **Parameter counts.** | Match section 3.11 within 0.5% |
| T10 | **Decode consistency.** Eval mode, fp32, deterministic routing (`learned`, `fixed` or `hash`). For every `kv_mode`, and with STOP or re-entry on where supported, decode token by token with the KV cache and compare with one full-sequence forward pass. | Max abs logit difference < 1e-4 at every position |
| T11 | **Distributed correctness.** Two ranks, fp32, one global batch, learned routing with gradient accumulation of 2. Arrange one case where a module is used on one rank only, and one where a module is used on no rank. Compare with a single process on the same global batch. | Loss, gradients, balancing counts and `beta` match within 1e-5; no hang or unused-parameter error |
| T12 | **Resume.** Train curriculum B with `max_steps = 1,000` and `stop_after = 200`, so that steps 151 to 200 run hard routing with exploration on. Save at step 100, resume, and compare steps 101 to 200 with an uninterrupted run (deterministic kernels). The checkpoint must hold `beta`, curriculum progress, and the routing and data generators, as well as the optimizer state. If activation checkpointing is used, the recomputed forward pass must reuse the stored routes. | Identical routes; losses within 1e-6 |
| T13 | **Schedule horizon.** Build the LR schedule and both curricula for `train.tokens = 2.5e9` and `batch_tokens = 524288`, without training. Also load a curriculum-A config and a curriculum-B config, each with `warmup_frac = 0.2`. | `max_steps = 4,768`. The LR rises linearly for 96 steps, then follows the cosine to 10% of peak at step 4,768. The phase boundaries match the table in section 4.2. Both `warmup_frac = 0.2` configs are rejected. |
| T14 | **Pipeline check.** Build the B-DOT model in the row 17 configuration (`r = 0`, `routing = fixed`, `step_film = false`) and give it the native dense reference's initial weights (T1 mapping). Train both models for 500 steps with exactly the same recipe: initial weights, data order, batch layout and LR schedule. Either train a fresh pair on a complete 500-step schedule, or reuse a completed native run: give the B-DOT model that run's schedule with `stop_after = 500` (for the current native baseline, `max_steps = 57,220` with 1,145 warm-up steps). | Mean training loss over steps 401–500, and periodic validation loss at step 500, agree within 0.01 nats. If they do not, train the native model a second time with the same recipe: a similar gap between the two native runs means the difference is numerical noise, not a pipeline bug. Report both models' tokens per second. |

---

## 8. Configuration reference

```yaml
model:
  arch: bdot                # bdot | ffn_moe (grid row 15)
  vocab_size: 151936        # Qwen3 tokenizer (section 4.1)
  context_length: 1024
  d_model: 1024             # Qwen3-0.6B dimensions from here to qk_norm
  n_heads: 16
  n_kv_heads: 8             # must divide n_heads
  head_dim: 128             # q_dim = n_heads × head_dim = 2048
  d_ff: 3072                # SwiGLU hidden width
  rope_theta: 1000000
  rms_norm_eps: 1.0e-6
  qk_norm: true
  tie_embeddings: true
  n_stem: 2
  n_coda: 2
  S: 8                      # middle steps per pass: 12 layers in all (the released model has 28)
  mode: compute_matched     # compute_matched | capacity | depth   (M = S, 2S, S/2)
  window: absolute          # absolute | relative | bucket | free
  r: 1                      # reach; bucket size for 'bucket'
  drift_cap: null           # 'relative' only: max |pos(m) - s|; this plan uses 2
  reuse: true
  kv_mode: step             # step | module_band | module_mixed | lol_chunk | exclude
  routing: learned          # learned | fixed | random | hash
  step_film: true
  visit_scale: none         # none | inv
  route_scope: qo_mlp       # qo_mlp | mlp (MLP-only routing; kv_mode = step only)
data:
  dataset: nht10/cx_sampled
  subset: qwen3_0.6b_base_en_30B   # English pool, about 30B tokens; used by every stage
  revision: null            # pin a commit hash and record it
  order: packed_shuffle     # pack the training split, then shuffle the sequences once (seeds.data)
  validation_manifest: null # path to the fixed periodic (1,024 seq) and final (10,240 seq) validation sets
  doc_mask: true            # attention blocked by document; positions restart per document
lol:                        # used only when kv_mode = lol_chunk
  chunk_size: 128
  chunk_stride: 64
ffn_moe:                    # used only when arch = ffn_moe
  n_experts: 10             # per middle layer
  expert_width: 768         # d_ff / 4
  top_k: 4                  # gates = softmax over the selected top-k logits
router:
  gate: st                  # st | scale
  default_prior: 1.0
  init_std_scale: 0.1       # weight std = init_std_scale / sqrt(d)
  balance_rate: 1.0e-3
  z_loss_coef: 1.0e-3
  explore_eps_max: 0.05
  curriculum: B             # A | B | none
  soft_phase_frac: 0.15     # curriculum B: end of soft top-2
  explore_end_frac: 0.5     # curriculum B: exploration reaches 0 here
  widen_every_frac: 0.10    # curriculum A
stop:
  policy: none              # none | random | learned | constant
  min_steps: 2
  train_random_prob: 0.05   # 'random'
  constant_prob: null       # 'constant' (Stage 3 baseline)
  threshold: 0.5            # 'learned'
  label_margin: 0.02        # nats
  label_rule: any_later     # next | any_later
reentry:
  policy: none              # none | random | repeat_only | forced | learned
  min_steps: 2              # first re-entry point is after step min_steps - 1
  point_every: 2            # steps between re-entry points
  max_extra_steps: 4
  random_prob: 0.05         # 'random' and 'repeat_only' (trunk training and Stage 4 controls)
  forced_point: last        # 'forced'
  forced_target: smallest_allowed
  reinject_rank: 64
  attention: first_pass_cache   # first_pass_cache | recompute
  label_margin: 0.02
heads:                      # STOP head and re-entry head training (trunk frozen)
  train_steps: 500
  lr: 1.0e-3
  batch_tokens: 65536       # 64 sequences × 1,024
  label_batches: 200        # re-entry labelling: fixed batches
  label_frac: 0.1           # re-entry labelling: share of tokens picked per batch
train:
  tokens: 2.5e9             # max_steps = floor(tokens / batch_tokens): one horizon for the LR and the curriculum
  epochs: null              # synthetic tasks: train for this many epochs instead of 'tokens'
  batch_tokens: 524288
  stop_after: null          # gate and grid runs: must be null (never stop a longer schedule early)
  lr: 3.0e-4
  lr_schedule: cosine       # cosine | constant
  warmup_frac: 0.02         # warm-up = ceil(warmup_frac · max_steps); must end before the first curriculum boundary
  warmup_steps: null        # if set, overrides warmup_frac
  min_lr_frac: 0.1
  weight_decay: 0.1
  grad_clip: 1.0
  precision: bf16
seeds:
  data: 0                   # data order (packed_shuffle); the same for every run
  model: 0                  # initialisation; varies across seeds of a run
  routing: 0                # exploration, random routing, random STOP/re-entry, label sampling
```

**Other sizes** (override only these keys):

```yaml
# Stage 1 uses small models built from the same Qwen3 layer (QK-norm, grouped-query attention)
# Stage 1, two-hop task (section 10.1): 2,000 entities + 200 relations + padding
model: {vocab_size: 2208, context_length: 4, d_model: 256, n_heads: 4, n_kv_heads: 2, head_dim: 64,
        d_ff: 768, n_stem: 1, n_coda: 1, S: 6}
train: {epochs: 2000, batch_tokens: 16384, lr_schedule: constant, warmup_steps: 1000}
        # 4,096 facts × 4 tokens; lr picked from {3e-4, 1e-3} (section 10.1)
# Stage 1, order task (section 10.2): 64 symbols + 3 operations + delimiter, padded
model: {vocab_size: 72, context_length: 1024, d_model: 256, n_heads: 4, n_kv_heads: 2, head_dim: 64,
        d_ff: 768, n_stem: 1, n_coda: 1, S: 6}
train: {tokens: 1.31072e9, batch_tokens: 65536, lr: 1.0e-3, lr_schedule: constant, warmup_steps: 500}
        # 20,000 training steps of 64 × 1,024 tokens
# Stage 5, size 1: the released Qwen3-0.6B depth (28 layers, 0.60B parameters)
model: {n_stem: 3, n_coda: 3, S: 22}
train: {tokens: 1.0e10, batch_tokens: 524288, lr: 3.0e-4}
# Stage 5, size 2 (optional): the Qwen3-1.7B architecture (28 layers, 1.72B parameters)
model: {d_model: 2048, n_heads: 16, n_kv_heads: 8, head_dim: 128, d_ff: 6144, n_stem: 3, n_coda: 3, S: 22}
train: {tokens: 2.0e10, batch_tokens: 1048576, lr: 2.0e-4}
ffn_moe: {n_experts: 9, expert_width: 1536}   # d_ff / 4; matches the capacity row within 1% (row 15)
```

**Run naming:** `{mode}_{window}{r}_{reuse}_{kv_mode}_{routing}`, then `_{section}.{key}-{value}` for every other setting that differs from the defaults above, then `_s{seeds.model}`.
- Examples: `cm_abs1_reuseOn_step_learned_s0`, `cm_abs1_reuseOn_step_learned_router.curriculum-A_s0`, `depth_abs1_reuseOn_step_learned_model.visit_scale-inv_s0`.
- Mode abbreviations: `cm`, `cap`, `depth`.
- Window abbreviations: `abs`, `rel`, `bkt`, `free`.

---

## 9. Experiment plan

**Cost assumptions:**
- An H100 at about 200 TFLOP/s effective.
- The 12-layer Qwen3 dense model costs about 2.2 GFLOP per training token, which is about 3.1 H100-hours per 1B tokens. The large LM head is 45% of the weight FLOPs, so this is about 2.8× the cost of a GPT-2-small-sized model.
- Routed runs are assumed to cost 1.5 to 2.5× dense until the dispatch is optimised.
- **Measured** on the implementer's 8-GPU node (2026-10-20; record the GPU type with the measurement): the native 12-layer dense model trained 2.62B tokens in about 51 minutes. That is about 6.8 GPU-hours, or 2.6 GPU-hours per 1B tokens. The H100-hour figures in this section are planning estimates; scale them by measured throughput.

### Stage 0a: minimal pilot (12-layer Qwen3, 2.5B tokens each, about 70 H100-hours)

Built from Phase A only (section 6).
- **Routed runs:** all use `compute_matched`, absolute `r = 1` and `kv_mode = step`.
- **Shared settings:** every run has the complete 4,768-step schedule of section 4.1, with the same data order, validation sets and batch layout.
- **Budget:** these runs use the Stage 2 budget on purpose. A run whose configuration matches a Stage 2 row becomes that row's seed-0 run (see Stage 2).

| Run | Setting | Purpose |
|---|---|---|
| D | the B-DOT model at `r = 0` with `routing = fixed`, `step_film = true`, `kv_mode = step` and curriculum `none`; not the native Qwen3 model | Reference |
| P1-A | learned, reuse on, curriculum A | Which curriculum works? |
| P1-B | learned, reuse on, curriculum B | |
| R | the P1 winner's setting with `routing = random` (curriculum `none`) | Chance control in the same window |
| N | the P1 winner's setting with `reuse = false` | Order without substitution |

Run D, P1-A and P1-B first, choose the P1 winner, then run R and N.

**Choosing the P1 winner.** Differences below 0.01 nats of validation loss count as ties.
- Measure the off-default share with the end-of-run diagnostics (section 5).
- A spike means the periodic validation loss, measured every 100 training steps, rises more than 0.1 nats above its running minimum. At each curriculum transition the running minimum restarts from the new phase's validation loss at the transition step; the jump itself is reported separately (section 5).
- Keep the curriculum with the lower validation loss among those that reach an off-default share of at least 5% and show no spike. Ties go to B.
- If neither qualifies but at least one did not spike, rerun the better non-spiking one once with `default_prior = 0.5` and `explore_eps_max = 0.1`, and keep these values for all later routed runs.
- If both spiked, continue with the one with lower loss and flag it in the report.

**Gate 0a (the pilot gate).** Build Phase B and run Stage 0b only if all six hold for the P1 winner:
1. **Stable:** no loss spike.
2. **Routing is used:** off-default share of at least 5%.
3. **The learned deviations matter:** the default-path test (section 5) raises its validation loss by at least 0.01 nats.
4. **Learned beats chance:** its validation loss is at least 0.01 nats below run R's.
5. **Quality:** its validation loss is at most 0.01 nats above run D's.
6. **Runtime:** routed training throughput is at least 60% of the native dense reference's (T8).

How to read the outcome:
- **All six hold, and the P1 winner also beats D by at least 0.01 nats:** a strong signal.
- **A criterion misses by less than its own threshold:** treat it as noise. Rerun the runs it depends on with `seeds.model = 1` and judge the 2-seed means. A noisy single result never justifies expansion.
- **Routing is unused (criterion 2) or no better than chance (criterion 4) after the rerun:** stop and report that bounded learned routing does not help at this scale. Do not try STOP, re-entry or a larger model as a rescue.
- **Run N** is reported but does not gate. It separates ordering (reuse off) from substitution (reuse on); see "How to read the table" in Stage 2.

### Stage 0b: remaining pilots (only if Gate 0a passed; 12-layer Qwen3, 2.5B tokens each, 1 seed, about 160 H100-hours)

Build Phase B first. Then run these pilots in parallel, with the curriculum chosen in Stage 0a (free windows always use B). As in Stage 0a, every run has the complete 4,768-step schedule, and matching runs carry over into Stage 2.

| Pilot | Runs (`compute_matched`, reuse on, unless stated) | Question |
|---|---|---|
| P2 | `kv = step`: absolute `r = 1` (the P1 winner, already run) vs relative `r = 1` with `drift_cap = 2` vs bucket 2 | Which window type? |
| P3a | absolute `r = 1`: `kv = step` (the P1 winner) vs `module_band` vs `module_mixed` | Which K/V handling works with a narrow window? |
| P3b | free window: `kv = step` vs `module_mixed` vs `lol_chunk` vs `exclude` | Does step K/V make free per-token routing stable where mixed K/V fails? How does it compare with LoL- and DNA-style routing? |
| P4 | `depth`, absolute `r = 1`, `kv = step`: `visit_scale = none` vs `inv` | Is visit scaling needed in depth mode? |

**Gate 0b.** Differences below 0.01 nats of validation loss count as ties.

- **Window for the rows marked W in Stage 2 (rows 3, 5, 7, 8 and 10):** the best in P2. Ties go to `absolute`, then `relative`, then `bucket`.
- **`kv_mode`:** the better of `step` and `module_band` in P3a. Ties go to `step`, which is cheaper.
- **`visit_scale` for grid rows 11 to 13:** the better in P4. Ties go to `none`. Row 16 and the Stage 1 looped runs always use `none`.
- **Record from P3b:** whether `module_mixed` was unstable (spikes or divergence), and how `step` compares with `lol_chunk` and `exclude`. This is context for interpreting rows 6 and 14; it does not gate anything.

### Stage 1: synthetic mechanism tests (tiny models, about 30 GPU-hours; up to about 100 if the two-hop runs must go to 10,000 epochs; assumes about 50 TFLOP/s effective, since tiny models use a GPU poorly)

The tasks are specified in section 10, and their configs in section 8 ("Other sizes").
- **Seeds:** run 3 seeds of each configuration. For the order task, each seed is a task seed (new permutations and a new held-out order) with its own initialisation seed, and every configuration uses the same 3 task seeds.
- **Dev and test sets:** choices made during training (the LR, whether to extend) use the dev sets only; Gate 1 uses the test sets (section 10).
- **Learned routing:** `kv_mode = step`, `reuse = true`, `step_film = true`, and the curriculum chosen in Stage 0a (B for the free window).
- **Random routing:** the same, with curriculum `none`.
- **Dense and forced re-entry runs:** `step_film = true` (the same FiLM option as the routed runs) and curriculum `none`.
- **Looped runs:** `step_film = false` (a standard looped transformer) and curriculum `none`.
- **Forced re-entry:** `reentry.policy = forced`, with the `rel2` positions as `force_mask`.

| Configuration | Two-hop task | Order task |
|---|---|---|
| Dense (`compute_matched`, `r = 0`) | ✓ | ✓ |
| Looped: the whole middle run twice (`depth`, `S = 12`, `M = 6`, `r = 0`, `kv_mode = module_band`) | ✓ | ✓ |
| Learned routing, absolute `r = 2` | ✓ | ✓ |
| Learned routing, free window | ✓ | ✓ |
| Random routing, absolute `r = 2` | ✓ | ✓ |
| Dense plus forced re-entry (section 10.1) | ✓ | |

**Gate 1** (compare 3-seed means; "beats" means by at least 5 points of accuracy):
- **(a)** Forced re-entry beats dense on the OOD-test set of the two-hop task (both hops OOD). Otherwise drop Stage 4.
  - Report the looped model as the reference.
  - If neither the looped model nor forced re-entry beats dense, the training budget was too short: extend all two-hop runs to 10,000 epochs before deciding.
- **(b)** On the order task, at least one learned configuration beats dense on held-out-order exact accuracy (all 64 held-out cases, mean over the 3 task seeds). Mutual information and the route-swap test are reported as diagnostics only, because a router can encode the instruction without executing it. Otherwise drop the order-only rows 2, 4 and 12 from Stage 2.

### Stage 2: main grid (12-layer Qwen3, 2.5B tokens, 2 seeds)

**Cost:**
- **Grid:** about 380 to 430 H100-hours for the runs not already done in Stages 0a and 0b, depending on how many carry over (about 480 for all 36).
- **5B reruns:** about 650.
- **If the budget is tight:** run the grid at 1.5B tokens (about 290 H100-hours) and keep 5B for the reruns. Then nothing carries over from Stages 0a and 0b.

**Runs carried over from Stages 0a and 0b.** A Stage 0a or 0b run whose configuration matches a row exactly, after the Gate 0b choices, is that row's seed-0 run. A seed-1 rerun from Stage 0a is likewise that row's seed-1 run. Before reusing a run, check it:
- load its final checkpoint with the current code: parameter names and shapes must match;
- re-evaluate it: its final validation loss must match the logged value within 1e-3, and its routes on the route sample must be identical;
- confirm in version control that the training code for its features (router, balancing, curriculum, data pipeline) has not changed since it ran.

If any check fails, rerun it.

| Gate 0b outcome | Carried-over runs |
|---|---|
| Any | D → row 1; P3b's free-window `step` run → row 6; P3b's `lol_chunk` run → row 14; the P4 run with the chosen `visit_scale` → row 13 |
| `step` K/V kept | Also N → row 2 |
| `step` K/V and the absolute window both kept | Also the P1 winner → row 3 and R → row 7 |
| Another window or `module_band` chosen | The winning P2 or P3a run becomes row 3, if its configuration matches |

| # | `mode` | Window | `reuse` | `routing` | Notes | Purpose |
|---|---|---|---|---|---|---|
| 1 | compute_matched | `r = 0` | – | fixed | | Dense baseline, with the same FiLM option as the routed rows |
| 2 | compute_matched | absolute 1 | off | learned | | Order alone |
| 3 | compute_matched | W 1 | on | learned | | Order plus substitution |
| 4 | compute_matched | absolute 2 | off | learned | | Wider reordering |
| 5 | compute_matched | W 2 | on | learned | | Wider, with substitution |
| 6 | compute_matched | free | on | learned | curriculum B | Free pool |
| 7 | compute_matched | W 1 | on | random | | Chance control |
| 8 | compute_matched | W 1 | on | hash | | Fixed-assignment control |
| 9 | capacity | `r = 0` | – | learned | | Mixture of whole-layer experts |
| 10 | capacity | W 1 | on | learned | | Choice on top of capacity |
| 11 | depth | `r = 0` | – | fixed | | Looped baseline (our parameterisation) |
| 12 | depth | absolute 1 | off | learned | | Reordered loop |
| 13 | depth | absolute 1 | on | learned | | Loop with substitution |
| 14 | compute_matched | free | on | learned | `kv_mode = lol_chunk` | Library-of-Layers-style baseline |
| 15 | FFN-MoE | – | – | – | see below | Standard MoE matched to row 9 |
| 16 | depth | `r = 0` | – | fixed | `kv_mode = module_band`, `step_film = false` | Standard looped transformer (281.5M) |
| 17 | compute_matched | `r = 0` | – | fixed | `step_film = false` | Plain dense transformer |
| 18 | compute_matched | W 1 | on | learned | `route_scope = mlp` | MLP-only routing (ReXMoE-style) |

**How to read the table:**
- **`W`** is the window type chosen at Gate 0b:
  - "W 1" means absolute `r = 1`, relative `r = 1` (with `drift_cap = 2`), or bucket 2;
  - "W 2" means absolute or relative `r = 2` (with `drift_cap = 2`), or bucket 4.
- **Curriculum:** learned rows use the one chosen in Stage 0a (always B for the free window). The `r = 0` rows (1, 9, 11, 16, 17) and the `random` and `hash` controls (rows 7 and 8) use curriculum `none`.
- **`kv_mode`:** the Gate 0b choice applies to the `compute_matched` rows 2–5, 7 and 8. Rows 1 and 17 use `step`, which at `r = 0` is identical to `module_band`. Rows 6, 9–13 and 18 always use `kv_mode = step`:
  - a band over a free window (row 6) would compute K/V for every module;
  - a band in `capacity` mode would break the parameter and FLOP matching with row 15;
  - a band in `depth` mode would add K/V FLOPs to rows 12 and 13 relative to their baselines (rows 11 and 16);
  - row 18's MLP-only routing is defined for `step` only.
- **Step FiLM** is on in every B-DOT row, including the `r = 0` references 1, 9 and 11, so routing is the only difference between a routed row and its reference. Rows 16 and 17 are plain baselines without FiLM; row 1 vs row 17 shows what FiLM alone does.
- **Order versus substitution:**
  - A win by a reuse-on row (3, 5, 10, 13) shows that per-token module selection helps, which may come from substitution or repetition.
  - It shows that *order* helps only if the matching reuse-off row (2, 4 or 12) also wins.
  - Report the two claims separately.
- **Attention routing:** row 3 vs row 18 shows what routing `W_Q` and `W_O` adds over MLP-only routing, when row 3 uses `step` K/V. If Gate 0b picks `module_band`, row 3's modules also own their K/V, so compare row 18 with the `step`-K/V run of row 3's window instead (the P1 winner for the absolute window, otherwise the P2 winner), and give that run a second seed.

**Row 15:** the 12-layer dense model, but each of the 8 middle layers' MLP is a mixture of 10 SwiGLU experts of width `d_ff / 4` (768 here). It uses top-4 softmax routing, so the active MLP width stays 3072, and loss-free balancing (a per-layer `beta`, as in section 3.7).
- **Parameters:** 457.7M total and 344.4M active, against row 9's 453.6M and 344.5M. The totals differ by 0.9%, because no whole number of experts of this width matches exactly.
- **Stage 5:** with 10 experts at size 1 and 9 at size 2, it stays within about 1.2%.

**5B reruns:**
- Pick the four learned B-DOT rows (from 2–6, 10, 12, 13 and 18) with the largest validation-loss gain over their own mode's `r = 0` row (2.5B tokens, mean of 2 seeds).
- Rerun them at 5B tokens with 3 seeds, together with each matching `r = 0` row (row 1, 9 or 11) and row 14.

**Gate 2** (on the 5B reruns): at least one rerun B-DOT row beats its own mode's `r = 0` row, in one of two ways:
- its mean validation loss is lower by more than twice the standard error of the difference;
- or its benchmark average is higher by at least 1 point.

If none does, stop and write up the grid as a controlled negative result. Do not start Stages 3 to 5 as rescue runs.

**The best Gate 2 row** is the passing rerun B-DOT row with the largest mean validation-loss gain over its own `r = 0` row (never row 14). Stages 3 to 5 use it. STOP and re-entry runs use its reuse-on counterpart (2 → 3, 4 → 5, 12 → 13).

### Stage 3: STOP (12-layer Qwen3, 5B tokens, about 130 H100-hours)

- **Runs** (all with 5B tokens and 1 seed):
  - the reuse-on counterpart of the best Gate 2 row, and its `r = 0` counterpart. Train each with `stop.policy = random`, then train its STOP head (section 4.3);
  - a dense reference (`compute_matched`, `r = 0`, `stop.policy = none`).
- **Baselines** (each gives points on the loss-vs-LE curve):
  - dense models with 4, 5, 6 and 7 middle layers (5B tokens);
  - the dense reference with 1 to 4 middle layers removed greedily (each time, the layer whose removal raises validation loss least), each fine-tuned from those weights for 0.5B tokens (fresh warm-up, cosine decay from LR 3e-4);
  - each STOP trunk with `stop.policy = constant`: one `constant_prob` per swept threshold, set to match the learned head's mean depth at that threshold.
- **Report:** loss and accuracy against `LE`, and against measured speed (decode at batch 1 and 32, and prefill).

**Gate 3** (judged on the routed STOP trunk; report the `r = 0` trunk beside it): at an operating point within 0.01 nats of the model's own full-depth validation loss, three things must hold:
- learned STOP saves at least 10% of layer-equivalents;
- the measured batch-1 decode speed-up is at least 1.1×;
- its loss-vs-LE curve lies below all three baselines.

### Stage 4: re-entry (only if Gate 1a passed; 12-layer Qwen3, 5B tokens, 1 seed, about 40 H100-hours)

- **Trunk:** the same row as in Stage 3.
  - Train it with `reentry.policy = random`, with 2% two-hop text (section 10.3) mixed into pretraining.
  - Then train the learned re-entry head (section 4.4).
- **Policies compared at evaluation, all on this one trunk:**
  - `learned`;
  - `random` and `repeat_only`, each with `reentry.random_prob` set so that its mean LE on the probes matches the learned head's within 1%. Matching the re-entry rate is not enough, because the two policies add different numbers of runs per re-entry;
  - `none`;
  - `forced` on the probe tokens (an upper reference).

**Gate 4:** on the OOD two-hop probes, `learned` beats both `random` and `repeat_only` by at least 3 points.

### Stage 5: scale-up (whatever passed; about 170 to 460 H100-hours at size 1; the optional size 2 adds about 900 to 2,100)

- **Sizes:**
  - **Size 1** is the released Qwen3-0.6B at its full depth of 28 layers.
  - **Size 2 (optional)** is the Qwen3-1.7B architecture. Run it only if size 1 confirms the gain.
- **Runs at each size** (1 seed, 2 if the budget allows):
  - the best Gate 2 row and its `r = 0` row, plus row 15 if the best row is a `capacity` row;
  - if Gate 3 passed, a separate STOP run, with the Stage 3 recipe (`stop.policy = random`, then a STOP head);
  - if Gate 4 passed, a separate re-entry run, with the Stage 4 recipe (`reentry.policy = random`, then a re-entry head);
  - STOP and re-entry runs use the reuse-on counterpart of the best row, and are never combined in one run.
- **Data:** mix the section 10.3 two-hop text (2%) into every Stage 5 run, so all of them can be scored on the probes.
- **Training:** as in section 4.1, except for the values in this table.

| Size | `d` | Query / K/V heads (`head_dim` 128) | `d_ff` | `n_stem` / `S` / `n_coda` | Parameters | Tokens | Peak LR | Batch |
|---|---|---|---|---|---|---|---|---|
| 1: Qwen3-0.6B, 28 layers | 1024 | 16 / 8 | 3072 | 3 / 22 / 3 | 0.60B | 10B | 3e-4 | 0.5M tokens |
| 2 (optional): Qwen3-1.7B | 2048 | 16 / 8 | 6144 | 3 / 22 / 3 | 1.72B | 20B | 2e-4 | 1M tokens |

At this stage, also report knowledge-heavy and reasoning-heavy evaluations separately (for example ARC and TriviaQA vs the section 10.3 probes, and GSM8K). Looped models typically lose on the first kind and gain on the second. TriviaQA and GSM8K need the harness's `generate_until` (greedy, with KV-cache decoding).

**Total plan:**
- Through Stage 5 at size 1: roughly 1,600 to 2,050 H100-hours, if every gate passes. Most of it is Stage 2.
- The optional size 2 adds about 900 to 2,100.
- The gates stop spending early: Stage 0a alone (about 70 H100-hours) decides whether anything else runs.

---

## 10. Synthetic task specifications

### 10.1 Two-hop composition (after Grokked Transformers)

- **Notation:** `a`, `b`, `c` are entities and `rel` is a relation. They have nothing to do with the reach `r` or the state `h`.
- **Facts:**
  - 2,000 entities and 200 relations; each entity has 20 outgoing relations, giving 40,000 atomic facts `(a, rel) -> b`, with `b` a random entity.
  - Two-hop facts `(a, rel1, rel2) -> c`, where `b = atomic(a, rel1)` is the bridge entity and `c = atomic(b, rel2)`.
- **Split** (as in the paper: over facts, not entities):
  - Split the atomic facts at random: 95% in-distribution (ID), 5% out-of-distribution (OOD).
  - Training contains every atomic fact, ID and OOD.
  - Training also contains two-hop facts whose two hops are both ID atomic facts, sampled at φ = 7.2 two-hop facts per ID atomic fact. A larger φ speeds up grokking.
- **Dev and test sets:**
  - ID: 2,000 unseen two-hop facts whose two hops are both ID, split at random into ID-dev and ID-test (1,000 each).
  - OOD: all two-hop facts whose two hops are both OOD (about 2,000 exist), split at random into OOD-dev and OOD-test halves.
  - Choices made during training (the LR, whether to extend) use only the dev sets. Gate 1a uses OOD-test.
- **Format:** one token per entity and per relation. Atomic facts are `a rel b`, right-padded with a PAD token (no loss); two-hop facts are `a rel1 rel2 c`. The loss is on the answer token only.
- **Model:** small Qwen3 layers: `d = 256`, 4 query heads and 2 K/V heads of `head_dim = 64`, `d_ff = 768`; `n_stem = 1`, `S = 6`, `n_coda = 1` (8 layers). The full config is in section 8.
- **Training:** AdamW, weight decay 0.1, batch 4,096 facts, 1,000 warm-up training steps, then constant LR. One fact per sequence.
  - Pick the LR from {3e-4, 1e-3} by the dense baseline's ID-dev accuracy after 500 epochs, then use it for every configuration.
  - Train for 2,000 epochs (about 153k training steps), logging dev accuracies every 50 epochs. If any configuration's OOD-dev accuracy is still rising at 2,000 epochs, extend all two-hop runs together to 10,000 epochs, and judge Gate 1a at 10,000 epochs.
  - **Horizon:** `max_steps` is the 2,000-epoch step count. The extension is the one allowed continuation of a run (section 4.1): the LR is constant after warm-up, so continuing changes no schedule. The curriculum never restarts or stretches; the extra epochs stay in its final phase (hard top-1, no exploration).
  - Picking the LR after 500 epochs is a training choice made on dev data, not a comparison between configurations, so it is allowed.
  - OOD composition, if it appears, appears late. Published looped models needed thousands of epochs.
- **Forced re-entry:** at the last re-entry point (after step 3), the `rel2` token always re-enters at step 0, for a total of 10 module runs. This is a capacity check.
- **Optional:** train on chains of up to 2 hops and test on 3 hops, allowing more re-entries or loops.

### 10.2 Order-required task

- **Operations:** an alphabet of 64 symbols and three operations `f_A`, `f_B`, `f_C`, each a fixed random permutation of the alphabet.
- **Example:** `O1 O2 O3 x y`, where `(O1, O2, O3)` is an ordering of `{A, B, C}`, `x` is a symbol, and the target is `y = f_O3(f_O2(f_O1(x)))`. The loss is on `y` only.
- **Training mix:** one-operation examples (`O1 x y`, all three operations), two-operation examples (all six ordered pairs), and three-operation examples in 5 of the 6 orders. Each of the three kinds makes up a third of the examples.
- **Test:** three-operation examples in the held-out order. There are only 64 such cases, one per input symbol, so report exact accuracy on all 64; repeated packing does not create more independent cases.
- **Task seeds:** each task seed draws new permutations `f_A`, `f_B`, `f_C` and a new held-out order, and comes with its own initialisation seed. Use the same task seeds for every configuration, so results are paired, and report the spread across task seeds.
- **Context:** pack many examples into each 1,024-token context, separated by a delimiter token. Sample fresh examples for every batch.
- **Model:** the same small Qwen3 layers as section 10.1; `n_stem = 1`, `S = 6` (`M = 6`), `n_coda = 1`. The full config is in section 8.
- **Training:** 20,000 training steps, batch 64 sequences, AdamW with LR 1e-3 (constant after 500 warm-up training steps), weight decay 0.1.
- **Diagnostics** (none of these gates anything; Gate 1b uses held-out accuracy):
  - accuracy on the seen orders and on the held-out order;
  - for each step `s`, the mutual information between the module chosen at the `x` token and the instructed order (6 classes). Report the maximum over steps, next to the value for random routing. A router can encode the instruction without executing it, so this alone proves nothing;
  - **route-swap test:** run each held-out example with the route the model chose, on the same `x`, for the reversed order. Measure how often the output follows the reversed order. If modules execute the operations, it should.

### 10.3 Two-hop facts inside pretraining (Stage 4)

- **Entities:** 2,000 made-up names, each written as two Qwen3 tokens (for example two random syllable tokens), all distinct.
- **Relations:** 200 fixed English nouns or noun phrases, such as "mentor" or "home town".
- **Facts:** sampled and split as in section 10.1:
  - 20 relations per entity;
  - atomic facts split 95% / 5% into ID and OOD;
  - two-hop training facts only when both hops are ID.
- **Text templates:**
  - atomic: `The {rel} of {a} is {b}.`
  - two-hop: `The {rel2} of the {rel1} of {a} is {c}.`
- **Mix:**
  - Pack these sentences into their own 1,024-token sequences, each sentence a document: separated by the end-of-text token and document-blocked like the pool.
  - In every training step, 2% of the sequences, rounded down, are two-hop sequences: 10 of 512, or 20 of 1,024 at Stage 5 size 2. Their positions in the step are drawn with seed 0; the other sequences continue the pool order of section 4.1.
  - The two-hop sequences cycle in a fixed order, reshuffled with seed 0 at each pass.
  - All Stage 4 and Stage 5 runs share this mixed order.
- **Probes:** the OOD two-hop facts (both hops OOD), written with the two-hop template. A probe counts as correct if greedy decoding after `is` produces `c` exactly. The `forced` policy applies to the token `is`.

---

## 11. Baselines and matching rules

- Every comparison uses the same data order, number of tokens, optimiser, schedule, batch size and context length.
- A longer schedule stopped early is never a reference or a comparison arm (section 4.1).
- The main comparison for any routed setting is **the same mode at `r = 0`**. Random and hash routes are diagnostics, not the bar.
- Report for every run:
  - unique parameters;
  - active parameters per token;
  - FLOPs per token and `LE`;
  - KV-cache bytes per token;
  - measured tokens per second.
- Use 2 seeds for the grid and 3 seeds for the finalists.
- Never report FLOPs alone as a speed-up.

---

## 12. Known pitfalls

From the v1 prototype:

1. **In-place updates of routing state break autograd.** The router reads the visit-count tensor, which autograd saves for the backward pass. Update visit counts and quota state out of place: `visits = visits.scatter_add(...)`, not `scatter_add_`.
2. **Weak hash routing collapses.** A linear hash such as `(id * a + s * b) mod small` gave only 4 distinct paths. Use this 32-bit avalanche mix (`mix32`):
   ```
   x = (id * 2654435761 + (s + 1) * 2246822519) & 0xFFFFFFFF
   x ^= x >> 15; x = (x * 2246822519) & 0xFFFFFFFF
   x ^= x >> 13; x = (x * 3266489917) & 0xFFFFFFFF
   x ^= x >> 16
   ```
3. **Narrow-to-wide training can lock in the default path.** With the +1 prior and no exploration, the router never left the default order (0% off-default). Use exploration at each widening, or curriculum B.
4. **Routers deviate mostly by substitution, not reordering.**
   - With reuse on, 63 to 96% of v1's buckets contained a repeated module.
   - With reuse off, only 2% of steps left the default order.
   - Keep reuse as an explicit factor.
5. **Soft top-2 needs two allowed candidates.** Fall back to top-1 when the mask allows only one.
6. **STOP tied to group boundaries can become impossible.** Allow STOP after every step `s` with `s + 1 >= stop.min_steps`.
7. **Easy tasks hide routing effects.** A copy task was solved by the fixed stem and coda alone. Use the section 10 tasks and real text.
8. **Naive masked loops are slow** (about 0.25 s per training step for a tiny model on CPU). Dispatch by sorting tokens per module and use grouped matrix multiplies.
9. **Avoid expert-choice routing.** Selecting the top tokens per module over the sequence is non-causal and breaks decoding.

Found while checking this spec:

10. **A greedy deadline rule is not enough for `reuse = false`.** Forcing a module once its remaining available steps equal its remaining quota still reached dead ends in about 1.5% of random `depth`-mode paths (absolute `r = 1`, `S = 8`). Use the feasibility table (section 3.4).
11. **Relative windows have degenerate corners.** At `r = 0` a token repeats one module forever. With `reuse = false` and `r = 1`, only the default path is feasible (or also its reverse, without a drift cap).
12. **Empty attention rows give NaN.** Every attention mask must include the token's own position. This matters for `exclude` and `lol_chunk`.
13. **Token dropping breaks causality.** Capacity-factor dropping makes a token's output depend on later tokens. No routed layer may drop tokens.
14. **Balancing from the current batch leaks information.** Update `beta` only after the backward pass, from that training step's all-reduced counts, never inside a forward pass (section 3.7).
15. **Dynamic routing and DDP.** A module unused on one rank gets no gradient there. Enable unused-parameter handling (or an equivalent) and keep T11 passing.
16. **Activation checkpointing re-runs the forward pass.** Store the routes and random draws from the first forward pass and reuse them during recomputation, or the backward pass will not match the forward pass (T12).

Found while preparing the pilots:

17. **A long schedule stopped early is not a short run.** Take a 30B-token schedule (57,220 steps) stopped after 5,000 steps:
    - its LR is still at 99% of peak, so every loss is measured before any substantial decay;
    - its routing curriculum, measured against the long horizon, never reaches hard routing (curriculum B) or never opens the window (curriculum A).

    Give every run its own complete schedule (section 4.1).

---

## 13. Expected outcomes (written before running)

| Comparison (grid rows) | Prediction |
|---|---|
| Order alone vs dense (2 vs 1) | Tie within seed spread |
| Reuse on vs off (3 vs 2) | On slightly better; most deviations are substitutions |
| Free pool vs dense (6 vs 1) | Worse, unless curriculum B rescues it |
| Learned vs random and hash (3 vs 7, 8) | Learned beats random; close to hash |
| Whole-layer experts vs dense (9 vs 1) | Better at equal FLOPs |
| Choice on top of capacity (10 vs 9) | About 1 benchmark point or less |
| Whole-layer experts vs FFN-MoE (9 vs 15) | FFN-MoE as good or better |
| Looped vs dense (11 vs 1) | Worse (same FLOPs, 16% fewer parameters) |
| Per-step K/V loop vs standard loop (11 vs 16) | Slightly better (3% more parameters) |
| Choice on top of depth (12, 13 vs 11) | Small gain at most, mainly on reasoning-heavy evaluations |
| Per-token routing vs LoL-style (6 vs 14) | Per-token training is stable with step K/V; quality similar or better |
| FiLM alone (1 vs 17) | No difference within seed spread |
| Routing attention as well as the MLP (3 vs 18) | Small gain or none |
| STOP (Stage 3) | 20 to 40% fewer middle steps within 0.01 nats; routed and `r = 0` trunks about the same |
| Re-entry (Stage 4) | No effect on web-text loss; an OOD two-hop gain only if Gate 1a passed |

Any prediction that fails in the hopeful direction becomes the headline result. The gates decide the rest.

---

## 14. References

- Library of Layers (LoL): Adaptively Ordering Transformer Layers at Runtime, ICLR 2027 submission: https://openreview.net/forum?id=O4y4PcNmYD
- Towards Distributed Neural Architectures (DNA): https://arxiv.org/abs/2506.22389
- Mixture-of-Modules (MoM): https://arxiv.org/abs/2407.06677
- ReXMoE (experts reachable from neighbouring layers): https://arxiv.org/abs/2510.17483
- Mixture of Attention Heads: Selecting Attention Heads Per Token (shared `W_K`/`W_V`, per-expert `W_Q`/`W_O`): https://arxiv.org/abs/2210.05144
- CALM, Confident Adaptive Language Modeling (state propagation for exited tokens): https://arxiv.org/abs/2207.07061
- FlexiDepth, Adaptive Layer-skipping in Pre-trained LLMs (skipped tokens still compute K/V): https://arxiv.org/abs/2503.23798
- Hash Layers for Large Sparse Models: https://arxiv.org/abs/2106.04426
- Auxiliary-loss-free load balancing: https://arxiv.org/abs/2408.15664
- Mixture-of-Recursions: https://arxiv.org/abs/2507.10524
- Ouro, Scaling Latent Reasoning via Looped Language Models: https://arxiv.org/abs/2510.25741
- Huginn, recurrent-depth latent reasoning: https://arxiv.org/abs/2502.05171
- Reasoning with Latent Thoughts: On the Power of Looped Transformers: https://arxiv.org/abs/2502.17416
- RecurTrace (supervised halting head; ACT/PonderNet collapse): https://arxiv.org/abs/2609.03379
- Closing the Loop (looped models at matched compute; halting collapse): https://arxiv.org/abs/2610.00673
- Grokked Transformers are Implicit Reasoners: https://arxiv.org/abs/2405.15071
- Loop, Think, & Generalize (recurrent depth and OOD composition): https://arxiv.org/abs/2604.07822
- Hopping Too Late (back-patching): https://arxiv.org/abs/2406.12775
- IOT: Instance-wise Layer Reordering for Transformer Structures: https://openreview.net/pdf?id=ipUPfYxWZvM

---

## Appendix A. Checklist

**Phase A** (section 6)
- [ ] Infrastructure, including the data recipe of section 4.1 (pinned pool revision, `packed_shuffle` order, validation manifest, document-disjointness check), document masking and the separate random streams.
- [ ] Native dense reference trains with a normal loss curve (no spikes).
- [ ] Minimal B-DOT: `compute_matched`, absolute window, `step` K/V, fixed/learned/random routing, reuse on and off, router extras, `step` decoding.
- [ ] T1 to T5 and T8 to T14 pass for this subset; T8 throughput recorded.

**Stage 0a** (minimal pilot, 2.5B tokens, complete schedules)
- [ ] D, P1-A and P1-B; P1 winner chosen; then R and N.
- [ ] Gate 0a recorded (all six criteria, with second-seed reruns where a result was within noise). If it fails: report and stop.

**Phase B and Stage 0b** (only if Gate 0a passed)
- [ ] Windows, modes, K/V variants, `route_scope = mlp`, `hash` routing, decoding for every `kv_mode`; their tests pass.
- [ ] P2, P3a, P3b and P4.
- [ ] Gate 0b decisions recorded: window, `kv_mode`, `visit_scale`, and the P3b notes.

**Phases C to F**
- [ ] Before Stage 1: re-entry mechanics and policies; T7 passes.
- [ ] Before Stage 2: FFN-MoE baseline.
- [ ] Before Stage 3: STOP; T6 passes.
- [ ] Before Stage 4: learned re-entry head.

**Stage 1** (synthetic)
- [ ] Two-hop and order tasks, 3 seeds per configuration.
- [ ] Gate 1a (re-entry) and Gate 1b (order rows) recorded.

**Stage 2** (2.5B tokens)
- [ ] Rows 1 to 18, 2 seeds (seed-0 runs carried over from Stages 0a and 0b where they match).
- [ ] 5B reruns of the top four learned rows, their `r = 0` rows and row 14, 3 seeds.
- [ ] Gate 2 recorded. If it fails: write up the negative result and stop.

**Stage 3** (STOP)
- [ ] Two STOP trunks and a dense reference (5B tokens each), STOP heads, threshold sweep, three baselines, measured speed.
- [ ] Gate 3 recorded.

**Stage 4** (re-entry, only if Gate 1a passed)
- [ ] One trunk, its re-entry head, and five policies compared on it.
- [ ] Gate 4 recorded.

**Stage 5**
- [ ] The runs listed in section 9, Stage 5, for whatever passed.

**Every run**
- [ ] `config.json` (with the dataset revision and seeds), `metrics.jsonl`, `final_eval.json`, `routes_sample.npz`.
- [ ] `max_steps`, warm-up steps and curriculum boundaries recorded.
- [ ] Gate and grid runs only: `stop_after` unset, and the run reached `max_steps`.
- [ ] Parameters (unique and active), nominal LE, executed training FLOPs, KV bytes and measured speed reported.

---

## Appendix B. Change log

### Version 1.4: schedule horizon, pilot budget and data

Version 1.4 answers the curriculum-horizon review of 2026-10-20 (see `BDOT_curriculum_context_review_response.md`). The changes:

- **One horizon** (sections 4.1, 4.2 and 11; pitfall 17).
  - Every run has one complete schedule, `max_steps = floor(tokens / batch_tokens)`, shared by the LR schedule and the routing curriculum. Epoch-based runs use `epochs × steps per epoch`.
  - A longer schedule stopped early is never used by a gate or a comparison. `train.stop_after` is defined (it ends a run without changing `max_steps`) and is allowed only in tests and exploratory runs.
  - The Stage 1 two-hop extension is the one allowed continuation of a run (section 10.1). It now has a fixed end at 10,000 epochs.
- **Transitions after warm-up.**
  - Warm-up is `ceil(0.02 · max_steps)` steps, and it must end before the first curriculum boundary; the config loader checks this.
  - Section 4.2 lists the boundaries for 2.5B tokens, and new test T13 checks them.
- **Pilot budget.** Stages 0a and 0b train 2.5B tokens (4,768 steps), up from 1B, so that a viable method is less likely to be stopped for lack of training.
  - Matching runs carry over into Stage 2, as seed-0 runs (and Stage 0a's seed-1 reruns as seed-1 runs).
  - Before reuse, each carried-over run passes a check of its checkpoint, routes and code.
- **Data.**
  - The prepared English pool `nht10/cx_sampled` (`qwen3_0.6b_base_en_30B`) replaces FineWeb-Edu.
  - It is read in the `packed_shuffle` order, which nothing else may reshuffle, with fixed periodic and final subsets of the pool's shuffled validation split and a document-disjointness check.
  - Stages 4 and 5 interleave the two-hop sequences at 2% of every step (section 10.3). Runs that continue from a trunk read on after the trunk's data.
- **Reference runs.** D is stated explicitly as the B-DOT model at `r = 0` with FiLM, `step` K/V and curriculum `none`, not the native Qwen3 model. New test T14 checks the B-DOT training pipeline against the native model.
- **Validation** (sections 5 and 9).
  - Periodic validation uses the routing currently being trained. Control runs use their own routing, and STOP and re-entry trunks use policy `none`.
  - At each curriculum transition, the same weights are validated under the old and the new routing. The spike rule restarts from the new value.
  - The route diagnostics and the route sample are named sequences of the final validation set.
- **Tests.**
  - T8 times training steps only, with the same layout for every model. Gate 0a's runtime bar is now the native dense reference.
  - T12 now runs curriculum B with `max_steps = 1,000`, so exploration is on after the resume.
  - The Stage 0b test list matches section 6.
- **Stage 2 notes.**
  - Rows 1 and 17 use `step` K/V.
  - Gate 0b's window choice applies to the rows marked W.
  - Row 3 vs row 18 is read as attention routing only with `step` K/V.
- **Costs.** Stage budgets updated, and the measured dense throughput added (section 9).

### Version 1.3: Qwen3 architecture

- **Layer:** every layer (stem, modules, coda, baselines) is now a Qwen3 decoder layer, as in the released Qwen3-0.6B. That means QK-norm, grouped-query attention with 16 query and 8 K/V heads of size 128, `d_ff = 3072`, RoPE base 1e6, RMSNorm eps 1e-6, no biases, and a tied 151,936-token embedding with the Qwen3 tokenizer. Models are trained from scratch with Hugging Face initialisation.
- **Depth:** the main experiments use 12 layers (2 + 8 + 2), a 344M-parameter dense model. Stage 5 uses the full 28-layer Qwen3-0.6B, with Qwen3-1.7B as an optional second size.
- **Who owns the norms:** `q_norm` belongs to the module and `k_norm` to whoever owns `W_K` (the step in `step` mode, the module in `module_band`).
- **Numbers recomputed:** parameter counts and `kv_frac` (2/15); row 15 (10 experts of width 768, within 0.9% of row 9); the costs (about 3.1 H100-hours per 1B tokens, 45% of it in the LM head); and every stage budget. The peak LR is now 3e-4.
- **STOP runs** keep Qwen3's 8 K/V heads; they no longer switch to 4.
- **T1** now compares against Hugging Face `Qwen3ForCausalLM`.

### Version 1.2: answers to the implementation review

Version 1.2 answers the implementation review of version 1.1. The changes:

- **Scope.**
  - The build is now phased (section 6): Phase A covers only the minimal pilot.
  - Stage 0 is split into Stage 0a (minimal pilot, with a six-part pilot gate, Gate 0a) and Stage 0b (the remaining pilots, built only if Gate 0a passes).
- **Positioning.**
  - Section 1 now says what is new and what is prior art (Mixture of Attention Heads, ReXMoE).
  - It states the central hypothesis.
  - The DNA result is reported per variant.
- **New controls in the grid:**
  - row 17, a plain dense transformer;
  - row 18, MLP-only routing (`route_scope = mlp`);
  - row 1 now uses the same FiLM option as the routed rows.
- **Interpretation rules:** order versus substitution, and attention routing versus MLP-only routing (Stage 2 notes).
- **Router.**
  - Exact soft top-2 semantics.
  - Tie-breaking by lowest module id.
  - The straight-through gradient described as a surrogate.
  - Balancing against default-path targets, with all-reduced counts, skipped empty steps and fixed timing.
  - A defined z-loss population.
- **Compute.** Three measures: nominal hard-path LE, executed training FLOPs (including the soft phase) and wall-clock time.
- **Data.**
  - Pinned dataset revision.
  - A fixed validation manifest across budgets.
  - Document-blocked attention with per-document positions (this replaces cross-document attention).
  - Separate random streams.
- **Tests.**
  - T1 now also compares gradients and one optimizer update.
  - New T11 (distributed training with unused modules) and T12 (resume and route reproducibility).
- **Decoding:** an eager cache policy for `module_band`; the re-entry recompute ablation is training-only.
- **Gate 0a uses the new default-path test.** The random-route test stays a diagnostic.
- **STOP and re-entry labels** are described as heuristics, to be validated by the deployed policy.
- **Synthetic tasks.**
  - Dev and test splits for the two-hop task.
  - Exact accuracy on the 64 held-out cases and paired task seeds for the order task.
  - Gate 1b uses held-out accuracy only; mutual information and the new route-swap test are diagnostics.
- **Library of Layers:** the description was checked against the paper's text, and the source is quoted in section 3.5.1.
