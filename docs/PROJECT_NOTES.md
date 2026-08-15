# Project Notes

## Training Infrastructure

- **h100-1**: 8x H100 80GB HBM3 — runs V2-attn and compositional experiments
- **h100-2**: 8x H100 80GB HBM3 — runs baseline experiments
- **main branch**: dev machine (4x A100 40GB) — code development, not training
- Each training machine pulls from its own git branch (`h100-1`, `h100-2`)
- Jobs submitted via `commands.sh` (mode #1 = run, #2 = pull logs)

## Hardware Adjustments (H100 80GB vs original H200 141GB plan)

- Batch size reduced: 16 → 4 per device
- Gradient accumulation increased: 4 → 16
- Effective batch unchanged: 4 × 16 × 8 = 512 sequences × 2048 = ~1M tokens/step
- Gradient checkpointing tested but reverted (batch 4, accum 16 fits without it)

## Experiments Completed

### De-risk run: 10K steps (of ~35K total)

All experiments train on full data schedule (~35K steps cosine LR) but are manually
stopped at checkpoint-10000 via `run_experiments.py --stop-at-step 10000`.

**2026-08-08: first-round compositional runs were INVALID (see Debugging Log below)
and are being retrained with fixed code. Baseline was always correct and is kept.**

| Arm | Machine | Round 1 (invalid) | Round 2 (fixed code) |
|-----|---------|-------------------|----------------------|
| Original ANT | h100-1 | 16× grads + lam=1e-3 collapse — deleted | Retraining (started 2026-08-08, lam=1e-6) |
| ANT (ours) | h100-2 | 16× grads — deleted | Retraining (started 2026-08-08) |
| V2-attn | h100-1 | 16× grads — deleted | Retraining (queued after Original ANT) |
| Baseline | h100-2 | **Valid** (~12h, loss 3.147 @10K) | Kept — reference run, no retrain |

### Training Configuration (identical across all arms)

```
Model:          Qwen/Qwen3-0.6B (from scratch, not pretrained)
Data:           CulturaX multilingual (~35B tokens: 30B en + 5x1B vi/zh/ru/de/ar)
Tokenizer:      Qwen/Qwen3-0.6B (vocab 151,936)
Block size:     2048
Seed:           42
Precision:      bf16 (entmax in fp32)
Batch:          4 per device × 16 accum × 8 GPUs = 512 seqs = ~1M tokens/step
LR:             3e-4, cosine with min_lr_rate=0.1 (decays to 3e-5)
Warmup:         500 steps
Weight decay:   0.1
Adam betas:     (0.9, 0.95)
Grad clip:      1.0
Logging:        every 10 steps
Checkpoints:    every 250 steps
```

### Arm-specific differences

| Arm | Script | Training script | Embedding params | Key difference |
|-----|--------|-----------------|------------------|----------------|
| Baseline | train_qwen3_0.6b_baseline.sh | train.py | 155.6M (standard, tied) | Standard Qwen3, HF Trainer |
| Original ANT | train_original_ant.sh | train_original_ant.py | ~626M (T: N×K + A: K×d) | YOGI optimizer (HybridOptimizer), L1 proximal on T |
| ANT (ours) | train_ant_ours.sh | train_compositional.py | ~23.7M (X + A + router) | Entmax router, AdamW, CompositionalTrainer |
| V2-attn | train_v2_attn.sh | train_compositional.py | ~23.8M (X + A + router + LocalEncAttn) | Context-conditioned selection, the contribution |

### Fair comparison notes

- All arms use same data, seed, LR schedule, batch size, block size
- Baseline: `tie_word_embeddings=True` (default Qwen3). All others: `False` (custom input embedding can't be tied)
- All arms train in bf16 (Qwen3 config default). Custom embeddings cast to bf16 via `.to(torch.bfloat16)`
- entmax is the only fp32 component (explicit `s.float()` cast for sum-to-1 precision)
- Baseline uses HF Trainer directly. Compositional arms use CompositionalTrainer (subclass) with EmbeddingShim
- Original ANT uses HybridOptimizer (AdamW for backbone, YOGI for embedding) via OriginalANTTrainer
- Load-balance loss OFF (lambda_div=0) for all current runs

## Debugging Log (2026-08-07 → 08-08)

First eval round showed all compositional models at astronomical PPL (~35,000+) while
baseline was normal. Root-causing this uncovered five real bugs plus two infrastructure
hazards. Everything below is verified (smoke tests / md5 / on-machine probes), not guessed.

### Bug 1 — 16× inflated loss AND gradients in custom trainers (the big one)

- **Where**: `CompositionalTrainer.compute_loss` (train_compositional.py) and the
  original-ANT trainer (train_original_ant.py).
- **What**: HF Trainer passes `num_items_in_batch` (total token count across the whole
  accumulated batch) to `compute_loss` so the model returns sum-CE / total-tokens, and
  then skips its own per-microbatch normalization. Our overrides **ignored** it, so each
  microbatch returned mean-CE, and the Trainer (believing normalization was handled)
  did not divide by gradient_accumulation_steps=16.
- **Impact**: not just logging — **real gradients were 16× too large**. Old runs logged
  step-10 loss ~194 (= 16 × 12.1) and grad_norm 17–21× baseline's, so every step hit the
  grad-clip ceiling (norm 1.0). Training "worked" (curves went down; ant_ours reached
  3.182, v2_attn 3.182 at 10K) but under perpetual clipping — not the intended optimizer
  dynamics, and not comparable to baseline. Hence full retrain of all compositional arms.
- **Fix**: forward `num_items_in_batch` to the model call in both trainers.
- **Why baseline was fine**: train.py uses HF Trainer's default compute_loss, which
  handles all of this correctly.

### Bug 2 — /8 deflation after fix 1 (caught by smoke test v1)

- **What**: with `average_tokens_across_devices=True`, HF's default path multiplies the
  loss by `num_processes` (8) because `num_items_in_batch` is the **global** token count
  all-reduced across devices while each rank's loss is local. Our fixed compute_loss
  initially omitted that multiply → loss/gradients 8× too **small** (step-10 loss ~1.51
  instead of ~12.1).
- **Fix**: `lm_loss *= accelerator.num_processes` when num_items_in_batch is in use.
- **Verification (smoke v2, 30 steps, 8 GPUs)**: step-10 loss 12.11 (orig-ANT) / 12.14
  (ant_ours) vs baseline's historical 12.1096. Grad norms exactly 8× smoke-v1's
  (8.875 = 1.109×8; 5.219 = 0.652×8), and old buggy run's grad norm / 128 (=16 accum ×
  8 procs) = smoke-v1's exactly (83.5/128 = 0.6523). Three independent consistency
  checks — the scaling math is airtight.
- Aux losses (e.g. div_loss) are scaled by 1/gradient_accumulation_steps when
  num_items_in_batch is active, matching how Trainer treats the returned loss.

### Bug 3 — evals loaded RANDOM embeddings (all round-1 eval results invalid)

- **What**: `train_config.json` was only written at the *end* of training. Runs stopped
  at 10K by run_experiments.py (kill) never wrote it. `is_compositional()` required that
  file, so eval fell back to a plain `from_pretrained` load — checkpoint has no
  embed_tokens weights (custom embedding lives in embedding.pt) → **freshly initialized
  random embedding matrix**. PPL ~35,616 for a model whose true PPL is ~34.
- **Fix (both sides)**:
  1. `save_train_config()` is now called **before** `trainer.train()` (rank-0 gated).
  2. `compositional/loading.py`: `is_compositional()` now keys on `embedding.pt`
     existing, and if train_config.json is absent the arm + hyperparams are **inferred
     from embedding.pt tensor names/shapes** (T→original_ant; A/X/W_q→ant; localenc.*→
     v2/isolation_control; etc.).
- **Verification**: on-machine probe of the old ant_ours checkpoint via fixed loader:
  en PPL 34.42 (was 35,616 through the broken path).

### Bug 4 — Original ANT collapse: lam=1e-3 was ~1000× the paper's value

- **What**: round-1 original_ant used a placeholder `--lam 1e-3`. The L1 proximal step
  drove T's support from 4096 → 0 nonzeros by step ~1300; loss flatlined after.
- **Paper (Liang et al. 2021)**: λ₂ ∈ {1e-6, 1e-5} for LM tasks, 1e-6 best.
- **Fix**: `scripts/train_original_ant.sh` now uses `--lam 1e-6` (paper-identical, fair
  comparison). Round-2 confirmation: at step ~1190, nnz still ~3936 and declining
  gently, dead_rate 0 — no collapse past the old failure point.
- HybridOptimizer itself was audited and is correct (AdamW decoupled decay for backbone;
  YOGI + per-coordinate proximal `max(p − λ·step_size/denom, 0)` on T = paper's eq. 3).

### Bug 5 — "ant_ours collapsed" was a false alarm caused by Dropbox file cross-delivery

- **What**: pulling files from both machines **simultaneously with identical job names**
  delivered th2's files under th3's names — ant_ours' "trainer_state.json" was a
  byte-identical copy of original_ant's (md5 on-machine 1c349b… ≠ delivered 4a649b…).
  This fabricated an "ant_ours collapsed at step 1300" story; re-pulling solo proved
  ant_ours never collapsed (nnz ~45 throughout).
- Same mechanism later cross-delivered **commands**: th3 once executed th2's cleanup
  script (proven: th3's log contained th2's marker text). Only the shared `logs` dir
  was affected; recovered by a solo retried run.

### Infrastructure protocol (adopted after the above)

1. **Never push to both machines simultaneously** — solo, staggered pushes only.
2. **Machine-unique job names** (`th2-…` / `th3-…` prefixes), never reused across
   machines.
3. After a push: wait 2–5 min, check `_RUN_STATUS_.log`, wait another 1–2 min before
   downloading outputs; verify md5 when a file's content is decision-critical.
4. Quick commands get `+120+a` (auto-pull log); max 10 files per `-f-` pull.
5. Don't re-push a machine's branch while it is mid-job unless the new push is intended
   to run.

### Clean slate + retrain (2026-08-08)

- Killed all runs; deleted every round-1 output (originals + `_old16x` archives + smoke
  dirs + logs) on both machines; **baseline kept** on h100-2.
- Wiped all HF datasets caches and `cache-*`/`tmp*` in the data dir (round-2 runs
  re-preprocess ~30 min each); GPUs verified 0 MiB; accelerate config re-copied.
- Relaunched with fresh-dir guards (hard fail if output dir exists → no silent
  checkpoint resume): h100-1 `--experiments 0 2` (original_ant then v2_attn),
  h100-2 `--experiments 1` (ant_ours), both `--stop-at-step 10000`.
- **Early verification passed on both machines**: step-10 loss 12.11 / 12.14 with the
  exact smoke-v2 grad norms; ~3.44 s/it → ~9.5 h per 10K run.

## Round-2 Results (2026-08-10, all arms at checkpoint-10000, fixed code + fixed eval)

All three retrains completed cleanly at exactly step 10000 (~10.3h each). Evals ran with
the fixed loader ("Loaded compositional model: arm=..." confirmed in logs); baseline
results dumped directly from machine. All numbers below are trusted.

### Held-out PPL @ 10K (10M tokens/language)

| Model | en | vi | zh | ru | de | ar | avg |
|---|---|---|---|---|---|---|---|
| baseline | **33.05** | **21.32** | **84.92** | **18.73** | **29.41** | **24.02** | **35.24** |
| original_ant | 34.44 | 22.06 | 95.20 | 19.71 | 31.23 | 25.04 | 37.94 |
| ant_ours | 34.08 | 22.17 | 94.98 | 19.92 | 31.06 | 26.51 | 38.12 |
| v2_attn | 34.25 | 21.99 | 96.17 | 19.89 | 30.86 | 26.28 | 38.24 |

- PPL curves monotone decreasing for every arm/language — no collapse anywhere.
- Compositional arms sit ~8% avg PPL behind baseline at 10K (en gap only ~3-4%).
- ant_ours/v2_attn (~23.7M embed params, ~45 active anchors) ≈ original_ant
  (~626M embed params, ~4000 active) — huge efficiency win for entmax routing.
- v2_attn ≈ ant_ours so far: context-conditioned routing not yet separating from
  static routing at 10K on PPL.
- Benchmarks @10K: averages within noise of each other (baseline 0.3702,
  ant_ours 0.3679, v2_attn 0.3667, original_ant 0.3660); most multilingual tasks
  still near chance at this scale — not yet discriminative. Clearly-above-chance
  tasks only: hellaswag-en (~0.30 vs 0.25), xstorycloze-en (~0.59 vs 0.50),
  xcopa-vi (~0.55), paws-en/de (~0.53); "best per task" is spread across all four
  arms (noise pattern). Full 26-task tables: temp/r2_eval/<model>/checkpoint-
  10000_eval_benchmarks.json.
- Training loss @10K: baseline 3.147 < ant_ours 3.19 ≈ v2_attn 3.192 < original_ant 3.203.

### Eval-PPL trajectory (avg over 6 languages, checkpoints 1k-10k)

| step | baseline | original_ant | ant_ours | v2_attn |
|---|---|---|---|---|
| 1000 | 178.90 | 327.50 | 263.82 | 265.46 |
| 2000 | 79.13 | 178.43 | 113.04 | 114.94 |
| 3000 | 57.14 | 72.80 | 72.93 | 73.18 |
| 4000 | 49.03 | 58.83 | 56.86 | 56.87 |
| 6000 | 41.51 | 46.13 | 46.30 | 45.99 |
| 8000 | 37.45 | 40.65 | 41.14 | 40.85 |
| 10000 | 35.24 | 37.94 | 38.12 | 38.24 |

Compositional arms start far behind (composition is harder to learn early), catch up
fast, and are still closing on baseline at 10K (Δavg-PPL 9k→10k: baseline −1.27 vs
ant_ours −1.88) — relevant to the full-35K go/no-go: the 10K snapshot catches them
mid-trajectory.

### Result data locations (per analysis)

| Analysis | Local files (dev machine, gitignored temp/) | Machine files (/opt/dlami/nvme/sparse_emb_outputs/) |
|---|---|---|
| Round-2 PPL curves | `temp/r2_eval/{baseline,ant_ours,v2_attn,original_ant}/checkpoint-{1000..10000}_eval_ppl.json` | `<model>/checkpoint-<step>/eval_ppl.json` |
| Round-2 benchmarks | `temp/r2_eval/<model>/checkpoint-<step>_eval_benchmarks.json` | `<model>/checkpoint-<step>/eval_benchmarks.json` |
| Freq-binned PPL (per-token-id NLL) | `temp/r2_eval/bytoken/<model>_eval_ppl_bytoken.npz` + `_summary.json` | `<model>/checkpoint-10000/eval_ppl_bytoken.npz` |
| Train-corpus token freqs (1/10 sample) | `temp/r2_eval/bytoken/misc_token_freq.npz` (+ committed copy `resources/token_freq_sample10.npz`) | `token_freq.npz` + `token_freq_meta.json` |
| Anchor-usage distribution | metrics JSON transcribed in probe logs `temp/{th2,th3}__run-*-anchor-usage*.log` / `*probe-anchor-results.log` (npz NOT pulled yet) | `<model>/checkpoint-10000/anchor_usage.{npz,json}` |
| Training loss logs (round 2) | `temp/h100-{1,2}__run-*tail-train-loss*.log`, `*check-progress*.log` | `logs/<model>.log` (full tqdm+loss stream) |
| Cross-lingual Phase 1 | not pulled yet | `crosslingual/<model>/<model>@10000.json` + `crosslingual/<model>.log` |
| Smoke tests (loss-scaling fix) | `temp/*smoke*-fixed*.log` | (smoke output dirs deleted in cleanup) |

Note: `temp/` is gitignored — these local copies do not survive a dev-machine switch;
the machine-side copies and the tables in this document are the durable record.

### Paper framing: why entmax routing beats Original ANT (worth a paragraph in the paper)

Both use the same K=4096 anchor codebook; the difference is how tokens select anchors,
and round-2 gives hard evidence for our side:

1. **ANT's params are huge by construction, and its promised compression never
   materialized.** T is a dense N×K matrix from step 0: 151,936 × 4096 ≈ 622M trained
   params (+A ≈ 4.2M) — the "compression method" more than doubles Qwen3-0.6B. The ANT
   paper's efficiency claim is about *storage after training*: L1 prunes T, you store
   only nonzeros (N×s + K×d). But at the paper's own best LM value λ=1e-6, our run ends
   at **nnz ≈ 4003/4096 (s ≈ K, 98% dense)** — nothing was pruned, so post-hoc storage
   ≈ the dense table. And at λ=1e-3 (round 1) T collapsed to all-zeros by step ~1300.
   L1-prox sparsity sits on a knife edge between "not sparse" and "dead".
2. **Ours is compressed unconditionally.** Coefficients are computed (attention query
   from X ∈ N×128 + entmax), not stored: ~23.7M params (27× less than ANT) during
   training, at deployment, regardless of any hyperparameter. entmax gives ~45 active
   anchors per token *structurally* — 90× sparser activation than ANT's ~4000 — at
   equal PPL (38.12 vs 37.94 avg) with plain AdamW (no YOGI/prox machinery).
3. **Shared routing geometry**: ANT's per-token coefficient rows are independent
   parameter blocks ("dog" teaches "dogs" nothing); our tokens share the routing space
   X and query projection, so selection knowledge transfers across the vocabulary
   (basis for the rare-token hypothesis below).
4. **Context routing is only possible in our formulation** (coefficients = f(query) →
   swap in a context-dependent query = V2-attn); a stored table T cannot depend on
   context by definition.
- Honesty caveats for the paper: K=4096 (matched to our arms) inflates ANT's N×K vs the
  small K used in their paper; and our arms untie lm_head (+155.6M dense output layer),
  so efficiency claims must be scoped to the *input* embedding.

### Planned evals to show the method's advantage (2026-08-10 discussion)

- [ ] **ALBERT-style low-rank baseline** (Lan et al. 2020; trains from scratch like us):
  Embedding(151936, 128) + Linear(128, 1024) ≈ 19.6M params ≈ ant_ours budget, no
  routing. The critical control: does anchor routing beat a plain linear bottleneck?
- [ ] **Hash embeddings** (Svenstrup et al. 2017) as second control: anchors selected by
  hash instead of learned content-based routing.
- [ ] **Frequency-binned PPL** (standard diagnostic, cf. Baevski & Auli ICLR'19 who
  report PPL by word-frequency buckets): same PPL formula, computed on subsets of
  positions binned by the target token's training frequency. Implementation: ppl.py
  with reduction='none', scatter-add nll_sum[id]/count[id] (vocab-sized arrays) per
  checkpoint; bins defined offline (equal-mass head/torso/tail). Bins decompose overall
  PPL exactly (count-weighted geometric mean). Hypothesis: compositional arms win/close
  the gap on tail tokens via shared anchors + shared routing geometry.
- [x] **Frequency-binned PPL — DONE 2026-08-10.** Per-token-id NLL arrays
  (eval/ppl_bytoken.py, self-verified: max chunk gap 9.5e-7; per-lang recombination
  matches round-2 eval to ~3e-9) + train-corpus counts on 3.5B-token 1/10 sample
  (scripts/count_token_freq.py). Equal-mass bins: head=19k types (90% mass),
  torso=44k (9%), tail=89k (1%). Pooled PPL @10K:
  | model | head | torso | tail | (ratio vs baseline) |
  |---|---|---|---|---|
  | baseline | 19.97 | 144.2 | 894.7 | — |
  | original_ant | 20.86 | 159.7 | 1069.2 | 1.045 / 1.108 / 1.195 |
  | ant_ours | 21.12 | 159.3 | 1061.5 | 1.058 / 1.105 / 1.186 |
  | v2_attn | 21.15 | 157.5 | 1033.4 | 1.059 / 1.093 / 1.155 |
  Findings: (a) **rare-token hypothesis refuted** — compositional gap WIDENS with
  rarity (+6% head → +16-20% tail); anchors/routing are dominated by frequent-token
  gradients while baseline keeps dedicated rows. (b) **First v2_attn separation**:
  tied with ant_ours on head, clearly better on torso (1.093 vs 1.105) and tail
  (1.155 vs 1.186, ~2.7% better tail PPL) — context routing compensates where static
  routing is weak. Tells the v2 story via frequency split; motivates ambiguous-token
  analysis. (c) original_ant best-compositional on head (memorizing table) but worst
  on tail (1.195) — supports shared-routing-generalizes narrative.
- [x] **Anchor-usage distribution — DONE 2026-08-11** (scripts/analyze_anchor_usage.py;
  exact over vocab for static arms, 3.0M eval positions for v2; arrays in
  checkpoint-10000/anchor_usage.npz on machines). Traffic-weighted load (θ mass):
  | metric | original_ant | ant_ours | v2_attn |
  |---|---|---|---|
  | dead anchors | 0 | 0 | 0 |
  | Gini | 0.066 | 0.536 | 0.485 |
  | effective anchors (1/HHI) | 4041 | 304 | 582 |
  | effective anchors (exp-entropy) | 4068 | 1593 | 2251 |
  | top-1% (41 anchors) load share | 1.3% | 26.4% | 15.7% |
  Findings: (a) original_ant "uniform" only because ~98% dense selection — no routing
  structure; (b) ant_ours concentrates load (Zipf + large θ on few anchors for frequent
  tokens; per-type selection itself fairly even, Gini 0.14); (c) **v2_attn spreads load
  ~2× better than ant_ours** — context routing diversifies anchor use, consistent with
  its torso/tail PPL advantage; (d) zero dead anchors with lambda_div=0.
  Follow-up ablations suggested: small lambda_div for ant_ours; smaller K.
  **Paper figure from the npz files**: each checkpoint-10000/anchor_usage.npz holds
  per-anchor load arrays (static arms: sel_type/w_type/sel_freq/w_freq, each float64[4096];
  v2: sel_occ/w_occ + n_positions). Plot sorted-load (rank vs load share, log-y) or
  Lorenz curves of w_freq/w_occ for the three arms in one figure — shows original_ant's
  flat no-structure line vs ant_ours' concentration vs v2_attn in between. Files are
  small (~80KB); pull via -f- when making figures.
- [x] **Cross-lingual transfer battery — DONE 2026-08-12** (crosslingual/ port of the
  embeddings-hub probe suite; t6 BLI/CSLS + McNemar, t8 MEXA 500 FLORES sents, probe_b;
  checkpoint-10000 all arms; JSONs in temp/xling_results/, report.md there too).
  **HYPOTHESIS REFUTED — compositional embeddings HURT cross-lingual transfer:**
  | metric | baseline | ant_ours | v2_attn | original_ant |
  |---|---|---|---|---|
  | T6 BLI mean P@1 (embed layer) | **0.278** | 0.013 | 0.006 | 0.049 |
  | T8 MEXA avg (all layers) | **0.556** | 0.428 | 0.432 | 0.429 |
  | Probe B cos gap (embed layer) | **0.076** | 0.016 | 0.013 | 0.054 |
  All deficits McNemar/MWU p ≈ 0 (e.g. ant_ours all-pairs: baseline-only hits 1338 vs
  model-only 7, n=3511). Findings: (a) dense baseline word embeddings align languages
  far better than any compositional arm; (b) severity tracks routing sparsity —
  original_ant (~4000 near-dense coefficients) preserves 4-8x more word-level alignment
  than the entmax arms (~45 anchors), i.e. sparse quantized routing fragments the
  cross-lingual geometry (consistent with anchors encoding identity/frequency, not
  meaning — cf. anchor-usage analysis, and with the tail-PPL deficit: less smooth
  embedding space); (c) the deficit shrinks but persists at depth (MEXA −0.13 at
  layers 16-22 peaks); (d) v2_attn ≈ ant_ours on sentence-level MEXA (0.432 vs 0.428)
  but worst on isolated-word tests — expected, its context routing is inactive on
  single words. Paper framing: cross-lingual alignment is a COST of sparse
  compositional embeddings at this scale, to report honestly alongside the
  efficiency/tail tradeoffs; do NOT claim a transfer advantage.
- [ ] V2-attn-specific: PPL on high-routing-variance (ambiguous) tokens; routing entropy
  vs polysemy analysis; cross-lingual anchor-overlap analysis.
- [ ] **Vocab-expansion demo**: add unseen tokens post-training; ours learns only a
  128-dim routing vector per new token vs baseline's full 1024-dim row.

## Cross-lingual transfer testing (started 2026-08-11)

Question: do compositional embeddings (shared anchor codebook) give better
cross-lingual transfer than the dense baseline? Structural motivation: unlike the
failed additive "EmbHub" project (/disk/thuat/cross_lingual_embeddings_hub — its
hypothesis was a controlled negative), our models FORCE every token's embedding to be
composed from shared anchors (bottleneck, not bypass), so cross-lingual sharing could
be structural. Known prior: the baseline develops some alignment on its own from
monolingual data; the question is the gap over baseline.

### Test battery (`crosslingual/`, ported from the EmbHub project)

Six architecture-independent tests (standard forward passes only, so compositional
checkpoints work via the existing loader; hidden_states[0] = our module's output):

| Test | What | Role |
|---|---|---|
| t6 BLI/CSLS | translation retrieval P@1/P@5 + exact McNemar paired vs baseline | headline (word-level) |
| t8 MEXA | FLORES-200 parallel-sentence mutual-NN per layer (500 sents/lang) | headline (sentence-level, published metric) |
| probe_b | translation-vs-random cosine gap + Mann-Whitney | secondary (gameable — never headline) |
| t5 layer sweep | in-context per-layer gap (fixed: matches space-prefixed tokens too) | diagnostic (where alignment lives) |
| t7 code-switch | logP(translation)−logP(random) | weak (frequency-dominated at this scale) |
| t1 XNLI probe | English-trained linear probe tested on 6 langs | functional transfer; risky (may be all-chance at 0.6B/10K) |

Drivers: `run_crosslingual.py` (loads model once, runs chosen tests, per-test
incremental JSON writes), `run_parallel.py` (eval_parallel-style GPU queue,
labels `<model>@<step>`), `merge_report.py` (cross-machine report merge).
Assets committed: `resources/frequent_translations_llm.json` (4,804 GPT-4o tuples)
+ `resources/flores200/` (500 sents × 6 langs).

Methodology guardrails (inherited from EmbHub's hard lessons): per-pair loanword
filter (worth ~20× on its own), single-token words, related (en-de,en-vi) vs
distant (en-zh,en-ru,en-ar) split, baseline measured at the same layer, en-vi
n=58 → do not over-read. Single-token pair counts (Qwen3 tokenizer): zh 2381,
ar 763, de 181, ru 128, vi 58.

Validation (dev machine, 2026-08-11/12):
- Specificity: random tiny models → probe_b gap −0.0008 (p=0.68), t6 P@1 ≈ chance,
  McNemar p 0.38–1.0 (no false positives); both plain and compositional load paths.
- Sensitivity: pretrained multilingual Qwen3-0.6B → t6 P@1 mean 0.62 (zh 0.85),
  probe_b gap +0.260 (p≈0), MEXA best 0.86–1.00 at layers 18–23 (mid-layer peak
  matches the MEXA paper). Pair counts reproduce EmbHub's documented n's exactly.
- Dictionary transferability to OUR data verified against token_freq counts: every
  test word appears in our training sample (0% unseen), median exposure 37k–130k
  occurrences per word.
- McNemar implementation unit-tested against scipy binomtest.

Interpretation notes for reading results: t6/probe_b feed words in isolation → for
v2_attn they measure its *static* routing; t5/t8 use real sentences → exercise
context routing. Our non-English exposure (5×1B tokens) is ~3× EmbHub's — never
compare absolute numbers across projects, only against our own baseline. If an arm
wins, the ALBERT low-rank baseline doubles as the no-routing control for the
cross-lingual claim too.

### Status

- [x] Battery built + validated (commits 6b3ce43, 9f9267f, f0b95eb, 6d75c8d, f48e8a8)
- [x] **Phase 1 DONE (2026-08-12)**: t6+t8+probe_b on all 4 arms @checkpoint-10000.
  Results pulled + verified (md5 match on re-pull, per-pair recomputation exact).
  See "Cross-lingual battery results" above for full tables + McNemar.
- [ ] Phase 2 (if Phase 1 interesting): θ-based anchor-overlap probe (translations
  share anchors? — the mechanism test, loanword-filtered) + θ-based language
  decodability (do anchors encode language identity or meaning?); alignment-vs-step
  curves (t6/t8 at 2k..10k, checkpoints exist every 250); t1 at mid layers; t5.

### Remote runner semantics (docs/commands.md — learned the hard way)

`#1` = submit only, NO log ever uploads (completion is silent by design);
`#1 +W+a` = submit + wait W sec + pull log (partial if still running); `#2` = pull
only, executes nothing (`-0/-1` recency, `+a` full, `-f-` files; folders with
trailing `/`; ≤10 items, ≤50MB). NEVER re-push the same `#1` commands to fetch a
log — every `#1` push re-executes the script (duplicate run). Never launch
never-ending jobs (dummy.py) with `+W+a`. If a push shows no `_RUN_STATUS_.log`
entry in ~10 min, report to the user and wait (runner outages: 3 so far).

## ALBERT + Residual ANT experiments (started 2026-08-13)

Two new arms: the critical matched-params controls for the compositional story.

### Arms

| Arm | Architecture | Embed params | What it tests |
|---|---|---|---|
| ALBERT (lowrank) | `e = X[ids] @ proj` (V×128 + Linear(128→1024)) | 19.6M | Does anchor routing beat a plain linear bottleneck at matched rank? |
| Residual ANT | `e = X[ids] @ W_up + θ @ A` (identity + codebook) | 23.9M | Does the codebook add value when the identity path handles discrimination? |

Both verified against the official Google ALBERT repo (google-research/ALBERT):
factorization itself (V×E lookup → Linear(E→H, bias, no activation)) is identical;
init matches HF PyTorch convention (normal(0.02), not TF's truncated_normal — same
as HF's own modeling_albert.py). Omissions (pos/type embeddings, LayerNorm at E,
dropout) are BERT-block components that Qwen3 doesn't have.

Residual ANT inherits from ANTEmbed, adds W_up = Linear(128→1024, bias=True).
Forward: `e_id + e_code, theta`. At init both paths contribute (~37% identity,
~63% codebook by norm). The identity path provides a dense gradient to X regardless
of routing quality (breaking the bootstrap problem).

### Training status

| Arm | Machine | Status | Train loss @10K |
|---|---|---|---|
| ALBERT | h100-1 | **DONE** (35,851s ≈ 10.0h) | **3.160** |
| Residual ANT | h100-2 | Running (~step 5,000+) | TBD |

### Early results: ALBERT

**ALBERT train loss 3.160 vs baseline 3.147 — only +0.4% gap** with 8× fewer
embedding params (19.6M vs 155.6M). This means the 128-d linear bottleneck is
nearly sufficient for Qwen3-0.6B at 10K steps.

Critically, ALBERT **beats every compositional arm** (ant_ours 3.190, v2_attn
3.192, original_ant 3.203) despite having *fewer* params than any of them
(19.6M vs 23.7–23.9M). This raises the bar for what anchor routing must
demonstrate: it isn't enough to beat the dense baseline — it must beat ALBERT
at matched rank to prove the codebook adds value over plain factorization.

**Eval PPL + benchmarks running now** (th2, checkpoints 1k–10k, 8-GPU parallel).

### Interpretation matrix (from the design doc)

| ALBERT vs Baseline | Residual ANT vs ALBERT | Meaning |
|---|---|---|
| ALBERT ≈ Baseline (3.160 vs 3.147 ✓) | Residual < ALBERT | Codebook adds value on top of low-rank |
| ALBERT ≈ Baseline ✓ | Residual ≈ ALBERT | Codebook ignored by optimizer |
| — | Residual > ALBERT | Codebook hurts (unlikely given ant_ours trains fine) |

The quantity `gap_residual - gap_albert` = the codebook's marginal value.

## Fine-tune Benchmark Results (2026-08-15)

Generative fine-tuning: train with causal LM loss on (prompt + completion)
matching lm-eval-harness format exactly, eval with lm-eval-harness in-memory.
3 tasks × 6 arms × 3 seeds = 54 runs total. Right padding, loss on completion
tokens only. Each task trains on English, evals on English + multilingual.

### Tasks

| Task | Train data | Eval (English) | Eval (multilingual) |
|---|---|---|---|
| HellaSwag | 40K (EN) | hellaswag (acc_norm) | ar, de, ru, vi |
| ARC-Easy | 2.3K (EN) | arc_easy (acc_norm) | ar, de, ru, vi, zh |
| XNLI | 393K (EN MultiNLI) | xnli_en (acc) | vi, zh, de, ru, ar |

### Full results (acc_norm where available, else acc; mean±std over 3 seeds)

| Benchmark | baseline | lowrank | ant_ours | v2_attn | original_ant | residual_ant |
|---|---|---|---|---|---|---|
| hellaswag EN | **0.359** | 0.356 | 0.352 | 0.349 | 0.343 | 0.356 |
| hellaswag ar | 0.275 | 0.271 | 0.272 | **0.274** | **0.276** | 0.274 |
| hellaswag de | 0.282 | **0.283** | 0.281 | 0.280 | 0.279 | 0.281 |
| hellaswag ru | 0.282 | 0.282 | 0.277 | **0.284** | 0.276 | 0.273 |
| hellaswag vi | 0.301 | **0.307** | 0.302 | 0.302 | 0.293 | 0.305 |
| arc_easy EN | 0.457 | **0.467** | **0.467** | 0.459 | 0.443 | 0.464 |
| arc ar | 0.226 | 0.235 | **0.241** | 0.238 | 0.238 | 0.223 |
| arc de | **0.228** | 0.224 | 0.222 | 0.224 | 0.219 | 0.224 |
| arc ru | 0.219 | **0.228** | 0.225 | 0.218 | 0.212 | 0.226 |
| arc vi | 0.219 | 0.223 | 0.219 | 0.216 | **0.225** | 0.223 |
| arc zh | 0.241 | 0.254 | 0.254 | **0.258** | 0.246 | 0.242 |
| xnli EN | 0.623 | 0.651 | 0.651 | **0.659** | 0.633 | 0.644 |
| xnli de | **0.371** | 0.335 | 0.339 | 0.336 | 0.344 | 0.344 |
| xnli ru | **0.367** | 0.343 | 0.360 | 0.340 | 0.354 | 0.344 |
| xnli vi | 0.345 | 0.337 | 0.346 | 0.350 | **0.351** | 0.340 |
| xnli zh | 0.335 | 0.335 | **0.346** | 0.343 | 0.336 | 0.337 |
| xnli ar | 0.333 | 0.334 | 0.333 | 0.333 | 0.334 | 0.332 |
|---|---|---|---|---|---|---|
| **AVERAGE** | 0.3213 | 0.3214 | **0.3228** | 0.3214 | 0.3177 | 0.3194 |
| **English avg** | 0.4795 | **0.4912** | 0.4898 | 0.4889 | 0.4731 | 0.4880 |
| **Non-English avg** | **0.2874** | 0.2850 | 0.2870 | 0.2855 | 0.2844 | 0.2833 |

### Key findings

1. **English tasks: compressed embeddings BEAT baseline after fine-tuning.**
   XNLI-EN: v2_attn 0.659 > ant_ours/lowrank 0.651 > baseline 0.623 (+3.6%).
   ARC-Easy: lowrank/ant_ours 0.467 > baseline 0.457. HellaSwag: baseline
   leads slightly (0.359 vs 0.356). English avg: lowrank 0.491 > baseline 0.480.
   This is the first evidence that compressed embeddings learn *better*
   task-specific structure — the bottleneck acts as implicit regularization.

2. **Cross-lingual: baseline leads on DE/RU after XNLI fine-tuning** (0.371/0.367
   vs ~0.335-0.360 for compositional arms). But the gap is much smaller than the
   embedding-level BLI deficit (0.278 vs 0.013) — fine-tuning partially recovers
   cross-lingual ability through the backbone. VI/ZH are competitive or better
   for compositional arms, suggesting the deficit is language-pair-dependent.

3. **ant_ours wins overall average (0.3228)** — the entmax-routed codebook-only
   architecture, with 6.5× fewer embedding params than baseline, produces the
   best aggregate downstream performance after fine-tuning. This was not visible
   in zero-shot benchmarks (which were near chance) nor in raw PPL (where
   baseline led by 8%).

4. **original_ant is consistently worst** (0.3177 avg, 0.473 EN avg) — its
   626M-param near-dense embedding table neither helps nor is parameter-efficient.

5. **residual_ant ≈ lowrank on English** (0.488 vs 0.491) — the codebook adds
   negligible value over the plain identity path on downstream tasks, consistent
   with the training loss result (3.172 vs 3.160).

### Data locations

Result JSONs: `/opt/dlami/nvme/sparse_emb_outputs/finetune/` on both machines.
Per-run logs: same directory, `*.log` files. Local copies: `temp/finetune_gen/`.
Code: `finetune/{tasks,train,run_all}.py`.

## TODO

- [ ] Run eval PPL + benchmarks for ALBERT and Residual ANT (checkpoints on machines)
- [ ] Run frequency-binned PPL + cross-lingual battery on both new arms
- [ ] Decide go/no-go for full 35K run based on all results
- [ ] Pull training metrics (loss curves, ppl) from wandb offline logs
