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
  still near chance at this scale — not yet discriminative.
- Training loss @10K: baseline 3.147 < ant_ours 3.19 ≈ v2_attn 3.192 < original_ant 3.203.

## TODO

- [ ] Decide go/no-go for full 35K run based on round-2 de-risk results
- [ ] Pull training metrics (loss curves, ppl) from wandb offline logs
