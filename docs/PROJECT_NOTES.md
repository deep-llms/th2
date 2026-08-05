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

| Arm | Machine | Status | Wall time | Notes |
|-----|---------|--------|-----------|-------|
| Original ANT | h100-1 | Done 10K | ~10h | |
| ANT (ours) | h100-2 | Done 10K | ~10h | |
| V2-attn | h100-1 | Done 10K | ~12h | |
| Baseline | h100-2 | Done 10K | ~12h | Standard Qwen3-0.6B, tied embeddings |

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

## TODO

- [ ] Pull training metrics (loss curves, ppl) from wandb offline logs
- [ ] Compare baseline vs V2-attn vs Original ANT at 10K steps
- [ ] Decide go/no-go for full 35K run based on de-risk results
- [ ] Evaluate checkpoints (ppl + benchmarks) using eval/eval_checkpoint.py
