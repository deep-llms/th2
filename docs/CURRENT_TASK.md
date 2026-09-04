# Current task: Tiered RankLift-512 and language-balanced GroupReduce

Update (2026-09-04): launch the controlled pair
`tiered_ranklift_lb_t4_c512` followed by `groupreduce_matched_lb_t4`, using the
same language-balanced static membership, data, optimizer, BF16, eight-GPU
B200 configuration, and 10k cutoff. Tiered preserves GroupReduce's stored
widths `1024/512/192/64` and adds lift widths `0/0/320/192`; it has 20,096,000
interface parameters, +672,768 over the 19,423,232-parameter control (about
0.15% of the full model). The control is required to distinguish architecture
gain from grouping gain. See `docs/language_balanced_tiered_ranklift.md`.

The historical task below is retained for provenance.

# Pure-local G16 R128 structural control

Status: implementation and local CPU validation complete; remote training has
not been launched.

## Purpose

The fair B200 checkpoint-10k results show that SharedLocal G16 tied does not
clearly outperform global low-rank R128 tied: mean held-out PPL is 33.93 versus
33.97, zero-shot accuracy is 0.3673 versus 0.3686, and finetuned accuracy is
0.3135 versus 0.3122. The next experiment isolates whether SharedLocal's single
global rank-64 basis is useful at all.

Pure-local G16 R128 represents token `i` in contiguous vocabulary group `g` as

```text
e_i = z_i @ L_g.T + b
```

There is no globally shared basis. Each of the 16 groups owns a rank-128 basis,
while the same factors are used for input lookup and exact tied output. Qwen's
151,936-token vocabulary divides evenly into 16 groups of 9,496 tokens, so the
production output path is a padding-free grouped batched GEMM in token-ID order.

The interface has 21,545,984 parameters: 19,447,808 token coefficients,
2,097,152 group-basis parameters, and one 1,024-dimensional shared bias. It has
983,040 more interface parameters than SharedLocal G16 64+64, because all 128
basis directions are replicated across every group. This run is the natural
equal-rank structural control, not an exactly parameter-matched control.

## Implementation

- New arm: `pure_local` with `--pure_local_rank` and the existing
  `--num_groups` option.
- New embedding: `PureLocalEmbed` in `compositional/embeddings.py`.
- New exact tied output: `TiedPureLocalHead` in
  `compositional/tied_head.py`; it does not materialize a dense vocabulary
  table.
- Training requires `--tie_output`, preventing an accidental dense output head.
- Checkpoint loading supports config-based and configless reconstruction and
  retains the existing strict sidecar/HF-state integrity checks.
- Production script: `scripts/train_pure_local_tied_g16_r128.sh`.
- Sequential-runner entry: index 17, `pure_local_tied_g16_r128`, with a fresh
  output requirement and the complete eight-rank checkpoint manifest.

## Validation completed

- Python compilation and shell syntax checks pass.
- `python -m compositional.tests` passes, including pure-local group coverage,
  shapes, and nonzero finite gradients.
- The complete repository pytest suite passes: 62 tests.
- Tests cover exact equivalence between the factorized tied head and an
  explicitly materialized embedding table, both divisible G16 and uneven
  padded vocabularies, gradient equivalence, unused-padding gradients, single
  parameter ownership, Qwen forward/save/load, configless inference, CLI
  construction, and rejection of a dense output configuration.
- A production-dimension CPU smoke (`V=151936`, `d=1024`, `G=16`, `r=128`)
  produced the exact 21,545,984 parameters, 9,496 tokens/group, finite input
  embeddings and full-vocabulary logits, finite CE loss, and finite gradients
  for every parameter.
- A real four-A100 BF16/NCCL DDP smoke at the exact production interface shape
  passed with `find_unused_parameters=False`: every group received finite,
  nonzero basis and token-factor gradients even though the inputs used only
  group zero; the SGD update was identical across all four ranks. Rank 0
  reported finite loss 11.959379, logits shape `(1, 8, 151936)`, and 240.7 MiB
  peak allocated memory. All four GPUs were verified free afterward.
- The actual `train_compositional.py`/HF Trainer path completed step 1 on a
  tiny Qwen model, saved a complete periodic checkpoint and `embedding.pt`,
  reloaded through the evaluation loader, and generated successfully. It then
  automatically resumed from checkpoint 1 to step 2 with optimizer, scheduler,
  RNG, HF state, and Pure-local sidecar validation intact.

## Next actions

1. Prepare `commands.sh` for th2: verify eight B200 GPUs are free, verify the
   output directory and relevant dataset caches are fresh, copy the accelerate
   configuration, and launch runner index 17 with `--stop-at-step 10000`.
2. Evaluate checkpoint 10k with the same PPL, zero-shot, and finetuning protocol
   used for the four fair B200 controls.
3. Interpret the result as a structural ablation. If Pure-local wins, the
   shared-global branch is not justified; if SharedLocal wins clearly, the
   global-plus-local decomposition receives direct support.
