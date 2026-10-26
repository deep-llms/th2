# KV-sharing recovery: 27 October 2026

Longer recovery helps, but this conversion is still too lossy to justify MLP
pruning. Both runs completed; the next question is where sharing is applied.

| Additional steps | Dense A PPL | Shared B PPL | B penalty |
|---:|---:|---:|---:|
| 0 | 20.0379 | 26.2246 | 30.88% |
| 512 | 20.1985 | 24.4859 | 21.23% |
| 1024 | 19.7000 | 23.1817 | 17.67% |
| 1536 | 19.3272 | 22.5403 | 16.62% |
| 2048 | 19.2379 | 22.3897 | 16.38% |

Final BF16-weight inference PPL: **A 19.2416, B 22.3924** (+16.37%).
Paired B−A NLL: **+0.151645 nats/token**, 95% document bootstrap CI
[0.145347, 0.158700]. B is worse on all 256 validation documents. The upper CI
exceeds log(1.05), so the predefined gate for pruning after recovery fails.
The late curve flattens while LR decays; this does not prove further learning
is impossible. It does show that this recovery budget has not made sharing
competitive. No downstream, long-context or novelty claim follows.

Each arm starts from its old 512-step checkpoint, with a new matched AdamW/cosine
phase: 2,048 steps, LR 5e-5, 102 warmup, .5CE+.5KL, all parameters train, eight B200s,
8 documents/GPU, length 1024. Each sees 131,072 fresh document prefixes (134.22M
input tokens). Old training/calibration/validation hashes are excluded; old
256-document validation reused exactly. This is development validation.

A/B measured training times (including in-loop validation/saves) were
522.31s/491.75s. Whole queue including smoke/loading finished **07:39:16 SGT**;
burn handoff completed **07:40:27 SGT**. Read-only health at 07:40:39 independently
confirmed eight burns, advancing collectives and a released guard. Current
workers 615740–615747 under launcher 615671; re-inspect before any stop.

Checks pass: actual per-rank sample orders match, exact one-pass coverage,
zero old-document overlap, unchanged validation, parent per-document NLL
reproduction and checkpoint reload differences both zero. Native eight-GPU BF16
Accelerate cache configuration verified. Environment/backend unchanged.

Run: `6y3o-efficient-kv-recovery-20261027-a02`, launch `a1847c0`, health `ce33588`.
[Machine-readable summary](results/kv_recovery_20261027/supervised/run/summary.json),
[completion](results/kv_recovery_20261027/supervised/run/complete.json),
[artifact receipts](results/kv_recovery_20261027/artifact-receipts.json).
All 14 exported artifacts reconstructed from the read-only log and SHA256-verified.
Raw local evidence: `temp/kv_recovery_tick01_health.log`; extracted originals:
`temp/kv_recovery_results_20261027/`. Large checkpoints stay on B200.

## Next controlled hypothesis

Compare full 50% sharing with two 25% sharing layouts: evenly spread pairs and
upper-half pairs. All three are newly converted from the same dense A-512
checkpoint, then use the exact same 2,048-step phase/data as the completed dense
A control. This isolates placement at matched 25% cache capacity and retains a
matched 50% conversion control. The old B continuation is historical context, not
a matched initialization for these new conversions. No MLP pruning or new
backend/loss. This is a conversion-baseline study, not a claimed novel method.
