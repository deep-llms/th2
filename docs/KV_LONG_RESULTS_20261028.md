# Longer KV-sharing recovery results

Both runs completed successfully at **00:38:46 SGT, 28 October 2026**.
Recipe: [KV_LONG_RECOVERY_20261027.md](KV_LONG_RECOVERY_20261027.md).

Each arm starts from the same original dense A-512 checkpoint and receives
1,360,003,072 recovery tokens (20,752 steps), with the same frozen teacher,
0.5 CE + 0.5 forward KL loss, data order and full-budget cosine schedule.
All parameters train. B shares K/V from the preceding layer at seven uniformly
spaced consumers, retaining 75% of the dense model's KV-producing layers.

| Arm | Final BF16-weight PPL ↓ | Gap to A | Training time |
|---|---:|---:|---:|
| Dense A | 17.657243 | — | 82.25 min |
| Uniform25 B | 18.200775 | +3.078% | 79.40 min |

Evaluation is the frozen 256-document × 1,024-target prefix probe, not the full
CulturaX validation set or downstream tasks. Paired B−A NLL is +0.030318 nats/token,
95% document-bootstrap CI [0.028418,0.032295]. This corresponds to a PPL increase
of 2.883–3.282%. B improves 4/256 documents. B remains worse than A, but passes the
prespecified upper95%CI <= log(1.05) quality screen.

Training-validation learning curve (FP32 parameters with BF16 autocast; slightly
different from final BF16-weight inference above):

| Step | Recovery tokens | Dense A PPL | Uniform25 PPL | Gap |
|---|---:|---:|---:|---:|
| 2,048 | 134.22M | 19.74698 | 20.86774 | +5.68% |
| 5,188 | 340.00M | 19.07789 | 19.84761 | +4.03% |
| 10,376 | 680.00M | 18.31229 | 18.91082 | +3.27% |
| 20,752 | 1,360.00M | 17.65549 | 18.19898 | +3.08% |

The gap narrows with training, with smaller improvement in the second half.
The earlier short-run gap was 6.34%, but that run had a shorter schedule and a
different shuffled order from its smaller dataset. The table above is the
within-run comparison; this is not an exact continuation of that earlier run.
One seed and reused development validation limit generalization. Passing this
screen supports considering inference and broader quality measurements, but
does not demonstrate actual latency benefits or downstream equivalence.
No additional evaluation/training jobs were launched automatically.

Data/model hashes, identical parent files and per-rank sample-order hashes,
once-only sample coverage, and parent/reload numerical checks all pass.
Both arms retain 23 full checkpoints, including optimizer/scheduler and all 8
rank RNG states. Local recomputation exactly reproduces the node summary.

The queue took 2h46m including numerical checks, smoke tests and setup. CPU data
preparation took~2h48 beforehand; see the user correction in PROJECT_NOTES.md.
Future larger training should reconsider that diagnostic-specific filtering.

Communicating GPU burns restored and verified **00:39:52 SGT**. Independent live
inspection at **00:41:31 SGT** confirms 8 burn workers, released guard and advancing
collectives through 130 cycles / 144.35 GiB logical payload per rank.

Local evidence: [summary](results/kv_long_20261027/final/supervised/run/summary.json),
[A result](results/kv_long_20261027/final/supervised/run/A/result.json),
[B result](results/kv_long_20261027/final/supervised/run/B/result.json),
[checkpoint receipts](results/kv_long_20261027/final/checkpoint-receipts.json),
[artifact hashes](results/kv_long_20261027/final/artifact-receipts.json).
All 18 exported artifacts verified. Raw local receipt `temp/kv_long_health03.log`,
SHA256 `a32aa5a6b72563ea3e04b972f245abbe461bf26aaf9455550f826777eeb97e51`.
Large checkpoints remain under the B200 run
`/mnt/local/_outputs/deep-llms_th2/6y3o-efficient-kv-long-20261027-a01`.


Downstream follow-up is complete: [nine-task results](KV_DOWNSTREAM_20261028.md).
Eight-task MC average A60.60% versus uniform25 58.43%; LAMBADA accuracy52.28%
versus50.26%. The5%PPL screen did not establish downstream parity.
