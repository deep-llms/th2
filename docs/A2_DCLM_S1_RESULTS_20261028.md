# A2 DCLM S1 pair — 28 October 2026

Completed10:53:01SGT in60.94minutes. Seven new fits; three earlier P300 grid fits
reused. Both parents selected LR3e-4 by tuning loss only. No divergence or retries.
The S1 student removes layers16–21 from24; pure KD from its own original parent.
Each budget is independently initialized and has a fresh full optimizer schedule.

| Budget | P300 final loss | P1T final loss | P300 damage | P1T damage | P1T−P300 damage |
|---|---:|---:|---:|---:|---:|
| 0.03B | 3.095975 | 3.070728 | 0.358182 | 0.383819 | +0.025637 |
| 0.05B | 3.045107 | 3.013827 | 0.307314 | 0.326918 | +0.019604 |
| 0.2B | 2.935242 | 2.895235 | 0.197449 | 0.208326 | +0.010877 |

Damage is final loss minus that arm's unpruned teacher loss. Original parents:
P3002.737793, P1T2.686909. Unhealed S1 loss: P3003.763775, P1T4.004721.
P1T remains better in absolute healed loss, but its remaining pruning damage at
0.2B is0.010877nats larger. The large unhealed damage gap shrinks substantially
with healing. This is one dataset and one removal pattern, not the preregistered
aggregate: no full-pilot GO/STOP or causal training-length conclusion yet.

Every fit passed exact eight-rank coverage/order, all-update LR verification,
expected tokens/steps, and zero saved-BF16 reload loss difference.78frozen source
files match a89c08e. Native supervisor exit0, communicating burns verified10:54:11
and independently healthy10:59:12SGT (workers644045–644052, guard released;
collective progress360→370cycles /399.73→410.83GiB per rank). Reverify identities
before the next authorized reclaim.

Raw receipt: temp/a2_pair_health02.log; SHA256
4479ea899ea17e10a289377d50478fdadcc985f02288b6b58c21a1efecdafbee.
Local metrics and full SHA/health receipt:
[results](results/a2_s1_pair_20261028/final/results.json),
[health](results/a2_s1_pair_20261028/final/health_1059.json).
Prior P300 grid: results/a2_early_grid_20261028/results.json.
Checkpoints remain under the original B2006y3o-a2-s1-pair-20261028-a01 root.

Next registered subset: FineWeb-Edu P300/P1T S1, each with its own3-point grid,
190-step short and762-step main heal. Same CulturaX healing/evaluation text,
different pretrained parent corpus. No new architecture/data/runtime change.
