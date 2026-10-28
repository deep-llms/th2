# A2 FineWeb-Edu S1 results — 28 October 2026

All ten fits completed at 12:27:02 SGT in 78.12 minutes, without divergence or
retry. Both parents selected LR 3e-4 using their own tuning grids. Same S1 removal
(layers16–21), pure KD and fresh original-parent initialization per budget as
DCLM; healing/evaluation text unchanged, pretrained parent corpus is FWE.

| Budget | P300 final loss | P1T final loss | P300 damage | P1T damage | P1T−P300 damage |
|---|---:|---:|---:|---:|---:|
| 0.03B | 3.193430 | 3.175876 | 0.332799 | 0.367169 | +0.034370 |
| 0.05B | 3.143627 | 3.115962 | 0.282996 | 0.307255 | +0.024259 |
| 0.2B | 3.036157 | 2.996828 | 0.175526 | 0.188120 | +0.012595 |

Damage is final loss minus its own original unpruned teacher loss.

| Parent | Original PPL | S1 healed 0.2B PPL |
|---|---:|---:|
| P300 | 17.4726 | 20.8251 |
| P1T | 16.5885 | 20.0219 |

PPL is exp(token-weighted final NLL), derived without additional evaluation.
These S1-only comparisons do not establish the preregistered aggregate decision;
S2 and the remaining dataset/pattern contrasts are still required.

All ten fits pass exact eight-rank data coverage/order, full LR schedule,
steps/tokens, matched parent baseline/provenance and zero BF16 save/reload loss
difference. All 78 source files verified against cfeb74c. Supervisor exit0 and
complete artifacts verified. Burns restored12:28:13SGT and independently live
12:31:49SGT, guard released; collective cycles260→270, payload288.69→299.79GiB/rank.

Full [results](results/a2_fwe_s1_20261028/final/results.json), compact
[summary](results/a2_fwe_s1_20261028/final/summary.json), and
[health/SHA receipt](results/a2_fwe_s1_20261028/final/health.json).
Raw local receipt temp/a2_fwe_health05.log SHA256
9fa109bdb9962be1553fa2e7fdc716f04464e67191505c4bff4db64e209b858f.
Extracted artifacts temp/a2_fwe_health05/. Checkpoints remain under B200 output
6y3o-a2-fwe-s1-20261028-a01. Next registered subset: NEM P300/P1T S1 pair.
