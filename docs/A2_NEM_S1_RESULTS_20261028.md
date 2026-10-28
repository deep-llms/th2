# A2 NEM S1 results — 28 October 2026 SGT

All ten fits completed at **13:56:15 SGT**, in **78.22 minutes**. Each parent
selected LR **3e-4** from its own three-point tuning grid. No divergence/retry.
Every budget starts from the original parent with a fresh optimizer/schedule;
S1 removes groups 8/9/10 (layers 16–21). This is pure KD, not continued grid weights.

| Budget | P300 NLL | P1T NLL | P300 damage | P1T damage | P1T−P300 damage |
|---|---:|---:|---:|---:|---:|
| 0.03B | 3.111010 | 3.089194 | 0.336759 | 0.365811 | +0.029051 |
| 0.05B | 3.061876 | 3.030614 | 0.287625 | 0.307231 | +0.019606 |
| 0.2B | 2.956839 | 2.913690 | 0.182588 | 0.190307 | +0.007719 |

Original parent NLL: P300 **2.774251**, P1T **2.723383**. At 0.2B, PPL is
**19.2371 / 18.4247**, respectively (exp of existing final NLL, no extra evaluation).
P1T retains lower absolute NLL but has slightly greater damage relative to its
own unpruned parent. This S1-only result is not a full-pilot GO/STOP conclusion;
S2/S3/S4 and longer-budget/noise measurements remain required.

Verified: all ten run recipes, full LR schedules, exact order/coverage on eight
ranks, original parent baselines, zero save/reload difference and 78 source hashes.
Raw receipt: `temp/a2_nem_health04.log`, SHA256
`6d9a52a2b3dfd3d432ee33907165532191844633a4005a045e8db38787937761`.
Compact metrics: `docs/results/a2_nem_s1_20261028/status_1356/`.
At that snapshot training completed and eight burn ranks were ready, but the
terminal supervisor/progress receipt was still pending. Do not treat startup
alone as completed GPU handoff; a separate fresh health request is submitted.

Final handoff independently verified at **14:01:32 SGT**: supervisor exit0,
guard released, eight communicating burn workers, cycles290→300 and collective
payload322.00→333.11GiB. Burns restored13:57:26. Final rawreceipt
`temp/a2_nem_health05.log`, SHA256
`459875ab2aa3b9a0b3214a85176e8dd084025ced8ac021d498073fc32db58db0`.
Full small-artifact archive: `docs/results/a2_nem_s1_20261028/final/`.
