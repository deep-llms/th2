# A2 branch surveys and S2 prerequisites — 28 October 2026

All six branch surveys finished successfully. The first prerequisite stage then failed in CPU metadata assembly; no healing ran. The survey outputs remain valid and are reused by exact SHA.

| Dataset | Parent | Final NLL | PPL | FP32/AMP gap |
|---|---|---:|---:|---:|
| dclm | S72 | 2.908575 | 18.3307 | 0.000115 |
| dclm | S188 | 2.886707 | 17.9341 | 0.000093 |
| fwe | S72 | 3.042152 | 20.9503 | 0.000078 |
| fwe | S188 | 3.023808 | 20.5695 | 0.000100 |
| nem | S72 | 2.952464 | 19.1531 | 0.000092 |
| nem | S188 | 2.926709 | 18.6661 | 0.000070 |

All six pass T1 precision and native forward/RoPE sanity. These are unpruned branch-parent measurements, not healed results.

BI selection uses all four eligible parents per dataset, with the preregistered rules:

| Dataset | Selected S2 groups |
|---|---|
| dclm | [1, 2, 3] |
| fwe | [1, 2, 3] |
| nem | [1, 2, 5] |

The CPU replay and GPU stage selected these groups; the full-pattern GPU numerical gate passed at15:00:30 SGT. No GO/STOP conclusion follows from these measurements alone.

The initial error was missing `parent_sanity` metadata in the exact original DCLM/P300 survey. Its precision evidence is preserved; parent sanity is independently verified in the pinned completed baseline. The compatibility fix accepts only that original SHA. A subsequent retry was refused before GPU acquisition because the inherited handoff helper required a successful prior job. The new explicit failed-handoff lock verifies this failed CPU job’s terminal cleanup and restored burns; live ownership checks remain unchanged.

At the independent 14:37:57 SGT snapshot, all eight communicating burns were running, guard released, and cycles advanced 80 → 90. The a03 recovery only freezes S2 and runs the full-pattern numerical check. All six surveys and 30 completed S1 fits are reused.

Local evidence: [verification](results/a2_branches_20261028/final_surveys/verification.json), [original failed run](results/a2_branches_20261028/final_surveys/run.json), [supervisor](results/a2_branches_20261028/final_surveys/supervisor.json), and the six survey JSON files in the same directory. Raw downloaded health: `temp/a2_branch_health03.log`, SHA `ce97553cda2f46cf51daadefca6a33c2561ffbb8c8d41aac4d0770b9be2475f9`.

After successful numerical and burn checks, 28 Phase0A fits remain. With the original measured 1.199 seconds/update and load/evaluation/save allowance, estimated remaining baselines + fits are 6.62 node-hours; the timeout envelope including cleanup is 11.00 hours. Prior conservative ledger is 5.00 hours plus the new prerequisite stage. Retry costs are charged to the 18-hour phase cap.


Recovery a03 completed successfully at15:00:30 SGT. The numerical check took
47.23seconds (earlier10–20min estimate was conservative); total jobs49.19seconds.
The15:03:12 snapshot verifies supervisor exit0, released guard, eightburnworkers,
and collective progress120→130. All86frozen files matchbb72766. Originalprune,
reload and numerical receipts are in results/a2_branches_20261028/final_check.
Actual fullgate assembly verifies allsixannealed parents and unchanged numerical
locks. Next ledger18600seconds includes the recovery and handoff reserve.
