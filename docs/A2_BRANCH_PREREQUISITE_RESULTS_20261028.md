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


At15:27SGT, the remaining Phase0A queue completed five of six all-pattern
baselines and was scoring NEM/P1T. Each completed parent's finalNLL matches its
original S1-stage baseline within1e-6. The five receipts, compact unhealed loss
table and SHA records are in results/a2_remaining_phase0a_20261028/status_1527.
No healing fit had finished at this snapshot. S2 pruning causes large initial
loss increases; recovery conclusions require the registered healing runs.


At15:56:41 SGT, allsixbaselines and the firstthree healing fits passed. Allthree
have190updates, 12,160sequences, exact8rankorder/fullLR schedule, zero saved-model
roundtrip difference, and no divergence/retry. DCLM/P300 short-budget results:

| Set | Final NLL | PPL |
|---|---:|---:|
| S1 (previous completed run) | 3.045107 | 21.0123 |
| S2 | 2.929940 | 18.7265 |
| S3 | 2.890030 | 17.9939 |
| S4 | 2.877159 | 17.7637 |

S2's large unhealed damage does not imply poor recovery: its post-healing loss
is lower than S1 here. This is only DCLM/P300 at0.05B; P1T/otherdatasets/mainbudgets
are still needed for the preregistered decision. No full-pilot GO/STOP is inferred.
Actualresults and SHAreceipts: results/a2_remaining_phase0a_20261028/status_1556.
S2–0.2B was active at logged660/762updates. Measured~1.19seconds/update with
~90seconds whole-fit loading/scoring/saving overhead gives21:00–21:40SGT ETA
(using90–180seconds overhead perremaining fit), without retries.


At16:57:33 SGT, 11/28 new healing fits are complete, all exact recipe/data order/
LR/roundtrip checks pass, with no divergence or retry. FWE/P300/S2–0.2B is active
at logged460/762 updates. Supervisor656246 still owns the guard; all86 frozen
files matchd3a8a0b; eightGPUs91–96% utilization. Currentqueue elapsed1h48min.

DCLM paired completed results (NLL; lower is better):

| Set/budget | P300 final NLL | P1T final NLL | P1T−P300 excess damage relative to each original parent |
|---|---:|---:|---:|
| S2 / 0.05B | 2.929940 | 2.900690 | +0.021634 |
| S3 / 0.05B | 2.890030 | 2.852705 | +0.013559 |
| S4 / 0.05B | 2.877159 | 2.849714 | +0.023439 |
| S2 / 0.2B | 2.849884 | 2.813267 | +0.014267 |

P1T retains lower absolute loss, but its degradation from the original parent
is slightly larger. DCLM's registered mean across S1/S2 main-budget contrasts
is +0.012572 nats; the other two datasets are incomplete, so no full-pilot
GO/STOP conclusion follows. FWE/P300 short S2/S3/S4 NLLs are
3.033330/2.997276/2.986001; matching P1T results are still pending.

With17original fits remaining including the active fit, 10,780 logged remaining
updates at measuredmedian1.190seconds/update plus90–180seconds/fit overhead,
ETA is21:00–21:25SGT (about4.0–4.4hours from16:57), without retries. Phaseledger
is18600seconds plus currentelapsed, about6.97nodehours so far, below15/18h caps.
Verified compact results: results/a2_remaining_phase0a_20261028/status_1657.
Raw local receipt: temp/a2_remaining_health05.log, SHA
ba807e563df5eac920ccb0e2e41a91e860573fdcd3b03c38944c3d2831a1d8c0.


At17:27:35 SGT, 15/28 new fits passed all recipe/order/LR/roundtrip checks, with
no divergence/retry. FWE/P1T/S2–0.2B is active, logged340/762updates. EightGPUs
95–97%, guard656246, 86frozen sourcefiles unchanged. Queueelapsed2h18min; cumulative
Phase0A ledger~7h28min (prior18600seconds included).

| New FWE result | Final NLL | Damage vs original parent |
|---|---:|---:|
| P300 / S2 / 0.2B | 2.949261 | +0.088629 |
| P1T / S2 / 0.05B | 3.002164 | +0.193456 |
| P1T / S3 / 0.05B | 2.959809 | +0.151102 |
| P1T / S4 / 0.05B | 2.959262 | +0.150554 |

FWE's short-budget S2/S3/S4 excessdamage contrasts P1T−P300 are
+0.020757/+0.014457/+0.025185nats. As with DCLM, the later parent still has lower
absolute NLL; these contrasts measure degradation from its own original loss.
The main-budget FWE S2 pair and NEM are pending; fullpilot remains INCOMPLETE.

13original fits remain includingactive, 9568loggedremainingupdates. Atmedian
1.1895seconds/update plus90–180seconds/fit overhead, remaining~3.5–3.8hours
from17:27; ETA21:00–21:20SGT including handoff allowance, without retries.
Archive results/a2_remaining_phase0a_20261028/status_1727; rawlocal
temp/a2_remaining_health06.log, SHA
8e57c2b81dda159afa05dec00e61c680000faf2e55017380a848dd96e10219f4.


At17:57:03 SGT, 19/28 new fits passed all recipe/order/LR/roundtrip checks, no
divergence/retry. NEM/P300/S2–0.2B is active, logged190/762updates; eightGPUs95–96%,
guard656246 and all86frozenfiles unchanged. Queueelapsed2h48min; Phase0A ledger
~7h58min including prior18600seconds.

| New result | Final NLL | Damage vs original parent |
|---|---:|---:|
| FWE / P1T / S2 / 0.2B | 2.914824 | +0.106117 |
| NEM / P300 / S2 / 0.05B | 2.932966 | +0.158715 |
| NEM / P300 / S3 / 0.05B | 2.927950 | +0.153699 |
| NEM / P300 / S4 / 0.05B | 2.928426 | +0.154175 |

FWE main S2 excessdamagecontrast is +0.017488nats, and its registered S1/S2
mean is +0.015041 (DCLM +0.012572). P1T retains lower absolute loss. NEM remains
incomplete, so no full-pilot decision is available. Ninefits remain, including
two longer0.8B NEM curves and both T14 controls. Remaining8386updates atmedian
1.1894seconds/update plus90–180seconds/fit overhead: ~3.0–3.2hours from17:57,
ETA21:00–21:15SGT including handoff allowance, without retries.
Archive results/a2_remaining_phase0a_20261028/status_1757; fullrawreceipt
temp/a2_remaining_health07.log, SHA
ae0a5a2cebdadfe6cb8d57bccb2a07df6ff7819fada8322e5bc4d2b6513e4fc8.

## 18:31 SGT check: 20/28 remaining fits complete

NEM/P300/S2 0.2B completed with final NLL **2.871260569** (own parent
2.774250937; damage +0.097009631). NEM/P1T counterpart is still pending;
no complete NEM contrast or full-pilot decision yet. Current NEM/P300/S1
0.8B logged 1,080/3,051 updates. All20 published fits pass exact recipe,
full LR schedule, eight-rank sample order/coverage, provenance and checkpoint
roundtrip checks; no divergence/retry. All6 pre-baselines remain validated.

Eight GPUs96%,45,882MiB each; guard owner656246 and86 frozen source hashes
match d3a8a0b. Queue elapsed3h22, cumulative Phase0A about8h31 including prior
18,600seconds. Estimated remaining2.5–2.75h, finish21:00–21:15SGT if no retry.
Artifacts: [compact18:31 archive](results/a2_remaining_phase0a_20261028/status_1831).
Full original receipt: temp/a2_remaining_health08.log, SHA256
3726bf541baf4c9933e9b1ec57e984c6495b1fa905243f42ed48031d1b61330d.
No extra GPU job or modification to running training.

## 18:56 SGT check: long NEM heal advancing

Still20/28 remaining fits complete, all6baselines verified. NEM/P300/S1 0.8B
advanced to2,400/3,051updates, from1,080 at18:31. Recent training KL~0.1323
is not final validation NLL; no new completed fit or scientific decision.
All20 receipts pass the unchanged LR/order/provenance/roundtrip checks, no retry.
Eight GPUs95–96%,45,882MiB each, supervisor656246/86frozen source hashes intact.
Elapsed3h47/current, totalphase~8h57. Remaining~2–2.3h; ETA21:00–21:15SGT.
[Compact archive](results/a2_remaining_phase0a_20261028/status_1856); rawlocal
temp/a2_remaining_health09.log SHA256
ca24a139e88251e26532491b3152a61e365d4f151224dbe93536245e012d2051.

## 19:26 SGT check: 24/28 remaining fits complete

| Newly completed NEM fit | Final NLL |
|---|---:|
| P300 S1, 0.8B | 2.884719484 |
| P300 S1, 0.05B, T14a identical repeat | 3.061869316 |
| P300 S1, 0.05B, T14b data shuffle | 3.059590097 |
| P1T S2, 0.05B | 2.901267885 |

P300 S1 0.8B PPL17.898546, NLL improves0.072119574 from0.2B2.956839059.
Both budgets started from the original parent with their own schedules; this
is not resume. Remaining parent-relative NLL damage is0.110468547. P1T long
counterpart is unfinished, so no paired long-budget conclusion yet.
T14a differs from original3.061875752 by−0.000006436; T14b by−0.002285655.
These are one repeat and one shuffle, not a confidence interval.

All24 completed receipts validate exact recipe/fullLR/eight-rank order/roundtrip,
with no divergence/retry. Snapshot is between fits: P1T/S2short ended19:26:23,
P1T/S3short worker group began19:26:31. GPUs temporarily0%/4MiB, not evidence
of failure or queue completion. Supervisor656246 retains guard;86frozen hashes
match. Fourfits remain: P1TS3/S4short,S2main,S1long. Queueelapsed4h17/phase9h27;
remaining~1.5–1.8h, ETA21:00–21:15SGT.
[Compact archive](results/a2_remaining_phase0a_20261028/status_1926); rawlocal
temp/a2_remaining_health10.log SHA256
8d3d0a11c1fb62d4eab23f6358907804cb7a8a6b532eb9de41382653e0eb8140.

## 19:56 SGT check: 27/28 remaining fits complete

NEM/P1T shortS3/S4 finalNLL2.884076917/2.896807783; mainS2 finalNLL
2.832700506. All27 results pass exactrecipe/fullLR/eight-rankorder/roundtrip,
no divergence/retry. Lastfit NEM/P1T/S1 0.8B logged110/3,051steps.
EightGPUs76–100%,45,882MiB,guard656246/86frozen source hashes verified.
Elapsed4h47/current,totalphase9h57. ETA21:00–21:10SGT, about one hour remaining.

All three datasets now have the required main S1/S2 .2B contrasts:

| Dataset | Mean excess damage, P1T minus P300 (nats) |
|---|---:|
| DCLM | +0.012571973 |
| FWE | +0.015041272 |
| NEM | +0.010013559 |
| Mean | **+0.012542268** |

All12short contrasts have positive sign. The registered **main statistic** now
satisfies its GO rule; this is not a final whole-pilot conclusion or authorization
to launch Phase0B. The NEM long-budget paired result and terminal artifact/burn
handoff remain pending. Positive means P1T loses slightly more relative to its
own parent after healing; P1T still has lower absolute NLL.
[Compact archive](results/a2_remaining_phase0a_20261028/status_1956); rawlocal
temp/a2_remaining_health11.log SHA256
fe0b525d744a8dcba1d6a4c4cd4eda7e19a7ecc37ae1f938a58d72df8d68ec70.

### 28 October 20:31 SGT — final long fit advancing

Read-only health `ee3243c` confirms the final NEM/P1T/S1 0.8B fit at
1,890/3,051 updates. All eight GPUs are active at 95–96%, using 45,882 MiB
each. Guard owner 656246 and all 86 frozen source files are unchanged.
27/28 new fits and all six baselines remain verified; no divergence/retry,
recipe/order/LR or save/reload failures. No new final validation result.

Main-rule E remains +0.012542268 nats with 12/12 positive short contrasts.
The main statistical rule returns GO, but the full phase and terminal handoff
are not complete. No Phase0B job was launched. ETA including validation and
burn handoff remains 21:00–21:10 SGT (29–39 minutes from this snapshot).
Elapsed queue ~5h22; conservative total phase ledger ~10h32.

Evidence: `results/a2_remaining_phase0a_20261028/status_2031/`; full local
receipt `temp/a2_remaining_health12.log`, SHA256
`25c7ad59c9164531bda219b2d9fc79d989fc47a5e8432f34651bcdd793a6e0b2`.
