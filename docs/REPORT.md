# Option 2 — deviation được người dùng phê duyệt, 31/10/2026 SGT

Người dùng đã chọn **Option 2 và yêu cầu launch ngay**. Batch 2 được quyết định bởi accuracy rule đã PASS: MC mean +0.004797435 điểm phần trăm, LAMBADA 0. Ngưỡng **0.1 nats/token chỉ áp dụng cho sáu gate đại diện**, cả sáu đã PASS. Trên full set, LL là **diagnostic**: max **0.10355663299560547 nats/token**, request **71485**; token mismatch **1309/1396626 = 0.0937258793693%**. Giữ nguyên artifact FAIL lịch sử; không đổi thành PASS. Việc mã cũ dùng full-set LL làm hard stop là deviation đã được điều chỉnh theo xác nhận này.

Continuation tái dùng P300-parent batch2, chấm năm model còn lại ở batch2, chạy paired bootstrap 10,000 seed0 và replay độc lập. Kiểm tra khoa học ghi PASS/FLAG và tiếp tục phần kế hoạch còn thực hiện được; không chờ trả lời hay tạo kết quả thay thế cho artifact thiếu. Native supervisor kiểm tra đúng quyền sở hữu GPU trước khi thu hồi burn. Trạng thái thực thi hoàn tất được tách khỏi trạng thái kết quả khoa học đầy đủ/đã xác minh.

File kết quả chung: [A2_STAGE1B_FWE_RESULTS_20261030.md](A2_STAGE1B_FWE_RESULTS_20261030.md). Chưa khẳng định scores/CI/replay của continuation đã hoàn tất.

Còn mở: launch/receipt continuation, năm model, paired CI, replay, final report và native handoff. Các ghi chú chờ lựa chọn phía dưới là lịch sử, đã được Option 2 thay thế.

---

# Stage1b FineWeb-Edu — report in progress

Status: All15 registered training fits are complete. Final downstream failed31Oct00:12:10SGT on the full P300 batch comparison: max0.10355663299560547 nats/token exceeds0.1. Six representative gates passed, and the accuracy rule accepts batch2; the full numerical failure is retained. Five other production scores and paired CI are absent. Native failed-job burn handoff verified00:33:07SGT. This is not a full completion report.

Part1 common/own-best LR is1e-3. CulturaX final NLL: P3003.2B2.955812352510089; P1T3.2B2.9273495106589977. E(3.2B)=+0.023461150403907727, `persists`, with DCLM/FWE length finding `replicated`. A_init(3.2B)=+0.01299365118963669 (`anneal first`); E(0.8B)=+0.02281977941126856; A_init(0.2B)=+0.07847837360805077 and A_init(0.8B)=+0.04245379935293281; A(3.2B)/A(0.8B)=0.30606568523151645. Full validated tables and preregistered caveats are in [REPORT_interim.md](REPORT_interim.md).

## Downstream deviation authorized on 30 October 2026

This records the user decision following the failed FineWeb-Edu P300 parent numerical gate. The final `REPORT.md` must incorporate this deviation together with actual comparison results; this document contains no new GPU result.

## Preserved failure

The original batch2 gate failed before production scoring at 30Oct14:04:31SGT: batch/production error **0.011164188385009766 nats/token**, exceeding the original 0.01 threshold. Manual-reference error was 0 and recorded request-level greedy mismatches were 0. The native supervisor restored communicating burns and released its guard. Raw receipt SHA256: `5005f261292a74d572dd099ed19fcc322147bf58ceb25d6b101dcf918c0abe1b`; archive: [downstream_part1_gate_failure](results/a2_stage1b_fwe_20261030/downstream_part1_gate_failure/verification.json). The original receipt is unchanged.

The DCLM P300 regression is independently PASS: all nine task accuracies match Stage1 exactly and maximum aligned per-request LL difference is zero. This checks a different checkpoint and does not substitute for the FWE comparison.

## Authorized replacement protocol

1. Complete the five registered Part2 heals first, in the original order. Downstream and other reported-only failures never block these heals.
2. After independent native terminal/burn handoff, evaluate all six FWE models in one screen: P300/P1T/S72 parents and P300/P1T/X72 3.2B students. Retain the exact 78,259 canonical requests, fixed nine tasks, zero-shot seed42, FP32 parameters/BF16 AMP, and paired bootstrap10,000 resamples seed0.
3. For every model, record the representative batch1/batch2 per-token check. Hard numerical limits are manual-reference error ≤0.00002 nats/token, batch difference ≤0.1 nats/token, and greedy token mismatch rate ≤0.1% of compared continuation tokens. The original 0.01 batch limit is diagnostic. Historical request-level exact-match booleans are not token mismatch counts; collect token argmax evidence explicitly.
4. Before other production scoring, score the entire FWE P300 parent request set at both batch1 and batch2. Accept batch2 if absolute differences in the eight-task MC mean and LAMBADA accuracy are each ≤0.1 percentage points. Individual MC tasks are reported, not separate pass thresholds. Reuse its batch2 scores for production on PASS. Otherwise use batch1 for all six models, reusing its batch1 scores, and explicitly record departure from Stage1's batch2 protocol.
5. Report each task's accuracy difference, prediction flips (including changes between two wrong MC choices), per-request LL max/mean differences, and whether differing requests are in padded pairs. Preserve all deciding-check scores and SHA locks. LAMBADA correctness flips and actual generated-token prediction flips must be distinguished; the historical score format alone stores only exact-match correctness.
6. Time limits are report-only under the subsequent 17:30 SGT user override. Continue the registered work and report overruns; retain numerical validity and ownership checks.

## Implementation and actual results

Part2 launch was submitted as `ff50d1bc26ea23b5564a61875ea8f51658965e71`, freezing training source `42f7820ce72d932ff3d75edc96e16538dbd85df5`. GPU startup was independently verified at30Oct17:19:54SGT: supervisor725565, eight free GPUs after exactknownburn723690–723697 reclaim; source126files42f7820 and27corehashmatch. Follow-up health at17:25:34SGT verified310/12207updates and eight GPUs99%; rawSHA6674ab36afc9c0ba3e96ec1a10f299aac00ca718c0d822efb5557a746e918a58. The new downstream gate and comparison helpers are local preparation only; final six-model orchestration, actual GPU gates, complete batch comparison and final confidence intervals remain pending. No BF16-root-cause conclusion is claimed from the original failure alone.


## Remaining evidence required

Actual Part2 terminal/source commit and five new fit receipts; independent supervisor/burn handoff; full78259 P300 batch1/batch2 decision; six-model production scores and every-model numerical checks; prediction flips and padded-pair LL diagnostics; paired confidence intervals; final compute ledger and cross-dataset E/A tables. Reported-only failures are recorded and never cancel registered heals.

30Oct2026 USER OVERRIDE (17:30 SGT turn): No hard runtime/deadline stop; time estimates and former20/24nodeh/31Oct04:06 deadline are REPORT-ONLY. Continue registered work correctly and report overruns/issues; repeat outstanding issues/questions at END OF EVERY REPLY, including watcher replies. Preserve numerical validity and ownership checks; never mark failed diagnostics as passed. New immutable authority resources/a2_stage1b_time_policy_20261030.json SHA e9cf0904bf92c8365b209ce940150cb33ce83c5987512242d3dcb2ef4d79a3ac. Existing Part2 source42f7820 remains frozen and healthy; its already-running supervisor timeout32893sec is still embedded (not live-patched). Existing watcher894175 likewise still has old04:06 process deadline; reconcile same-watch continuation before then if task remains unfinished, no duplicate watcher. Future final downstream launcher must use report-only ledger/no total-time termination. This latest instruction supersedes all older harddeadline/cutdownstream text below.

At 17:31:48 SGT, Part2 X72 3.2B was at 700/12,207 updates with all eight GPUs at 99% utilization; source `42f7820` and ten inherited fit receipts were unchanged. No new final validation is available. Evidence: `results/a2_stage1b_fwe_20261030/part2_1731`.

The final six-model orchestration is CPU-prepared: ten integration/admission tests and eight revised-gate tests passed, including score-preserving argmax capture on a pinned tiny native model. All 66 original regression scoring/model files remain byte-identical; six new adapters implement the authorized checks, comparison and final orchestration. Actual full-model GPU checks and full P300 batch comparison remain pending. Future downstream has no total elapsed-time termination under the latest user time policy; projected and actual usage will still be reported.

At 18:01:31 SGT, source126files42f7820, the ten inherited fit receipts and native ownership were verified unchanged. Raw health SHA256: `0610ffc9e6c2b7d1366d5474f9375d8736b5df7b51664c6036ea122d9142a828`; archive: `results/a2_stage1b_fwe_20261030/part2_1801`.

The existing watcher was continued serially in the same session, state, thread and pane under the exact time-policy SHA. Old PID894175 exited before PID1003079 acquired the watch; no overlapping watcher or GPU restart occurred. Its deadline is now null, with the next scheduled check at18:30SGT. The frozen Part2 process still contains its legacy job/per-fit timeouts and admission checks; these remain an open operational issue to resolve before they can interrupt unfinished registered work.

The independent final verifier is CPU-prepared (two artifact/gate verification tests passed); two watcher policy tests also passed. The verifier recomputes six-model scores/replays, paired intervals, full P300 batch decision and token diagnostics from SHA-bound artifacts. Actual final GPU evidence and a separate fresh native-burn handoff are still required; no final downstream success is claimed.

At 18:30:58 SGT, Part2 healtha05 confirmed continued X72 progress at about0.96016seconds/update, unchanged source/input/runtime/provenance and ten prior fit receipts. No new final validation. Evidence: `results/a2_stage1b_fwe_20261030/part2_1830`; rawSHA `0073055c5d155d3abfc0fc4539a1a2082c1d6cb910b7451acdbaa5660a7202d0`. X72 completion is provisionally20:40–20:50SGT; all heals23:30–00:00 and full downstream/report00:30–02:00. The local Part2 terminal verifier is syntax-checked only; actual15-fit terminal and fresh native-burn evidence remain required. Legacy loaded Part2 timeouts remain monitored, not imminent at this snapshot.

At19:01:20SGT, healtha06 independently verified continued X72 progress at about0.95907seconds/update, unchanged source/input/runtime/provenance and ten prior fits; no new final validation. Evidence: `results/a2_stage1b_fwe_20261030/part2_1901`, rawSHA `d5835e16272b4be88214180d4a2b027b39138e8668e1759f851eda8060ee10e6`. Remaining training is about94minutes plus validation/save; provisional X72 completion20:40–20:50SGT and overall00:30–02:00SGT. Legacy loaded timeouts remain an open issue, with no imminent boundary at this snapshot.

At19:31:23SGT, healtha07 independently verified X72 at8170/12207steps, about0.96446seconds/update, unchanged source/input/runtime/provenance and ten prior fit receipts. No new final validation. Evidence: `results/a2_stage1b_fwe_20261030/part2_1931`; rawSHA `a7ece95cd1a13aa9574f487e445c3977ecd137c85ff31db8d26947b0501a10f3`. Remaining training is about65minutes plus validation/save. X72ETA20:40–20:50SGT, full stage00:30–02:00SGT provisional. Legacy timeouts remain monitored, not imminent at this snapshot; downstream actual evidence remains pending.

At20:00:56SGT, healtha08 independently verified X72 at10020/12207steps, about0.95827seconds/update, unchanged source/input/runtime/provenance and ten prior fit receipts. No new final validation. Evidence: `results/a2_stage1b_fwe_20261030/part2_2000`; rawSHA `05a07193f23024ebcc64996e994c1aa3a08a15a914ed0b619947b726e887cf37`. Remaining training is about35minutes plus validation/save. X72ETA20:40–20:50SGT; full stage00:30–02:00SGT provisional. Legacy timeouts remain monitored, not imminent at this snapshot; downstream actual evidence remains pending.

At20:31:21SGT, healtha09 independently verified X72 at11920/12207steps, about0.96017seconds/update, unchanged source/input/runtime/provenance and ten prior fit receipts. No new final validation. Evidence: `results/a2_stage1b_fwe_20261030/part2_2031`; rawSHA `7ba47ccb190f2a23480ac12314dbd82b0a55d43a57567350bb2b00a8c94bff99`. Remaining training is about5minutes plus validation/save; the next health check must verify the X72 final result and transition to P3000.8B. Full stage00:30–02:00SGT remains provisional. Legacy timeouts remain monitored, not imminent; downstream actual evidence remains pending.

At30Oct21:01:08SGT, healtha10 verified X723.2B complete: finalNLL2.9688060036997257, tuningNLL2.9494744594565536, FWEheldout2.420236999085872; full LR schedule, exact8rank order, roundtrip0 and no divergence. A_init(3.2B)=X72−P300=+0.01299365118963669 nats, label `anneal first`; FWEheldout A=+0.013620394355744914 is diagnostic. This contrast does not isolate LR state. P3000.8B is active1480/3051 under the same supervisor725565; all ten prior fit receipts and source126files42f7820 are unchanged. Evidence: [receipt](results/a2_stage1b_fwe_20261030/part2_2101/receipt.json), [X72 summary](results/a2_stage1b_fwe_20261030/part2_2101/x72_3_2b_summary.json); rawSHA `fe1de27675a49a7287db250add3cf4dab14cfd07cd25c7e4b9926e1b2d8b12ae`. All heals23:30–00:00/full00:30–02:00SGT remains provisional. Four short fits, terminal/burn and downstream evidence remain pending. Legacy loaded timeouts remain monitored, not imminent.

At30Oct21:31:18SGT, healtha11 verified P3000.8B complete: CulturaX final2.9970963238272885, tuning2.9777841032682884, FWEheldout2.444843807379002, delta_post0.1364649979704735. Full LR schedule, exact8rank order, roundtrip0 and no divergence were verified; training2929.928263354115seconds. P1T0.8B is active210/3051 under supervisor725565 with eight GPUs98–99%; source126files42f7820, inputs/runtime/provenance and eleven prior fit receipts unchanged. Evidence: [receipt](results/a2_stage1b_fwe_20261030/part2_2131/receipt.json), [P300 summary](results/a2_stage1b_fwe_20261030/part2_2131/p300_0_8b_summary.json); rawSHA `dfcfd05ddf392d3eacc65e96829e36af3d4ace840a813c656e8558a7698bdab5`. P1T0.8B/X720.8B/X720.2B, terminal/burn and downstream remain pending; E/A at0.8B cannot yet be computed. ETA all heals23:30–00:00/full00:30–02:00SGT remains provisional. Legacy timeouts remain monitored, not imminent. Next read-only health collection raises the per-artifact read cap from5MB to10MB to accommodate the growing runs.json; training source is unchanged.

Healtha12, retrieved22:30SGT, independently verifies the22:01:42SGT snapshot: P1T0.8B2110/3051(69.2%), eight GPUs98–99%, unchanged source126files42f7820/inputs/runtime/provenance and twelve prior fit receipts; no new final validation. Evidence: `results/a2_stage1b_fwe_20261030/part2_2201`; rawSHA `104b51a351d20e5eea4f1966820ac21408091e84daba6d37b461fdd9c5fc1cdd`. Fresh healtha13 is needed to verify P1T completion and X720.8B transition.

At30Oct22:32:12SGT, healtha13 verified P1T0.8B complete: final2.967992110983558, tuning2.949299474694987, FWEheldout2.416288128300919, delta_post0.15928477738174207; full LR schedule, exact8rank order, roundtrip0 and no divergence. E(0.8B)=+0.02281977941126856; diagnostic FWEheldout E=+0.024077046610296193. E(3.2B)/E(0.8B)=1.028105924298394. X720.8B is active870/3051 under supervisor725565, eight GPUs98–99%; source126files42f7820, inputs/runtime/provenance and twelve prior fit receipts unchanged. Evidence: [receipt](results/a2_stage1b_fwe_20261030/part2_2232/receipt.json), [P1T/E summary](results/a2_stage1b_fwe_20261030/part2_2232/p1t_0_8b_summary.json); rawSHA `0a76f23d1bedd186cd9b39ac27dc242440a585e06582501c3880ffcbe2bf9500`. X720.8B/X720.2B, terminal/burn and downstream remain pending. All heals23:30–00:00/full00:30–02:00SGT remains provisional; legacy timeouts remain monitored, not imminent.

At30Oct23:01:19SGT, healtha14 independently verified X720.8B2690/3051(88.2%), about0.960149seconds/update, eight GPUs98–99%, unchanged source126files42f7820/inputs/runtime/provenance and thirteen prior fit receipts. No new final validation. Evidence: `results/a2_stage1b_fwe_20261030/part2_2301`; rawSHA `0115b32f5addca4975da6fb3e1625166ddeb7b0953c1dd3d4578db6603cc22d8`. Conservative elapsed ledger is19.619330102606146nodehours; report the20hour crossing at the next check and continue under the user override. Remaining training is about6minutes plus validation/save, then X720.2B. The next receipt must check both remaining fit results and, if terminal, independently verify native burn progress before final downstream. ETA all heals23:30–00:00/full00:30–02:00SGT remains provisional; legacy timeouts remain monitored, not imminent. A local bundle builder is syntax-checked and requires the actual independent15fit terminal verification before writing Part2 SHA locks; no terminal result is inferred.


30Oct23:31SGT: All15 training fits and native handoff independently verified; source42f7820 unchanged. Evidence: [part2_final](results/a2_stage1b_fwe_20261030/part2_final/verification.json). Native training ledger71801.2573sec (19.9448h); conservative controller ledger has crossed20h including gaps/handoff, reported before downstream. Legacy timeouts ended with Part2; final downstream has no elapsed-time termination.


## Validated cross-dataset training comparison (30Oct23:31SGT)

CulturaX final is the deciding endpoint. Differences below are FWE minus DCLM. Both datasets independently selected common LR1e-3 and own-best LR1e-3 for P300/P1T.

| Budget | DCLM E | FWE E | Difference E | DCLM A_init | FWE A_init | Difference A_init |
|---|---:|---:|---:|---:|---:|---:|
| 0.2B | 0.021096293348 | 0.026224221566 | +0.005127928219 | 0.072724703707 | 0.078478373608 | +0.005753669902 |
| 0.8B | 0.020786422564 | 0.022819779411 | +0.002033356848 | 0.038910028864 | 0.042453799353 | +0.003543770489 |
| 3.2B | 0.021670884570 | 0.023461150404 | +0.001790265834 | 0.012113131874 | 0.012993651190 | +0.000880519316 |

Ratios 3.2B/0.8B: DCLM E=1.042549986824, A_init=0.311311305279; FWE E=1.028105924298, A_init=0.306065685232. Both yield `persists` and `anneal first`; FWE length finding is `replicated`. These are registered single-seed recipe comparisons, not isolated LR-state or exposure effects. Own-corpus held-out remains diagnostic.

Actual Part1 source7dd9da0, Part2 source42f7820, inherited preflight provenance9f5601d with27core hashes unchanged. All15 fits use full registered LR schedules, exact8-rank coverage, roundtrip0 and no divergence. Final downstream sourcef0c8154 preserves all66 original regression eval/model files and adds only the six registered adapters. GPU validation is pending.


30Oct23:41:25SGT actual downstream startup verified: sourcef0c8154808bef431c76553a47fd9727d3ba8bd53/132files, original66 eval/model files byte-identical; six registered adapters. Native supervisor730334 reclaimed exact Part2burn729392–729399 with matching start identities, verified all8free, then started23:40:10. P300 revised gate PASS (manual0,batch0.011164188385009766,tokenmismatch0/502); original0.01 diagnostic failure retained. P1T gate running; full batch comparison and production scores pending. [Receipt](results/a2_stage1b_fwe_20261030/downstream_final_2341/receipt.json). Runtime cap/deadline null; priorledger72844.667764sec, forecast22.418nodeh before actual preparation.


## Actual six-model numerical gates, 31Oct00:01SGT

All six revised gates PASS; each compares502 continuation tokens with0 mismatches. Original0.01 threshold diagnostics remain failures for P300-parent, P1T-parent and X72-student. [Gate evidence](results/a2_stage1b_fwe_20261030/downstream_final_0001/receipt.json).

| Model | Manual nats/token | Batch nats/token | Token mismatch | Old0.01 diagnostic |
|---|---:|---:|---:|---|
| P300-parent | 0 | 0.011164188385 | 0/502 | FAIL |
| P1T-parent | 2.65370244565e-07 | 0.0292415618896 | 0/502 | FAIL |
| S72-parent | 0 | 0.00355625152588 | 0/502 | PASS |
| P300-student | 0 | 0 | 0/502 | PASS |
| P1T-student | 2.64221455628e-07 | 0 | 0/502 | PASS |
| X72-student | 0 | 0.0159463882446 | 0/502 | FAIL |

Full P300 batch2 produced78259 scores (SHAae05ee601822e3af5c24b70b1e6ea1c974b07fb9967ed72acc9743953adca3a3); batch1 remains active at snapshot00:01. No batch decision or final accuracy/CI is claimed. Source132filesf0c8154 and all six identity locks remain unchanged; ownership chain to supervisor730334 verified.


## Full P300 comparison failure — 31Oct00:33 SGT receipt

Both batches contain all78,259 scores. The accuracy rule accepts batch2: eight-task MC mean difference +0.004797435042615955 percentage points, LAMBADA0. Full token mismatches1,309/1,396,626 (0.09372587936927997%) meet0.1%. However, maximum per-request normalized LL difference0.10355663299560547 nats/token exceeds0.1, so the deployed full-comparison check failed and stopped before the five other production scores. This additional full-set enforcement is distinct from the six representative model gates, which all passed. No threshold or failed diagnostic has been changed. Independent failed-batch recomputation reproduced the result at31Oct00:39:22SGT (retrieved01:00SGT).

Raw LL max0.5551643371582031, mean absolute0.011516675116967773; differences also occur in unpadded pairs. MC prediction flips by task: HellaSwag7, ARC-easy5, ARC-challenge2, PIQA2, WinoGrande2, BoolQ3, OpenBookQA0, SciQ1. LAMBADA has1 token-prediction flip and0 correctness flips. [Full diagnostic and SHA locks](results/a2_stage1b_fwe_20261030/downstream_final_batch_failure/batch_decision.json).

Job exit1 at00:12:10; native supervisor restored eight burns and released its guard00:13:16. At00:33:07 the fresh native burn workers730982–730989 under730913 were verified, cycles1370→1380 and collective payload1521.18→1532.29GiB. [Failure and handoff receipt](results/a2_stage1b_fwe_20261030/downstream_final_batch_failure/receipt.json). Raw SHA256 `d2d4e9c420033532c7b8491305be387668f84a87b2c800c64ca317e3cd39286c`. All15 training results remain valid; full Stage1b downstream remains incomplete.

Conservative elapsed ledger:20.8000654871 node-hours at failed downstream terminal00:12:10,21.1492799282 at the00:33:07 verification; these include controller gaps and are report-only. No runtime timeout caused this failure. The open protocol question is whether to extend fallback to batch1 for this full-set LL exceedance, or confirm that the0.1 numerical threshold applies only to the representative gates while the full-set batch choice follows the stated MC/LAMBADA rule. No GPU retry is submitted while this question is unresolved.

Independent CPU recomputation at31Oct00:39:22SGT verified all132 frozen source files, all78,259 aligned scores, binary-item accuracies, token/prediction flips, full per-request LL/padding rows and the exact recorded decision. Only one request exceeds0.1 nats/token: index71485, SHA `9efed81f23070082278106d9c7cc6ed4fcb255319d08f48d9fb46c5cb9ccb99d`, context53 tokens, continuation1 token, unpadded pair71484/71485, difference+0.10355663299560547, no greedy exact-match flip. This confirms the failed diagnostic; it does not establish successful final downstream completion. [Independent receipt](results/a2_stage1b_fwe_20261030/downstream_final_batch_failure/independent_failed_batch_replay.json), raw SHA `aa54ba2393fc93c28aeaca73f0a13c0bd9bf2b0ed7bcdae6341752dd209ddb46`. Awaiting clarification of threshold scope/fallback; no GPU retry.
