# Stage1b FineWeb-Edu — report in progress

Status: Part1 training complete; Part2 running, verified at19:31:23SGT: X72 3.2B8170/12207 updates (66.9%), eight GPUs99–100%. Part2 actual GPU startup was verified at17:19:54SGT; final downstream results are pending. This is not a full completion report.

Part1 common/own-best LR is1e-3. CulturaX final NLL: P3003.2B2.955812352510089; P1T3.2B2.9273495106589977. E(3.2B)=+0.023461150403907727, `persists`, with DCLM/FWE length finding `replicated`. A_init and0.8B findings await Part2. Full validated tables and preregistered caveats are in [REPORT_interim.md](REPORT_interim.md).

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
