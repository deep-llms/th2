2026-11-13T02:53:31.393114+08:00: Submitting unique authorized v1.5 E1b launch a01, source f9da3512ba6443fa07282201f239080a58340b63. CPU tests15+6PASS; native jobs schema and shell syntaxPASS; code26184971bytes<25MiB. Prior pending export resolved; informational probea87d833 freshly retrieved but ACK absent, no mutation and no duplicate probe. Launcher performs fresh exact identity/hash/native checks before GPU handoff. Actual node admission/tests/train not yet observed. Retrieve launch receipt before further writes. Watch3388692 alive; actual first delivery still unconfirmed.

# Current task — DOT-pool v1.5 E1b

Updated 2026-11-13T02:51:32.663725+08:00. User explicitly authorized docs/DOT_pool_pilot_spec_v1.5.md and requires a live watcher. Spec SHA256 `41aea5e294682e4ee703cff1df4e95a2de84662b7eba880eb862b08067ff2065`.

## Current state

Implementation prepared locally, NOT submitted/NOT running on GPU yet. Eight fresh E1b D/L/SO/DOT × seeds0/1, no stem/coda,8 steps,one GPU/run. T1/T2/T7/T9 production checks queued before training. Train first200epochs then preserve fullstate and continue to common N: largest500multiple fitting11h for both DOT seeds from first200 speed and actual elapsed transition, minimum2000 with flag. AdamW/LR.001/batch4096 and exact existing KG/splits. Checkpoint500epochs,last2+final. No E2/pilot reruns.

Local CPU verification:15 core/new/schedule tests PASS;6 new tests also PASS with explicit v1.5 protocol. Actual trainer uninterrupted vs split-resume weights exactly equal, retention/fullstate tests PASS, report fixture/replay/duplicate refusal PASS. Tests logs `temp/dot_pool_v15_watch_20261113/cpu_tests_full.log` and `cpu_tests_v15.log`. GPU tests NOT yet observed. Source and environment unchanged outside small DOT extensions; native supervisor untouched.

Old pilot R4b computed locally into `results/dot_pool_20261112/v15_preparation_20261113`, provisionally from regenerated KG arrays. Whole local KG manifest hash differs (metadata/archive byte identity not yet resolved); DO NOT claim exact original manifest. Launcher explicitly compares OOD-test/bridge npy actual hashes against local recreation and verifies every original node KG file against original ready/manifest before any GPU action. Canonical R4b is recomputed CPU on node original data; original pilot outputs immutable.

## Watcher

Live PID3388692, state `temp/dot_pool_v15_watch_20261113`, thread `01a120fe-ea0e-7030-a337-dd3f52f3c625`, interval1800s. First message queued18:42:41UTC12Nov (02:42:41SGT13Nov), id01a1d111-b10f-7a81-9e5b-03fee2b0c586; active-writer warning is expected while this turn is running, queue accepted but actual delivery NOT yet observed. Next03:12:39SGT. Deadline20Nov02:42SGT, renew if work unfinished. Old v1.2 watcher stopped and preserved. Keep this watcher through report/replay/backup/native handoff, record actual deliveries, no duplicate watcher/subagent.

## Remote and next action

Authorized main remote `origin` git@github-share:deep-llms/th2.git, wc69 `thiennh-p6-wc69-worker-0`,8B200, Python `/mnt/local/conda-py311/envs/dot_pool/bin/python3.11`. Existing data `/mnt/local/_outputs/deep-llms_th2/dot_pool_20261112/data`. New fresh planned execution-e1b-v15-a01 and run-e1b-v15-a01 under same base. Launcher `scripts/launch_dot_pool_e1b.py`; pin `resources/dot_pool_e1b_v15_admission.json` must regenerate after source edits before push. Admission validates latest native burn PID/start/commands/GPU UUIDs, data, source, dependencies; unchanged supervisor rechecks before reclaim and auto-restores native burn after completion.

Fresh controller retrieval before work: `temp/dot_pool_v15_watch_20261113/controller_before_v15.log`, SHA c3c04a5b42e3315da53d4234bf224b4aa68b85de0ec5f83b6d668eb648ba7e92. Prior export4c0c375 resolved. Informational #3 probea87d833 still not acknowledged; no pending mutation or export, do not resubmit. No new infra error observed. Retrieve receipts before writes; stale share is not job failure. Only small Tier1 exports, weights/fullstate remain node-only. Commands currently#0. Next review/test/pin/code-size checks then one unique authorized E1b launch; retrieve receipt, verify production tests/first200 and N. No duplicate/relaunch.

## Completed predecessor

DOT v1.2 20train/10downstream/20diagnostics +2 SR diagnostic corrections complete/localhash/replay verified; report `results/dot_pool_20261112/final_recovery_a01/REPORT_WITH_SR_CORRECTION.md`, SHA5a101183713ca3db3d22b93998e7ef3c2802135b394b0567847f94971669ed3d. Final proof FINAL_VERIFICATION.json; ledger1081. Native handoff verified12Nov22:55SGT and snapshot13Nov00:22SGT:8 burnworkers190953–190960 under190884, source sr-recheck-a01/source/resources/llm_pretrain_burn.py. Old timeoverrun/E1inconclusive/node-onlyweights flags retained. A2 experiments complete; no A2 work authorized now.

Full previous task ledger preserved losslessly at `docs/history/CURRENT_TASK_20261113_before_e1b.md.gz` (uncompressed SHA256 f6a10663f1576e529d88a5b2f6afa7cc8a37d6015abcb6d91d9f1facc9e246c2); gzip verified. No scientific artifacts removed.
