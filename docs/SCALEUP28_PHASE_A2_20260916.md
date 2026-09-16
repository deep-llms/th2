# 28L/12B Phase-A2 recovery queue — 2026-09-16

Recovers the Phase-A queue after its external kill (2026-09-15 21:44:55 UTC:
seed-29 Shallow ranks SIGKILL'd by another party's mistaken node use; zero
cgroup OOM events; foreign `deepeyes` training and burn-style jobs then
occupied the GPUs). The user confirmed on 2026-09-16 that the node is ours and
explicitly authorized: stop all current GPU jobs, remove the interrupted
seed-29 outputs plus caches, and restart from seed-29 Shallow onward.

## What a2 does (one tmux owner, `scripts/pilot_scaleup28_phase_a2.py`)

1. Core/env preflight and the full pinned input gate, now also verifying the
   **completed seed-17 section** of the a01 root (gates + record hashes) —
   carried forward, never rerun or deleted.
2. **Authorized clear** of all current GPU jobs: enumerate GPU compute PIDs,
   require every one to match a python-workload allowlist (mp-spawn workers,
   `deepeyes`, the burn script; anything else aborts), record identities and
   the user authorization, SIGTERM parents then workers via identity-pinned
   pidfds, wait, SIGKILL only verified survivors, then double free checks.
   A new unexpected process appearing mid-clear aborts the queue.
3. **Guarded cleanup** (exactly the authorized targets, nothing else):
   `<a01>/shallow12_seed29` (refused if it ever looks completed or its
   run contract mismatches), `<a01>/shallow29_train.log` (evidence archived on
   the dev machine first), the partial data root
   `scaleup28_v3_20260915_a01` (refused if a manifest/complete marker exists;
   exact-path pinned), and any `cache-*`/`tmp-*` files under the a01 root.
4. Fresh `prepare-scaleup` into `scaleup28_v3_20260916_a01` (background CPU),
   the seed-29 Shallow follow-up with all gates and the Deep-vs-Shallow
   report, extension validation, the decided smoke pair (32 / 1,024 updates),
   the 38,147-update 10B common run, validation, burns, idle observer.

Output root: `ccm_scaleup28_phase_a2_20260916_a01`; burn labels
`..._a2_20260916_a01_b{n}`, ports 30801+. Commands for the 28L queue remain
test-enforced byte-equal to `ccm scaleup-jobs --queue common`.

Notes: this project runs under **torchrun**; no Accelerate configuration
exists or is copied (`resources/accelerate_config.example.yaml` is an unused
template and is not touched). The `ccm/` research core is unchanged
(`214d57c4...`); only scripts/tests were added.

## Still not launched by this queue

28L compile/Stage-1/Stage-2 panel, 12L locked `D_val` evaluations, replication
seeds, LR fallback — each needs separate authorization.
