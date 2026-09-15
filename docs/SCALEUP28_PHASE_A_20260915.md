# 28L/12B Phase-A queue — 2026-09-15

Implements plan v3 §23.A / §26 as one guarded sequential remote handoff. The
user authorized launch on 2026-09-15 after the implementation review and the
recorded pre-launch decisions (`SCALEUP28_PRELAUNCH_DECISIONS_20260915.md`).

## Scope

One persistent tmux owner (`ccm_scaleup28_phase_a_20260915_a01`,
`scripts/pilot_scaleup28_phase_a.py`) performs, in order:

1. Core/environment preflight (code hash, Transformers 5.9.0) and the static
   input gate: pinned historical corpus/vocabulary/coverage, both common
   checkpoints (seeds 17/29), Shallow+Contextual table artifacts, historical
   Stage-2 Contextual evaluations, raw shards, Base assets, ≥250 GiB free.
2. Starts `prepare-scaleup` as a **background CPU child** (extension corpus
   `scaleup28_v3_20260915_a01`: val28 → continuation extension → common
   extension, historical prefix untouched).
3. Cooperatively disarms the seed29 queue's idle observer through its exact
   `STOP_IDLE_WATCH`, verifies its voluntary exit, then stops only verified
   original burn workers and passes two 30 s free checks.
4. 12L Stage-2 Shallow follow-ups, seed 17 then seed 29 (study
   `pilot12-shallow-followup`): 3,815 updates each from the seed's own common
   checkpoint with its own frozen Shallow table, full validation gates, dev
   evaluation with the seed's Contextual diagnostic table, and the
   Deep-vs-Shallow 10,000-replicate doc_id/independent comparison against the
   verified historical Contextual evaluation.
5. Waits for preparation, runs `validate-data` plus the extension binding gate
   (exact historical prefix, frozen vocabulary/mapping hashes, quotas).
6. The decided 28L common queue, byte-equal to
   `ccm scaleup-jobs --queue common` (tests enforce equality): 32-update
   pipeline smoke → 1,024-update stability smoke (past the 763-update warmup
   at peak LR 3e-4) → fresh full 38,147-update / 10B-token common training
   with `--save-every 5000` → `validate-run` plus contract/LR-schedule gates.
7. Restores original all-eight communicating burns, writes `complete.json`,
   and stays as the idle observer until its own `STOP_IDLE_WATCH`.

Output root: `/mnt/local/_outputs/deep-llms_th2/ccm_scaleup28_phase_a_20260915_a01`.
Data root: `/mnt/local/_data/deep-llms_th2/ccm/scaleup28_v3_20260915_a01`.

## Explicitly NOT in this queue

- 28L compilation, Stage-1, Stage-2 panel (needs `theta_10B_28L`; separate
  authorized launch, including the distributed-compiler B200 equivalence gate).
- 12L locked `D_val` evaluations (require frozen interpretation choices after
  the Shallow results are reviewed).
- Any 28L replication seed, seed 43, capacity changes, online writing.
- The 2e-4 LR fallback: it requires a checksummed objective-failure record and
  a regenerated queue; this queue runs 3e-4 only.

## Guards

- Fresh output/data roots; no automatic retry/resume/cleanup. A failed stage
  preserves everything; a partial extension has no manifest and is unusable.
- GPU reclaim only after the previous owner's verified completion record,
  fresh heartbeat, cooperative observer exit, and per-PID re-verified burn
  workers (`GPU_SAFETY.md`); on failure before reclaim, burns are untouched.
- Preparation is the controller's own CPU child; on queue failure it is
  terminated (own child only) and its partial output left for inspection.
- Every stage is followed by a fail-closed validator
  (`scripts/validate_scaleup28_phase_a.py`) writing `validated_*.json`;
  `complete.json` appears only after all gates and active burns.
- Rough expected wall time: preparation ~4–5 h in parallel with ~2.5 h of
  Shallow follow-ups, then smokes ~1 h, then full 10B common training
  (order of a day; no measured 28L throughput exists). Heartbeats each minute.

## After completion

Pull small results via `#2` exports (validated gates, metrics, reports,
stability records, train logs). Next authorized steps, separately: review
Shallow results → freeze 12L choices → 12L `lock-final`/`D_val`; and the 28L
panel queue against `theta_10B_28L`. Before any later GPU job, disarm THIS
queue's `STOP_IDLE_WATCH` and verify its observer exit per `GPU_SAFETY.md`.
