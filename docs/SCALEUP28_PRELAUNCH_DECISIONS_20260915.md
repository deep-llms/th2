# Decision request: three pre-launch choices for the 28L / 12B scale-up

Date: 2026-09-15.
Addressed to: the owner of `ccm_28layer_scaleup_plan_v3_12B.md` (no code knowledge required).
Prepared by: implementation review of the plan's local implementation.

The v3 plan is implemented and locally tested (135 CPU tests pass). A code-level
review found no correctness deviations from the plan, but three items need a
protocol-owner decision **before launch**. Every implementation fact you need is
stated below; it was verified directly against the code and tests on 2026-09-15.
Treat the "Facts" blocks as given.

None of these decisions change hypotheses, budgets, arms, capacity, thresholds,
data roles, or the reader/memory mechanism. Answering this memo does **not**
authorize a launch; deployment still goes through its own reviewed handoff.

Please return explicit numbered answers. Per the plan's own freeze rule, each
answer will be recorded as an explicit scale-up decision (in
`SCALEUP28_IMPLEMENTATION.md` / `PROJECT_NOTES.md`) before any 28L result exists.

---

## Decision 1 — Length of the pre-training stability smoke (plan §9)

### Facts

- Plan §9 requires "a short 28-layer stability check" before the full 10B
  common run, and permits the 2e-4 LR fallback only on **objective numerical
  failure at 3e-4** (non-finite loss/gradients/optimizer state).
- The common schedule warms up over **763 updates** to peak LR 3e-4.
- Implementation fact: the generated queue's default smoke is **32 updates**,
  a strict prefix of the real 38,147-update schedule. At update 32 the LR is
  about **1.3e-5** (32/763 of peak). The default smoke therefore never trains
  at or near 3e-4; it validates the pipeline, data, and precision environment,
  not the peak-LR stability question the fallback rule is written around.
- The smoke's checkpoint is discarded either way; the full run restarts from
  fresh initialization. The smoke verifies optimizer-state finiteness at its
  endpoint and records a checksummed pass/fail certificate that the full run
  requires.
- The implementation explicitly allows "a longer predetermined smoke chosen
  before training, without shortening the scientific schedule."
- Rough cost (scaling extrapolation, not a measured 28L throughput): the 12L
  4B common run took ~3.4 h on eight B200s; a 1,000-update 28L smoke is
  ~1/38 of the 28L common run, i.e. on the order of half an hour of node time.

### Question

Choose the predetermined smoke length N, recorded before training:

- **(a)** Keep N = 32. Accepts that 3e-4 stability is first tested by the real
  run itself; a mid-run non-finite failure would then trigger the documented
  fallback (discard, restart at 2e-4) at the cost of the wall time consumed.
- **(b)** Set N past warmup at peak LR, e.g. **N in 900–1,500**, so the smoke
  actually exercises 3e-4 before the 10B commitment.
- **(c)** Both: keep the 32-update run as a fast pipeline check, then a longer
  predetermined smoke before the full run.

Reviewer recommendation: **(b) or (c)** with N ≈ 1,000–1,200 (≥ 763 + a few
hundred peak-LR updates).

### Related sub-decision (same §9 territory)

Plan §9 cites "the existing training safety monitor triggers a divergence
abort" as an example of objective failure. Implementation fact: **no such
monitor exists**; only non-finite loss/gradient/optimizer-state detection is
implemented. A finite-but-diverging run would consume its full budget.

- **(1b-i)** Accept non-finite-only detection as the complete objective-
  instability definition and record that reading of §9. (Reviewer
  recommendation: this; non-finite failure almost always follows true
  divergence, and any finite-loss threshold would itself need preregistration.)
- **(1b-ii)** Require a preregistered finite-divergence abort rule (state the
  exact threshold) before launch; this is a code change and a new gate.

---

## Decision 2 — When may `D_val_28` be consumed; is a seed-17-only final lock legitimate? (plan §19 vs §23)

### Facts

- Plan §19: after all decisions "for a given replication panel" are frozen,
  evaluate the required models **once** on `D_val_28`.
- Plan §23 sequences: (11) if both primary CI criteria pass, start replication
  seeds → (12) freeze the 28L panel → (13) evaluate once on `D_val_28`.
- Implementation fact: the final-lock tool accepts **either** a seed-17-only
  six-arm panel **or** the full {17, 29, 43} panel. Locking is a manual,
  explicitly confirmed action; no automatic queue ever generates a final
  evaluation.
- Implementation fact: each named checkpoint gets exactly one locked heldout
  evaluation. Evaluating seed 17 early would **not** technically block later
  seeds' evaluations — but the seed-17 `D_val_28` results would then be known
  while replication seeds are trained and while any replication-scope choices
  are made.
- The 12-layer precedent reserved its locked set until the panel was final;
  the 12L `D_val` remains untouched to date.

### Question

Pick the rule now, before any 28L Stage-2 result exists:

- **(a)** `D_val_28` is evaluated only after the **full replication panel** is
  frozen (§23 reading). A seed-17-only lock is permitted **only** if the dev
  decision is `diagnose`/`ambiguous` and the study ends there, i.e. solely for
  honest final reporting of a one-backbone result.
- **(b)** Permit the seed-17 lock and `D_val_28` evaluation immediately after
  the dev decision in every case, accepting that heldout results are known
  during replication.
- **(c)** Another rule (state it exactly).

Reviewer recommendation: **(a)**.

---

## Decision 3 — Checkpoint save interval, retention, and restart strategy (plan is silent)

### Facts

- Checkpoints contain the bf16 model **plus fp32 master weights and both Adam
  moment tensors**. Approximate sizes at 28L (≈ 596M backbone parameters):
  **≈ 8–9 GB** per common / non-Grad Stage-2 checkpoint; **≈ 12–13 GB** for the
  Grad arm (its 268M-parameter table adds master + moments).
- Implementation fact: the save interval defaults to **every 1,000 updates
  plus the final update**, and the generated queues do not override it.
  Consequences if left unchanged:
  - common run: ~39 checkpoints ≈ **340 GB**;
  - each Stage-2 arm: 8 checkpoints (≈ 67 GB non-Grad, ≈ 100 GB Grad),
    six arms ≈ **435 GB**;
  - Stage-1 arms save once each (≈ 50 GB total); compiled tables ≈ 6 GB;
  - panel total on the order of **0.8–1 TB**, roughly 5× the 12L run's
    footprint. Free disk on the node must be verified at launch.
- Implementation fact: **no resume path exists.** Optimizer snapshots are
  saved, but there is no implemented or tested entry point that continues a
  run from a checkpoint; the documented policy is fresh restart in a new
  output root. Intermediate checkpoints therefore **cannot** currently be used
  to resume — their only uses are post-hoc analysis (e.g. future diagnostics
  at intermediate training states) and any *future* implemented-and-tested
  resume feature.
- Loss-vs-tokens reporting does **not** need intermediate checkpoints: per-
  update training logs are always saved.
- Nothing is cleaned up automatically, by long-standing project policy;
  deletions require explicit authorization.
- Rough duration (scaling extrapolation, not a measured claim): the 12L 4B
  common run took ~3.4 h; the 28L 10B run is plausibly of order **~20 h** on
  the same eight GPUs. A mid-run failure without resume costs the entire run.

### Questions

**3a — save interval / retention.** Choose one:

- **(a)** Keep every-1,000 saves everywhere (~1 TB; maximal post-hoc
  flexibility).
- **(b)** Sparser saves, e.g. every 5,000 + final for the common run
  (~8 checkpoints, ≈ 70 GB) and every 2,000–4,000 + final for Stage-2 arms.
- **(c)** Keep dense saves but preregister a post-validation deletion policy
  for intermediates (deletion executed only under explicit authorization).

Reviewer recommendation: **(b)**, unless you foresee intermediate-state
diagnostics for the paper; only the final checkpoint per run is consumed by
the protocol itself.

**3b — restart strategy for the long common run.** Choose one:

- **(a)** Accept restart-from-scratch as the failure policy for the ~20 h
  common run (consistent with the 12L pilot, which never needed it).
- **(b)** Require an implemented **and tested** resume path before launching
  the 28L common run (new code, new tests, and an explicit decision about
  whether bitwise schedule/data-order continuation is required for the
  scientific contract, since a resumed run is not the preregistered
  single-process trajectory unless made exactly so).

Reviewer recommendation: **(a)** for the first attempt, reconsidering only
after an actual failure; (b) adds meaningful new surface area to a frozen
protocol.

---

## Recorded answers — 2026-09-15

The plan owner returned explicit answers on 2026-09-15. They are recorded here
and in `SCALEUP28_IMPLEMENTATION.md` before any 28L result exists.

1. **(c)** — a fixed 32-update pipeline smoke, then a **1,024-update**
   stability smoke (past the 763-update warmup, at peak LR). Implemented: the
   `common` queue now generates both smokes; the full run consumes the
   1,024-update stability certificate; the generator refuses a stability smoke
   that does not run past warmup.
   1b. **(i)** — non-finite loss/gradient/optimizer-state detection is recorded
   as the complete objective-instability definition for the §9 fallback rule.
   No finite-divergence monitor is added; §9's "training safety monitor"
   wording is read as referring to this non-finite detection.
2. **(a)** — `D_val_28` stays untouched until the full replication panel is
   frozen. A seed-17-only final lock/evaluation is permitted **only** if the
   study stops at seed 17 (dev decision `diagnose`/`ambiguous`), solely for
   honest one-backbone final reporting. Initially recorded as an operator
   rule; subsequently enforced in code the same day — a scientific seeds=[17]
   scale-up lock requires the explicit `--single-seed-terminal` flag and
   records the declaration in the lock.
3a. **(b)** — scale-up save intervals: common every 5,000 updates + final;
   Stage 2 every 2,000 + final; Stage 1 final-only. Smokes save only their
   endpoint. The 12L Shallow follow-up keeps the historical every-1,000
   convention of the arms it is compared against. Implemented in the queue
   generator and locked by tests.
3b. **(a)** — restart-from-scratch remains the failure policy for the 28L
   common run; no resume path is added before launch. Reconsider only after
   an actual failure.

## Not being asked

- No change to the two primary hypotheses, the Shallow secondary hypothesis,
  the decision rule, the 12B budget split, arms, capacity, seeds, or the
  bootstrap policy.
- No opinion is sought on B200 deployment mechanics (GPU handoff, burn
  observers, runner protocol); those gates are unchanged and separately owned.
- Findings already fixed or merely documentational (e.g. re-recording the
  tested source hash) are not decisions and are handled by the implementation
  side.
