# Exposure recovery: retain optimizer state after mixture exposure

Status: a01 CPU rejected insufficient unused rows before any GPU work. Recovery a02 implements a shared authenticated post-2B heal slice; local and frozen tests must pass before fresh CPU admission. Predecessor: `results/soft_exploration_20261119/REPORT_FINAL.md`, verified compact export (116 members, 84 replay checks). Original objective: `docs/warmup_DOT_idea_sketch.md`; user authorizes continued controlled experiments after negative screens.

## Evidence and question

The registered exposure schedule failed its quality criteria. SF soft loses to matched dense C by 0.003243 nat on DCLM and 0.003493 on FWE. Hard projection also loses. Noncanonical route diversity alone is not a useful result. Do not describe this follow-up as confirming a winner.

The final 500 updates were floor-free, but body LR had already decayed from about 2.45e-5 to 1e-5. Paired training-batch losses show some recovery after exposure, with noisy late behavior: SF minus C in the last four logged batches is about 0.00338/0.00398 for seeds 0/1. These are training observations, not a held-out convergence test. The next question is whether a fixed additional recovery period removes the exposure penalty, while equally continued controls prevent crediting ordinary additional training to routing.

## Registered continuation

Continue all five parents C/G/S/GF/SF, seeds 0/1: ten fits, each 2,000 additional updates of batch 32, length 512. Keep the architecture and soft objective unchanged. Use one common new 64,000-row order per seed, disjoint from earlier matched continuation rows and every router/evaluation/diagnostic reservation. Authenticate the row reservation and corpus identities before launch; reject insufficient capacity rather than silently reuse rows.

Start from each parent's terminal full state at update 2,000. Retain weights, Adam moments and counters, and Python/NumPy/Torch/CUDA RNG states. This is an explicitly separate recovery phase in a fresh output, not a mutation of the completed experiment's recipe. Use constant LR equal to each parent's actual terminal LR: 1e-5 for body and 1e-4 for router. No restart warmup, optimizer reset, floor, temperature adjustment or new loss. Preserve parameter-group order. Local resume within this phase restores its own full state and row position exactly. Count parent and new-phase compute separately.

Evaluate all terminal soft and hard models on both existing exploratory domains. Require SF to improve over equally continued C, GF and S by at least 0.005 nat for every seed and domain before a soft quality claim. A discrete-order claim additionally requires the analogous hard result and input-dependent noncanonical paths. Report all ten fits and paired document CIs regardless of sign. Do not select a stopping point or LR using final sets. Reused-final/multiple-comparison, two-seed, conditional-suffix and unequal-compute limitations remain.

## Implementation and gates

Reuse existing model, data and independent-reference evaluation code. A new authenticated phase loader must verify the full-state hash, terminal metadata, model/seed/arm/step, parameter schema and parameter-group settings before restoring optimizer/RNG. Parent numerical outputs must be exactly reproduced at phase entry; compare to the actual parent, not to dense initialization. Check both original soft-route and exposure checkpoints; the latter alone have an exploration-update buffer. Disable exposure explicitly and verify the zero-floor behavior, rather than restarting its schedule.

Before GPU launch require local and frozen-entrypoint CPU tests for both schemas, exact interrupted/uninterrupted optimizer/RNG continuation, row disjointness, ten-fit queue/report contracts, and export tamper rejection. Node CPU admission must load all ten actual parents. Keep production numerical, same-shape causality, first-40-step resume, fullset independent NLL/route parity, source/data/weight hash, ownership and native-handoff checks. Source-pack cap and fresh ownership must pass. Do not relax failed gates.

Weights/full states remain on node. Export only compact results and proof. Expected backup is twenty new weights/full states plus one report archive; verify actual hashes. Keep predecessor outputs unchanged and watcher 9736 alive through report/replay/backup and reassessment. No old DOT-RL resume.

## a02 data-capacity amendment (before any recovery GPU launch)

CPU a01 passed six node tests but failed reservation with `Insufficient new rows`; the original output is retained. Verified Horizon pool has 2,645,376 eligible rows, of which each parent used 2,560,000, and the matched continuation consumed another 64,000. This leaves at most 21,376, before additional reservations, below 64,000 required. This is a project data-capacity failure, not infrastructure.

Use fresh output `exposure_recovery_20261119_a02`. All ten recovery fits instead share a newly extracted DCLM heal input from original source tokens [2,000,000,000, 2,199,999,488). The restored 30B source already exists; fully verify its original token/boundary hashes and manifest on CPU. The old DOT E2 corpus is a permutation of exactly the first 2B source tokens. Authenticate that provenance and all prior reservation indices. Skip the document crossing the 2B cut; reject every 512-token row with an internal document start. Select 128,000 unique eligible source windows using fixed seed 19220 and split into two disjoint 64,000-row seed orders. Set row-start boundary flags as in original E2 packing. Save exact source offsets and hashes; never overwrite old data.

This preserves source-document-instance and token disjointness from all previous 2B training/router windows, and the heal/final text-hash partition. It is not a claim of corpus-wide text deduplication. The chronological source slice is a data change shared by every control and must remain a reported limitation. Optimizer, LR, steps, objective, gates and quality criteria are unchanged. Derived data stays on node; small provenance reports only. CPU a02 must pass boundary-straddling/selection and full ten-fit extension-pipeline tests before GPU admission.
