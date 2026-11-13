# DOT-RL implementation decisions (v1.2)

Authorized spec: `docs/DOT_RL_router_spec.md`, SHA256
`9c8bc06bf1c702f40e334fd23f7b0604b99661a65381e1f12b0883bb4efbe33b`.
The owner supplied the file, not a digest; this is the computed immutable digest.

The study extends `efficient_llm/dot_pool`, reusing Qwen3 blocks, initialization,
parameter groups, seed/RNG utilities and the existing verified supervisor. Pilot,
E1b and E2 outputs remain immutable. No internet calls from GPU jobs.

- New KG uses seed2026 and the original atomic edge generator. Permutation entries
  0:4000 are OOD-RL, 4000:6000 OOD-test, remaining34000 ID. Draw ID-dev/test before
  excluding them from244800 training compositions. All40000 atoms remain trained.
  Exact arrays and split-index hashes are pinned in `resources/dot_rl_data_expected.json`.
- Whole-question grouping uses SDPA, with no token-level layer routing. L's loop
  forward is tested exactly against the existing E1b model. Only the policy uses
  detached FP32 normalized answer-position states; its RMSNorm has no learned scale.
- CE backward occurs before rollout backward, but neither optimizer steps until
  both have used the same layer weights. RL rollout blocks have no autograd graph;
  only policy logits retain gradients. Reward normalization is over all512×4
  centered rewards, population std; identical-path groups receive zero advantage.
- The independent RL stream shuffles without replacement per pass. Its permutation,
  cursor, generator and both optimizers/all RNGs are included in full checkpoints.
- Predeclared N speed reference is J-RL seed0 at200epochs, including periodic dev.
  N is atomically published while all8 workers continue; no first200 relaunch.
  WS switches at the next epoch boundary after the planned N-derived warmup; any
  late switch is explicitly flagged. If seed0 fails, seed1's measured first200 is
  the flagged fallback. Never silently estimate a speed from missing data.
- Each P-rand GPU runs FR-adaptive then FR-fixed after its parent completes.
  Both retain actual frozen-layer before/after hashes, selected(best RL-dev,
  earlier ties) and last-router evaluations. Main/FR config files record seeds.
- Final eval uses8 seeded samples per question where required, always restoring
  training RNG afterward. Path comparisons include all training compositions,
  RL-train, OOD-test and atomic facts; total-variation/overlap are descriptive.
- Preserve the fixed threshold readings, explicit unequal reward-data caveat,
  seed-noise flags, sanity checks and runtime overruns. A non-winning result is
  not a code failure. Incomplete arms cannot publish a successful final receipt.
- Tiny CPU entrypoint tests set `development_only` and N=1; that branch rejects
  CUDA or d>32. Production configs never set it and require N2000..4500/multiple500.

Local verification:12 CPU tests, four-arm real-entrypoint smoke, T11 loop policy
probability0.9999468 and T13 uniformity minimum p0.045768. Full production GPU tests
are a separate queued Phase0; do not claim them before their receipts arrive.

Tier1 includes router `.pt` files (tiny), configs, metrics, routes and report.
Model/fullstate stays node-only. Backups are exact small archives with manifests;
never export a raw output tree. Native GPU handoff remains the existing supervisor.
