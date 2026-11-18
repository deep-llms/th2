# Frozen-body budgeted refinement

Registered follow-up to ADAPTIVE_REFINEMENT_20261119.md. User authorizes continued
iterations after negative screens. Parent body fits and all earlier studies are
preserved. No body training or old DOT-RL resume.

The completed screen selected the extra block for99.35–100% of final target
positions and sparse Python dispatch was slower than dense. J7's NLL gain over
C6 was only~0.0006–0.0007nats. Neither registered criterion passed. The next
question is whether predictable token-level gains can support a useful cost
tradeoff, with a batched implementation instead of one Python attention call per
sequence. This is a targeted diagnostic, not a claim of a winning design.

For each F7/J7 seed0/1 frozen endpoint, fit one causal gain regressor:
LayerNorm768→128SiLU→1 on normalized canonical hidden state at t, predicting
baseNLL(x[t+1])-refinedNLL(x[t+1]). MSE, AdamW.001/wd.01, eight epochs, batch4096
positions; use the existing4096fit/512document-disjoint validation rows and
counterfactual caches. Private permutation seed3489+seed. Select best epoch by
validation hard NLL at a fixed50% calibration budget. No final-score selection.

Calibrate three scalar thresholds on validation scores at the25/50/75% refine
quantiles. Inference applies the fixed scalar to each causal score independently;
never sort/rank within a test sequence or use its future states. Ties skip.
Report achieved rates and distribution shift rather than forcing final budgets.
Compare always-base/refine, C6, validation-rate matched random, and target oracle.
All three budgets are reported. Reused final sets are explicitly exploratory,
including FWE which was fresh only in the parent screen.

Batch active sequences, pad their selected queries to the batch maximum, keep
original positions and full causal input K/V, and gather only real query outputs.
Test dense parity, gradients, arbitrary masks (including empty sequences), future
independence, actual NLL replay, head fullstate resume and the complete export.
Count both logical and physically padded queries and full K/V work. Benchmark
optimized dense-refine versus sparse with alternating order/warmup, five measured
repeats, batch8: body-only and CE-inclusive full-context timing. No decoding claim
and no assumed equal FLOPs; a5% median speed signal still needs confirmation.

For each arm and budget, require all two seeds/two domains:
- learned beats random by≥0.001nats for a routing prediction signal;
- learned beats C6 by≥0.01nats for the inherited quality signal;
- ≤0.01nats worse than refine and≥25% actual queries skipped for an adaptive
  tradeoff (alone this does not prove routing is useful);
- ≥5% body latency reduction versus optimized always-refine for a timing signal.

Source/input/checkpoint hashes and fresh native ownership gate admission. Four
single-GPU router/eval jobs, report/raw replay, five node backup files (four head
fullstates plus compact archive), native handoff. Only small Tier1 archive/proof
is exported. Watcher continues across the next decision; preserve all failures.
