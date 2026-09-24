# B200 joint-v2 results — 2026-09-24

All six sequential eight-B200 experiments and the final CPU report completed
successfully at 03:25:08 UTC. Each arm completed the fixed 6144 updates /
201326592 input tokens using the full pretrained 28-layer model. The report
verified matched data/config/source, final checkpoints and complete histories.

recommend_distillation_design

Exploratory fixed-split validation; no automatic follow-up training.

| Seed | Base NLL | Shallow NLL | Deep NLL | Deep − shallow, 95% CI | Pass |
|---|---:|---:|---:|---|---|
| 1 | 2.93309613 | 2.93286550 | 2.92608501 | -0.00678049, [-0.007079702553694977, -0.006493299234988326] | True |
| 2 | 2.93310189 | 2.93297846 | 2.92608777 | -0.00689069, [-0.0071941112603086995, -0.006595813416454321] | True |

Deep used actual strict-past block-20 states at block 4. This is positive
exploratory evidence for the joint teacher; estimated-deep student behavior,
latency benefits and generalization beyond this fixed dev split remain untested.
No follow-up training was launched. The controller reports that its GPU burn
resumed after the queue exited and the temporary disable marker was removed.

Source root: `/mnt/local/_outputs/deep-llms_th2/joint-v2-b200-20260923-a03`.
Retrieved evidence: `temp/b200-final-20260924-a01/`. Report JSON SHA256:
`00cb05c9bdb3a1432c25bd01e21be8e928ceb3919772faf0170535bae9b66744`,
verified against the queue completion record. Earlier protocol and deployment
checks: `PCC_B200_LAUNCH_20260923.md`.

## Download and live GPU follow-up

All 45 result files were downloaded and checksum-verified locally under
`artifacts/joint-v2-20260924/`, including plots, CSV curves, six full training
histories, identities, completion records and per-context evaluation arrays.
The archive is `artifacts/joint-v2-20260924-download/results.tar.gz` (3639009
bytes), SHA256 `67d4c81cd2d2c60349b69163c16135cad0298cd830af6993be542f1f6547d147`.
Local verification checked all six 6144-update histories, finite losses/gradient
norms, fixed token totals, eight ranks, matched identities and receipt hashes.
Model/optimizer checkpoints remain on B200.

At 2026-09-24 20:46 UTC, live inspection confirmed burn workers 80990–80997,
one on each GPU, under launcher 80920. The script is now
`/mnt/local/deepeyesv2/code/p6_jobs/polite_burn.py`, using the deepeyes Python
runtime; its SHA256 matches the previously inspected polite burn. All eight
worker CPU counters advanced between samples. vLLM servers are also running
on all eight GPUs. These workloads were not modified or signaled.

The initial export's `gpu-burn-verification.json` retains its strict controller
path/exclusive-ownership check failure. It is superseded for live burn identity
by `gpu-burn-followup.json` and `gpu-identity-followup.log` in the extracted
artifact directory. Do not interpret the initial false flag as an idle node.
