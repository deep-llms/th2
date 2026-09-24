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
