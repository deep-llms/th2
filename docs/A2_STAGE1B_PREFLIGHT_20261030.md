# Stage1b FWE preflight — 30 October 2026

All required preflight checks passed. GPU preflight finished at 04:04:55 SGT; independent native burn progress, eight known workers and released guard verified at 04:07:02. The test summary was sent before Part1 submission.

| Parent | CulturaX final | CulturaX tuning | FWE held-out | S1 pruned final |
|---|---:|---:|---:|---:|
| P300 | 2.860631326 | 2.841494222 | 2.308842449 | 3.822488626 |
| P1T | 2.808707334 | 2.790512734 | 2.256209723 | 4.124982157 |
| S72 | 3.042152002 | 3.024812803 | 2.489572923 | 4.007307971 |

FWE held-out losses are diagnostic only and are not selection targets. S72−P300 final annealing gain: 0.18152067608948386 nats.

T1 and T3 passed for all three parents. Numerical self-KL/gradient/update tests passed; full LR schedule and exact eight-rank coverage passed. T17 original NEM/P300/S1 190-step repeat differs by −0.000144556073, within ±0.002. Saved/reloaded validation passed. T7 document-ID/content overlap zero.

T8: 0.950574577 seconds/update, 275774.26 tokens/second at MBS4/chunk256. Estimated remaining work: 15.6474 node-hours. Stage1b report threshold20h, hard24h, per-fit cap1.5×, deadline31Oct04:06SGT. Part1 launch prior ledger3152.574seconds includes elapsed preflight and handoff.

Reviewed tests SHA256: `0ddedf66933134338cc87f872b28e5df9d8a67ba0cf6460d482c90ebe0321b8b`. Raw receipt SHA256: `15755798c4a6ef346907955a2fb2e9ddd46f525ba31a788188ca082dd4fd215f`. Source9f5601d (120 files); all27 numerical/provenance core files unchanged in launch source7dd9da0. Archive: `results/a2_stage1b_fwe_20261030/preflight_final`.

Continuation: FWE grid and P300/P1T3.2B, four-model downstream/interim report; then authorized Part2 and final cross-dataset report. No scientific result is claimed by this preflight report.
