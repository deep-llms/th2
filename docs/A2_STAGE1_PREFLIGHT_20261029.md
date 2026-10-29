# Stage 1 preflight — 29 October 2026

All required preflight checks passed. GPU work completed at 09:05:49 SGT; communicating burns restored at 09:06:56 and independently verified at 09:09:01. Test results were sent to the user before submission of the scientific queue.

## Parent measurements

| Parent | CulturaX final | CulturaX tuning | DCLM held-out | S1 pruned final | Δ pre final |
|---|---:|---:|---:|---:|---:|
| P300 | 2.737793032 | 2.722719408 | 2.506344141 | 3.763775144 | 1.025982112 |
| P1T | 2.686908992 | 2.671354022 | 2.452078286 | 4.004721485 | 1.317812493 |
| S72 | 2.908575296 | 2.896920584 | 2.677280175 | 3.854544222 | 0.945968926 |

S72−P300 annealing gain on final: 0.170782264 nats. These are pre-healing measurements, not Stage 1 outcomes. The DCLM held-out set may have been seen in parent pretraining, more by P1T; it is not used for selection or decisions.

## Tests

- T1: P300/P1T/S72 precision gaps 0.000098401 / 0.000112374 / 0.000114932 nats, all below 0.002; native sanity passed. T3 branch identity passed.
- T5 zero self-KL; finite gradients. T12 loss difference 4.77e-7, gradient max 2.53e-7, accumulated-update max 8.94e-7. Prune/reload probes exact.
- T6 full actual T8/T17 LR schedule verified; registered budgets use the unchanged schedule formula. T16 saved/reloaded T17 validation difference zero.
- T7 train/held-out document-id and exact-content-hash overlap both zero. T18 complete sequence count, EOS rate and first/last decode recorded; all three parents have lower DCLM held-out loss than CulturaX final.
- T17 original NEM/P300/S1 190-step CulturaX repeat: 3.061888517607 versus original 3.061875752042, difference +0.000012765565, passing ±0.002. Exact 12,160-sequence eight-rank order, full LR schedule and source/data identities verified.
- T8: 0.948931813 seconds/update, 276251.67 tokens/second; 16 updates, first five excluded for timing; MBS4/chunk256, global64×4096. Exact coverage and teacher no-gradient check passed.

## Timing and continuation

Estimated remaining work 15.623 node-hours, including a possible registered edge pair, evaluation and cleanup. Per-run estimates: 0.2B 15.1min; 0.8B 51.3min; 3.2B 196.1min. Each run has its registered 1.5× time cap; Stage1 hard cap36hours. Conservative prior ledger1800seconds covers admission, preflight and handoff.

Queue: six paired LR-grid fits, at most one extra edge pair if required by the fixed selection rule, then three main3.2B, three mid0.8B and X72short0.2B. Fresh original parents/optimizers/schedules for every run. Common LR chosen only on mean tuning loss. Downstream and final reporting remain required after training.

Reviewed tests SHA256: `739847de9e35a517fd7240f46b69a47bad7a55183503abfe2668a402b24380ef`. Source `0c1c1488477f1b836989182986bd785fbcd667d8` (103 frozen files). Raw receipt SHA256: `e85283a72b5ed41fa90f8bed8946fcdf22bb17a90b72f5ad7829d71a801a9881`. Compact audit: `results/a2_stage1_20261029/preflight_final/receipt.json`; full original receipts retained locally and on the node.
