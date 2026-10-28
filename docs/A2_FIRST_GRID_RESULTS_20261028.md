# A2 first DCLM/P300 S1 grid — 28 October 2026

Completed **08:44:30 SGT**, 13m54s including mechanism checks, baseline scoring,
three fits, validation and exports. Three fits, 114 updates / 29,884,416 KD tokens
each; fresh optimizer/schedule at each LR. All eight B200s per fit. No retries.

| Model / LR | Tuning loss | Final validation loss | Final PPL |
|---|---:|---:|---:|
| Original P300 | 2.722719 | 2.737793 | 15.4528 |
| S1 removed, unhealed | 3.758345 | 3.763775 | 43.1109 |
| 3e-05 | 3.305984 | 3.312775 | 27.4612 |
| 1e-04 | 3.180288 | 3.188723 | 24.2574 |
| 3e-04 | 3.086757 | 3.095975 | 22.1088 |

The preregistered tuning rule selects **3e-4**, the top of the tested grid.
Final validation did not select LR. Healing recovers some pruning damage but
remains worse than the original parent. This short LR grid alone cannot answer
the research comparison; P1T and longer matched healing are not yet run here.
No expansion of the registered LR grid is implied by its boundary winner.

Actual model removal, KD/accumulation and save/reload gates passed. All three
fits consumed exactly the same 7,296 sequences across eight ranks, followed the
114-update LR schedule, and had zero BF16 saved/reloaded small-set loss difference.
Each full fit took 230–233 seconds; actual training took 137–138 seconds.

Automatic communicating burn handoff passed **08:45:36 SGT**. Independent
inspection **09:31:39 SGT** found eight burn workers and advancing collective
cycles/payload; the guard was released. No further training queued.

Raw metrics: [results.json](results/a2_early_grid_20261028/results.json).
Completion and SHA/health receipts: [complete.json](results/a2_early_grid_20261028/complete.json),
[final health](results/a2_early_grid_20261028/final_health_0931.json).
Full locally downloaded artifacts: `temp/a2_grid_health02/`;
raw export: `temp/a2_grid_health02.log`. Parent weights remain on B200.
