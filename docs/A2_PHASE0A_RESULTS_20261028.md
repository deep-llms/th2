# A2 Phase0A results — 28 October 2026

All 58 fits completed; no divergence or retries. The final queue finished at 20:55 SGT; independent inspection verified communicating GPU burns at 20:56 SGT.

We remove six of 24 native OpenSci layers, then heal with pure teacher–student KL. P300 and P1T are annealed parents trained for 300B and 1T tokens. Each fit starts from its own original parent with a fresh optimizer and complete schedule. All six LR grids selected 3e-4.

## Main result

The preregistered **screening label is GO**: mean excess recovery damage `E = +0.012542268` nats; all three dataset means and 12/12 short-budget contrasts are positive. Positive E means P1T loses more relative to its own parent than P300 does. P1T still has lower absolute NLL. This is a screening result, not a significance test or proof of novelty; the contrasts share parents/data. Phase0B and deferred phases have not run.

| Dataset | Mean E, S1/S2 at 0.2B |
|---|---:|
| dclm | +0.012571973 |
| fwe | +0.015041272 |
| nem | +0.010013559 |

## Held-out NLL

| Dataset | Parent | Original | S1, 0.2B | S2, 0.2B | S1, 0.8B |
|---|---|---:|---:|---:|---:|
| dclm | P300 | 2.737793 | 2.935242 | 2.849884 | — |
| dclm | P1T | 2.686909 | 2.895235 | 2.813267 | — |
| fwe | P300 | 2.860631 | 3.036157 | 2.949261 | — |
| fwe | P1T | 2.808707 | 2.996828 | 2.914824 | — |
| nem | P300 | 2.774251 | 2.956839 | 2.871261 | 2.884719 |
| nem | P1T | 2.723383 | 2.913690 | 2.832701 | 2.839079 |

NEM 0.8B recovery damage is +0.110468547 for P300 and +0.115696093 for P1T: contrast +0.005227546 nats. Both improve substantially over 0.2B; the remaining contrast is smaller.

T14 checks, NEM/P300/S1/0.05B: identical repeat changes NLL by −0.000006436; fixed alternate data order by −0.002285655. These individual checks are not confidence intervals.

## Verification and evidence

- All fits: complete LR schedules, exact eight-rank coverage/order, save/reload NLL difference zero; all six pre-baselines match their original parents.
- [Final metrics and aggregate](results/a2_remaining_phase0a_20261028/final/aggregate.json), [per-fit summary](results/a2_remaining_phase0a_20261028/final/summary.json), [terminal evidence](results/a2_remaining_phase0a_20261028/final/terminal-verification.json).
- Full local original receipt: `temp/a2_remaining_health13.log`; SHA256 `52acc0b66ad7c55447cdb2856794fb8560f44c08d992974769ece936909e5a83`.
- Original checkpoints and full training artifacts remain on B200 under the roots recorded in `resources/a2_completed_surveys_20261028.json` and `resources/a2_phase0a_terminal_20261028.json`.

Supplementary downstream evaluation started at 21:06 SGT after verified GPU handoff: 14 fixed checkpoints, nine existing English tasks, zero-shot seed 42. The first attempt stopped before scoring because batch-eight numerical error exceeded the fixed gate; batch-two recovery is being prepared. No downstream results are available yet.
