# Stage1b FineWeb-Edu: grid note

Verified and sent 30 October 2026, 06:31 SGT snapshot. All eight registered fits completed with full LR schedules, exact eight-rank data coverage, zero roundtrip difference and no divergence. Common LR and both individual best LRs are **1e-3**. No further edge extension.

| LR | P300 tuning NLL | P1T tuning NLL | Pair mean |
|---|---:|---:|---:|
| 0.0001 | 3.151107980759 | 3.125343523082 | 3.138225751921 |
| 0.0003 | 3.084298514214 | 3.053515230555 | 3.068906872385 |
| 0.001 | 3.037060038496 | 3.011759343502 | 3.024409690999 |
| 0.003 | 3.052761722209 | 3.051705073696 | 3.052233397952 |

At 0.2B on CulturaX final, P300 Δ_post = 0.195721988347, P1T Δ_post = 0.221946209913; **E = +0.026224221566**. Own-best uses the same LR and gives the same E. FineWeb-Edu held-out diagnostic E = +0.026352782881. These are short-budget results; the registered 3.2B replication decision and downstream results remain pending.

P300 3.2B is active at 1310/12207 updates in this snapshot. Running source 7dd9da0 and its 124 frozen files remain unchanged. Numeric evidence and preregistration hashes: [grid_note.json](grid_note.json); full snapshot checks: [receipt.json](receipt.json).
