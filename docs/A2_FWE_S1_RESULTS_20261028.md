# A2 FineWeb-Edu S1 — interim results, 28 October 2026

Snapshot 11:59:25 SGT: eight of ten fits completed without retry. Both parents
selected LR 3e-4 using tuning loss. P300 main is at 240/762 steps; P1T main remains.
Eight GPUs are active. Expected completion 12:28–12:35 SGT, including scoring and
normal burn handoff, if there is no retry. This is not a completed queue.

Same S1 removal, pure KD and fresh original-parent initialization per budget as
DCLM. The healing/evaluation text is unchanged; the pretrained parents are FWE.

| Budget | P300 final loss | P1T final loss | P300 damage | P1T damage | P1T−P300 damage |
|---|---:|---:|---:|---:|---:|
| 0.03B | 3.193430 | 3.175876 | 0.332799 | 0.367169 | +0.034370 |
| 0.05B | 3.143627 | 3.115962 | 0.282996 | 0.307255 | +0.024259 |

Damage is final loss minus the same parent's original unpruned loss:
P300 2.860631; P1T 2.808707. At these short budgets P1T has better absolute
loss but slightly larger remaining pruning damage. The main budget is pending;
this is not the full preregistered multi-dataset/multi-pattern decision.

All eight completed fits passed exact eight-rank sample coverage/order, every
LR update, expected steps/tokens and zero BF16 save/reload loss difference.
All 78 frozen source files match cfeb74c. Supervisor and guard remain active;
final completion and burn restoration are not yet verified.

Local [summary](results/a2_fwe_s1_20261028/status_1159/summary.json) and
[SHA/health receipt](results/a2_fwe_s1_20261028/status_1159/health.json).
Full raw receipt: temp/a2_fwe_health03.log, SHA256
68aa9c31b9f112cbb7462ee9b43631ad2182061a4ad104fdabd8ede4523d506e.
Extracted full artifacts remain in temp/a2_fwe_health03/; checkpoints remain on B200.
