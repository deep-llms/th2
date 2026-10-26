# Full-model B200 training pilot — 17 October 2026

Historical template validation only. Machine names, paths and job IDs below
are provenance, not new-project configuration or current run status.

## Recipe

Full 28-layer Qwen3-0.6B scratch baseline using the existing pinned local config
and tokenizer, and existing English cx_sampled_old text. No downloads or
package/driver changes. Each backend used all eight B200 GPUs, BF16, activation
checkpointing, microbatch 16, accumulation 4, sequence 2048, seed/data_seed 42.
Nominal batch: 1,048,576 input tokens. EOS separates documents; attention is
isolated and positions reset per document.

Fixed input slice: 100,000 training documents and 256 validation documents from
the first sorted existing sources. This is test-only text selection; normal
train.py tokenization, packing and HF caching remain unchanged. Packing yielded
40,497 training sequences. Evaluation used 95 sequences and 194,222 valid targets.

Full schedule 28,600 steps, warmup 1,430. Train to 25, save native checkpoint,
resume to 50, verify checkpoint and final standalone evaluation. Both backends
start fresh with the same inputs/seeds. Source snapshot: template_validation;
launch 931839d, output q359-template-realdata-20261017-a02.

## Results

| Backend | Completed steps | Validation loss at 25 | Validation loss at 50 | HF training time, both stages |
|---|---:|---:|---:|---:|
| SDPA | 50 | 12.078832 | 11.692120793 | 156.85s |
| FA4 | 50 | 12.078842 | 11.692118750 | 153.91s |

All 12 queue stages passed. Each backend's standalone loss exactly matched its
step 50 periodic validation. Final loss difference between backends is about
0.00000204 nats. This short pilot tests execution and numerical behavior, not
long-run equivalence or model quality. Times are HF train_runtime totals for
the two invocations, including in-training validation; they exclude
preprocessing, process startup and standalone evaluation. They are not a
controlled throughput benchmark. Observed GPU memory was about 74 GiB per GPU;
no OOM occurred.

Checkpoint checks confirmed weights, optimizer, scheduler and all eight rank RNG
files. Native HF resume loaded the step 25 checkpoint and reached 50. Logged
learning rates matched 3e-4*(step-1)/1430; gradients were finite. Exact
resumed-versus-uninterrupted tensor comparison was covered by the earlier tiny
GPU suite, not repeated for this full-model pilot.

## Configuration issue caught and fixed

Initial pilot a01 mistakenly combined warmup_ratio=0 with warmup_steps=1430.
HF 5.9 prioritizes the deprecated ratio, producing no warmup. The owned pilot
supervisor was stopped through its verified PID, and its burn handoff was
verified before the fresh a02 launch. No a01 checkpoint was resumed or used
for the results above. The helper now omits the ratio and verifies effective
HF arguments before GPU reclamation; positive and injected-conflict tests pass.
The native training core was unchanged.

HF's aggregate train_loss after resume uses cumulative global_step and is not
used here for quality comparisons. Step-level logs and held-out loss are the
appropriate quantities.

Local raw receipts: /disk/thuat/deep2shallow/temp/template-realdata-20261017/.
Final evidence: [structured results](results/template_realdata_pilot_20261017.json).
Queue completed at 19:22:38 UTC; supervisor returned zero and restored verified
burns at 19:23:49. The 19:26:07 read-only check confirmed eight matching burn
workers, a released guard, and collective cycles advancing from 90 to 170.
commands.sh was returned to #0.
