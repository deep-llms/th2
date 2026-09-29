# B200 checkpoint and microbatch investigation — 2026-09-29

Recommendation: keep microbatch 16 and accumulation 4, and disable decoder, LM-loss
and F/G auxiliary-loss checkpointing. This preserves the 1,048,576-token global
batch and is the fastest tested compatible setting for B, F and G.

These are disposable performance probes, not additional scientific training.
All use the existing train_env, eight B200 GPUs, BF16, SDPA with the implicit causal
backbone mask, LM chunks of 512, the full 28-layer model and real EOS-packed English data.
No package, driver, system CUDA, production recipe or scientific checkpoint changed.

## Remaining checkpointing at microbatch 16

All rows below disable decoder checkpointing and retain the same 1,048,576-token
global update (16 examples/GPU × 4 accumulation × 8 GPUs × 2048 tokens).
Each trial completed 18 real HF Trainer optimizer updates; timing covers updates 6–18.
Profiling and weight/optimizer checkpoint writes were disabled only for these disposable
benchmarks. No timing includes startup, final evaluation or model saving.

| Arm | LM checkpoint | Auxiliary checkpoint | Seconds/update | Peak allocated GiB | Peak reserved GiB |
|---|---|---|---:|---:|---:|
| B | True | N/A | 2.2063 | 98.32 | 103.38 |
| B | False | N/A | 2.0906 | 111.71 | 112.65 |
| F | True | True | 2.8512 | 98.38 | 103.51 |
| F | False | True | 2.7174 | 111.78 | 112.78 |
| F | True | False | 2.5928 | 98.38 | 103.51 |
| F | False | False | 2.4745 | 113.15 | 116.03 |
| G | True | True | 3.2220 | 98.45 | 103.51 |
| G | False | True | 3.0849 | 111.84 | 112.78 |
| G | True | False | 2.8345 | 103.04 | 105.66 |
| G | False | False | 2.7053 | 121.64 | 124.28 |

## Observed capacity

In the configuration names, `lm1`/`aux1` enable loss checkpointing and
`lm0`/`aux0` disable it. The auxiliary flag has no effect on B.

These searches disable decoder checkpointing. A passing size completed 18 updates
on all 8 ranks; each upper boundary is a typed CUDA OOM in a fresh process.

| Configuration | Largest tested passing microbatch | Next failing size |
|---|---:|---:|
| B-lm1-aux1 | 30 | 31 |
| B-lm0-aux1 | 26 | 27 |
| F-lm1-aux1 | 30 | 31 |
| F-lm0-aux0 | 26 | 27 |
| G-lm1-aux1 | 30 | 31 |
| G-lm0-aux0 | 24 | 25 |

Search assumes broadly monotone memory growth; these are empirical boundaries, not
proofs for every untested batch or a long-run OOM guarantee. Intermediate sizes use
accumulation 1 and therefore a different global batch. Sizes 16/32/64 use 4/2/1 to
preserve the original global batch. The boundary consequently includes this policy;
it is not a uniform accumulation 1 sweep.

## Fixed-global-batch recommendation

- B: `B-layers0-lm0-aux1-b16`, accumulation 4, 2.0906 s/update, 65.7 GiB reserved-memory headroom.
- F: `F-layers0-lm0-aux0-b16`, accumulation 4, 2.4745 s/update, 62.3 GiB reserved-memory headroom.
- G: `G-layers0-lm0-aux0-b16`, accumulation 4, 2.7053 s/update, 54.1 GiB reserved-memory headroom.

The decoder-checkpointed microbatch 64 comparator completed at 2.6532 s/update
and 81.67 GiB peak allocation; reserved 110.97 GiB.

## Correctness and operational evidence

- 21 CPU tests passed. All 8 combinations of the 3 checkpoint toggles across all 7 arms
  were checked for objective, gradient, next AdamW weights/moments and scheduler
  agreement after restoring a common model/optimizer/scheduler state. These are
  tiny FP32 tests, not a full CUDA/HF checkpoint-resume test.
- Every successful GPU trial passed the existing CUDA BF16 objective/gradient check
  on all 8 ranks before training, including a nonzero auxiliary branch. This checks
  tolerance-based agreement, not bitwise equality of long training trajectories.
- After the matched 18-update runs, the largest absolute final LM-loss change
  within an arm was 0.0000253. Final evaluation used the same 16 contexts. This is
  evidence against a gross training discrepancy, not proof of identical convergence.
- Fixed-global-batch trials matched the first 512 packed input-row hashes and data
  fingerprints. Intermediate capacity probes are not scientific comparisons.
- Each attempt used a fresh output/process. Only structured CUDA OOM receipts
  permitted continuation after failure; owned-child cleanup and all-GPU free checks
  ran before the next attempt. Failures and their receipts remain preserved.
- Production train.py and its strict resume configuration checks are unchanged.
  The model adds default-on checkpoint_lm/checkpoint_aux switches; the benchmark
  is the only CLI exposing them so far. Adopting settings for an existing scientific
  resume still needs an explicit, tested metadata policy rather than bypassing checks.

Local evidence: `artifacts/capacity-20260929-a01/` (source hashes verified); CPU log
`temp/capacity-tests-20260929.log`. Remote root:
`/mnt/local/_outputs/deep-llms_th2/capacity-20260929-a01`. Launch commit `2fbc273`.

The queue completed at 14:58:33 UTC: 41 attempts, 25 successful 18-update runs and
16 controlled CUDA OOMs. The supervisor verified free GPUs and automatic enhanced
burn recovery at 14:59:39. Independent audit `f353a59` at 15:01:32–15:01:45 verified
all eight approved workers, guard released, 100% utilization and 155212 MiB/GPU.
Collective cycles advanced 150→160 and payload 166.55→177.66 GiB.

All 429 exported artifacts and the remote source hashes were verified locally.
The raw audit log is `temp/capacity-final-20260929-a01.log`, SHA256
`17b85a9b0a2dc1023df59530421eb522dbd4d851c75a02b972e2a18ef76734f5`.
`commands.sh` is restored to inactive mode. No further probes are queued.
