# Same-weight attention verification at update 200

All eight comparisons passed the unchanged numerical thresholds. This follow-up
compares both backends at identical trained weights, rather than comparing the
different weights produced by the two 200-update trajectories.

## Scope and method

Use both final checkpoints from document-stability-200-20261004-a01: one trained
with dense document-isolated SDPA and one with document-isolated FA4. For each
checkpoint, compare the first 2 and first 16 packed rows of both the training
and held-out pools. Sequence length is 2048. The micro2 batches are subsets of
the corresponding micro16 batches; these are not eight independent data samples.

Each of the eight B200s executes one checkpoint/batch case independently. There
are no optimizer updates or distributed gradient reductions in this diagnostic.
Each case captures full-model loss, final hidden states, sampled logits (every
128 positions), and gradients of every unique model parameter. The same masks,
positions, loss normalization and weights are used for each backend. Repeat both
BF16 captures to measure pipeline repeatability. Micro2 cases additionally use
FP32 math SDPA with TF32 disabled. FP32 micro16 was not attempted because its
memory requirement had not been verified.

The strict checkpoint loader verifies that saved copies of tied parameters
agree, fills only known tied aliases when deduplicated, then requires exact state
keys and shapes. Parameter SHA256 values before and after every case agree.
Input hashes match between corresponding cases for the two checkpoints. All
captures completed without nonfinite gradients or OOM.

## Results

Ranges below are relative L2 errors of the full parameter-gradient vector,
not differences between gradient norms.

| Comparison | Gradient relative L2 difference |
|---|---:|
| FA4 vs dense SDPA, identical weights, all eight cases | 0.181%–0.532% |
| Dense full pipeline vs its repeated capture | 0.119%–0.302% |
| FA4 full pipeline vs its repeated capture | 0%–0.111% |
| Dense BF16 vs FP32 math reference, four micro2 cases | 0.539%–1.209% |
| FA4 BF16 vs FP32 math reference, four micro2 cases | 0.540%–1.163% |

All eight FA4/dense cases satisfy the 3% gradient, 2% hidden/logit relative L2,
and 0.01 absolute loss limits. Both backends also satisfy these limits against
the FP32 reference in all four micro2 cases. The limits were not relaxed.

Maximum FA4/dense loss difference is 0.00012493. Maximum relative hidden-state
and sampled-logit differences are 0.270% and 0.180%. Gradient cosine similarity
is at least 0.9999858. Repeated forward captures are identical in loss, hidden
states and sampled logits; repeated backward captures can differ slightly.
This is repeatability of the complete model/loss/backward pipeline, not an
isolated measurement of attention-kernel nondeterminism.

For micro16 specifically, full-gradient differences between backends range from
0.181% to 0.266%. Peak allocated GPU memory across captures is 101.74 GiB for
micro16 and 34.13 GiB for micro2 (including the FP32 comparison). These are
single-batch checks with no optimizer state; they are not training throughput
or training-memory measurements.

## Interpretation

The results support ordinary accumulated numerical drift as an explanation for
the earlier trajectory divergence. They provide no evidence of a large FA4
backward error in these trained-checkpoint/batch cases: FA4 and dense BF16 have
similar errors against the higher-precision reference. Small backward variation
also occurs when repeating the same backend at unchanged weights.

This does not prove that accumulated drift caused every difference in the
200-update run. The earlier 48% figure compared gradient norms at different
trained weights; the new maximum 0.532% compares full gradient vectors at the
same weights. These are different measurements, and the earlier failed gate
remains a failure in its historical record.

No additional long baseline training run is needed merely to repeat this kernel
check. A repeated dense training control would be useful if quantifying the
expected trajectory variation is a research objective. This evidence supports
proceeding to controlled integration of document-isolated FA4 for the baseline;
it does not promise bitwise reproducibility or cover untested custom auxiliary
arms. Production training code and its attention policy remain unchanged.

## Execution and provenance

Initial 3279ab4/a01 attempt failed before GPU captures: safetensors.load_model
rejected a Trainer checkpoint with both cloned tied-embedding copies. No weights
were changed. Burn recovery passed at 13:49:25 UTC. The diagnostic loader was
fixed and tested against deduplicated, cloned and conflicting tied-weight cases.

Retry 3d5070b/a02 loaded both actual checkpoints on CPU before reclaiming GPUs.
The existing attention_bench environment was used. All ten focused CPU tests
passed remotely; original train_env and NVIDIA driver were not changed.
Accelerate configuration was copied and verified, known burn workers
18263–18270 were identified and stopped, and all eight GPUs were verified free.
Every GPU diagnostic worker exited 0.

Source parameter fingerprints:

- Dense-trained: a5280b6471a66bf7ebafd20bd953a8275e9798d2600667e1f1d2fcadfd7ce1d0
- FA4-trained: 7dec2fc15ea2919c4c3e55104abfe846a79daab386a9e8cf3313ea3f9c1aed4c

Launch/capture log: temp/trained-attention-launch-a02.log, SHA256
7b687f383ee48701f475737223de28cf93ef1b3f80d72f9ce3f513aa768e01ac.
Remote output: /mnt/local/_outputs/deep-llms_th2/trained-attention-check-20261004-a02.
The final supervisor receipt reports passed=true at 13:57:24 UTC. All eight
burn workers 19173–19180 were automatically restored with collective progress
verified. Read-only collector 736802d independently confirmed those live worker
identities and the released GPU guard at 13:58:16 UTC.
Collector log: temp/trained-attention-collect-a02.log, SHA256
e984b1c32c44de2b3bcb1e361ea18438828db729d3a91785d2ff5e549b969eef.
Result archive SHA256:
dc7d0645149fac6628bcf3ea510ffca99cfa0a6fb3557dcd512691bf93555680.

Verified all 21 source-file hashes after retrieval to
artifacts/trained-attention-check-20261004-a02/. Eight receipts match the summary;
final supervisor passed and burn collective progress passed. commands.sh is #0.
