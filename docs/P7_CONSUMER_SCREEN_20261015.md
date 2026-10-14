# P7 consumer-aware screen

The user authorized further method exploration and training on 15 October 2026.
This is a fresh four-run screen, not a continuation of the completed runs.

The previous P6 consumer/cosine and consumer-only arms were already evaluated:
PPL 36.1313/36.2192 and seven-task accuracy means 45.4036%/45.5942%, versus
A 36.2769 and 45.5798%. They do not establish a downstream improvement.
P7-simple-targetgrad had small accuracy advantages at both seeds (+0.312pp at42,
+0.213pp at1042), despite worse PPL. The combined targetgrad+LM variant hurt.

## New objective

Keep the P7-simple architecture, locations (blocks2,4,6,8 in12layers), tokenwise
predictor, four-block MLP-sum target, and four joint-attention query heads.
The native and proxy entries still share one causal same-document softmax.
EOS packing and per-document RoPE resets are unchanged. No new inference module.

At each proxy location, retain its normalized shallow input, prediction `p`, and
the standardized/clipped deep target `r` already computed by the backbone.
Use the actual current consumer projections, frozen for this auxiliary computation:

- Teacher: joint attention over native entries and memory entries computed from
  `stopgrad(r)`; detach the whole response.
- Student: the same queries/native entries and projection weights, with memory
  entries computed from `p`. Gradients reach only the predictor through `p`.
- Compare the four selected head outputs **before** `o_proj`, residual addition,
  and the decoder MLP. Native Q/K/V are computed once and shared between branches.

For each query token, `L_effect = mean((student-teacher)^2) /
(mean(teacher^2)+1e-6)`, where both means cover the four heads and head dimensions.
Compute in FP32; average over valid tokens and proxy sites using the existing
Trainer global-token normalization. Teacher/student projection and attention
precision follow the normal BF16 recipe. Target is cast to predictor dtype before
projection. All native inputs, targets, RoPE values and projection parameters are
detached in this term. The teacher uses causal visibility `j <= t`, exactly as
the existing P7 consumer. This training-only deep teacher is not evidence that
online oracle decoding would behave identically; inference uses only predictions.

| Run label | Arm | Auxiliary loss before outer weight | Outer maximum |
|---|---|---|---:|
| consumer-only | P7-simple-consumer-only | effect | 0.1 |
| consumer-cosine | P7-simple-consumer | effect + cosine | 0.1 |
| consumer-targetgrad | P7-simple-consumer-targetgrad | effect + live-target cosine | 0.1 |
| targetgrad-low | P7-simple-targetgrad | existing live-target cosine | 0.01 |

The response term never changes target/backbone/projection parameters directly.
In consumer-targetgrad, the separate cosine term can flow into the target
backbone and its upstream memory projections through the original forward graph.
Predictor inputs and the LM-to-predictor path remain detached in every new arm.
Teacher statistics stay detached. Log cosine and response losses separately.
Disabling proxy use makes the response term zero; no auxiliary work occurs in
ordinary likelihood inference. The inference weights/state layout are unchanged.

Matching consumer responses allows different hidden vectors to be equivalent
when their attention outputs agree. It still assumes the teacher response is
useful; a highly accurate proxy is not itself proof of better language modeling.
Pure response versus response+cosine tests whether full-vector alignment helps
or constrains learning. The targetgrad variant tests target adaptation separately.

## Protocol and implementation

All four: fresh random12layer initialization, training/data seed42, proxy seed43,
2500steps,125-step LR warmup, existing250-step auxiliary ramp,1,048,576tokens/update,
2048sequence length, micro16/GAS4/all8B200, FA4 BF16. Preserve every250step
checkpoint and existing data/cache. Reuse completed A/P7-simple/P7-simple-targetgrad
seed42 baselines. Run the same nine downstream benchmarks, same examples and
demonstrations/evaluation seed42, after all four training validators pass.
No fine-tuning, extra seeds or longer training in this queue.

`deep_kv/proxy_memory.py` implements the frozen consumer; `proxy.py` routes and
combines losses. Existing `train.py`, optimizer, scheduler, packing and FA4 kernel
are reused. `scripts/proxy_consumer_queue.py` reuses the confirmation workflow.
Historical24/11/27-run scopes remain frozen. The new queue requires predecessor
success and verified burn handoff, rechecks the environment/data, copies/verifies
Accelerate, safely reclaims known burns, verifies all GPUs free, then runs CUDA
numerical checks and eight-GPU smokes before production. The existing supervisor
restores/verifies communicating burns on success or failure.

Local acceptance passed:28tests in195.241s cover independent eager reference
and gradient routes, causal/document isolation, SDPA/FA4 and checkpoint
 equivalence, exact real Trainer resume/reload, two-rank training for each new
arm, queue contracts and historical regression checks. After audit changes,
11final targeted tests passed in3.838s, including FA4 teacher/student call and
Q/K/V gradient counts. Logs:temp/p7-consumer-acceptance-20261015.log and
temp/p7-consumer-final-core-20261015.log. Remote full-shape CUDA numerical and
all four eight-GPU smoke gates remain required before production.


## B200 acceptance and launch

All four full-shape CUDA numerical checks and all four eight-GPU3step smokes
with validators passed. Production consumer-only is advancing at97/2500steps
as of15October02:11:46SGT (~1.27s/update); other runs/evaluation follow.
Accelerate verified, known burns safely reclaimed, GPUs verified free before
launch. Automatic burn restoration remains enabled. Source launch36e5f71;
receipt temp/remote_logs/p7-consumer-health-20261015-a03.log, SHA256
c416719f19bbf9af13acdf3427594e512980eb0e13fc2442afa199ce8f00fd19.
No production result is available yet.
