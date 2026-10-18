# B-DOT local GPU verification — 2026-10-18

SDPA GPU correctness and preliminary throughput checks are complete. **FA4 training is not validated:** the pinned kernel rejects A100 during backward. No pilot was launched, no production model code was changed, and `commands.sh` remains `#0`.

## Machine and runtime

- Four local A100 PCIe 40GB GPUs, driver 525.125.06; inspected before launches. No existing workload was stopped.
- SDPA tests: existing `sparse_emb`, PyTorch 2.7.1+cu118, Transformers 5.9.0, Accelerate 1.13.0, Datasets 4.8.5.
- FA4 forward: isolated `temp/bdot_gpu_verification/fa4_compat`, PyTorch 2.8.0+cu126, flash-attn-4 4.0.0b33, CUTLASS DSL 4.8.1, quack 0.6.5; `pip check` passes.
- `train_env`, other existing environments and NVIDIA drivers were unchanged. The pinned PyTorch 2.14.1 CUDA 13 environment cannot use this driver. An isolated 2.14.1+cu126 attempt enumerated GPUs but failed tensor allocation. These local tests therefore do not validate the production runtime.

## Numerical results

- Full 12-layer, 344M-parameter fixed-route/no-FiLM model versus native Qwen3, FP32 and TF32 disabled: logits match exactly. With SDPA math as the attention reference, maximum gradient difference is **8.94e-8** and maximum difference after AdamW is **4.82e-5**, below T1’s 1e-4 threshold.
- **Preserved caveat:** default GPU SDPA produced maximum updated-weight differences of 7.54e-5 on the first trial and **1.0275e-4** on a repeat; the latter fails the strict threshold. Logits still matched exactly and gradient error was 9.69e-8. Native Qwen3 versus an identical native copy also produced gradient differences (1.15e-7) and updated-weight differences (5.92e-5). This supports a CUDA reduction/AdamW sensitivity explanation, but does not turn the failed default-backend trial into a pass. The math-reference result is reported separately.
- Reduced-width 12-layer fixtures retain S=8, GQA and head dimension 128. BF16 hard/soft routing, reuse on/off and nonreentrant checkpoint recomputation pass; checkpointed logits and gradients match exactly.
- Document isolation: changing another document changes dispatch group sizes and can alter BF16 rounding (up to 0.0078125 in this fixture). With routes fixed, unaffected logits match bitwise. Input gradients across document boundaries are exactly zero. Causal masking also passes.
- Two real CUDA ranks using NCCL and actual BDOTTrainer: hard routing with rank-local/globally unused experts, soft routing, accumulation=2 and nonreentrant checkpointing pass. Maximum gradient error against the full-batch FP32 reference is **1.79e-7**. Counts, balance updates and the optimizer update also pass.

## Complete-training-update throughput

Actual PackedTrainer/BDOTTrainer on one A100; full 12-layer model and vocabulary; BF16, sequence 1024, microbatch 2, accumulation 2 (4,096 input tokens/update), document fragments of 128 tokens, checkpointing off. Three warmup updates and ten measured updates. Includes forward, backward, clipping, AdamW, scheduler, Trainer and routing-state updates; excludes preprocessing, saving and evaluation. No other benchmark occupied that GPU.

“Mixed” explicitly sets the diagnostic initial router prior to zero to exercise non-default paths; production configs are unchanged. The mixed hard case has 57.4% off-default decisions. Fresh default-prior hard routing is entirely diagonal here and would give an overly optimistic speed estimate.

| Model / phase | Tokens/s | Seconds/update | % of D | % of native | Peak allocated GiB |
|---|---:|---:|---:|---:|---:|
| dense_native | 19,615 | 0.209 | 124.0% | 100.0% | 11.31 |
| D | 15,818 | 0.259 | 100.0% | 80.6% | 11.63 |
| P1B_hard | 15,628 | 0.262 | 98.8% | 79.7% | 11.82 |
| P1B_soft | 11,500 | 0.356 | 72.7% | 58.6% | 13.29 |
| P1B_hard_mixed | 12,101 | 0.338 | 76.5% | 61.7% | 11.88 |
| P1B_soft_mixed | 11,331 | 0.361 | 71.6% | 57.8% | 13.30 |

The mixed hard case is above 60% of both references locally, but only narrowly above native dense. Soft routing is above 60% of current D and below 60% of native dense. D includes FiLM and generic dispatch overhead; do not hide its roughly 19% lower throughput versus native when making efficiency claims. These short, single-A100 synthetic measurements **do not clear the final eight-GPU T8 gate**. Repeat at the intended batch, runtime and backend before pilot launch.

## FA4: actual kernel attempt

The [pinned package](https://pypi.org/project/flash-attn-4/4.0.0b33/) has an SM80 forward path. Its backward dispatcher explicitly requires compute capability 9.x or newer (`flash_attn/cute/interface.py:2278` in the installed wheel). The real native-Qwen3 probe reached backward and failed on that guard. It was not removed. Therefore no FA4 training throughput or gradient-equivalence result is available.

Forward-only B-DOT comparisons on the actual FA4 kernel gave roughly 0.33–0.34% relative logit L2 error for identical hard/no-reuse choices. The soft reuse-on case had 0.98% error and **one changed secondary route** (primary routes unchanged); its maximum individual logit difference was 0.1993. That case is a natural-routing comparison, not an identical-path kernel comparison. The updated probe checks both selected branches to avoid overlooking this. Forward-only evidence is not training approval.

An H100/B200-class execution target must be confirmed for FA4 backward, complete updates and distributed checks. DOT has no assigned execution repository or machine.

## Code and artifacts

- `scripts/benchmark_bdot.py`: now benchmarks the actual Trainer, supports SDPA/FA4, accumulation, checkpointing and explicit curriculum phases, and records runtime/config/source hashes.
- `tests/bdot/gpu_probe.py`: CUDA numerical/native/checkpoint/isolation checks; separate math reference and native-self control; traces primary and secondary choices.
- `tests/bdot/ddp_probe.py`: adds explicit CUDA/NCCL execution to the existing CPU probe. `test_model.py` respects an explicit CUDA_VISIBLE_DEVICES for this helper import.
- [Durable machine-readable results](results/bdot_gpu_20261018.json), including all benchmark records and their hashes.
- [Raw local artifacts](/disk/thuat/DOT/temp/bdot_gpu_verification): `sdpa_trace_final.json`, `t1_math.json`, `t1_self_control.json`, `sdpa_final.log` (preserved failure), `ddp_cuda_{hard,soft}/rank*.json`, `bench_sdpa_*.json`, `fa4_forward_traced.json`, and `native_fa4_torch28.log` (unsupported backward).

Example commands on an explicitly allocated compatible GPU (fresh output paths required):

```bash
CUDA_VISIBLE_DEVICES=0 python tests/bdot/gpu_probe.py \
  --full-native --native-only --reference-math --output temp/t1-new.json
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc_per_node=2 \
  tests/bdot/ddp_probe.py temp/ddp-new --device cuda --soft --checkpointing
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_bdot.py --device cuda \
  --config configs/bdot/P1-B.json --output temp/speed-new.json \
  --batch-size 2 --accumulation 2 --length 1024 --bf16 --progress 0.7 --router-prior 0
```

Use the matching environment; the default CUDA 13 train_env is not usable on this dev driver. The FA4 backward check additionally requires compatible hardware.
