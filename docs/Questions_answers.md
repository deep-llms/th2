# Answers to `Questions.md` (8 October 2026)

These answers refer to the **pretraining** path for the 28-layer Qwen3-0.6B proxy runs. The special `task_finetuning` path has different gradient routing. The seed-42 P6-iso final evaluation and gate values below come from the completed 2,500-step run; the newer seed-42 ablations and seed-1042 queue are separate runs.

## A. Design

**1. Gradient routing.** At each P6 location, `u_l = layer.input_layernorm(hidden)` in [`deep_kv/proxy.py`](../deep_kv/proxy.py#L473-L489). The actual routing is in [`AnticipatoryHead.routed`](../deep_kv/proxy_estimators.py#L86-L98):

| Path | P6-iso | P6 |
|---|---|---|
| Predictor input | `u.detach()` in the predictor **forward**; both LM-injection and auxiliary estimates are functions of this detached input | `u` in the LM path; `u.detach()` in the auxiliary backward path of `BlockMLP.apply` |
| Auxiliary cosine gradient into `u`, InputNorm, earlier backbone | **No.** Detached predictor input and detached target prevent it. | **No.** The second `_MLPPath` receives `x.detach()`; target is detached. |
| LM gradient into predictor weights | **No.** `value.detach()` is injected. | **Yes.** The first `_MLPPath` retains `x` and predictor-weight edges. |
| LM gradient into gate `alpha` and backbone | **Yes** to both. The gate multiplies the detached estimate; ordinary LM/backbone gradients still flow through the residual/model. | **Yes** to both. |
| Auxiliary gradient into predictor weights | **Yes.** | **Yes.** |

The non-iso two-output gradient split is implemented in [`BlockMLP.apply`](../deep_kv/proxy_estimators.py#L41-L61), and the target is detached in [`cosine_loss`](../deep_kv/proxy_estimators.py#L12-L15) and in the target construction in [`proxy.py`](../deep_kv/proxy.py#L614-L639). [`ProxyTrainer.compute_loss`](../deep_kv/proxy_training.py#L164-L190) combines LM and weighted auxiliary loss. Thus P6-iso is *not* an auxiliary-only backbone training method; its LM loss still updates the backbone.

**2. Initialization.** **Yes by construction, and T9 passed for an architecture-equivalent small model**: the Qwen backbone is built inside a seed-42 `fork_rng` before any proxy modules in [`deep_kv/model.py`](../deep_kv/model.py#L181-L188); the proxy heads use another `fork_rng`, seed 43 by default, in [`deep_kv/proxy.py`](../deep_kv/proxy.py#L284-L303). [`test_placement_budget_initialization_and_zero_gates`](../tests/test_anticipatory_proxy.py#L39-L57) compares *every* backbone tensor between A and each proxy arm with zero tolerance, and verifies that changing module seed changes the proxy but not the backbone. The test does not itself compare the complete B200 step-0 checkpoints bitwise; that stronger artifact-level check has not been recorded. The follow-up queue explicitly sets proxy seed to `seed+1` in [`scripts/proxy_followup_queue.py`](../scripts/proxy_followup_queue.py#L25-L33).

## B. Existing seed-42 checkpoint

**3. Gate magnitude.** Initial `alpha=0.1` per channel for P6/P6-iso in [`proxy.py`](../deep_kv/proxy.py#L62-L71). The values below are the **logged mean absolute value over the 1,024 channels at step 2,500**, extracted from the completed run's Trainer log, not a single scalar gate. The callback obtains this statistic in [`proxy_training.py`](../deep_kv/proxy_training.py#L83-L96).

| Layer | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20 | 22 | 24 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Mean \|α\| | .1128 | .0993 | .0888 | .0776 | .0774 | .0731 | .0672 | .0636 | .0616 | .0604 | .0636 | .0720 |

Layer 2 rose modestly above 0.1 (about 13%); no layer is *well* above initialization. Deeper gates fell to roughly 0.06–0.09, but none collapsed to zero. A read-only B200 extraction subsequently verified the final checkpoint tensors and the complete every-250-step log trajectory below. The checkpoint tensor means agree with the step-2,500 logged values to the displayed precision. Source: `trainer_state.json` and `checkpoint-2500/model.safetensors` under `/mnt/local/_outputs/deep-llms_th2/proxy-remaining-2500-20261006-a01/supervised/run/training/seed-42/P6-iso`; the retrieved extraction log is `temp/p6-questions-readonly-a01.log` (SHA256 `ae295d5b708ac29f64ccbe4954f1846a7f781371ab1a1e6fda74f65f9da591b3`).

| Step | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20 | 22 | 24 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | .1000 | .1000 | .1000 | .1000 | .1000 | .1000 | .1000 | .1000 | .1000 | .1000 | .1000 | .1000 |
| 250 | .1000 | .0999 | .0999 | .0998 | .0998 | .0998 | .0998 | .0997 | .0998 | .0997 | .0997 | .0997 |
| 500 | .1004 | .1000 | .0998 | .0996 | .0994 | .0992 | .0991 | .0991 | .0991 | .0991 | .0992 | .0993 |
| 750 | .1015 | .1005 | .0999 | .0992 | .0988 | .0984 | .0980 | .0978 | .0978 | .0979 | .0982 | .0986 |
| 1,000 | .1034 | .1015 | .0997 | .0978 | .0975 | .0967 | .0957 | .0951 | .0951 | .0954 | .0959 | .0971 |
| 1,250 | .1056 | .1024 | .0992 | .0956 | .0950 | .0935 | .0915 | .0905 | .0903 | .0907 | .0917 | .0941 |
| 1,500 | .1080 | .1030 | .0981 | .0924 | .0917 | .0892 | .0862 | .0845 | .0842 | .0844 | .0860 | .0900 |
| 1,750 | .1098 | .1027 | .0961 | .0885 | .0878 | .0844 | .0805 | .0782 | .0776 | .0776 | .0797 | .0852 |
| 2,000 | .1112 | .1019 | .0937 | .0845 | .0839 | .0801 | .0753 | .0726 | .0716 | .0713 | .0738 | .0804 |
| 2,250 | .1122 | .1007 | .0912 | .0808 | .0805 | .0763 | .0709 | .0677 | .0662 | .0655 | .0685 | .0761 |
| 2,500 | .1128 | .0993 | .0888 | .0776 | .0774 | .0731 | .0672 | .0636 | .0616 | .0604 | .0636 | .0720 |

**4. Auxiliary cosine.** These are full-validation `eval_proxy_layer_<layer>_cosine_0` metrics, i.e. mean cosine between the predictor and normalized target over valid tokens; they are **not** a cosine derived from scalar aggregate loss. See [`proxy_training.py`](../deep_kv/proxy_training.py#L16-L50). Sources: [P6-iso](../artifacts/proxy-remaining-results-20261007-a03/training/P6-iso/result.json), [short](../artifacts/proxy-followup-monitor-20261008-a04/seed-42/P6-iso-short/result.json), [sparse](../artifacts/proxy-followup-monitor-20261008-a04/seed-42/P6-iso-sparse/result.json).

| Layer | P6-iso | P6-iso-short | P6-iso-sparse |
|---:|---:|---:|---:|
| 2 | .6421 | .6816 | .6696 |
| 4 | .5631 | .5759 | — |
| 6 | .5127 | .5158 | .5422 |
| 8 | .5042 | .4970 | — |
| 10 | .4923 | .4874 | .5227 |
| 12 | .4792 | .4740 | — |
| 14 | .4753 | .4662 | .5053 |
| 16 | .4698 | .4672 | — |
| 18 | .4642 | .4613 | .4948 |
| 20 | .4736 | .4670 | — |
| 22 | .5039 | .4850 | .5381 |
| 24 | .5777 | .5217 | — |

**5. Same-code throughput.** A and P6-iso *were* benchmarked on the same current optimized code in the paired 25-update native-FA4 probe: median full optimizer update after startup (steps 10–25) was **2.0698 s for A** and **2.2399 s for P6-iso**; ratio 1.0822. Source: [`summary.json`](../artifacts/proxy-speed-monitor-20261008-a26/summary.json) and [`PROXY_B200_VALIDATION_20261008.md`](PROXY_B200_VALIDATION_20261008.md). Seed-1042 A will give a sustained same-code runtime, but is not the first same-code measurement. The older full 2,500-step runs took about 88.30 min for A and 96.48 min for P6-iso, under an earlier code revision; use the current paired probe for the provisional step calculation, then revise using sustained same-code runs if needed.

## C. Next steps

**6. Time-matched A.** The provisional target is `round(2500 × 2.2398694 / 2.0697830) = 2705` total steps: **205 additional A updates**, about **7.1 min of optimizer-update time**, plus model startup, evaluation, and checkpointing. This is a compute-time control only when the same batch/data/recipe and measured timing definition are used; it is not an exact FLOP match. The seed-42 A checkpoint can be resumed with the same full 28,600-step scheduler and a larger `--stop_after`, because [`train.py`](../train.py#L259-L307) loads the checkpoint via Trainer, checks the original schedule/config, and allows the cutoff to rise. Optimizer, scheduler, RNG and model checkpoint contents are restored. The data stream is deterministically reconstructed from the same seed, dataset and sampler/skip logic in [`train.py`](../train.py#L173-L218), but the DataLoader iterator itself is not a serialized object; first verify dataset fingerprint and the first resumed batch/rank against the uninterrupted stream. Recent six-arm checkpoint-24→25 checks verified matching data, scheduler, normalization and rank RNG states with only tiny numerical model/optimizer differences; they are strong evidence, **not** a bitwise guarantee for native FA4 at step 2,500 ([validation report](PROXY_B200_VALIDATION_20261008.md)). Preserve the original A output and checkpoint; stage the resume deliberately rather than creating a fresh model run.

**7. Twelve per-location ablations.** Feasible as evaluation only: load the seed-42 P6-iso checkpoint once, record the unablated validation, then set exactly one head's full-channel `alpha` to zero for each of the 12 locations, evaluate on the *same* validation set, and restore it before the next location. No optimizer step or checkpoint write is needed. The gate is read at [`proxy.py`](../deep_kv/proxy.py#L487-L489). The existing full-validation P6-iso evaluation reported **8.42 s**; twelve main passes are about **101 s of evaluation**, plus startup/I/O and baseline evaluation. [`ProxyTrainer.evaluate`](../deep_kv/proxy_training.py#L200-L217) also runs a no-proxy pass, so naively calling it twelve times roughly doubles the repeated evaluation work. A small evaluator that runs only the main pass should finish in a few minutes, subject to a short pilot on the actual machine. Keep the same held-out rows, tokenizer, backend and document isolation.

**8. Offline estimability dump.** Feasible, but needs a new **streaming, read-only export hook**. At layers 2/8/14/20 capture `u_l` immediately after InputNorm ([`proxy.py`](../deep_kv/proxy.py#L473-L475)) and the exact detached, normalized 4-block MLP target after the window closes ([`proxy.py`](../deep_kv/proxy.py#L614-L639), normalization at [L379–390](../deep_kv/proxy.py#L379-L390)). Use the checkpoint's stored normalization state, disable state updates, and keep train/held-out IDs separate. The current `block_observer` captures the post-block `hidden`, **not** the requested `u_l`/target ([`proxy.py`](../deep_kv/proxy.py#L608-L609)). For 5M tokens × 4 locations × 2 arrays × 1,024 dimensions × 2 bytes (BF16), raw arrays need **81.92 GB (76.3 GiB)**, plus token IDs (~20 MB as int32) and shard/index metadata. FP32 for both arrays doubles the main payload to 163.84 GB; if only targets are FP32, it becomes 122.88 GB. Write bounded shards directly to local disk; do not hold the dump in GPU/RAM. This is ~2,442 packed sequences at 2,048 tokens. A forward pass over 5M tokens should be minutes rather than hours on eight B200s, but transfer/disk/write overhead is unknown; benchmark a 100k-token shard before committing to that estimate. Fix the held-out split *before* fitting width-256/512/1024 or token-lookup predictors to prevent leakage.

**9. Queue priority.** Scientifically, **yes**: P7-simple-sparse and P7-simple-short do not feed A or P6-iso, so the seed-1042 replication can be moved earlier. There is a reporting tradeoff: the existing seed-42 within-seed comparison expects all six seed-42 arms, so omitting the two P7 variants means updating/skipping that report rather than interpreting it as complete. The current queue is a fixed sequential job list ([`scripts/proxy_followup_queue.py`](../scripts/proxy_followup_queue.py#L7-L48), [`run_experiments.py`](../run_experiments.py#L142-L202)); it has no live “skip next jobs” control. Reprioritization would require a controlled stop of the **verified owned** runner/job, preservation of completed P6 results, a fresh queue/output root for A and P6-iso seed 1042, and a separately validated comparison. Do not remove output/cache or kill by GPU PID. The latest locally verified status (8 Oct, **12:15 UTC**) was P7-simple-sparse at step 489/2500, with P7-simple-short and then the seed-1042 runs queued ([`CURRENT_TASK.md`](CURRENT_TASK.md)). That status may be stale; inspect the fresh runner state before any change. No workload was stopped for this answer.
