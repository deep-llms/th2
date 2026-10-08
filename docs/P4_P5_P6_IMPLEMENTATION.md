# P4/P5/P6 implementation

Forward-local target-normalization optimization is documented in
[the 8 October performance review](PROXY_OPTIMIZATION_20261008.md).

Implements revision 3 of [the specification](proxy_arms_P4_P5_P6_spec.md).
Uses the existing `train.py`, HF Trainer/Accelerate, EOS document packing,
document-isolated attention and per-document reset positions. No new training
loop or preprocessing stage. The CLI default remains dense SDPA; the selected experiment recipe explicitly uses FA4. These are fresh
random-initialization experiments, not conversions of trained P1 checkpoints.

| Arm | Injection | Initial channel gate | Estimator LM gradients |
|---|---|---:|---|
| `P4` | All value groups; native queries and keys | 1 | Yes |
| `P4-iso` | Same as P4 | 1 | No |
| `P4-4h` | Values for the last four query heads only | 1 | Yes |
| `P4-iso-4h` | Same four-head selection | 1 | No |
| `P5` | All values from a width-256 residual stream across depth | 1 | No |
| `P6` | Residual before the block, scaled by detached input RMS | 0.1 | Yes |
| `P6-iso` | Same as P6 | 0.1 | No |

Original-arm placement is blocks 2, 4, …, 24 in the 28-block model. P4/P6 use independent
two-linear SiLU estimators. P5 has distinct down/up projections at each block,
and residual two-linear updates after the first proxy block. Its stream mixes
depth only and remains connected across proxy blocks for auxiliary gradients.
Gates are trainable and excluded from weight decay; absolute channel means
are logged starting at optimizer step zero.

The four-head variants select **four query heads**, not four KV heads. With
Qwen3's 16 query heads / 8 KV heads, these are query heads 13–16 (zero-based
12–15), sharing KV groups 7–8 (zero-based 6–7). Native queries and keys are
preserved in each proxy block; the other twelve query heads read native values.
The existing value projection is sliced by output rows: native input for the
first six KV groups and proxy-enhanced input for the last two. Later blocks'
inputs can change through the normal residual path. Estimator architecture,
initial weights, gates, target, loss and isolation are identical to the respective
all-head parent. The arm name fixes the selection and is checked on resume.
Architectures that cannot represent exactly four query heads using complete
GQA groups are rejected. These variants do not reduce estimator parameter count.

For the original arms, each target is the detached sum of four consecutive MLP outputs starting at
the injection block, with the existing two-pass bootstrap, running per-channel
standardization, relative variance floor and clipping. Target identity is
`p4p6-r1`. Auxiliary cosine loss is averaged across tokens and layers and has
the existing ramp `0.1 * min(step / 250, 1)`. Global token denominators account
for DDP and gradient accumulation. Statistics remain fixed during each update,
checkpoint recomputation and evaluation; their buffers are reduced together
once per update. At step zero isolated estimators participate through a zero
auxiliary gradient so DDP hooks remain valid when the ramp becomes positive.

`deep_kv/proxy_estimators.py` contains the new estimator code. P4/P6 use one
forward with separate LM/auxiliary autograd outputs: only the LM output sends
gradients into its input. Isolated arms detach the injection and backbone
inputs; gates and native projections still receive LM gradients. P5 shares its
detached-input stream forward. `--proxy_aux_recompute true` selects the slower
reference implementation for acceptance comparisons. Normal training defaults
to no estimator recomputation.

`--proxy_module_seed` defaults to 43 independently of the backbone seed, and
is saved in the recipe. Adding heads does not alter backbone initialization or
the global RNG stream. The arm names fix isolation and initial gate values;
contradictory overrides are rejected. All new settings participate in strict
resume validation. Defaults of existing A/P1/P3 recipes remain compatible.

## P6-iso target variants: weighted and per-layer normalized (8 October 2026)

Two additional, separate arms change only the detached target used to supervise
P6-iso. Both keep all twelve injection blocks (2,4,...,24), a four-block target
window, the same 1024→256→1024 predictor and initialization, isolated gradient
routing, residual injection, gate initialization 0.1, and auxiliary schedule
`0.1 * min(step / 250, 1)`. They do not combine the sparse or short variants.

Let `m_i` be block i's MLP output in the actual current forward pass, and `N`
be the existing running per-channel standardization and clipping. Targets are:

| Arm | Target at block l |
| --- | --- |
| `P6-iso` | `stopgrad(N(m_l + m_(l+1) + m_(l+2) + m_(l+3)))` |
| `P6-iso-weighted` | `stopgrad(N(1.6*m_l + 1.2*m_(l+1) + 0.8*m_(l+2) + 0.4*m_(l+3)))` |
| `P6-iso-layernorm` | `stopgrad(N(LN(m_l) + LN(m_(l+1)) + LN(m_(l+2)) + LN(m_(l+3))))` |

Weights are fixed and ordered from current to deepest block; no learned target
weights are introduced. LN is parameter-free LayerNorm over each token's hidden
channels: subtract the channel mean, divide by the square root of the biased
channel variance plus 1e-6. It is not RMSNorm and does not mix tokens/documents.
The complete target construction is detached and FP32 under mixed precision.
For LN, each contributing block is normalized once and shared across overlapping
windows. The final window is blocks 24–27. Inference skips target construction.

Both normalization-bootstrap passes and subsequent running-statistic updates
use the transformed raw target. After summation, `N`, cosine loss and averaging
over tokens/active locations are unchanged. The target itself cannot train the
backbone or the gate. LM gradients still train the backbone/gate but not the
isolated predictor; auxiliary gradients train the predictor only.

The existing `p4p6-r1` standardization version is retained. Saved target metadata
records `quantity=weighted_mlp_window_sum` plus `layer_weights`, or
`quantity=layernorm_mlp_window_sum` plus the exact `per_layer_normalization`
settings (axis, epsilon, affine and dtype). Reports validate those definitions
before allowing the intended target difference. Resume requires identical arm
and saved recipe/target metadata; these experiments require fresh training.
Old arm definitions and checkpoint metadata remain unchanged.

Both reuse train.py's Trainer/Accelerate, packing/cache, SDPA/FA4, checkpointing,
evaluation and optional supervised fine-tuning. Parameters/inference architecture
are identical to P6-iso (6,303,744 extra parameters for Qwen3-0.6B). Target-only
arithmetic adds no learned parameters; its training-time cost is not included in
the architecture MAC estimate and must be measured on B200.

```bash
python -m deep_kv make-jobs --config proxy_heads.b200.json \
  --arms P6-iso-weighted P6-iso-layernorm --seeds 42 --stop-after 2500 \
  --output temp/p6-target-variants-jobs.json
```

This only writes a sequential eight-GPU queue. Neither arm has been launched;
full-size FA4 CUDA validation and research results remain pending.

Local validation: all 55 regression tests passed in 403.788 seconds. Four
focused final checks also passed in 61.152 seconds. Coverage includes exact
full-depth targets and parent-equivalent initial LM outputs/gradients, FP32/BF16,
independent LayerNorm/bootstrap/loss/gradient calculations, document isolation,
recomputation/compilation, two-rank CPU DDP, accumulation, exact Trainer resume,
metadata tampering, sequential queues, checkpoint evaluation and task fine-tuning.
FA4 used the independent CPU attention oracle; actual CUDA kernels were not run.
Evidence: temp/p6-target-variants-regression.log and
 temp/p6-target-variants-final-targets.log. The first focused run exposed an old
test expectation (last target block 25) that was corrected to 27 for these
four-block, twelve-location arms before the passing regression.

## P6-iso follow-ups: sparse placement and shorter targets

Two separate arms extend P6-iso; neither combines the two changes.

| Arm | Injection blocks (1-based, 28-block model) | Detached raw target at block `l` |
|---|---|---|
| `P6-iso` | 2, 4, …, 24 | `m_l + m_(l+1) + m_(l+2) + m_(l+3)` |
| `P6-iso-sparse` | 2, 6, 10, 14, 18, 22 | Same four-MLP sum |
| `P6-iso-short` | 2, 4, …, 24 | `m_l + m_(l+1)` |

Here `m_l` is the actual MLP output in the current forward pass. The short
variant keeps twelve locations; it does not add an injection at block 26.
Both retain the width-256 SiLU estimator, residual injection scaled by detached
input RMS, initial channel gate 0.1, and predictor isolation from LM gradients.
The backbone and gates still learn from LM loss; auxiliary gradients train only
the predictor. Running target normalization, cosine loss, and the auxiliary
ramp to 0.1 over 250 updates are unchanged. Auxiliary loss remains averaged over
tokens and active locations, so using six locations does not halve its weight.

Sparse initialization preserves the parent's initial predictor weights at each
retained location without retaining the unused modules. Its estimator forward
MACs/token drop from 6,291,456 to 3,145,728, and extra parameters including gates
from 6,303,744 to 3,151,872. The short variant has the parent's parameter count
and estimator cost. These counts do not predict total training throughput.

Select either arm by name in `train.py` or the existing sequential queue:

```bash
python -m deep_kv make-jobs --config proxy_heads.b200.json \
  --arms P6-iso-sparse P6-iso-short --seeds 42 --stop-after 2500 \
  --output temp/p6-variants-jobs.json
```

This writes a queue only. Dense SDPA and FA4 use the existing attention paths;
packing, document isolation, reset positions, optimizer and schedule are shared
with P6-iso. Omit `proxy_lookahead` to resolve it from the arm (4 or 2). Explicit
contradictory lookahead, placement, isolation or gate overrides fail. Existing
arms still resolve an omitted lookahead to 4, preserving their saved recipes.
New arms require fresh training; changing arm or target while resuming is
rejected. Saved target metadata records the exact locations and target length,
and matched reports validate them before allowing these intended differences.
Checkpoint evaluation and optional supervised fine-tuning support both arms;
fine-tuning enables task gradients into the predictor, as for P6-iso.

These variants are implemented but have no full-size CUDA validation or research
results yet. They are not added to the running B200 queue.

Local verification: 90 of 94 CPU tests passed in 481.898 seconds. All new-arm
checks passed: exact targets/placement and initial weights, LM/auxiliary gradient
routing, BF16/recomputation/compilation, document isolation, two-rank DDP, gradient
accumulation, exact Trainer resume, SDPA versus the independent CPU FA4 oracle,
checkpoint reload, queue arguments and report rejection checks. Four existing
evaluation-harness tests could not import the missing local lm_eval package;
no numerical/assertion failures were reported. The focused two-test run also
passed. Logs: temp/p6-variants-regression.log and temp/p6-variants-focused.log.
CUDA kernels were not exercised by these CPU tests.

Follow-up review (8 October): no production-code defect found. An additional
28-block regression passed in FP32/BF16 on SDPA and the CPU FA4 oracle, checking
parent-equivalent LM outputs/gradients and exact detached targets through block
25. The local dependency gap was resolved in the separate ignored
`temp/p6-review-env`; all 11 evaluation tests now pass, including the four
previously blocked by missing `lm_eval`. Existing train environments and B200
were unchanged. Logs: `temp/p6-full-depth-review.log` (2.997 s) and
`temp/p6-eval-review.log` (75.652 s). CUDA validation is still pending.

## Usage and comparison

Train directly with the existing entry point and recipe, selecting `--arm
P4-iso`, `--arm P6`, etc. Do not pass a legacy `proxy_target_version=r7` or
`proxy_alpha_init=0` override: target identity and gate initialization resolve
from the arm. Both command-line and JSON configurations work.

Generate the normal sequential, eight-GPU-per-experiment queue:

```bash
python -m deep_kv make-jobs --config proxy_heads.b200.json \
  --arms A P4-iso P6 --seeds 42 --stop-after 2500 \
  --output temp/p4-p6-jobs.json
```

To select only the four-head variants, use `--arms P4-4h P4-iso-4h` instead.
The direct `train.py` interface accepts the same arm names. Gate evaluation and
matched-arm reports also support them; existing P4/P4-iso keep all-head routing.

This only writes a queue. It does not submit or start training. A can be reused
with the existing `--reuse-baseline 42=/absolute/path/to/A` option, provided
its data, backend, schedule and stopping step match. P5 is deliberately absent:
queue it separately only after judging P4-iso. Optional P4 and P6-iso also need
separate experiment selection. The historical default P1/P3 screen is unchanged.

Use the matching FA4 A baseline for the selected FA4 experiments. Dense-SDPA
results remain separate controls; do not mix their backend with FA4 comparisons. Reports check backend,
data fingerprints, recipe, target identity and module seed, and label these
token-matched comparisons exploratory. Time-matched A and initialization-seed
spread are still needed for the screen success criterion. For an additional
A initialization seed while holding data order fixed, use the direct recipe
with `seed=1042, data_seed=42`; the historical queue `--seeds` option changes
both seeds. Resume A's existing schedule for a time-matched continuation.

## Verification and remaining launch checks

All 44 selected CPU acceptance/regression tests passed (169.987 seconds).
A separate tightened custom-backward check also passed: maximum absolute
error relative to each reference tensor's maximum magnitude is at most 1e-6
in fp32 and 1% in BF16. Logs are in `temp/p4-p5-p6-acceptance.log` and
`temp/p4-gradient-tolerance.log`.

Local acceptance tests live in `tests/test_anticipatory_proxy.py`: placement,
identical backbone initialization, zero-gate logits, native Q/K and all-value
routing, document isolation including targets and stream, separate LM/auxiliary
gradients, custom backward versus recomputation in fp32/BF16, checkpoint replay,
P6 injection scale, real HF Trainer save/resume, reporting, queue generation,
two-rank CPU DDP and the optional compiled path. Resume tests compare final
weights and normalization buffers bitwise against uninterrupted runs.

`--proxy_compile_estimator true` compiles estimator, normalized injection and
cosine-loss functions without changing parameter/state names. CPU AOT-autograd
tests check the compiled graph and gradients; CUDA Inductor performance and
numerics still need a GPU smoke. Leave compilation off until that passes.

Architecture-only extra MACs/token are 6,291,456 for P4/P6 and 8,454,144 for
P5, plus elementwise operations. These figures exclude auxiliary backward,
targets, statistics and communication; they do not predict wall time.
Before any full screen, Section 6 still requires an actual B200 profile of
estimator forward/backward, auxiliary recomputation, targets, statistics and
their all-reduce, and the remaining model, together with seconds/update versus
A. Neither the 3% overhead target nor CUDA numerical acceptance is established
by these CPU tests. No B200 run is launched by this implementation.

Interpretation cautions: P1 alpha-one used its proxies but did not outperform A;
that does not prove estimator collapse or its cause. Target detachment defines
gradient routing, not a stability guarantee. P4-iso is a useful direct test of
prediction-only estimators; whether it helps remains an empirical question.

## Follow-up review

Fixed the initial packed-dataset shuffle to use `data_seed`, falling back to
`seed` when it is unset. Previously, `seed=1042, data_seed=42` still changed
that first shuffle, despite fixing the Trainer sampler seed. The new real
Trainer test records consumed batches and confirms identical data order with
different backbone initialization, and different order when the data seed
changes. Existing runs with equal seeds keep their ordering. Historical runs
with unequal seeds cannot silently resume with the corrected shuffle: their
saved dataset fingerprint must match, otherwise resume is refused.

Result comparison now checks the actual four-MLP target windows, quantity,
normalization, epsilon and cosine loss, and agreement between the recipe and
saved result. A matching `p4p6-r1` label alone is insufficient.

Additional BF16 checks backpropagate LM and auxiliary losses separately for all
five variants, with decoder/loss checkpointing and AOT compilation, comparing
against the detached-input recomputation reference. These supplement the
original combined-objective tests. The B200/CUDA smoke and throughput profile
remain pending; this review does not launch training.

Follow-up verification: all 46 selected CPU tests passed in 183.264 seconds.
Log: `temp/p4-review-regression.log`.

Four-head extension verification: all 48 selected CPU tests passed in 194.643
seconds (`temp/p4-four-head-regression.log`), including dedicated 16-query/8-KV
mapping and real Trainer resume tests. The latest paired-report assertion also
passed separately (`temp/p4-four-head-report.log`). No GPU training was launched.


## CUDA gradient-routing correction (2026-10-06)

The first full-Qwen B200 check found finite auxiliary loss but NaN gradients
in P4-4h's original shared two-output autograd node. Anomaly detection identified
`ScaledDotProductCudnnAttentionBackward0`; the detached-input recomputation
reference passed on the same weights/environment. This used **dense SDPA**, not
FA4. There is no evidence of a hardware fault.

Returning `None` for the input gradient from that shared node still allowed
PyTorch to visit upstream backward nodes. A small CPU regression demonstrates
one upstream visit with the old implementation and zero with the correction.
`BlockMLP` now calculates its forward once and shares saved activations between
two `_MLPPath` nodes. The auxiliary node takes an explicitly detached input,
which structurally disconnects it from the backbone. LM input gradients and both
parameter-gradient contributions are preserved. P4/P6 use this corrected path;
isolated arms and A/P1/P3 do not use the defective shared node. State-dict names,
architecture, initialization, target definitions and loss weights are unchanged.

All16 anticipatory CPU tests passed, including save/resume, AOT compilation,
BF16 and two-rank DDP; all22 selected legacy/validator tests passed. The first
post-fix CUDA retry stopped before its P4-4h auxiliary check on a finite LM
gradient difference of1.07026%, just over the1% bound. Consequently, that retry
did not yet validate the original failing case. The numerical probe now enables
deterministic algorithms and adds a reference-vs-itself control. The1% bound is
unchanged. These settings are confined to the diagnostic process; ordinary
Trainer smokes and production retain the established performance settings.
PyTorch documents that CUDA/cuDNN SDPA can select nondeterministic algorithms:
[SDPA documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention).
CUDA revalidation and full-size smoke results must be checked before launch.


### Completed B200 revalidation and profile

All 38 selected CPU tests and the controlled CUDA checks passed. Reference
repeat controls were exactly equal. P4-iso-4h losses and gradients matched the
recomputation reference exactly; P4-4h matched separately for LM and auxiliary
losses. Its combined gradient maximum relative difference was 0.315%, below
the unchanged 1% bound. Both document isolation checks passed.

Normal, nondeterministic dense cuDNN SDPA completed 25 full-size eight-GPU
updates for each arm. No math fallback; finite gradients, checkpoint state,
and all eight rank RNG files passed validation. Microbatch16/GAS4/2048,
full 28600-step schedule, warmup1430, no activation checkpointing.

| Arm | Median seconds/update | Overhead vs A | Peak allocated GiB |
|---|---:|---:|---:|
| A | 2.4275 | — | 110.62 |
| P4-iso-4h | 2.6136 | 7.67% | 117.13 |
| P4-4h | 2.6586 | 9.52% | 118.64 |

Times use unprofiled updates10–25. This short measurement is not a full-run
ETA or a scientific result. The 3% target is not met. Per-update exclusive
summed kernel durations from rank0 at update6 are shown below; overlapping
kernels mean these are not a decomposition of critical-path wall time.

| Component (milliseconds) | A | P4-iso-4h | P4-4h |
|---|---:|---:|---:|
| Estimator forward | 0 | 3.757 | 5.954 |
| Estimator backward | 0 | 4.088 | 16.889 |
| Auxiliary cosine forward + backward | 0 | 68.070 | 68.070 |
| Target/mask/statistics setup | 3.511 | 42.079 | 42.109 |
| Setup backward | 5.408 | 5.396 | 5.408 |
| Target normalization | 0 | 22.934 | 22.956 |
| Statistics all-reduce + normalization update | 0 | 0.192 | 0.189 |
| Remaining decoder forward + backward | 2108.431 | 2148.161 | 2178.330 |
| LM loss forward + backward | 278.097 | 278.163 | 278.160 |
| Other model/optimizer/communication | 24.018 | 24.634 | 24.627 |

No auxiliary forward recomputation is used. Targets/normalization/cosine work
costs substantially more than the small estimator itself. Production retains
the tested implementation; this profile does not justify changing the objective.
Supervision completed and restored communicating burns. Full evidence:
`temp/p4-fix-monitor-04.log`, SHA256
`479573f1b78f3c3ac8654df0aa420d7a2a624e98519df8b59daf0e361dde5893`.


### FA4 selection and validation scope (2026-10-06)

The operator clarified that FA4 remains the selected experimental backend;
the dense-SDPA wording in the specification was stale. The shared P4 attention
path already supports FA4 without changes to projections, loss, or model state.
`proxy_heads.b200.json` now selects it explicitly. The CLI still accepts SDPA
as a reference. Replacement runs use the completed FA4 A, fresh initialization,
and separate outputs; no backend change on resume is permitted.

CPU verification in the matching `sampling_b200` environment passed all13
FA4 proxy/baseline/validator tests, then the5 proxy/validator tests passed again
after adding P4 document-isolation, exact Trainer resume and FA4 checkpoint
validation cases. Logs: `temp/p4-fa4-cpu-tests-02.log` and
`temp/p4-fa4-cpu-tests-final.log`. The added16-query/8-KV test checks that only
two KV groups serve the four proxy query heads; separate LM/auxiliary/combined
routes agree with the independent per-document CPU oracle. No dense mask may
be constructed by the FA4 path. CPU tests use an oracle, not the CUDA kernel.
An initial invocation in the older dev `train_env` correctly failed the pinned
Transformers version guard; the environment was not modified to bypass it.

SDPA queue cancellation05373dd was verified: only supervisor143985 was sent
SIGTERM after identity checks; it cleaned its owned workers, verified all8GPUs
free, and restored communicating burns. Outputs and dataset caches preserved.
Evidence `temp/p4-fa4-stop-handoff.log`, SHA256
`3a5f3b6ce38469beb20f9d57439e898397094e1e459117f892c62124b898a4ee`.

CUDA check submission498955f uses `attention_bench` (FA4 4.0.0b33), fresh
`p4-fa4-checks-20261006-a01`, verified Accelerate and approved burn handoff.
It requires same-backend routed/reference comparisons, cross-backend comparisons
at gate1, and25-step full-size8GPU training/checkpoint/profile checks before
production. This paragraph records submission, not their eventual outcome.


### FA4 CUDA and full-size smoke results

All required numerical and training gates passed in
`p4-fa4-checks-20261006-a01` (source498955f). Full-Qwen FA4 routed/reference
loss differences were zero; the worst per-parameter maximum gradient relative
error was0.315%, below the unchanged1% threshold. Reference repeats were
exact except a3.8e-8 relative difference in one P4-4h LM gradient.

At identical weights and gate1, FA4 versus dense SDPA:

| Arm | Loss absolute difference | Overall gradient relative L2 | Proxy gradient relative L2 |
|---|---:|---:|---:|
| P4-iso-4h | 0.0000763 | 0.5073% | 0.3859% |
| P4-4h | 0.0000763 | 0.5142% | 0.4310% |

The existing cross-backend thresholds were unchanged: loss<0.01, gradient
relative L2<3%, hidden/logits relative L2<2%. Hidden/logit errors were0.546%/
0.616%; cross-document output and gradient leakage were exactly0. Parameters
and normalization buffers were unchanged by these probes.

All three normal HFTrainer smokes completed25updates, all8GPUs, BF16,
micro16/GAS4/2048. Validator checks weights, finite logged gradients, optimizer,
scheduler, all8rank RNG files, normalization/gates, actual FA4 Q/K/V backward
receipts, exact recipe/data, and peak allocated memory below160GiB.

| Arm | Seconds/update | Overhead vs FA4 A | Peak allocated GiB |
|---|---:|---:|---:|
| A | 2.0692 | 0.00% | 103.62 |
| P4-iso-4h | 2.2533 | 8.90% | 112.37 |
| P4-4h | 2.2979 | 11.05% | 113.88 |

The3% overhead target remains unmet. Exclusive summed kernel durations at
profiled update6 (milliseconds; not critical-path wall time):

| Component | A | P4-iso-4h | P4-4h |
|---|---:|---:|---:|
| Estimator forward | 0.000 | 4.086 | 5.949 |
| Estimator backward | 0.000 | 4.110 | 16.937 |
| Target/mask/statistics setup | 8.458 | 45.694 | 45.639 |
| Target normalization | 0.000 | 22.946 | 22.956 |
| Auxiliary cosine forward/backward | 0.000 | 68.038 | 68.044 |
| Statistics all-reduce/update | 0.000 | 0.183 | 0.189 |
| Remaining decoder forward/backward | 1748.092 | 1786.759 | 1815.933 |
| LM forward/backward | 277.996 | 278.073 | 278.058 |
| Other model/optimizer/communication | 26.174 | 26.596 | 26.827 |

Unprofiled updates10–25 determine wall time. No auxiliary forward recomputation.
These bounded checks support launching, not a claim of better research results
or identical long training trajectories. Local receipts:
`artifacts/p4-fa4-validation-20261006/`. Full retrieval:
`temp/p4-fa4-checks-monitor-04.log`, SHA256
`22ebbfe7eb9af176d116ebb2eec8fc0fcd239de69f843e82ccf2b95e95591ce6`.

Fresh production submission: `p4-fa4-four-head-2500-20261006-a01`, P4-iso-4h
then P4-4h,2500updates each, matching completed FA4 A. All scientific settings
are unchanged except the selected attention backend. Preflight also requires
the check supervisor's completed, verified burn handoff and unchanged model
source hashes. Full production completion is not yet claimed.


Production follow-up: launch565c695 verified at step82 by monitor0337db7.
P4-iso-4h uses all8GPUs, finite gradients and~2.25s/update. Actual FA4 receipts
record28forward and28each Q/K/V backward calls. P4-4h remains queued;
2500steps each, fresh starts, matching FA4 A. Completed checks and automatic
burn handoff were verified before production. No long-run result is claimed.
Evidence `temp/p4-fa4-production-monitor-02.log`, SHA256
`e9d6302e300ca22c7bf6e84053c72a5864f76bfb04c29d8ccb8d5a9a32de5ea7`.
