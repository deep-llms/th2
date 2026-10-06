# P4/P5/P6 implementation

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

Placement is blocks 2, 4, …, 24 in the 28-block model. P4/P6 use independent
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

Every target is the detached sum of four consecutive MLP outputs starting at
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
