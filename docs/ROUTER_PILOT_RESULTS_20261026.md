# Learned-router pilot results — 26 October 2026

The learned joint router beats the capacity-matched independent router on this
held-out probe. It still trails dense Qwen substantially. The current decode
profile suggests modest potential savings, so these results do not establish an
efficient replacement for dense attention.

Protocol: [ROUTER_PILOT.md](ROUTER_PILOT.md). Original 28-layer pretrained
Qwen3-0.6B-Base, frozen backbone, six of eight GQA groups retained at every layer.
Both routers are identical linear classifiers initialized with seed 42; only
their target cost changes. Every policy keeps all native K/V groups in cache.

Training uses 1,024 English document prefixes (2,097,152 teacher input tokens),
131,072 sampled feature positions, 512 native HF Trainer updates on eight B200s.
The new evaluation set has 256 documents and 524,288 next-token targets, disjoint
from both training and the earlier 128-document diagnostic. PPL values therefore
must not be compared directly with the earlier report's PPL values.

## Held-out language modeling

| Policy | NLL (nats/token) | PPL ↓ |
|---|---:|---:|
| Dense original model | 2.942724 | 18.9674 |
| Static six groups | 3.380993 | 29.4000 |
| Raw-norm oracle | 3.140139 | 23.1071 |
| Projected-norm oracle | 3.114769 | 22.5282 |
| Joint reconstruction oracle | 3.091544 | 22.0110 |
| Learned joint router | 3.155486 | 23.4644 |
| Learned independent router | 3.184718 | 24.1605 |

Joint versus independent learned router: **−0.029232 nats/token**, paired
95% document-bootstrap interval **[−0.032145, −0.026263]**. Joint improves on
250/256 documents and lowers PPL by 2.88%. This is a consistent advantage on
one fixed probe/seed, not evidence of cross-seed or downstream generalization.

Joint router versus dense: +0.212763 nats/token, **23.71% higher PPL**. It also
trails the joint oracle by 0.063943 nats/token. Oracle policies see actual
attention outputs and are not cheap deployable routers. The learned router
makes its choice before attention; it receives no teacher outputs at inference.

## Did the routers learn?

Training cross-entropy falls from 4.55484 to 2.89571 for joint, and 4.55154 to
2.90147 for independent. Target entropies are 2.75922 and 2.76679, respectively.
Training top-1 target agreement rises from about 3.36% to 49.24%/49.08%.

On each router's own held-out trajectory, averaging per-layer normalized
reconstruction metrics:

| Router | Error / dense-output energy | Excess over local joint oracle | Joint-oracle top-1 agreement |
|---|---:|---:|---:|
| Joint | 0.06701 | 0.01882 | 45.22% |
| Independent | 0.06842 | 0.02007 | 42.93% |

These averages weight layers equally; they are not ratios of global sums.
Each layer has 32,768 held-out sampled positions. The local reference differs
between policies because their preceding routed hidden states differ. A high
classification accuracy is not sufficient to establish language-model quality.

Teacher collection plus initial metrics took 6.99 seconds; the cached-feature
Trainer loop took 4.85 seconds. These are small linear-router updates, **not**
512 full-Qwen LM training steps. The whole fit/evaluation subprocess took about
54 seconds, including loading and all seven held-out policies.

## Dense cached decoding

Native SDPA, BF16, eager HF StaticCache, no compilation/CUDA graphs. Two GPU
replicates per case, three uninstrumented repetitions each. Times cover a whole
model decode step, including logits; prefill is excluded. Values below show the
range across device replicates, not a statistical confidence interval.

| Batch | Context | ms/decode step | Instrumented SDPA-core share | Idealized six-group estimate |
|---:|---:|---:|---:|---:|
| 1 | 2,048 | 21.46–23.39 | 21.14–21.97% | 1.056–1.058× |
| 8 | 2,048 | 21.03–21.50 | 15.87–16.31% | 1.041–1.043× |
| 1 | 8,192 | 21.35–21.98 | 49.48–50.91% | 1.141–1.146× |
| 8 | 8,192 | 32.73–32.78 | 35.38–35.41% | 1.097–1.097× |

The last column uses `1/(1−0.25×core_share)`: roughly **4–15% potential throughput
increase**, before router/dispatch overhead. It assumes linear core-time scaling;
it is not achieved sparse performance or a universal bound. Core timing uses a
separate CUDA-event pass, whose hooks add overhead. Eager launch/CPU behavior and
backend choice can change this picture in an optimized serving engine. No sparse
kernel was implemented or timed; KV storage is unchanged.

## Correctness and operation

- 67 local tests passed, including a real two-rank HF Trainer fit and loss-bank
  independence, data exclusion, causal/cache checks and profiler passthrough.
- B200 GPU checks passed: frozen backbone, finite router gradients, exact export
  reload and causal prefixes. Trained-router FP32 full/cached maximum absolute
  logit differences were 0.000142 (joint) and 0.000273 (independent), within the
  declared tolerances. BF16 causal-prefix identity passed; these checks do not
  assert bitwise BF16 full-versus-cached equivalence.
- Native checkpoints at steps 256 and 512 both retain model, optimizer and
  scheduler state. No resume experiment was needed or claimed.
- Accelerate config was copied to the actual cache; `accelerate env` verified
  eight GPUs/BF16. Only verified previous burn workers were stopped; all eight
  GPUs were confirmed free before the correctness gate.
- All three jobs exited zero by **15:56:08 SGT**. Communicating burns were
  restored and verified by **15:57:19 SGT**, then independently observed healthy
  at **16:00:18 SGT**, with advancing collective payload and the guard released.
- Environments, drivers and pretrained checkpoint files were not changed.

## Artifacts and recommendation

Launch/source commit `5554c61`; read-only verification `d8e96c1`; export `fd97c19`.
Remote root: `/mnt/local/_outputs/deep-llms_th2/6y3o-efficient-router-pilot-20261026-a01`.
Raw local downloads, including the SHA256-verified 6.4 MB `routers.pt`:
`/disk/thuat/efficient_llm/temp/router_results_20261026/`.

Archived [evaluation](results/router_pilot_20261026/evaluation.json),
[training summary](results/router_pilot_20261026/training-summary.json),
[decode measurements](results/router_pilot_20261026/decode-profile.json),
[probe manifest](results/router_pilot_20261026/probe-manifest.json), and
[artifact hashes](results/router_pilot_20261026/artifact-hashes.json).
The training summary selects relevant Trainer fields; its original raw record
and checksum are referenced explicitly. The evaluation, profile, probe, source
and handoff records match the remote files byte-for-byte. Full checkpoints and teacher features remain on B200.

Keep the joint target as a supported hypothesis. Practical efficiency remains
unproven: the dense-quality gap is large and measured eager-decode headroom is
modest. Review those tradeoffs before investing in backbone adaptation, a sparse
kernel, or long training. No further experiment is queued; commands.sh returns
to inactive #0.
