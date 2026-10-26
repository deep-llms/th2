# Project notes

Initialized 26 October 2026 SGT at `/disk/thuat/efficient_llm` from
`/disk/thuat/new_project_template`. Joint-group diagnostic and matched learned-router pilot completed.
Report times and ETAs in Singapore time (SGT, UTC+8).

## Purpose and success criteria

See [PROJECT_IDEA.md](PROJECT_IDEA.md). Target similar useful quality with lower
inference cost. Distinguish latency, throughput, compute, cache capacity and total
memory. A specific novelty claim, baseline and acceptance criteria remain to be
defined. Normal fixed layer order is allowed; B-DOT routing and proxy mechanisms
are not requirements of this project.

First candidate: [joint attention-group selection](FIRST_ARCHITECTURE_HYPOTHESIS.md).
It asks whether subsets chosen to preserve the combined attention output are
better router targets than independent magnitudes. Frozen pretrained backbone,
dense KV writes, per-token group reads; no cache-size claim. Prior-work overlap
and practical usefulness remain unresolved. The first real-model diagnostic is complete; see JOINT_GROUP_RESULTS_20261026.md.
The user authorized B200 implementation and experiments after the verified migration.

## Infrastructure

Development folder: `/disk/thuat/efficient_llm`. Authorized execution repository:
`deep-llms/th2`, branch main, SSH alias github-share. Reuse the existing 6Y3O
B200 node; verify live identity before mutations. Migration checkout:
`/disk/thuat/efficient_llm-th2`. Remote code root: `/mnt/local/deep-llms_th2`.
Data/models/outputs remain under their separate `/mnt/local/_data`, `_models`,
and `_outputs` roots for deep-llms_th2; environments must be preserved.

Reuse local Dropbox credentials and the 6Y3O folder map under ignored DOT/temp
without committing their contents. No operator INSTRUCTION.md was found locally;
this is the same runner, using the confirmed commands.md protocol and prior
successful th2 commands (4831f9a inventory; 1676c12 guarded cleanup).
Keep credentials, private shared URLs and operator instructions outside Git.

Migration on 26 October: f009def reduced the GitHub tree to commands.sh while
preserving history; de02a34 completed inventory after correcting a logging-name
error. Guarded cleanup 2fca185 archived 419 old code files with SHA256 verification
at 08:16:55 SGT. Backup:
`/mnt/local/_outputs/deep-llms_th2/6y3o-efficient-migration-20261026-a01/archived_code`.
Only commands.sh and the runner's four metadata/log entries remained in the code
root. No process was signaled; workers 581907–581914 remained owned by burn
launcher 581838, running from an older immutable output snapshot. External
data/model/environment/output directory identities were verified unchanged.
Local raw receipts: `temp/th2_migration_20261026/`.

The development folder is now a Git repository following th2/main, preserving
the prior history. It serves both development and execution roles; never push
an unchanged executable commands.sh. The migration checkout remains separate.

## Data and reproducibility

The first diagnostic uses pretrained Qwen3-0.6B-Base and the pinned English
CulturaX pool; see JOINT_GROUP_DIAGNOSTIC.md for hashes and selection. The matched-router recipe is now implemented in ROUTER_PILOT.md. Backbone
adaptation, efficient sparse execution and a downstream suite are not selected.
Preserve the helper contracts below where applicable, and validate deliberate
changes required by the eventual design. Template examples do not select a model
or dataset for this research.

## Results and decisions

26 October first diagnostic: use full pretrained Qwen3-0.6B-Base, 6/8 and 4/8
GQA groups, frozen backbone and small disjoint English prefix probe. See
JOINT_GROUP_DIAGNOSTIC.md. Dense SDPA execution with output masking isolates
quality; no speedup or cache-size claim. Reused the old verified GPU supervisor
unchanged, with its ownership/failure tests. The earlier model directory held
only configuration/tokenizer, so acquire the exact pretrained weights through
controller #d and verify their published hash before GPU execution.

| Date (SGT) | Work | Evidence | Result | Decision |
|---|---|---|---|---|
| 2026-10-26 | Project initialization | CURRENT_TASK.md; PROJECT_IDEA.md | Template copied; helper code unchanged | Continue design here; no experiments yet |
| 2026-10-26 | Candidate design | FIRST_ARCHITECTURE_HYPOTHESIS.md | Joint-selection algebra checked on two synthetic fixtures; close priors documented | Review before real-model diagnostics; no training launched |

## Baseline contracts inherited from the template

- Native HF Trainer/Accelerate; default isolated SDPA, optional pinned FA4.
  Full-attention Qwen3/default RoPE is the validated isolated model path.
- Append a document-end token, reset positions per fragment, mask boundary
  targets; tokenization/packing use HF caches with per-map-batch remainders.
- Keep tokenizer, inputs, preprocessing workers/map batch size and seeds fixed
  for comparisons. Native loss is normalized over valid targets across ranks
  and gradient accumulation. Validate custom losses before enabling them.
- max_steps specifies the full schedule; stop_after specifies an absolute cutoff.
  Supply only one warmup option. Tested Transformers 5.9 prioritizes a supplied
  warmup_ratio over warmup_steps, including ratio=0. Prefer explicit warmup_steps.
- Keep all checkpoints by leaving save_total_limit unset. Native resume restores
  optimizer/scheduler/RNG/data skip under strict recipe and source/runtime checks.
  Changed code/configuration requires revalidation, not a silent guard bypass.
- Periodic and standalone validation use the same token-weighted packed loss.
  Validation preserves training RNG and module modes. Downstream task evaluation
  and custom architectures are new-project work.
- All assets are local, reporting defaults to none, HF network access is offline.
  commands.sh defaults to #0; GPU availability and execution authority are explicit.

See TRAINING.md for details and VALIDATION.md for historical baseline evidence.
These contracts do not select this project's scientific hypothesis or recipe.

## Backup and handoff

Record where source, small results and large checkpoints are retained and what
remains on temporary storage. Keep CURRENT_TASK.md concise and up to date.

## Deployment handoff, 26 October 2026 SGT

Deployment verified by 6ea92de/a04 at **09:05:45 SGT**: 56 source hashes matched,
old DOT code was absent, and all eight existing burn PID/start identities were
unchanged. The log's command SHA256 matches the pushed revision. The result file
was written under the migration output directory. See
[the migration record](results/efficient_llm_migration_20261026.json) and local
`temp/th2_migration_20261026/deployment04.log`. commands.sh is restored to #0.

Earlier verification had a burn-log format error before writing its result.
The export therefore failed because that result file did not exist. An a03/a02
receipt mismatch remains unexplained historically; the fresh a04 receipt verifies
current command delivery. Do not repeat cleanup or assume experiments launched.
The sole test-code change removes an upstream default-option assertion rejected
by the source scanner; the offline network-transport protection test remains and
passed locally. Production training code and installed environments were unchanged.

## First real-model result, 26 October 2026 SGT

[Report](JOINT_GROUP_RESULTS_20261026.md): 128 held-out long-document prefixes.
Dense PPL 17.7753; at 6/8 groups, joint 20.7038, projected norms 21.2005, raw
norms 21.7584, static 26.7351. Joint improves over projected norms on all 128
documents (−0.023705 nats/token; paired 95% CI [−0.025049, −0.022412]), but
remains worse than dense. At 4/8, joint PPL 55.8841: too much quality loss for
the present frozen model. These policies compute full attention and use its
outputs; they are not cheap learned routers and provide no efficiency evidence.

Run 6y3o-efficient-joint-groups-20261026-a01, launch 1f786de, finished 09:29:27 SGT.
59 local tests plus the pretrained GPU gate passed. Burns automatically restored
and communication verified 09:30:38 SGT; observed healthy 09:32:02 SGT, guard released.
At this diagnostic handoff, the burn source was its external source/resources/llm_pretrain_burn.py;
workers 590625–590632, launcher 590556 at that observation. Reinspect identities
before any future stop. Downloaded results/provenance are in
docs/results/joint_groups_20261026/. commands.sh is #0; no further job queued.

## Matched-router pilot approved, 26 October 2026 SGT

See ROUTER_PILOT.md. Two linear banks with identical initialization, data and
HF Trainer recipe; joint versus independent post-projection reconstruction
cost targets are the only controlled difference. Frozen backbone, 6/8 groups,
2M teacher context tokens, 512 router updates. No global clipping couples the
banks. Use 256 new validation prefixes excluding the earlier 128; no held-out
target scaling/tuning. A separate native SDPA cached-decode profile tests runtime
headroom. Dense masked execution establishes quality only, not sparse speed.
No backbone adaptation or further methods are queued automatically.

## Matched-router result, 26 October 2026 SGT

[Report](ROUTER_PILOT_RESULTS_20261026.md). Launch 5554c61, run
6y3o-efficient-router-pilot-20261026-a01. All jobs completed 15:56:08 SGT;
burns restored/verified 15:57:19, independently healthy 16:00:18, guard released.
Current workers at that observation: 594491–594498, launcher 594422, burn source
under this run's immutable source/. This supersedes the previous burn record.

New 256-document probe: learned joint PPL 23.4644 versus learned independent
24.1605 and dense 18.9674. Joint wins on 250 documents, paired NLL delta
−0.029232 [−0.032145, −0.026263]. Joint supervision has a repeatable document-level
signal within this seed/probe; dense-quality parity remains absent. Cached-decode
profile suggests only 4–15% idealized headroom in this eager SDPA setup, before
router/dispatch costs. No sparse speedup, adaptation benefit or downstream gain
has been demonstrated. No additional training or kernel work is queued.

67 local tests and real-model GPU gates passed. Native 256/512 checkpoints have
optimizer and scheduler state. All ten exported files verified by SHA256, with
raw data/compact learned-router weights in temp/router_results_20261026 and
small durable reports in docs/results/router_pilot_20261026. Restore commands.sh
to #0 after archival. Do not compare these PPL values directly to the previous
128-document probe. Continue only after reviewing quality/cost tradeoffs.

## Context-budget screen authorized, 26 October 2026 SGT

Latest user requests more research and B200 tests. See CONTEXT_BUDGET_PLAN_20261026.md.
Preserve every head's local attention, allocate half the GQA groups or layers to
global context. Matched joint/independent full-minus-local reconstruction teachers;
frozen backbone, reuse the proven native Trainer recipe. Compare fixed groups,
fixed layers, periodic layers, learned routers, oracle, dense and local-only.
Explicitly acknowledge DuoAttention/All-or-Here/On-Demand/Switch Attention overlap.
The 2k probe is reused development data; separate 8k prefixes test longer context,
with overlap reported. No backbone adaptation, driver or environment changes.
Static policies execute sliced attention for actual cached-decode timing; dynamic
references compute both branches. Full KV retained: no capacity-saving claim.

## Context-budget result, 26 October 2026 SGT

[Results](CONTEXT_BUDGET_RESULTS_20261026.md). Frozen original Qwen3, half the GQA
groups global; the rest keep sink+recent local attention. Learned joint PPL 19.1899
versus dense 18.9674 on the identical 2k development probe (+1.17%); 15.3083 versus
14.9860 on 32 long 8k prefixes (+2.15%). This is much better than whole-head removal,
but is not an equal-latency comparison. Independent router 19.2066/15.3270: joint
supervision has a small extra gain, not a strong standalone novelty claim.

Actual static grouped execution is 1.142× faster at batch8/context8k, with 3.52%
higher PPL, and slower in the smaller cases. Whole-layer allocation is 1.560× faster
there but has 11.45% higher PPL. Learned policy still computes both branches for
quality; no learned speed or KV capacity saving is demonstrated. Next useful gate
is efficient execution and retrieval preservation, not a large training sweep.
Closest existing local/global methods are explicitly acknowledged in the plan.

Source/launch 81ef234, run 6y3o-efficient-context-budget-20261026-a01; all jobs zero
exit by 19:03:33 SGT. Burns restored/verified 19:04:39 and independently healthy
19:09:24, guard released. Workers 597076–597083, launcher 597007 at observation;
burn source under this run's immutable source/resources/llm_pretrain_burn.py.
Reinspect before stopping. This supersedes earlier burn ownership records.

74 local tests + GPU gates passed. Thirteen exported files independently SHA256
verified. Raw results/router weights in temp/context_results_20261026; durable
reports in docs/results/context_budget_20261026. Short probe byte-identical to prior
pilot; 23 of the 32 long validation documents also occur in short validation.
No fitting/validation document overlap. commands.sh #0, no further queue.

## Selective execution screen authorized, 26 October 2026 SGT

The user's “keep going” approves the next execution/retrieval gate. See
SELECTIVE_DECODE_PLAN_20261026.md. Reuse frozen fitted routers and Qwen3; no training.
Inference-only split-K kernel reads native full/local KV without historical gathers;
prefill remains the reference. Dense with the same kernel is a required control.
A fixed balanced eight-choice record-lookup probe includes dense/near controls;
failed positive controls mean inconclusive retrieval evidence. All KV remains stored.
Use the unchanged supervisor and preserve prior outputs/environment. Fix cache-owner
identity using a retained object token; a Python numeric id may be recycled.

## Selective execution and retrieval results, 26 October 2026 SGT

[Report](SELECTIVE_DECODE_RESULTS_20261026.md). Kernel compiled FP32/BF16 math,
masking and poisoned-cache tests passed. BF16 whole-model native loss tolerance
failed; do not hide that failure or call experimental timings backend acceptance.
Eight-document controls recover agreement with FP32 compute (max 2.87e-5), while
BF16 hard choices differ 1.80%, mean joint loss difference +.000420 nats. Native
SDPA remains default. No production/long-generation equivalence claim.

Batch 8/context 8k: native dense 32.72 ms, selective joint 24.26 ms, same-kernel dense
20.74 ms. Kernel improvement accounts for the apparent native speedup; routing
still costs 16.9% versus dense same-kernel. Smaller cases also lose to native.
Reference-only synthetic retrieval: dense/fixed/joint/independent 128/128 each,
local 45/128 (32 near + 13 distant). Half-global allocation retains this task, but the
probe is saturated and does not establish a learned-router benefit. No KV saving.
Do not scale router training now; establish headroom against a strong dense
implementation in an attention-bound workload first.

Three immutable runs preserve failure and follow-up evidence: initial 9481bee,
numerics 25dc043, final measurement 8adb821. Latest root 6y3o-efficient-selective-research-20261026-a01.
Jobs completed 23:52:48 SGT; burns verified 23:53:59 and independently 23:54:51, guard
released. Workers 602209–602216, launcher 602140, burn source under latest run/source;
reinspect before stopping. Supersedes previous burn ownership. Eight final exports
verified by SHA256 (export b48013a), 48 timing records/640 lookup scores/128 frozen cases;
80 local tests across maintained suites. Raw archive temp/research_results_20261026,
durable docs/results/selective_decode_20261026. commands.sh #0; no further queue.

## Sink/window and compact scheduling, 27 October 2026 SGT

[Report](WINDOW_COMPACT_RESULTS_20261027.md). User authorized a specific mask and
scheduling ablation, not more training. Native SDPA on 256 × 2k documents: dense
18.9668 PPL, fixed half-global first-four+recent-128 19.3570, same groups with
recent-128 only 75.5163. Every document worsens; paired delta +1.361293 nats,
CI [1.330133, 1.393875]. Keep sinks for this frozen pretrained policy; this does
not condemn models trained with sliding windows or establish four as optimal.

Added opt-in triton_compact: no empty local tiles, direct local output, global-only
partial reduction. 82 local tests and 60 compiled attention cases passed; FP32
whole-model errors <=6.25e-5. Original kernel retained, native SDPA default unchanged.
BF16 whole-model original/compact comparisons still contain >.01-nat document
differences; prior native gate remains failed. No BF16 backend acceptance.

B8/8k attention-only graph microbenchmark: original fixed 51.50 us, compact fixed
48.57 us; zero/four sinks nearly identical. Whole-model eager: dense original
22.346 ms, fixed original 22.489, compact fixed 22.656, original joint 26.179,
compact joint 27.160. Scheduling helps larger attention kernels, but not whole-model
latency. Four extra keys are not the primary custom-kernel cost in these cases.
Do not promote compact to default or start router training from this result.

Run 6y3o-efficient-window-compact-20261027-a01, source 36e30fa, health b10b860,
export 4cf8964. All jobs zero exit 01:19:46; automatic burns verified 01:20:56 and
independently healthy 01:21:17 SGT. Eight ready ranks, collective sum36, advancing
40 cycles/44.41 GiB per rank; guard released. Workers 604193–604200 under launcher
604124 at observation; source latest run/source/resources/llm_pretrain_burn.py.
Reinspect identities before later stops. Supersedes all previous ownership records.

Eleven exports SHA256-verified against node receipts, exact source and prior fit
identity checked. 88 timing rows, 48 microbenchmark rows, 256 paired documents;
raw temp/window_results_20261027, durable docs/results/window_compact_20261027.
commands.sh #0; no new job queued. Environments/drivers/old outputs unchanged.
