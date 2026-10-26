# Project notes

Initialized 26 October 2026 SGT at `/disk/thuat/efficient_llm` from
`/disk/thuat/new_project_template`. First joint-group diagnostic completed; matched learned-router pilot now approved.
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
CulturaX pool; see JOINT_GROUP_DIAGNOSTIC.md for hashes and selection. A training
recipe, deployable router and downstream suite are not selected.
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
Current burn source is this run's external source/resources/llm_pretrain_burn.py;
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
