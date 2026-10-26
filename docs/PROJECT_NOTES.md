# Project notes

Initialized 26 October 2026 SGT at `/disk/thuat/efficient_llm` from
`/disk/thuat/new_project_template`. Research/design stage; architecture undecided.
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
and practical usefulness remain unresolved. This remains a design proposal with
no real-model evidence. The user has since authorized B200 experiments, with
migration required first.

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

Model, dataset, tokenizer, training recipe and benchmark suite are not selected.
Preserve the helper contracts below where applicable, and validate deliberate
changes required by the eventual design. Template examples do not select a model
or dataset for this research.

## Results and decisions

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
