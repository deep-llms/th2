# Current task

Status: B200 migration verified on 26 October 2026 at 09:05:45 SGT.
Initial architecture hypothesis drafted; no model implementation or training yet.
Working folder: `/disk/thuat/efficient_llm`.

## Objective

Idea document: [PROJECT_IDEA.md](PROJECT_IDEA.md), summarizing the user's direction
and our discussion. A detailed architecture specification has not been selected.

Investigate an efficient LLM architecture that preserves useful model quality
while reducing inference latency, compute or KV-cache memory. This is a separate
project from dynamic layer ordering/B-DOT and deep-to-shallow proxies.

Background: [literature review](EFFICIENT_LLM_RESEARCH_20261026.md).
Candidate: [joint attention-group selection](FIRST_ARCHITECTURE_HYPOTHESIS.md).
It targets decode work at fixed active-group count using joint post-projection
reconstruction targets. It retains dense KV writes; it does not claim cache
capacity savings or established novelty.

## Scope and authorization

User authorized using the existing th2 B200 for experiments, first replacing
the GitHub tree with only commands.sh, cleaning the old remote project code,
and uploading this project. Execution target: deep-llms/th2 main; existing
6Y3O node. Migration checkout: /disk/thuat/efficient_llm-th2. Historical DOT
source/checkpoints/results must remain recoverable. Data, models, environments,
runner metadata and live workload sources are outside the code-cleanup scope.

Minimal-tree commits f009def/de02a34 inspected /mnt/local/deep-llms_th2. The first
inventory had a logging-name error; the corrected read-only inventory completed.
Cleanup commit 2fca185 completed at 08:16:55 SGT, preserving runner metadata and
all eight burn workers. It archived 419 files to
/mnt/local/_outputs/deep-llms_th2/6y3o-efficient-migration-20261026-a01/archived_code.
External data/model/environment/output directory identities were unchanged.

Deployment commit 6ea92de completed on the expected node. Its a04 log contains
the matching job marker, expected commands.sh SHA256 and DEPLOYMENT_COMPLETE:
all 56 source hashes matched, old DOT code was absent, and all eight burn worker
PID/start identities were unchanged. It wrote deployment_result.json under the
migration output directory. Raw receipt: temp/th2_migration_20261026/deployment04.log.
Tracked evidence: docs/results/efficient_llm_migration_20261026.json.
commands.sh is restored to #0. Do not rerun cleanup.

The initial deployment de1207c was rejected by the controller scanner for a CPU
test's disabled-publication-option assertion. Removed that redundant assertion;
the stronger offline transport-blocking test still passes locally. Production
training code is unchanged. No environments or GPU workloads were modified.
This Git repository serves development and execution roles, so every push
requires command review. Current model/data choices still need verification.

Earlier verification failed by parsing burn key/value text as JSON, so no result
file was written. The subsequent export failures explicitly reported a missing
file, not a missing project folder. An a03/a02 log mismatch was also observed;
its cause remains unknown, but the fresh a04 receipt establishes correct command
delivery and deployment. No architecture experiments have been started.

## Next steps

1. Specify model, calibration/evaluation data and fixed group budget for the
   small diagnostic in FIRST_ARCHITECTURE_HYPOTHESIS.md, using inspected local
   B200 assets where suitable. Available model directory:
   /mnt/local/_models/deep-llms_th2/Qwen3-0.6B-Base_da87bfb6.
2. Implement and test the diagnostic before GPU execution; reuse helpers and
   evaluate the local-loss surrogate on the altered model's own causal states.
3. Follow GPU ownership checks for any launch. Current burns were preserved,
   not stopped; no model/data/environment compatibility has been freshly tested.

## Current evidence and handoff

The initial copy was checked byte-for-byte against all 55 template files before
editing project documentation. Training/helper code remains unchanged.
Two synthetic four-group fixtures verify the reconstruction objective and show
that magnitude ranking can be suboptimal. They are algebra checks, not model
results. New prior-work checks identified AHA/Switch Attention/ODA as precedents
for local/global switching. No real-model results exist for this project.
Template-only historical test evidence
in VALIDATION.md does not validate an unimplemented new architecture or describe
current remote machine status.
