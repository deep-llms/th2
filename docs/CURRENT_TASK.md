# Current task

Status: B200 migration in progress on 26 October 2026 SGT;
initial architecture hypothesis drafted, no model implementation or training yet.
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

Project code was uploaded by 3bf2626. Its verification passed all 56 file hashes,
Python syntax, absence of old code and unchanged GPU PID/start identities, then
failed in supplementary burn-log parsing (key/value text mistaken for JSON).
This conclusion follows the traceback location and command control flow;
the job did not emit its final completion marker.

Corrected read-only verification 6be699f was acknowledged by the controller as
a03, but its newly exported log is named a02 and repeats the old JSON parser
traceback byte-for-byte. The corrected command has no such parsing call at that
line. Result export 7c7625c is submitted; its receipt remains pending.
Do not rerun cleanup. Do not claim full completion until deployment_result.json
or the matching DEPLOYMENT_COMPLETE record is retrieved. The remote commands.sh
is export-only (#2). Local commands.sh is restored to #0 in an unpublished
handoff commit; publish it after verifying the pending receipt. No new commands
should be submitted merely to retrigger the unacknowledged request.

The initial deployment de1207c was rejected by the controller scanner for a CPU
test's disabled-publication-option assertion. Removed that redundant assertion;
the stronger offline transport-blocking test still passes locally. Production
training code is unchanged. No environments or GPU workloads were modified.
This Git repository serves development and execution roles, so every push
requires command review. Current model/data choices still need verification.

At the final 26 October morning check, the controller acknowledged 6be699f/a03
but exported a new a02 log at 08:38:19 SGT, identical to the earlier failure.
This is a controller receipt/synchronization mismatch; the cause is unconfirmed.
Stop remote writes and preserve state until command delivery can be established.
No architecture experiments have been started.

## Next steps

1. Review the candidate's prior-work overlap, local-loss surrogate, router
   predictability and execution cost before selecting it.
2. If pursued, specify model, calibration/evaluation data and fixed group budget
   for the small diagnostic in FIRST_ARCHITECTURE_HYPOTHESIS.md.
3. Reuse helpers for the approved experiment; validate consistent prefill/decode
   and real sparse execution before claiming speedup.

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
