# Project notes

Status: unconfigured general template. No inherited experiments or results.

## Purpose and success criteria

Describe the actual problem, working hypothesis, baseline and decision criteria.

## Infrastructure

Record non-sensitive repository/branch roles, active machines, GPU allocation,
runtime paths, environment versions, and Dropbox labels. Keep credentials,
private folder URLs and operator instructions in ignored local files.

## Reproducibility

Record input revisions/manifests, preprocessing, seeds, configuration, resource
budget, metrics and validation criteria. For ML projects, distinguish training
from held-out results, document batch/precision/scheduler settings, and retain
checkpoint/evaluation provenance. Do not claim speed from tiny smoke tests.

## Results and decisions

| Date (UTC) | Run / commit | Evidence location | Result | Decision |
|---|---|---|---|---|

## Operational lessons

Record confirmed failure causes and validated recovery procedures; distinguish
project exceptions from runner/infrastructure failures. Avoid live-status claims
without a timestamp. Do not treat a failed run as successful because a PID ended.

## Backup / portability

List which small result/config artifacts are retained locally, which large
artifacts remain on temporary remote storage, and the authorized backup method.
Keep shared code and docs on the development repository rather than only in a
temporary execution worktree. Track project guides except the local-only
`AGENT_GUIDE.md` and `DROPBOX_ACCESS.md`; never track populated
credentials or `temp/INSTRUCTION.md`.
