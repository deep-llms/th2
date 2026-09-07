# Git setup and push guide

## Current project mapping (2026-09-07)

The user authorized publishing this project to both repositories below, with
`commands.sh` in inactive `#0` mode. This does not authorize a training job.

| Role | Remote | URL | Branch |
|---|---|---|---|
| Development | `origin` | `git@github.com:nguyenhuuthuat09/stagewise_widening_transformer.git` | `main` |
| Execution | `runner` | `git@github-share:deep-llms/th2.git` | `main` |

Develop in `/disk/thuat/stagewise_widening_transformer`. The existing deployment
checkout is `/disk/thuat/th2_runner_clean_probe`; its remote is named `second`
and points to the same th2 repository. It is a separate repository, not a Git
worktree of Stagewise. Inspect/fetch it before each deployment.

The new development repository starts with Stagewise-only history. th2 retains
its historical commits. The initial deployment explicitly merges the new
Stagewise root into th2 and resolves `commands.sh` to the Stagewise `#0` version.
Verify matching **tree IDs**, not identical commit IDs. Do not merge th2's old
sparse-embedding history back into Stagewise development `main`.

Development authentication uses the already configured key with a scoped
override; th2 uses the existing `github-share` SSH alias:

```bash
GIT_SSH_COMMAND='ssh -i ~/.ssh/id_ed25519_thuat -o IdentitiesOnly=yes' git push origin main:main
```

No global SSH/key configuration changes are needed. Never use the old
`nguyenhuuthuat09/sparse_embedding` repository as this project's origin.
The generic workflow below applies with these confirmed destinations.

## Roles

| Role | Suggested remote | Branch | Meaning |
|---|---|---|---|
| Development | `origin` | `main` | Source code, documentation and tests |
| Execution | `runner` | operator-specified, usually `main` | A push can submit a job |

The current settings are recorded above. Some existing deployments
use remote names `second`/`third`; names alone do not establish their role.
One machine is not assumed; list each active target explicitly. Keep inactive
targets as history, never include them in an automatic push-all command.

## First setup (only after destinations are confirmed)

```bash
git init -b main
git remote add origin <DEVELOPMENT_GIT_URL>
# Add this only when a runner repository has been allocated:
git remote add runner <EXECUTION_GIT_URL>
git remote -v
git config user.name
git config user.email
```

The angle-bracket URLs are placeholders, not runnable shell values. Confirm
identity and SSH access with the operator; don't overwrite global Git or SSH
configuration. If a specific key is needed, scope it to that command:

```bash
GIT_SSH_COMMAND='ssh -i /absolute/path/to/approved_key -o IdentitiesOnly=yes' git fetch origin main
```

No key file or credentials belong in Git. On a new empty remote there may be
no `main` to fetch; distinguish that from authentication/network failure.

## Before every push

```bash
git status --short
git branch --show-current
git worktree list
git remote -v
git fetch origin main
git rev-list --left-right --count origin/main...HEAD
git log --oneline origin/main..HEAD
```

Use the intended remote/branch. A positive first count means remote commits
are missing locally. Inspect and integrate them; never force-push over them.
Preserve dirty work. Do not stash-and-forget newer docs/tests or blindly
overwrite a worktree to get a clean push. Back up and resolve overlaps.

Stage explicit paths, review the staged diff and ensure it contains no
secrets/data. Do not use `git add .` or `git add -A` in a research workspace.

```bash
git add <specific-reviewed-files>
git diff --cached --check
git diff --cached --stat
git diff --cached
git commit -m "Describe the actual change"
git push origin main:main
git ls-remote origin refs/heads/main
git rev-parse main
```

A successful development push does not deploy to the runner. Do not append
an execution push automatically. Verify GitHub's head matches the intended
commit; uncommitted files are not included in any push.

## Execution worktree

For an existing runner repository, fetch its head and select a clean worktree
that actually follows it. Do not choose a stale branch by its historical name.
Use an explicit project-owned path outside the main checkout when creating
a worktree; record the path and do not overwrite an existing directory.

```bash
git fetch runner main
git worktree list
git -C <runner-worktree> status --short
git -C <runner-worktree> rev-list --left-right --count runner/main...HEAD
```

If only behind, `git -C <runner-worktree> merge --ff-only runner/main` is
appropriate. If diverged, inspect both histories. Do not reset or force-push.
For a new execution repo, coordinate its initial branch/setup with the operator;
do not blindly merge unrelated pre-existing histories.

Bring approved shared code into the execution worktree using an explicit
merge/cherry-pick as appropriate. Keep shared work on development `main`,
not solely in a temporary execution branch. Before the execution push:

1. Read `commands.sh` in full. Inspect the complete pending commit delta.
2. For code-only synchronization, deliberately set line 1 to `#0`.
3. For execution, use the exact authorized action and unique job name.
4. Check shell syntax for `#1`, environment paths, output paths, GPU ownership,
   and the expected completion test. `#i`/`#d` are directives, not shell scripts.
5. Push **only** the authorized execution target and inspect its returned log.

```bash
git -C <runner-worktree> diff --check runner/main..HEAD
git -C <runner-worktree> log --oneline runner/main..HEAD
git -C <runner-worktree> show HEAD:commands.sh
git -C <runner-worktree> push runner HEAD:main
```

Even an unchanged executable `commands.sh` can be processed on a push; don't
re-push to refresh logs. Use Dropbox or a fresh `#2` request. Never put
`git push`, HF uploads, `scp`, `rsync` to an external host, or S3 uploads inside
runner commands. Results come back through the operator's `#2` mechanism.

## Multiple machines, when explicitly configured

Record a separate execution repository/branch, Dropbox label, machine identity
and job prefix for every node. Submit and verify targets individually rather
than pushing all remotes blindly. Never infer a second active node from an old
remote configuration. Keep shared experiment protocols versioned on `origin`.

## Credentials and history

`temp/`, `.env*` (except a placeholder `.env.example`), and `project.local.json`
are ignored. `docs/AGENT_GUIDE.md` and `docs/DROPBOX_ACCESS.md` are also local,
Git-ignored guides; copy them separately to new dev checkouts.
`docs/commands.md` and this guide are intentionally tracked.
`temp/INSTRUCTION.md` must never be committed or pushed. If a secret was ever
committed, stop distributing it and report it; revoke/rotate the credential
and coordinate any history cleanup. Deleting the current file is not enough.
