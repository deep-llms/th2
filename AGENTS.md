# Instructions for coding agents

Read `docs/AGENT_GUIDE.md`, `docs/commands.md`, and `docs/GIT_PUSH.md` before
working on runner commands or deployment. Read `docs/CURRENT_TASK.md` for
current scope and `docs/PROJECT_NOTES.md` for durable decisions.

This project starts with reusable helper code and a tested training baseline.
Read the user's project idea/design document before implementing the method;
record its path in docs/CURRENT_TASK.md. Follow that document and the user's
latest instructions for the research objective, architecture and experiments.
Reuse existing code where applicable. Keep working infrastructure unchanged
unless the new idea requires adapting it; validate any changed behavior.
Example models, datasets and recipes do not define the new project's scope.

Do not infer a GitHub repository, GPU machine, dataset, model, or credentials.
Do not push to an execution remote without checking the command and authority.
Default `commands.sh` is inactive (`#0`). Keep credentials and the
operator-provided `temp/INSTRUCTION.md` out of Git.

Do not kill unknown GPU processes or arbitrary process groups. Use read-only
GPU inspection first; explicit stopping of a known workload requires
authorization. Never match the runner burn path with `pkill -f`.

Do not spawn sub-agents unless the user explicitly asks for delegation.
