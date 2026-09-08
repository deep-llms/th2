# Instructions for coding agents

Read `docs/AGENT_GUIDE.md`, `docs/commands.md`, and `docs/GIT_PUSH.md` before
working on runner commands or deployment. Read `docs/CURRENT_TASK.md` for
current scope and `docs/PROJECT_NOTES.md` for durable decisions.

This project implements the Qwen3 six-layer English capacity-allocation pilot;
see `docs/CAPACITY_EXPERIMENTS.md`. The GPT-2/Llama pilot is superseded.
Reuse sparse-embedding's sampled HF text and batched multiprocess Dataset.map
workflow. Architecture changes belong in `capacity_allocation/modeling.py`.
Implementation is not a live run request.
Do not infer a GitHub repository, GPU machine, data revision, or credentials.
Do not push to an execution remote without checking the command and authority.
Default `commands.sh` is inactive (`#0`). Keep credentials and the
operator-provided `temp/INSTRUCTION.md` out of Git.

Do not kill unknown GPU processes or arbitrary process groups. Use read-only
GPU inspection first; explicit stopping of a known workload requires
authorization. Never match the runner burn path with `pkill -f`.

Do not spawn sub-agents unless the user explicitly asks for delegation.
