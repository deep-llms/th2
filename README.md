# Stagewise widening / capacity allocation

English from-scratch Llama-family experiments from
`docs/Capacity_Allocation_Research_Project_v0.5_Final_Reviewed.md`.
Models B0, A128/A256/A512 and stagewise C/D, offline data preparation, and the
training CLI are implemented. Read [the experiment guide](docs/CAPACITY_EXPERIMENTS.md)
for architecture details, tests and usage. Real data revisions, optimizer choices,
and a remote training run have not been configured.

The project also retains the generic Git-triggered runner utilities below.
The confirmed development and th2 repositories are documented in `docs/GIT_PUSH.md`.
Code-only publication keeps `commands.sh` at `#0` and submits no workload.

## Configure infrastructure

1. Work in this project folder. Do not copy another project's credentials
   or populated `temp/` folder.
2. Read `docs/AGENT_GUIDE.md`, then configure `project.local.json` from
   `project.example.json`. Populate the project notes and current task.
3. Configure and verify the development and execution Git remotes separately
   using `docs/GIT_PUSH.md`. Leave `commands.sh` at `#0` until a job is ready.
4. Choose project dependencies. Use runner `#i` to install, `#d` to download,
   and `#1` to verify/process/run. Follow `docs/commands.md`.
5. Run the local tests and harmless CPU example below before adding real jobs.

## Local verification (no network or GPU use)

```bash
python3 -m unittest discover -s tests -p test_utilities.py -v
python3 run_experiments.py --config jobs.example.json --list
python3 run_experiments.py --config jobs.example.json --run-dir temp/example-run-001
```

The example writes and verifies one JSON artifact, then publishes
`temp/example-run-001/complete.json`. It refuses an existing run directory;
use a new run ID for a retry. It does not train, download, or use GPUs.

## Included utilities

| File | Purpose |
|---|---|
| `run_experiments.py` | Sequential arbitrary commands, logs, timeouts, required artifacts, success marker |
| `scripts/gpu_status.py` | Read-only physical GPU/PID inspection; optional fail-if-busy check |
| `scripts/verify_manifest.py` | SHA-256/size manifest creation and verification for local files |
| `scripts/dropbox_access.py` | Dev-machine non-recursive shared-folder listing and safe file download |
| `scripts/example_job.py` | Tiny CPU example; not a project training script |
| `envs/runtime.txt` | Minimal runner environment example |
| `resources/accelerate_config.example.yaml` | Optional, deliberately single-process/no-mixed-precision example |
| `resources/llm_pretrain_burn.py` | Optional PyTorch/NCCL GPU stress workload; see `docs/GPU_SAFETY.md` before use |

See `docs/JOBS.md` for the job manifest and `docs/DROPBOX_ACCESS.md` for
credentials and result retrieval. ML tests require `requirements-capacity.txt`;
run them with `python -m unittest discover -s tests -v` in the ML environment.
Do not copy old sparse-embedding benchmarks or defaults into this experiment.
