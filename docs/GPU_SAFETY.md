# GPU ownership and safe handoffs

This template does not claim any GPU allocation. Configure physical indices
for each machine. Never use utilization or allocated memory alone to decide
who owns a process or whether a job completed successfully.

## Read-only preflight

```bash
python3 scripts/gpu_status.py
python3 scripts/gpu_status.py --gpus 0 1 --require-free
```

Replace indices with the actual allocation. The helper checks physical GPU
UUID/PID mappings and returns nonzero for query failures, unknown layouts, or
occupied selected GPUs. It never kills a process. It is not an atomic scheduler
lock: coordinate with the operator and recheck immediately before launching.
Unsupported MIG layouts fail closed instead of reporting the machine free.

Before stopping an existing workload, confirm the user authorized that exact
stop. Identify its job name, launcher, start time, command and descendants,
and map its worker PIDs to physical GPUs. The generic job runner may terminate
only a child session it created itself on timeout/cancellation; it never
derives a process group to kill from `nvidia-smi` output.

## Reclaim an operator-provided GPU burn

The known runner may provide `/tmp/llm_pretrain_burn.py` (singular). Verify it
exists and inspect/hash the file before assuming its behavior. Different
versions at the same path have used very different memory/collective payloads.
Use a burn only when explicitly authorized or required by operator policy.

The template includes an unchanged copy of the reviewed enhanced version at
`resources/llm_pretrain_burn.py`, SHA256
`2b32968798e2200a8148a3395f1d37ae06e92b6340a74a2f192bfe1a48bcf174`.
It is not necessarily the same as the runner's `/tmp/llm_pretrain_burn.py`.
Do not overwrite the runner-provided file merely to launch this copy.

### Enhanced version: defaults and limits

- Requires a compatible CUDA-enabled PyTorch with NCCL, and at least two
  visible GPUs by default. It self-spawns one worker per visible GPU; do not
  wrap it in `torchrun` or `accelerate launch`.
- Targets **85% total device-memory usage**, including existing device usage,
  within 256 MiB. Start only on verified free GPUs. Most of this occupancy is
  a retained, initialized allocation, not active model/optimizer state.
- Keeps at least **8 GiB free**. Consequently the default 85% target requires
  more than about 53.3 GiB total memory. It refuses a 40 GB A100 by default;
  `GPU_BURN_MIN_FREE_GIB=4` is an explicit smaller-device option, subject to
  checking capacity and availability. This is not an OOM guarantee: competing
  allocations, oversized overrides, or allocator/NCCL behavior can still fail.
- Each multi-GPU cycle performs **46 SUM all-reduces**: 45 buckets of 25 MiB
  and one of 12 MiB, totaling **1,137 MiB logical payload per rank**. GEMMs
  are interleaved with reductions; reduced buckets are averaged to keep values
  bounded. NCCL chooses the physical transport/algorithm, so logical payload
  is not a measured network-byte count.
- Calibrates toward **approximately 0.75 seconds per cycle**, not a guaranteed
  interval. The payload is modeled on a roughly 596M-parameter BF16 gradient
  set; it is configurable, not a universal training workload.

Check inherited `GPU_BURN_*` environment overrides before launching; they can
change these defaults. Keep `GPU_BURN_MIN_WORLD_SIZE=2` for communicating runs.
A single-GPU override cannot provide inter-GPU communication.

To verify a live run, require a `gpu_burn_ready` line from every expected rank,
the correct `world_size`, and `collective_probe_sum=N*(N+1)/2` (36 for eight
GPUs). Then check that multiple `gpu_burn_progress` records advance both
`completed_cycles` and `completed_collective_payload_gib`. Each recorded cycle
follows GPU synchronization after its bucket reductions; startup or worker
count alone is insufficient. Inspect `average_cycle_seconds` and actual memory
usage too. Static code review does not establish live NCCL or memory behavior.

**Never use `pkill -f llm_pretrain_burn` or a similar pattern on the script
path.** The sleeper's PID-1 command line can embed that path; matching it can
terminate the pod. Do not kill the sleeper, tmux server, or a process group
obtained from an arbitrary GPU worker. A per-GPU worker may not include the
burn script name in its own command line.

Safe procedure:

1. Query only the relevant physical GPUs and list their compute PIDs.
2. Trace those PIDs to the known burn launcher, excluding PID 1. Verify no
   actual training/evaluation process is included and that no selected process
   owns GPUs belonging to another live workload. If uncertain, stop and ask.
3. Record exact PID/start-time identities. Immediately before signaling,
   recheck they still represent the same workers (PIDs can be reused).
4. Signal only those verified GPU worker PIDs, **not their process groups**.
   On the known runner, SIGKILL of the verified burn workers is the established
   reclaim operation. Do not replace it with a name-based `pkill`.
5. Wait the user-requested interval, requery the same GPU subset and require
   no compute PIDs. If processes remain, inspect/report; do not escalate to a
   broad all-node kill.

The final signaling step, after verification, has this form:

```bash
# Replace with the exact, freshly verified GPU worker PIDs; never PID 1.
# This deliberately fails until the caller supplies a nonempty array.
set -euo pipefail
TASK_VERIFIED_GPU_PIDS=()
test "${#TASK_VERIFIED_GPU_PIDS[@]}" -gt 0
for TASK_PID in "${TASK_VERIFIED_GPU_PIDS[@]}"; do
  [[ "$TASK_PID" =~ ^[0-9]+$ ]] && (( TASK_PID > 1 )) || exit 1
done
kill -9 "${TASK_VERIFIED_GPU_PIDS[@]}"
sleep 30
python3 scripts/gpu_status.py --gpus 0 1 --require-free
```

The empty array is intentional; this is not a blind kill-all snippet. Replace
the GPU indices too. Failure to signal a stale PID should trigger reinspection,
not silently discard the error. Do not kill a real training job using the burn
procedure unless its cancellation is separately authorized and identified.

## Persistent burn launch, only after verified free GPUs

An unattended workflow must not assume that `python script.py &` survives
after its shell exits. On the previously observed runner, independent tmux
sessions kept the burn alive; redirecting all three standard descriptors
could close the pane and cause SIGHUP. Keep stdin attached to the tmux pane.

The following is a launch pattern, not a default template job. Change the
indices, interpreter, run ID and port after inspection. Verify the port is
unused and distinct from other distributed jobs before submitting it.

```bash
#1 +60+a
#project-approved-burn-YYYYMMDD-a01
set -euo pipefail
TASK_INDICES=0,1
TASK_SESSION=burn_YYYYMMDD_a01
TASK_PORT=29537
TASK_PYTHON=/mnt/local/conda-py311/envs/runtime/bin/python3.11
TASK_BURN="$PWD/resources/llm_pretrain_burn.py"
TASK_LOG=/mnt/local/_outputs/@PROJECT@/logs/burn_YYYYMMDD_a01.log
test -x "$TASK_PYTHON"
test -s "$TASK_BURN"
command -v tmux >/dev/null
python3 scripts/gpu_status.py --gpus 0 1 --require-free
test ! -e "$TASK_LOG"
if tmux has-session -t "$TASK_SESSION" 2>/dev/null; then
  echo 'REFUSE: session already exists' >&2
  exit 1
fi
mkdir -p "$(dirname "$TASK_LOG")"
printf -v TASK_CMD 'exec env CUDA_VISIBLE_DEVICES=%q MASTER_ADDR=127.0.0.1 MASTER_PORT=%q %q -u %q >%q 2>&1' \
  "$TASK_INDICES" "$TASK_PORT" "$TASK_PYTHON" "$TASK_BURN" "$TASK_LOG"
tmux new-session -d -s "$TASK_SESSION" "$TASK_CMD"
tmux set-option -w -t "$TASK_SESSION" remain-on-exit on
sleep 30
test "$(tmux display-message -p -t "$TASK_SESSION" '#{pane_dead}')" = 0
nvidia-smi
tail -n 30 "$TASK_LOG"
```

Do not append `</dev/null` to the tmux command above. Verify one expected
worker per selected GPU and a common correct launcher. For a communicating
burn, check actual collective progress supported by its version; independent
workers/high utilization alone do not prove communication. Recheck after the
workflow that launched it exits. For a subset, keep free checks, stop queries,
and CUDA visibility scoped to that same subset.

The selected environment must actually include CUDA-enabled PyTorch; the
minimal `envs/runtime.txt` does not install it by default. The example launches
the included enhanced version directly and leaves the runner's `/tmp` file
unchanged. Use the same verified-worker-PID stopping procedure for either file.

Do not automatically run a new burn on failure while the failed job may still
have workers. Preserve logs, determine ownership, and report uncertainty.

## Training → evaluation → next stage

Require successful exit and project-specific result/checkpoint validation,
then wait/recheck GPUs, apply the correct environment/distributed config, and
wait/recheck again if requested. Checkpoints need the expected step, weights,
config and any required state, not merely a directory with the right name.
Only publish a completion marker after the declared validation succeeds.
Monitoring with read-only PID/proc checks does not itself signal a workload;
nevertheless, PID disappearance is only a liveness observation, not success.
