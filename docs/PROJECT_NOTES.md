# Project notes

## Research scope

[BDOT_v2_spec.md](BDOT_v2_spec.md) v1.3 defines the research. Implement Phase A
first: compute-matched bounded routing, absolute windows and step-owned K/V.
Do not build capacity/depth modes, other KV designs, STOP or re-entry until the
specified pilot gate passes. Configurations reject unsupported modes explicitly.

The plain dense reference is native Qwen3ForCausalLM with 12 layers and random
initialization. B-DOT reuses those Qwen3 components. The matched dense D has
FiLM; native-equivalence tests turn FiLM off. They are different references.

## Training contracts

- Keep HF Trainer/Accelerate, AdamW/scheduler and checkpoint/resume behavior.
  BDOTTrainer adds separate global router-loss normalization, one balance update
  per optimizer step and ordinary DDP unused-parameter handling. It rejects
  unvalidated FSDP/DeepSpeed/compile/reentrant-checkpoint paths.
- Model initialization follows the model seed. Routing draws use a separate
  checkpointed stream/counter and rank; evaluation never advances that stream.
- B-DOT's document_stream mode freezes a validation ID manifest before training,
  shuffles documents with seed 0, carries packing remainders between map batches,
  and uses a sequential prefix. It rejects budgets larger than the corpus.
- Append document EOS, isolate documents and reset positions per fragment.
  LM target tokens and router decisions have separate denominators. Validation
  loss excludes router terms. The inherited packed_shuffle mode is unchanged.
- max_steps defines the full schedule; stop_after is an absolute cutoff. Keep
  all checkpoints by leaving save_total_limit unset. Strict resume compares
  data, source, runtime, model and training receipts before loading state.
- Pilot queue default: D, P1-A, P1-B. R and the matching N follow explicit
  curriculum selection. The queue generator does not launch or select winners.

## Runtime and evidence

The local train_env was recreated on 2026-10-18: Python 3.11.15, PyTorch 2.14.1,
Transformers 5.9.0, Accelerate 1.13.0, Datasets 4.8.5, Tokenizers 0.22.2,
Safetensors 0.8.0 and lm-eval 0.4.13. Direct requirements are in envs/train.txt
and envs/eval.txt. No other environment or NVIDIA driver was changed.

The installed CUDA 13 wheel cannot use this dev machine's older driver. Initial
tests used CPU. Subsequent local GPU tests use separate compatible runtimes;
the pinned production-runtime/backend checks remain separate launch gates.
See [BDOT_IMPLEMENTATION.md](BDOT_IMPLEMENTATION.md) and
[local validation receipt](results/bdot_phase_a_local_20261018.json).
Historical template GPU evidence does not validate this new model.

The 2026-10-18 follow-up review fixed HF checkpoint loading of derived routing
buffers: regenerate no-reuse quota tables and zero update accumulators after
meta-device materialization, without changing persistent state. Tests now cover
all pilot checkpoint types, CPU BF16, independent dispatch math and checkpointed
hard/soft DDP. See [review receipt](results/bdot_phase_a_review_20261018.json).

Local GPU verification on 2026-10-18 used free A100 GPUs, without changing the
driver or train_env. SDPA model/gradient/checkpoint tests and two-rank NCCL pass;
see [GPU report](BDOT_GPU_VERIFICATION_20261018.md) for the default-SDPA AdamW
roundoff caveat and measured throughput. Native dense is also reported because
the current fixed D shares generic dispatch overhead. Pinned FA4 4.0.0b33 has
an SM80 forward path but rejects SM80 backward. Do not bypass that kernel guard
or claim FA4 training validation from forward-only evidence.

## Infrastructure

On 2026-10-18 the user explicitly assigned the previous `deep-llms/th2` GitHub
repository (`main`), q359 B200 and its existing Dropbox folder to DOT. The
separate execution checkout is `/disk/thuat/DOT-th2-inventory`. Its initial
normal descendant commit `4831f9a` replaces the tracked old-project tree with
only commands.sh and submits a read-only directory inventory. Previous history
and the local old-project checkouts are preserved. This is not authorization to
delete datasets/checkpoints/environments or stop processes on the node.

The first inventory succeeded on `thiennh-p6-q359-worker-0` at 18:32:47 UTC.
The source tree retains 28 top-level entries, including old project code and
controller-managed logs/status files: this runner does not delete old source
files just because they were removed from GitHub. No remote deletion was
performed. Commit `5e395c7` returns the minimal execution tree's commands.sh to
#0. Evidence: [inventory receipt](results/dot_th2_inventory_20261018.json).

The user then explicitly authorized source-directory cleanup. Job `1676c12`
completed at 18:53:02 UTC, removing 41 old source/resource entries. Preserved
commands.sh, controller logs/status files and resources/llm_pretrain_burn.py,
which is used by the existing eight-GPU burn. All eight worker identities and
the launcher were unchanged; no signals, new GPU work or environment changes
occurred. Data/model/output/conda roots were not modified. Commit `162455c`
deactivates the command and removes its cleanup body. GitHub still contains
only commands.sh. See [cleanup receipt](results/dot_th2_cleanup_20261018.json).

DOT itself currently has no initialized Git repository; its commands.sh is #0.
Actual asset/environment paths still require verification. Assets and library
execution are offline; no weights/data have been downloaded for the research
run. Credentials and shared URLs remain outside Git. Existing approved Dropbox
credentials are read from the old development project's ignored temp directory;
the q359 folder mapping and downloaded receipts are in DOT's ignored
`temp/dot_th2_inventory_20261018/`.
