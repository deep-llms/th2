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

## Route study and scientific-review decisions

`eval/packed_ppl.py --route_study true` reuses packed evaluation for full selected
validation losses under as-trained/fixed/random routes. It collects a fixed
4,096-sequence diagnostic prefix and exports 256 sequences by default. Cross-seed
agreement requires matching exported token identities and evaluation contracts;
it is explicitly sample-scoped. The comparison can use all 4,096 by requesting
that artifact size for both runs. Fixed/random interventions are not replacements
for separately trained D/R controls. See the implementation guide for usage.

The user subsequently authorized the 2026-10-18 review fixes. Whole forced-choice
routers are bypassed (`grad=None`, including at the optimizer boundary during
the home-only phase); forced rows in mixed batches contribute zero z-loss.
Normalization deliberately remains **all input tokens × routing steps**, summed
over accumulation and ranks. AdamW momentum/decay must not update an inactive
router. The original reproduction changes from 52.9419% off-default at update
100 to 0%, with the default z-loss coefficient unchanged.

Periodic evaluation logs target hard-routing loss and the curriculum loss at
completed updates / full schedule, both without exploration. Gate 0a uses the
curriculum loss (`eval_spike_loss`) for stability; final studies use the target
policy. Native HF objective aggregates are explicitly labeled separately from
token-weighted LM loss. Cumulative LM sums and logging cursors are checkpointed;
interval wall throughput includes validation/checkpoint overhead. Peak memory
is the maximum across ranks. The training runtime gate compares against matched
D, with native Qwen timing also reported.

Standalone B-DOT evaluation and route studies visit all packed batches, including
all-boundary batches, so route counts and random-route streams agree for the
same layout/seed. No-reuse and curriculum CPU DDP/accumulation tests cover the
new loss semantics. See [resolution](BDOT_PHASE_A_CODE_REVIEW_20261018.md#resolution-after-authorized-fixes)
and [fix receipt](results/bdot_phase_a_fixes_20261018.json). GPU/FA4 validation of
these changes is still needed before a pilot. Strict source/state resume checks
remain in force; do not bypass them to resume checkpoints made before this fix.

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
deactivates the command and removes its cleanup body. GitHub contained only commands.sh at that point. See [cleanup receipt](results/dot_th2_cleanup_20261018.json).

DOT itself currently has no initialized Git repository; its commands.sh is #0.
Actual research assets still require selection and verification. Assets and library
execution are offline; no weights/data have been downloaded for the research
run. Credentials and shared URLs remain outside Git. Existing approved Dropbox
credentials are read from the old development project's ignored temp directory;
the q359 folder mapping and downloaded receipts are in DOT's ignored
`temp/dot_th2_inventory_20261018/`.


## B200 verification — 2026-10-18

Published DOT to the authorized th2 main branch from DOT-th2-inventory. Preflight
`0f37b2c` verified q359 and the existing eval_fa4 interpreter: all training-library
pins match, with FA4 4.0.0b33 and CUDA 13. No reinstall/driver changes. The
lm-eval version is older than our eval pin and was not used for these checks.
Launch `c3ce8d1` copied/verified Accelerate's eight-GPU configuration and reclaimed
only verified burn workers. The numerical suite and eight-GPU benchmarks passed
all 16 required cases; one optional natural near-tie route comparison differed.
Keep the attention backend fixed for an experiment. Same-route comparisons pass;
this does not promise identical routing trajectories across kernels.

Hard mixed routing is 89.2% of matched D throughput on SDPA and 85.9% on FA4;
relative to native Qwen these are 78.1% and 75.4%. Both pass the 60%-of-D runtime
gate in this synthetic setup. Full report, limitations and receipts are in
[BDOT_B200_VERIFICATION_20261018.md](BDOT_B200_VERIFICATION_20261018.md).
No research pilot or corpus download was launched. The source code remains in
DOT; the separate Git checkout publishes it without tracking temp credentials.

The supervisor finished at 19:27:32 UTC and a separate live check at 19:30:53 UTC
verified eight communicating burns, advancing collectives and released guard.
See [handoff receipt](results/bdot_b200_handoff_20261018.json). commands.sh returns
to #0 with the final code/results publication; no unattended pilot is queued.


## GPU FA4 resume evidence — 2026-10-18

Launch `82f6428` ran a 200-step uninterrupted reference, then fresh-process
checkpoint continuations 100→200 and 20→40 through real train.py on eight B200s.
All ranks match exactly for data/RNG, both routing branches, loss, model, AdamW,
scheduler and balancing state. Maximum loss difference is zero. The fixture
uses 12 layers with reduced width/vocabulary, BF16 and the existing FA4 defaults.
No deterministic override or production model change was necessary. See
[resume report](BDOT_FA4_RESUME_20261018.md) and its all-rank JSON receipt.
The supervisor restored communicating burns on success. A final live check at
19:56:35 UTC verified all eight workers, advancing collectives and released guard.
commands.sh returns to #0. No pilot was launched.


## Grouped expert execution, 2026-10-18

`bdot.expert_backend` now supports optional CUDA BF16 `grouped` projections;
`reference` remains the default. Original expert parameter names and unused
parameter/AdamW semantics are preserved. CPU tests (40 B-DOT, 49 utilities),
real B200 SDPA/FA4 numerical cases and three checkpointed two-rank BF16 tests
pass. Same-route fixture logits/losses match exactly; gradient relative error
is at most 0.325%, with explicit first-AdamW roundoff sensitivity recorded.

Eight-rank A/B resume and 50-update full-size timings were submitted under the
normal guarded burn handoff, but remote telemetry has not advanced past
22:07 UTC as of 22:20 UTC. Their outcome and final burn status are unverified;
no speed claim or default switch is justified yet. See
[verification notes](BDOT_GROUPED_GEMM_20261018.md). No pilot was launched.

## Replacement node environments — 2026-10-18

The user confirms q359 is dead. Submit only the controller installation directive
to th2: `#i envs/train.txt envs/eval.txt +a`. The two self-contained specifications
create fresh Python 3.11 `train`/`eval` environments with matching training pins
and the previously verified CUDA 13 FA4 4.0.0b33, CUTLASS DSL 4.8.0 and Quack 0.6.5.
Eval retains the project's lm-eval 0.4.13 pin, with HF support and SentencePiece.
Installation and GPU imports must be verified later; do not monitor Dropbox
until the user asks. Replacement node identity/mapping is pending, so old
q359 launchers and environment paths must not be reused without inspection.
