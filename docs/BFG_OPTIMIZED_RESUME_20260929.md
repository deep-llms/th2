# B/F/G checkpoint-only continuation to 10,000 updates

Completed on September 30 at 04:22:53 UTC. All three arms reached 10,000 steps;
final LM losses: B 3.029353, F 3.029287, G 3.030396. Automatic burns were
verified at 04:24:04 and remained active on all eight GPUs at 05:23 UTC.
Final artifacts: `artifacts/BFG-10000-final-20260930/`.

The following records the launch and resume validation.

Production B resumed successfully from step 5,000. The 16:34:12 UTC audit on
2026-09-29 observed step 5,040, finite losses/gradients, continued LR about
0.0002884, and one train.py worker on each of eight B200 GPUs. Each GPU used
119,192 MiB with 98–99% utilization. Early B updates took about 2.5 seconds each.
F and G follow sequentially; each arm stops at 10,000 total updates.

## Final recipe

Only decoder, LM-loss and auxiliary-loss activation checkpointing are disabled:
`checkpoint_layers=false`, `checkpoint_lm=false`, `checkpoint_aux=false`.
Keep original `causal_attention=false` (explicit attention mask) and `lm_chunk=128`.
The additional causal/chunk optimizations were not selected for this continuation.

Everything else stays fixed: native HF Trainer/Accelerate, eight GPUs, BF16,
microbatch 16, accumulation 4, sequence 2048, 1,048,576 input tokens per update,
full schedule 28,600, warmup 1,430, original optimizer/LR settings, seeds, language
mix, EOS packing, sampler/data skipping, evaluation and checkpoint-save cadence.
The cutoff is 10,485,760,000 cumulative input tokens per arm.

`train.py` exposes the five execution settings while preserving their historical
defaults. Changing them on resume requires explicit
`--allow_performance_change_on_resume true`. All scientific recipe and data
fingerprint checks remain strict. Each change archives the full previous and
requested recipe in `resume-transition-<step>-<UTC>.json`. Production B's record
changes exactly the three checkpoint switches.

## Validation and the initial rejected variant

The initial combined recipe also enabled implicit causal SDPA and LM chunks 512.
Its six real-checkpoint smoke runs completed, and all 48 rank receipts confirmed
exact initial model/optimizer/scheduler/RNG restoration and matched input order.
However, B's updated parameter relative L2 difference was 1.12609e-5, exceeding
the predeclared 1e-5 bound. Production did not start. The queue failed at 16:14:53;
automatic burn recovery was verified at 16:16:04. That attempt remains preserved.

The fresh checkpoint-only retry reused the completed controls read-only and ran
three new eight-GPU resumes, each taking one update from original step 5,000,
saving full state and evaluating the same 4,882 rows. No tolerances were relaxed.

| Arm | Parameter relative L2 | Adam first-moment relative L2 | Adam second-moment relative L2 | LM loss delta |
|---|---:|---:|---:|---:|
| B | 4.19410e-6 | 0.002110 | 1.53924e-5 | -4.09e-6 |
| F | 4.04550e-6 | 0.002117 | 2.04061e-5 | +1.68e-6 |
| G | 4.14861e-6 | 0.002157 | 1.73025e-5 | +1.38e-6 |

All passed: parameter difference <=1e-5 and <=5% of the control update;
first/second moments <=3%; LM loss delta <=0.001. Scheduler states and optimizer
step counters matched exactly. All ranks restored exact source state and consumed
the same next four microbatches; input batches also matched across B/F/G.
This supports numerical correctness, not bitwise-identical future trajectories.
The smoke wrapper observes native Trainer behavior without replacing loading,
RNG restoration, data skipping or optimization. Production calls train.py directly.
Local CPU checks also covered native resume, strict metadata rejection, all-arm
model/gradient behavior and the smoke/queue workflow.

## Launch and artifacts

Source checkpoints remain intact at:
`/mnt/local/_outputs/deep-llms_th2/deep-kv-BFG-5000-20260929-a01/production/run`.
Active root:
`/mnt/local/_outputs/deep-llms_th2/deep-kv-BFG-10000-20260929-a02`.
Production arms are under `production/run/continuation/{B,F,G}`.
They use fresh checksum-verified original step-5,000 copies, not smoke outputs.

The launch copied resources/accelerate_config.yaml into the actual cache,
`/dev/shm/.cache/huggingface/accelerate/default_config.yaml`, compared bytes and
verified eight-process BF16 MULTI_GPU with `accelerate env`. Only freshly verified
burn worker identities were stopped. All eight GPUs were empty at 16:22:52 before
the retry. Native production B started at 16:31:44 after the successful gate and
fresh checkpoint staging. No environment, CUDA, driver or package changes occurred.
The existing supervisor restores and verifies enhanced burns after queue success
or failure, after cleaning only its own descendants and checking GPUs are free.

Launch commit: 4daa75c. Startup audit: 967843b.
49 small source artifacts, source hashes, production copy manifests, CLI cutoffs,
transition settings and all-rank receipts verified locally under
`artifacts/optimized-resume-20260929-a02/`. Initial attempt artifacts remain in
`artifacts/optimized-resume-20260929-a01/`.
Final startup log SHA256:
`63dcc975edd3167c4bf7b830ab72858a74de3ddf930040de46611f3a19607f43`.
