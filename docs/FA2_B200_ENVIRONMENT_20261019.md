# Isolated FA2 environment for the replacement B200

The user authorized creating a separate FA2 test environment. The submitted
request was:

```text
#i envs/train_fa2.txt +a
#th2-fa2-isolated-env-20261019-a01
```

The runner names this Python 3.11 conda environment `train_fa2`. The spec uses
`fresh: false` and names no existing production environment. It retains all
model/data pins from `envs/train.txt`: torch 2.14.1, Transformers 5.9.0,
Accelerate 1.13.0, Datasets 4.8.5, Tokenizers 0.22.2 and Safetensors 0.8.0.
It installs FA2 instead of the FA4/CUTLASS/Quack dependencies. Einops is pinned
to 0.8.2, matching the successful local FA2 test runtime.

The production `train`/`eval` specifications and NVIDIA driver are unchanged.
There are no training launches, process stops or GPU allocations in this
installation request. The user subsequently authorized routine Dropbox checks
on 2026-10-19 now that the replacement folder is available.

## Observed installation status — 2026-10-19

The controller reports `train_fa2` installation failure (exit 3) for commit
`960bbc9`. Read-only inspection `5f06832` at 23:05 UTC confirms the conda
environment exists and the shared torch/HF pins and core imports pass, but
**flash-attn is not installed**. FA2 is not usable yet. The controller summary
alone did not contain the actual installer error; an ordinary #2 export could
not retrieve installer details.

The subsequent read-only inspection `d3434c5` at **23:18:04 UTC** found the
saved per-environment status at `/mnt/local/_env_status/train_fa2.status.json`.
It explicitly reports:

```text
spec: flash_attn-2.8.3.post2+cu13torch2.14cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
result: missing
source: gap
overall: incomplete
```

This wheel is its only gap; other requested packages succeeded. The inspected
node wheel cache has no matching file. The installer handles a `.whl` entry
as an exact cached file, so this request does not compile FA2 from source.
The failure occurred before FA2 kernel execution. See the
[diagnostic receipt](results/fa2_install_failure_20261019.json).

The specification requested the correctly named compatible artifact without
confirming it was available. Reusing this prebuilt-wheel request requires a
matching build in the supported wheel store. A controlled offline source build
is another option, described below. Repeating the unchanged failed request
will not supply the missing file.

On `thiennh-p6-6y3o-worker-0`, the separate production `train` and `eval`
environments have their expected versions and successfully import FA4 4.0.0b33.
Their runtime is Python 3.11.15, torch 2.14.1, CUDA 13.0, C++11 ABI enabled.
The node has eight B200s, driver 580.167.08. No CUDA workload or process stop
was performed. See the [verified inspection receipt](results/b200_6y3o_environment_20261019.json).
commands.sh returns to #0. Do not resubmit an installation without diagnosing
the failure or confirming the required wheel is available.

## Required compiled wheel

FA2 is a compiled PyTorch extension. The matching artifact requested through
the runner's documented special-wheel mechanism is:

```text
flash_attn-2.8.3.post2+cu13torch2.14cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
```

This name follows the upstream [2.8.3.post2 build script](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3.post2/setup.py).
The local pinned torch 2.14.1 package reports CUDA 13.0 and C++11 ABI enabled.
The replacement-node inspection confirms these runtime and ABI values.

The matching wheel is absent from the upstream release asset list, rechecked
through the official GitHub API on 2026-10-19 at 23:14:57 UTC. Its 48 assets
contain CUDA 12 / torch 2.4–2.8 wheels, with no torch 2.14 wheel. The runner
reports the requested wheel **missing**; its controller-side store was not
separately audited. A replacement prebuilt wheel must be built against
torch 2.14.1/CUDA 13/Python 3.11 with SM100 support. Do not rename a torch 2.8
wheel, silently downgrade torch, or run network package installation in a `#1`
job. The runner's special-wheel rule was checked in the local operator guide;
that private guide is not included in Git.

## Source-build alternative and node health

The user pointed out the official
[PyPI source distribution](https://pypi.org/project/flash-attn/2.8.3.post2/#files).
Dev-only archive inspection verified SHA256
`d0f9f0b08a986cdeadbffcf5ec8b3a8f514366c68ad1853d3ab061bc83cc340b`
and the bundled `csrc/cutlass/include/cutlass/cutlass.h`. The GitHub tag's
automatic source archive omits that submodule header. Both setup scripts
include an SM100 build target and `FLASH_ATTENTION_FORCE_BUILD`.

Thus a pre-existing operator-provided wheel is not the only possible route.
An offline source build still needs the matching CUDA compiler, PyTorch and
build dependencies verified first. External inputs must use controller #i/#d;
any local build must suppress automatic wheel downloads, stay within the
separate `train_fa2` environment/build directory and limit compiler parallelism.
Do not modify the production environments, system CUDA or NVIDIA driver.
No source build has been submitted.

At the user's request, read-only health inspection `161d804` completed at
23:27:14 UTC. All eight GPUs respond, volatile uncorrectable ECC counters are
zero, existing burn workers remain active, production package versions/imports
match the previous snapshot, and cgroup OOM counters are zero. RAM and disk
have ample free capacity. There is no observed sign of node damage from the
missing-wheel failure; this is not a complete hardware stress test. See
[health receipt](results/b200_6y3o_health_20261019.json). commands.sh returns to #0.

## Verification before the comparison

The import/identity inspection above completes part of this checklist; kernel
execution and training comparisons still require their own verification:

1. Resolve the missing wheel, then inspect every package result in a deliberate
   retry. A conda directory or successful torch installation alone does not
   mean FA2 installed.
2. Resolve the actual `train` and `train_fa2` interpreter paths. The expected
   new path is `/mnt/local/conda-py311/envs/train_fa2/bin/python3.11`; never use
   an unqualified system `python` or mutate the production environment.
3. Read package versions from both environments and verify shared pins, CUDA
   runtime and ABI. Compare common dependency versions as well; the top-level
   pin list is not a full transitive lockfile.
4. Import `flash_attn` and `flash_attn_2_cuda` using only `train_fa2`. Validate
   the compiled extension against the actual runtime before allocating GPUs.
5. Verify machine/GPU identity and workloads, then use the established safe
   GPU handoff and explicit Accelerate configuration for any distributed test.
6. Reuse the [local numerical and full-update comparison](FA2_LOCAL_VERIFICATION_20261018.md)
   with separate output folders and explicit actual-kernel labels. Keep batch,
   precision, data, model, checkpointing and routing settings matched. Add the
   save/resume check before adopting FA2 for production training.

The local probe currently uses a test-only FA2 substitution, not a public
`--attention_backend fa2` option. No production default is changed by this
environment specification. FA2 installation is incomplete; B200 FA2 numerics,
throughput and resume behavior remain untested.
