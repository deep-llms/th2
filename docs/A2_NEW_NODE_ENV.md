# A2 environment setup for the replacement node

Requested 17 November 2026. Install only; experiments remain paused.
The existing deep-llms/th2 main runner repository is the requested destination.
The replacement node identity/runtime are not yet verified. No wc69 paths or
GPU assignments are used by this installation command.

- `a2_env`: Python 3.11, historical A2 train/native scoring pins, Transformers 4.57.1.
- `eval`: Python 3.11, lm-eval 0.4.13 and its separate Transformers 5.9.0 runtime.
- Added direct data/replay imports match the standalone A2 reconstruction recipes;
  these are not complete transitive locks. Do not silently substitute failed pins.
- `fresh: true` is intended for the replacement node, not an environment in use.

After the node is allocated, retrieve each install outcome before further writes.
Verify interpreter paths, pip check, project imports, actual package versions and
CUDA compatibility before GPU work; save pip freeze. Installation submission is
not proof of environment readiness. No data/model download or training is queued.

The old wc69 liveness probe f1a18710 failed with rc=5: no Running worker pod
for thiennh-p6-wc69 (last controller log timestamp 2026-11-16 20:09:44;
controller timezone unconfirmed). Do not resubmit that probe or old experiment.
The user separately authorized preparing this environment command for a new node.
