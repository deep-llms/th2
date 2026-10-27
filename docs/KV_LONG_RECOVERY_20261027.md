# Longer matched KV-sharing recovery

27 October 2026, Singapore time. User approved longer training after reviewing
CLA and SwiftKV. This tests whether the earlier 134M-token result was limited by
recovery budget. It is not a reproduction of either paper or a new-method claim.

| Setting | Both arms |
|---|---|
| Model | Original pretrained 28-layer Qwen3-0.6B-Base |
| Initialization | Same immutable dense A-512 final from `6y3o-efficient-kv-mlp-20261027-a03` |
| A | Dense, all parameters trainable |
| B | Uniform25: layers 3,7,11,15,19,23,27 reuse preceding layer K/V (zero based); all remaining parameters trainable |
| Recovery budget | 20,752 updates × 65,536 tokens = 1,360,003,072 tokens per arm |
| Batch | 8 GPUs × 8 documents × 1,024 targets, accumulation 1 |
| Loss | Existing 0.5 CE + 0.5 teacher-forward-KL, temperature 1 |
| Teacher | Frozen original pretrained model, unchanged |
| Schedule | Fresh AdamW, peak LR 5e-5, cosine over all 20,752 updates, 1,038 warmup steps (5%) |
| Other recipe | Weight decay .01, gradient clip 1, seed/data seed 42, FP32 parameters/BF16 autocast, native SDPA, no activation checkpointing |
| Validation | Same frozen 256 document prefixes, 262,144 scored tokens; final BF16-weight evaluation too |
| Save/evaluate | Every 1,024 updates, plus 5,188 / 10,376 / final20,752; retain every saved full checkpoint |

Fresh schedule from the common parent, not a resume from the expired short
schedule. Intermediate 134M / 340M / 680M results belong to this longer schedule;
they are not exact repetitions of the earlier short schedule. No MLP pruning.

Data is the already-downloaded pinned CulturaX English pool. Select 1,328,128
distinct sufficiently long documents with the existing seed9042 prefix policy.
Reuse validation exactly and exclude all original pilot train/calibration/eval
documents. Extend the old recovery selection; verify the first131,072 document
records AND token tensors equal the old selection. Actual shuffled sample order
will differ from the short run because dataset size changes, but must match
between the new A/B runs on every rank, with exact once-only coverage.

Batched tokenizer preparation (256 documents, 16 CPU tokenizer threads) preserves
selection order, token IDs, EOS behavior and duplicate filtering. It changes no
model forward/backward path. Verify hashes of each consumed source shard and the
saved token tensor; training checks model and probe identities again.

Before launching: verify node/burn ownership and completed parent artifacts;
freeze source; prepare and verify data while burns remain running; copy the
eight-GPU resource Accelerate config to its actual HF cache and run `accelerate
env`; native supervisor holds the guard, rechecks burn worker PID/start identities,
stops only verified workers, waits30 seconds and verifies all GPUs free.
Run the uniform25 GPU numerical gate and separate eight-step eight-GPU A/B smoke
tests before long A then B. On success/failure the existing supervisor restores
and verifies communicating burns. Preserve all earlier outputs and environments.

Final summary verifies same parent bytes, recipe, probe, per-rank sample order,
step count and validation points. Report paired document-bootstrap confidence
intervals and PPL curves. Upper95% CI(shared-minus-dense NLL) <= log(1.05) remains
a screening gate, not evidence of downstream equivalence. This reused small
validation probe and one training seed limit generalization. Passing allows
proposing measured inference/long-context evaluation; no extra training is queued.

Earlier measured training time projects roughly 3 hours for the pair, excluding
CPU preparation and verification. Confirm from actual longer-run progress.

Implementation: `efficient_llm/kv_long.py`, existing `kv_mlp_study.py` trainer,
`diagnostic_data.py` selection and `scripts/launch_group_diagnostic.py` launcher.
Core `kv_mlp.py`, loss, backend, GPU supervisor and burn implementation unchanged.
