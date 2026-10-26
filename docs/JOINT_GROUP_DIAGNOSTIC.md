# First joint-group diagnostic

26 October 2026 SGT. Implements the first screen in
[FIRST_ARCHITECTURE_HYPOTHESIS.md](FIRST_ARCHITECTURE_HYPOTHESIS.md).

Use the original, frozen, 28-layer Qwen3-0.6B-Base pretrained weights, revision
`da87bfb608c14b7cf20ba1ce41287e8de496c0cd`. This is not an old 12-layer scratch
checkpoint. Keep 6/8 GQA groups first and 4/8 second, at every layer. Compare:

| Policy | Retained groups |
|---|---|
| Dense | All eight; unchanged model |
| Static | One subset per layer minimizing total calibration reconstruction error |
| Raw norm | Largest group norms before the output projection |
| Projected norm | Largest norms of group contributions after the output projection |
| Joint | Subset minimizing the squared norm of the sum of omitted projected contributions |

The last three inspect actual attention outputs. They are offline references,
not deployable pre-attention routers. All attention is computed, then complete
groups are masked at `o_proj` input. All K/V groups remain available. Qwen's
ordinary output projection applies its bias once. FP32 contributions and Gram
matrices define the selection objective; the evaluation backbone uses BF16 SDPA.
Exact ties prefer lower indices/lexicographically earlier subsets. No parameters
are trained or changed. There is no sparse kernel or inference speed claim.

Every modified model is evaluated on its own causal hidden states. Separate
dense-trajectory statistics describe local reconstruction; they are not used as
a substitute for whole-model language-model loss.

## Data and measurements

Use the existing pinned `nht10/cx_sampled` English pool, revision
`dd1b15134dc133a26d04f7edf45d9fe8315f4f52`. Choose 64 calibration documents from
train and 128 evaluation documents from validation, seed 42. Shuffle rows in
bounded, pinned source shards. Take one 2,049-token prefix per sufficiently long
document, yielding 2,048 next-token targets. Deduplicate document and token-prefix
hashes across both splits. There is no cross-document attention or packing in
this probe. Append `<|endoftext|>` at a real document end if needed; never insert
it inside a truncated prefix.

This is a long-document English screen, not full validation PPL or a uniform
sample of the 30B-token pool. Freeze tokens and row identities before scoring.
Verify the pool metadata, all files in consumed shards, tokenizer/config and
pretrained weights against pinned hashes. Do not rehash unused 134 GB of data.

Calibration and held-out local statistics sample every 16th position. Report
each layer's summed reconstruction error divided by summed dense-output energy.
LM loss uses every next-token target. Report token-weighted loss/PPL, per-document
losses and paired document-bootstrap 95% intervals for joint minus each reference.
Bootstrap intervals describe this fixed probe, not variation across training runs.

## Code and validation

- `efficient_llm/group_selection.py`: whole-group masking, exact subset enumeration.
- `efficient_llm/diagnostic_data.py`: frozen disjoint local probe and provenance.
- `efficient_llm/diagnostic.py`: calibration and eight-rank inference/reduction.
- `efficient_llm/gpu_check.py`: pretrained GPU identity, causal and cached checks.
- `scripts/launch_group_diagnostic.py`: immutable source, preparation, Accelerate
  copy/verification and the reused guarded supervisor.

CPU tests cover projection/bias algebra, GQA layout, exact keep-all identity,
causality, cached equivalence, loss shifting, shard corruption, EOS handling,
and serial versus uneven two-rank evaluation. Eleven mechanism/scoring tests
and eleven reused supervisor tests passed before submission. The GPU gate uses
FP32 for cached equivalence, plus BF16 identity/causality/finite-output checks;
it does not claim bit-identical BF16 cached routing across different kernels.

Preparation occurs before reclaiming GPUs. Copy `resources/accelerate_b200.yaml`
to the interpreter's actual Accelerate cache and run `accelerate env`. The
supervisor verifies exact burn workers, holds the GPU guard, stops only those
workers, waits and requires all eight GPUs free. One short GPU correctness job
precedes the eight-GPU study. On successful completion or job failure it cleans
only its own descendants, requires free GPUs and restores the communicating
burn. Unknown ownership fails closed. Live source resides outside the synced
code directory. Existing data, checkpoints and environments remain intact.

If joint selection improves local MSE but not LM loss relative to projected norm,
do not spend compute fitting more router variants automatically. A useful LM
advantage would justify a later matched-capacity router comparison, followed by
real sparse decode measurement. This screen alone cannot establish deployment
efficiency or novelty.
