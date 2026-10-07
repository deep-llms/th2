# Checkpoint evaluation

Reuse the existing LM Evaluation Harness for downstream likelihood tasks. The
custom loader reconstructs A, V1/V3, P1/P3, P4–P7, and legacy DeepKV arms from
`train_config.json`, then strictly loads `model.safetensors`, including proxy
parameters and normalization buffers. Keep the recipe beside the model or in
the parent of `checkpoint-N`. Ordinary Hugging Face checkpoints still work.

The training wrapper computes vocabulary logits in chunks and returns loss
sums/counts. `models.EvaluationModel` exposes
`wrapped.backbone.lm_head(wrapped.hidden_states(context)[0])` as `.logits`.
This executes the full custom architecture with its proxy active. It does not
call the native backbone's forward and skip the proxy. No auxiliary targets,
auxiliary losses, optimizer, or normalization updates are needed at inference.
The training implementation and checkpoint format are unchanged.

## English benchmarks

```bash
python eval/eval_checkpoint.py \
  --checkpoint /path/to/P7-simple/checkpoint-2500 \
  --bench-only --english-only \
  --tasks hellaswag xnli belebele xstorycloze paws-x \
  --batch-size 8 --num-fewshot 0 --seed 42 \
  --output-dir /path/to/fresh-evaluation
```

`--english-only` selects HellaSwag, XNLI English, Belebele `eng_Latn`,
XStoryCloze English, and PAWS-X English from the existing groups. XCOPA has no
English subset: the default English suite skips it with a message; explicitly
requesting it raises an error. Additional supported English tasks are `piqa`,
`arc_easy`, `arc_challenge`, and `winogrande`. Omitting `--english-only` retains
the existing multilingual group selection. Individual listed task names are
also accepted. Dataset repository corrections are applied in memory, without
editing the installed harness.

Each benchmark request (including any few-shot examples and answer) is one
document. Separate requests do not share context. Literal EOS inside a prompt
does not create a new document. Explicit packed segment IDs are supported by
the adapter; each ID must occupy one contiguous segment per row. Attention
masks must be binary, with padding outside the valid token span. Invalid layouts
are rejected so SDPA and FA4 cannot interpret the same input differently.
Positions reset at document boundaries. P3/P7-ems receive isolated
trailing padding to complete internal EMS chunks; this padding cannot affect
real tokens. The harness's causal right padding is never scored.

Custom checkpoint evaluation preserves FP32 parameters and uses the saved BF16
autocast setting. The saved attention backend is used unless explicitly
overridden with `--attention-backend sdpa` or `fa4`; overrides are recorded.
FA4 requires the existing compatible CUDA/BF16 environment and pinned FA4
package; there is no silent backend fallback. Use SDPA explicitly for CPU
checks of FA4 checkpoints. Current custom support is likelihood scoring, not
generation or incremental KV caching. Benchmark context is capped at the
trained context length (and 2048).

Use `--tokenizer-name /local/tokenizer` if the tokenizer path recorded in the
recipe is unavailable. Model and tokenizer loading are local-only. Benchmark
datasets must be pre-cached on an offline worker. `--limit N` is diagnostic
only; omit it for complete benchmark results.

For controller-downloaded raw files, pass both `--dataset-root /local/snapshots`
and `--dataset-manifest resources/downstream_english_20261007.json`.
The manifest pins repository revisions, file sizes/SHA256 and split mappings.
The evaluator verifies all files before loading weights, then uses the local
Parquet/JSON/TSV loaders with the original task prompts and scoring. It makes
no Hub requests. This avoids relying on a pre-existing Hugging Face cache or
downloading unneeded language subsets. Metadata records the manifest hash.
XNLI uses the harness's validation split (2,490 examples), not its test split.

`envs/eval_fa4.txt` supplies a separate pinned environment with both the harness
and the same FA4 version used for training. It does not replace `train_env`,
`attention_bench`, or the existing `eval` environment.

Outputs include metrics, full harness configuration/task versions, per-example
scores (`eval_samples.jsonl`), and checkpoint/recipe hashes, precision, backend,
package versions, arguments and completion status (`eval_metadata.json`). Use
a fresh output directory for each comparison. Match evaluation seeds, task
versions, prompts, few-shot counts, checkpoints' training steps, and precision.

## Multiple checkpoints

```bash
python eval/eval_parallel.py \
  --checkpoints /path/to/A/checkpoint-2500 /path/to/P7-simple/checkpoint-2500 \
  --bench-only --english-only --tasks hellaswag xnli \
  --num-gpus 2 --gpu-ids 0 1 --batch-size 8 \
  --output-dir /path/to/fresh-comparison
```

One evaluation process runs per selected GPU. This is not an Accelerate launch.
The launcher forwards task, precision/backend, tokenizer, seed and limit
options; writes separate outputs; and exits nonzero if any child fails.
Without explicit IDs it honors `CUDA_VISIBLE_DEVICES`. It does not reclaim
GPUs or stop other workloads: use the normal runner ownership checks first.

## Perplexity

`--ppl-only --eval-dir /path/to/validation` accepts a single local Arrow dataset,
`shard_*` Arrow directories, local Parquet files, or a parent containing
language folders (`--langs en` to select English). Documents are tokenized
individually, receive one `<|endoftext|>`, and never share attention. Long
documents use overlapping windows; each next-token target is counted once.
Short documents are included. This independent-document metric has a different
context policy from training's packed validation metric and must be reported
separately. It also intentionally replaces the old newline-concatenated PPL.

## Local verification

`python -m unittest tests.test_eval -v` uses tiny models and local synthetic
datasets, including an actual Trainer checkpoint and real harness scoring.
It checks logits/loss equivalence for all proxy arms, strict restore, active
proxies, unchanged buffers, variable lengths/padding, English selection,
perplexity target counts, CLI behavior, and failure propagation. FA4 in CPU
tests uses the existing reference attention oracle, not the CUDA kernel.
Full-model downstream GPU acceptance and actual benchmark dataset availability
must be checked before a production evaluation run.

Review on 7 October 2026: all ten local test groups passed (70.408 seconds).
Additional checks exercise the five real English task templates with synthetic
local datasets through the full CLI, numeric result serialization, repeated-run
output protection, and invalid layouts/padding under both attention paths.
Dataset retrieval is mocked in that test; it does not establish availability
of the actual benchmark datasets on B200.

`scripts/check_downstream_eval.py` additionally checks real dataset documents,
full-checkpoint CUDA loss equivalence/isolation/active proxies/unchanged buffers,
and complete results with matching per-example hashes across checkpoints.
Smoke and full evaluations must have separate output directories. Full results
are accepted only with no example limit and all nine task counts verified.
