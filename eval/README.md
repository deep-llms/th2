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

### Expanded English suite (12 October 2026)

Implemented locally; production B200 numerical gates and smoke runs are still
required before a full study. Defaults for existing commands remain unchanged.
The explicit sequential suite is `python -m eval.suite`; it retains existing
ARC/HellaSwag/PIQA/WinoGrande/XNLI/PAWS/XStoryCloze recipes and adds:

| Benchmark | Protocol | Primary reporting |
|---|---|---|
| XCSR | English X-CSQA (1,000) and X-CODAH (300), zero-shot labeled validation | Accuracy for each and their mean; public test labels are hidden |
| TruthfulQA | Zero-shot canonical harness MC1 and MC2 | Both metrics, one benchmark vote via their mean |
| Belebele | English, zero-shot; no demonstrations from evaluation pool | Accuracy |
| LAMBADA OpenAI | Zero-shot final-word likelihood, including all tokens of the final word | Accuracy and target-word perplexity; PPL excluded from accuracy averages |
| BLiMP | Zero-shot, all 67 subsets | Macro-subset accuracy and linguistic-phenomenon breakdowns |
| MMLU | 5-shot, first five official subject dev examples, all 57 subjects | Accuracy weighted by evaluated examples across subjects |
| OpenBookQA | 5-shot from train | Length-normalized accuracy (raw also retained) |
| Social IQa | 5-shot from train | Accuracy |
| BoolQ | 5-shot from train, no fine-tuning | Accuracy; separate from supervised BoolQ |
| GSM8K | Pinned harness 5-shot worked examples, greedy decoding, 256 new tokens | Strict numeric-answer exact match; flexible extraction also retained |
| GLUE | Supervised CoLA/SST-2/MRPC/QQP/STS-B/MNLI/QNLI/RTE/WNLI | Official task metrics, with both MNLI splits; local validation aggregate |
| SQuAD v2 | Supervised causal span head, sliding context windows | Exact match, F1, development-calibrated no-answer threshold |

All likelihood definitions come from `lm_eval==0.4.10`; XCSR uses the authors'
HF dataset with a small normalized local adapter. X-CODAH's blank question is
normalized to an empty string so the harness correctly scores whole candidate
sentences from its prefix token. English-only results imply no multilingual claim.
SciQ, Natural Questions and TriviaQA remain optional and are not part of this suite.

GSM8K uses `eval/generation.py`: full-prefix forward passes through the complete
custom architecture, with the learned proxy active. This avoids introducing an
untested incremental proxy cache. It is intentionally slower than cached decoding.
EOS and text stops are honored; prompt plus generation budget exceeding the
trained context raises an error instead of silently deleting demonstrations.
No sampling, chat template or instruction-tuning transformation is added.
Likelihood tasks retain the existing harness left-truncation policy; inspect its
warnings/context audits when interpreting nominal shot counts.

GLUE uses official `nyu-mll/glue` data, separate from the older normalized STS-B
source. Scores remain on the official 0–5 regression scale. CoLA uses Matthews
correlation; MRPC/QQP report accuracy and F1 (their mean selects checkpoints);
STS-B reports Pearson/Spearman (their mean selects checkpoints). Other tasks use
accuracy. Training-derived development splits group duplicate inputs; public
validation is final holdout. Both MNLI validation splits are excluded from
training/development, and the study queue reuses one fitted model for both.
WNLI is included explicitly. These are local held-out scores, not submissions
to the official hidden-test GLUE leaderboard.

SQuAD groups training/development by context, excludes final-validation context
overlaps, places the question before the context, and keeps causal attention.
Token offsets and overlapping windows map predictions back to original text.
EOS after each window is the no-answer position, whose hidden state sees the
whole window. Span length is capped at 30 tokens. Threshold selection maximizes
F1 on development only; the saved threshold is reused on final validation.
The head, Trainer, source hash checks, frozen proxy statistics and task-only
backbone gradients follow the existing supervised implementation.

The existing fine-tuning study builder supports the new tasks, with development
LR selection and three confirmation seeds. It keeps all epoch checkpoints.
Per-task results remain visible; averages use unrounded values. The likelihood
suite averages benchmarks once, separates generation/language diagnostics from
multiple-choice tasks, and verifies matching examples/prompts across checkpoints.
The supervised report includes a labeled task mean and, for all GLUE tasks,
a nine-task local aggregate with MNLI's two splits combined into one vote.

#### Prepare pinned local inputs (networked dev machine)

```bash
python -m scripts.prepare_expanded_benchmarks \
  --output /path/to/benchmarks --source-lock /path/to/sources.json \
  --tasks xcsr truthfulqa belebele lambada blimp mmlu openbookqa social_iqa boolq gsm8k \
          hellaswag piqa arc_easy arc_challenge winogrande xnli paws-x xstorycloze \
  --supervised glue_cola glue_sst2 glue_mrpc glue_qqp glue_stsb \
               glue_mnli_matched glue_mnli_mismatched glue_qnli glue_rte glue_wnli squad_v2
```

The source lock pins HF commit IDs before downloads. The portable parquet
manifest records file hashes, sizes, split mappings and source revisions. Reuse
that lock to reproduce snapshots; different existing outputs are rejected.
GPU nodes must receive inputs through the established controller download path,
then use the manifest offline. This command is for the dev machine only.

#### Sequential evaluation and supervised job construction

```bash
python -m eval.suite --checkpoints /path/to/A /path/to/P6-iso \
  --dataset-root /path/to/benchmarks --manifest /path/to/benchmarks/manifest.json \
  --output /path/to/fresh-evaluation --attention-backend fa4 --plan-only

python -m scripts.finetune_study build \
  --root /path/to/fresh-study --project /path/to/project \
  --data-root /path/to/benchmarks --manifest /path/to/benchmarks/manifest.json \
  --checkpoints /path/to/checkpoints.json --step 20000 \
  --tasks glue_cola glue_sst2 glue_mrpc glue_qqp glue_stsb glue_mnli_matched \
          glue_mnli_mismatched glue_qnli glue_rte glue_wnli squad_v2
```

`checkpoints.json` maps arm names to checkpoint directories, e.g.
`{"A": "/path/to/A", "P6-iso": "/path/to/P6-iso"}`. The supervised builder only
writes configs/jobs for the existing experiment runner. It includes numerical
gates, smoke/reload checks, selection, fitting and final validation. No process
management or remote submission is performed. `eval.suite --plan-only` similarly
writes a plan; remove that flag **with a fresh output** to execute sequentially.
Use `--limit 2` for a diagnostic likelihood/generation smoke run. That result is
marked diagnostic and must not be presented as a full benchmark score.

References: [XCSR](https://github.com/INK-USC/XCSR),
[TruthfulQA](https://github.com/sylinrl/TruthfulQA),
[Belebele](https://github.com/facebookresearch/belebele),
[LAMBADA](https://arxiv.org/abs/1606.06031),
[BLiMP](https://arxiv.org/abs/1912.00582),
[GSM8K](https://github.com/openai/grade-school-math),
[MMLU](https://github.com/hendrycks/test),
[OLMES task suites](https://github.com/allenai/olmes/blob/main/oe_eval/configs/task_suites.py),
[GLUE](https://gluebenchmark.com/),
[SQuAD](https://rajpurkar.github.io/SQuAD-explorer/).

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
checks of FA4 checkpoints. Custom generation uses full-prefix recomputation;
incremental KV caching is not implemented. Benchmark context is capped at the
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

## Supervised sentence-pair fine-tuning

`python -m eval.finetune /path/to/config.json` uses the standard HF Trainer
with a task classification head and the full custom backbone. This is separate
from the inference-only `EvaluationModel`. The initial supported arms are A,
P6 and P7-simple, on English PAWS-X/NLI. See
[the fixed protocol](../docs/SUPERVISED_FINETUNING_20261007.md) for gradient
routing, seeds, LR selection, split handling and the gated B200 study.

For later checkpoints, fine-tuning configs must set `expected_step` to the
requested pretraining step (default 2500 preserves older studies). The
downstream validator accepts an explicit checkpoint specification with labels,
steps and weight hashes, so two checkpoints of A can be compared safely.
`scripts/checkpoint_downstream_study.py` composes the established few-shot and
fine-tuning studies without changing model forward, loss, or optimization.
Fine-tuning now retains all epoch model snapshots; the best epoch is still
selected on development data.
