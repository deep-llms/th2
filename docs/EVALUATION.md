# Checkpoint evaluation and task fine-tuning

**Open pre-launch gate:** the review found that the benchmark harness disables
the wrapper's outer BF16 autocast. Until its explicit mixed-precision option is
configured, benchmark model calls run in FP32 even when BF16 is requested.
The documented BF16 protocol below is the intended protocol, not a verified
current harness behavior. On2026-09-09, `scripts/check_eval_precision.py` reproduced
this on B200 for all six arms; the bounded production-checkpoint smoke confirmed
the same issue. BF16 fine-tuning updates, save/reload and PPL-path tests passed,
but the overall smoke failed on benchmark precision. Fix/retest before a full
BF16 evaluation; see CURRENT_TASK.md for reports. No fix has been applied yet.
PPL and new diagnostic model calls do not use the affected harness wrapper.

Current Stagewise run: **English only**. These tools are separate from
pretraining and do not alter its model code, schedule, caches, or running job.
They load stock B0 and custom A128/A256/A512/C/D checkpoints through registered
HF model classes. No pretrained Qwen weights are substituted.

## Protocol

All entry points default to `--languages en`. No language is inferred from
the tokenizer, checkpoint name, or directories present on disk.

- PPL: score only selected saved `eval/<language>` datasets. Report each
  language separately plus a target-count-weighted aggregate NLL/PPL. Never
  pack across language boundaries or average language PPL values. Use the
  current pretraining settings: 2048 tokens, 160 map workers, batch1000.
- Zero-shot: official **lm_eval 0.4.10**, zero few-shot examples, no chat
  template, BF16 autocast, maximum context2048. The default now includes
  BLiMP, LAMBADA-OpenAI, HellaSwag, PIQA, ARC-Easy, WinoGrande, ARC-Challenge,
  BoolQ, plus the earlier `xnli_en`, `belebele_eng_Latn`, `xstorycloze_en`
  and `paws_en`. BLiMP expands to all67 official subtests:78 individual task
  results in total. See the English suite section below for task IDs/metrics.
  Do not compare averages across different benchmark selections or count
  each BLiMP subtest as a separate family in an overall benchmark average.
- Fine-tuning: independent original-checkpoint reload for each task/seed;
  never train the next task from a previous task's fine-tuned weights.
  English HellaSwag, ARC-Easy and XNLI training splits; evaluate only their
  English counterparts by default. Validation/test are never training fallbacks.
  Fine-tuning is full-model, generative next-token CE, not a new classifier head.

Inherited fine-tune hyperparameters: three epochs, LR2e-5, AdamW weight decay
0.01, linear schedule with10% warmup, clip1, sequence length256, batches16
(HellaSwag) /32 (ARC-Easy and XNLI), one GPU per job. Three seeds default to
42/123/456. Six checkpoints × three tasks × three seeds means **54 separate
fine-tunes**, including18 XNLI jobs. English-only evaluation does not shrink
the English XNLI training split. Select seeds explicitly when choosing a budget.

Two deliberate improvements over the old fine-tuner must be recorded in
comparisons: FP32 master weights/Adam state with BF16 autocast (old code loaded
weights in BF16), and continuation tokenization matching HFLM's joint BPE
boundary/empty-context EOS-prefix behavior. Prompts and answer choices come
directly from the same official task object used for scoring. Prompt/padding
labels are masked; real EOS targets are not masked. Overlength examples are
right-truncated at256; examples with no scored completion are skipped and
their counts are reported. These changes apply equally to every Stagewise arm.
Old results are not an exactly identical fine-tuning protocol.

## Offline environment and inputs

Use the reviewed dependencies in `envs/swt_eval.txt`. Installing on B200 must
go through `#i`, and only when authorized; never install into the active
training environment. An already-installed compatible environment can be
verified and reused, but it must contain the unmodified official task YAMLs
(the old evaluation helper patched those files in place). Unexpected paths
are rejected rather than trusted. This file does not itself install anything.

```text
#i envs/swt_eval.txt +a
#project-install-swt-eval-UNIQUE_ID
```

Model/tokenizer loading is local-only. All workers force HF offline flags
before importing the relevant libraries. Download benchmark snapshots using
authorized `#d` first, then preserve this layout under a chosen dataset root:

```text
benchmarks/
  facebook/xnli/
  facebook/belebele/
  juletxara/xstory_cloze/
  google-research-datasets/paws-x/
  Rowan/hellaswag/
  allenai/ai2_arc/
  nyu-mll/blimp/
  EleutherAI/lambada_openai/
  baber/piqa/
  allenai/winogrande/
  aps/super_glue/
```

Only snapshots for selected tasks/languages are required. The code overrides
official task dataset paths in memory before loading them; it does not mutate
installed lm-eval YAML files, download missing inputs, or use a remote fallback.
Missing snapshots/configurations/splits fail. Verify real B200 inputs and run
a destination smoke before scheduling the full evaluation battery.

## English pretraining benchmark suite

Definitions come from the pinned [lm-eval v0.4.10 task code](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.10/lm_eval/tasks).
No custom prompts, scoring normalization or few-shot demonstrations are added.

| Group flag | Official task(s) | Evaluated split | Reported metrics |
|---|---|---|---|
| `blimp` | All67 `blimp_*` subtests | Each configuration's `train` split (official benchmark split) | Subtest accuracy and unweighted suite mean |
| `lambada` | `lambada_openai`, configuration `default` | test | Last-word exact-token accuracy and target-word perplexity |
| `hellaswag` | `hellaswag` | validation | acc, acc_norm |
| `piqa` | `piqa` | validation | acc, acc_norm |
| `arc_easy` | `arc_easy` | test | acc, acc_norm |
| `winogrande` | `winogrande`, configuration `winogrande_xl` | validation | acc |
| `arc_challenge` | `arc_challenge` | test | acc, acc_norm |
| `boolq` | `boolq`, configuration `boolq` | validation | acc |

`lambada` deliberately selects the [OpenAI variant](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.10/lm_eval/tasks/lambada/lambada_openai.yaml),
not `lambada_standard` and not the multilingual `en` configuration. Its
perplexity scores the withheld target word, not corpus PPL. It is a
loglikelihood task, unlike the multiple-choice tasks. WinoGrande uses the
official distinct candidate contexts and shared suffix scoring.

BLiMP membership and aggregation follow the [official group](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.10/lm_eval/tasks/blimp/_blimp.yaml).
It compares complete grammatical/ungrammatical sentence likelihoods. The suite
summary is stored separately under `benchmark_summaries.blimp`; all67 original
results remain under `benchmarks`. A partial suite cannot produce a suite mean.
No overall average mixing these families is generated.

For **only these eight families** (74 task results), pass:

```bash
--task-groups blimp lambada hellaswag piqa arc_easy winogrande arc_challenge boolq
```

For the previous six-task English suite, explicitly select:

```bash
--task-groups xnli belebele xstorycloze paws-x hellaswag arc_easy
```

New groups are English-only. Future multilingual runs retain their existing
task mappings and report unsupported group/language pairs explicitly. Adding
these evaluation tasks does **not** expand the fine-tuning task list: it remains
HellaSwag/ARC-Easy/XNLI. Evaluation accepts checkpoint5000 without code changes.

### Download preparation for later B200 use

The following is a **template only**, not a submitted command. Confirm the
destination root and authorize downloads separately. ARC-Challenge reuses the
ARC-Easy repository, so do not download it twice. Preserve org/repo directories.

```text
#d +a
#datasets
--hf-dataset nyu-mll/blimp /mnt/local/_data/@PROJECT@/benchmarks/nyu-mll/blimp
--hf-dataset EleutherAI/lambada_openai /mnt/local/_data/@PROJECT@/benchmarks/EleutherAI/lambada_openai
--hf-dataset baber/piqa /mnt/local/_data/@PROJECT@/benchmarks/baber/piqa
--hf-dataset allenai/winogrande /mnt/local/_data/@PROJECT@/benchmarks/allenai/winogrande
--hf-dataset allenai/ai2_arc /mnt/local/_data/@PROJECT@/benchmarks/allenai/ai2_arc
--hf-dataset aps/super_glue /mnt/local/_data/@PROJECT@/benchmarks/aps/super_glue
--hf-dataset Rowan/hellaswag /mnt/local/_data/@PROJECT@/benchmarks/Rowan/hellaswag
```

This covers only the eight-family core. Default evaluation additionally needs
the four earlier XNLI/Belebele/XStoryCloze/PAWS-X repositories listed above.
The `#d --hf-dataset` form uses the runner's resolved snapshot revision; compare
downloaded files to the tested manifest before use. Main branches may change.
Use complete parquet snapshots for WinoGrande and SuperGLUE, as in the whole-repo
commands above. Their README metadata lists multiple configurations; the current
HF local-directory loader may inspect a different configuration during format
inference even when `winogrande_xl` or `boolq` is selected. A hand-pruned directory
containing only the selected configuration can therefore fail to load. This
does not change which split/configuration the evaluator scores.
For exact revisions, use the supported direct resolve-URL mode per file with
operator-managed credentials when necessary, never direct GPU-node downloads.

`resources/english_core_benchmark_files_20260908.json` records the tested
seven repository revisions and SHA256/size of133 required metadata/parquet files.
It covers full parquet snapshots for WinoGrande/SuperGLUE and selected English
files for LAMBADA; extra downloaded files are allowed. Verify later downloads:

```bash
python scripts/verify_manifest.py verify --root /absolute/benchmarks \
  --manifest resources/english_core_benchmark_files_20260908.json
```

These hashes establish identity with the tested pinned snapshots, not model
accuracy. Do not regenerate the reference manifest from a mismatching B200
download merely to make verification pass; inspect any revision difference.

### Dev-only smoke test on real snapshots

After retrieving the snapshots on dev (or through an authorized #d on B200),
run in the evaluation environment with a fresh output directory:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
python -m scripts.smoke_english_benchmarks \
  --dataset-root /absolute/benchmarks --output-dir temp/fresh_benchmark_smoke \
  --examples-per-task 2 --arms B0 C
```

This CPU-only test validates full dataset split counts, then scores just two
real examples for each of74 tasks on tiny random B0/C models with a byte-level
smoke tokenizer. Its outputs are explicitly marked `smoke_only`: they are not
research metrics. No checkpoint/data downloads or GPU operations occur inside
this script. Production eval has no sample-limit option and checks full coverage.

## Multi-checkpoint evaluation

Optional frequency loss, spectra and fixed-probe gradient workers can share
this queue via `--diagnostic-bundle`; see [DIAGNOSTICS.md](DIAGNOSTICS.md) for
preparing immutable inputs and the additional job count. Default queue behavior
is unchanged when no bundle is supplied.

Example only: select actual checkpoint paths and a fresh external output root.
Activate the verified evaluation environment and ensure the selected physical
GPUs are free first. The queue cannot stop burns and never signals GPU PIDs.

```bash
python -m eval.eval_parallel \
  --checkpoints B0=/absolute/run/B0/checkpoint-10000 A128=/absolute/run/A128/checkpoint-10000 \
  --eval-dir /absolute/Qwen_Qwen3-0.6B/eval \
  --dataset-root /absolute/benchmarks \
  --preprocessing-cache-dir /absolute/swt/qwen_en_map160_batch1000 \
  --languages en --precision bf16 --gpus 0 1 2 3 4 5 6 7 \
  --output-dir /absolute/fresh_eval
```

Add A256/A512/C/D entries identically. `--dry-run` prints the exact job plan
without GPU queries or execution (checkpoint config paths must exist).
PPL and benchmarks are separate workers per checkpoint: six checkpoints
create12 queued jobs, up to eight simultaneous, each with its own physical
`CUDA_VISIBLE_DEVICES`. There is no DDP or Accelerate launch for these jobs.
Never start burns on a GPU merely because its last job finished: pending
queue jobs may still need it. Handle authorized burns only after the whole
queue exits and a fresh free-GPU check, or explicitly exclude burn GPUs.

Single checkpoint: `python -m eval.eval_checkpoint --checkpoint ... --eval-dir ...
--dataset-root ... --languages en --output ...`. Optional `--ppl-only` or
`--bench-only`. The old `evaluate_capacity.py` remains the original pooled
PPL-only entry point; use the new entry point for per-language reports and
benchmarks. With English only, PPL uses the same packing/scored-token logic.

## Fine-tune then evaluate

```bash
python -m finetune.run_all \
  --checkpoints B0=/absolute/run/B0/checkpoint-10000 A128=/absolute/run/A128/checkpoint-10000 \
  --dataset-root /absolute/benchmarks \
  --tasks hellaswag arc_easy xnli --seeds 42 123 456 \
  --train-language en --languages en --precision bf16 \
  --gpus 0 1 2 3 4 5 6 7 --output-dir /absolute/fresh_finetune
```

Each checkpoint/task/seed is one independent single-GPU worker. It saves a
reloadable HF model plus `training_complete.json`, then runs the selected
benchmarks in memory and writes `result.json` only after successful full
evaluation. A training marker alone does not establish evaluation success.

Queue outputs are never written into pretraining checkpoints. Both queues
require fresh output roots, validate worker exits/result provenance and
coverage, and write `complete.json` only when every requested job succeeds.
After failure, no more jobs are launched; already-running workers finish
naturally. Failed/partial directories are retained, never silently skipped
or overwritten. There is no automatic resume of partial fine-tuning.

## Future multilingual runs

- PPL/zero-shot: `--languages en,ar,de,ru,vi,zh`. Available task mappings
  live in `eval/benchmarks.py`; optional `--task-groups ... xcopa` adds XCOPA.
  Unavailable group/language pairs are reported, not invented or replaced
  with English. Every selected language must have some benchmark coverage.
- Fine-tune training language is independent: `--train-language en
  --languages en,vi` means English fine-tuning plus English/Vietnamese scoring,
  **not** mixed-language fine-tuning. A different single train language can
  be selected only when that task provides an actual training split.
- Every requested eval language must exist for the chosen fine-tuning task.
  For example, Chinese HellaSwag is not in the registered suite; request
  supported tasks/languages separately instead of silently skipping it.
- Mixed-language supervised fine-tuning in one worker is not implemented.
  It needs an explicit mixing/budget protocol before adding it.
- Add required language-specific snapshots via `#d`; model/task tokenization
  remains identical across languages and no code fork is required.

## Verification

Basic tests run in `swt`; actual harness integration tests additionally need
the pinned evaluation dependencies. They use synthetic local fixtures, never
download benchmarks or access GPUs. Run:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  python -m unittest discover -s tests -v
```

These tests do not establish real benchmark availability, B200 evaluation
memory/throughput, or research accuracy. No evaluation/fine-tuning job has
been appended to the currently running training-and-burn workflow.
