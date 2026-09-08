# Qwen3 capacity-allocation experiments

Current implementation, 2026-09-08. This supersedes the GPT-2/Llama setup in the
historical research drafts. All models train **from scratch**. No pretrained
weights, research training, or remote jobs are launched by this rewrite.

## Model arms

| Arm | Residual widths | Vocabulary interface | Parameters |
|---|---|---|---:|
| B0 | 1024 × 6 | Stock Qwen3, exactly tied 1024-wide table | 249,969,152 |
| A128 | 1024 × 6 | Independent input 128 / output 896 | 251,017,728 |
| A256 | 1024 × 6 | Independent input 256 / output 768 | 251,017,728 |
| A512 | 1024 × 6 | Independent input 512 / output 512 | 251,017,728 |
| C | 256,256,512,512,1024,1024 | Independent 128 / 896 | 198,485,504 |
| D | 512,512,1024,1024,1280,1280 | Independent 128 / 896 | 247,117,568 |

B0 uses stock HF Qwen3ForCausalLM with Qwen3-0.6B dimensions and six layers:
vocabulary 151936, width 1024, FFN 3072, 16 query heads, 8 KV heads, head dimension
128, Q/K RMSNorm, RoPE theta 1,000,000, maximum positions 40960.
Training blocks remain 2048 tokens. No dropout or attention bias.
Uniform arms also support 12 layers; C/D are defined only at depth six.

Custom models reuse HF Qwen3 blocks, masking, rotary embeddings, causal LM loss,
generation and dynamic cache. For stage width d, FFN width is 3d, query heads
are 2d/128, and KV heads are d/128. This preserves Qwen3's query/KV ratio of two
and query projection width 2d. Each block has 15d² + 2d + 256 parameters,
including two per-head Q/K norms. T1 transitions project after whole blocks,
only at width changes. Final RMSNorm precedes the output adapter.

Custom interfaces and transitions retain the explicit fan-in initialization:
adapter std 1/sqrt(fan_in); output-table std 0.02*sqrt(1024/output_rank).
The body retains HF initialization. B0 has no custom modules. The arms are not
exactly total-parameter-matched; compare the actual counts above.

## Reuse the old sparse-embedding data and training workflow

There is no new sampling prerequisite or custom binary-token format.
Use the old prepare_data.py outputs, saved as Hugging Face text datasets:

```text
Qwen_Qwen3-0.6B/
  train/en/shard_0000/    # saved Dataset, text column
  train/en/shard_0001/
  eval/en/               # saved Dataset
```

Pass the parent train/ and eval/ directories; --languages en is the default.
Other languages are opt-in (comma-separated). Missing selected languages fail.
The B200 candidate root is
/mnt/local/_data/deep-llms_th2/data/Qwen_Qwen3-0.6B;
verify those sampled directories before any launch. Raw parquet alone does not
establish that sampling outputs exist.

The old corpus contains approximately 30B sampled English tokens and a separate
10M-token English evaluation set. Reusing it does **not** create the former
20M validation + 20M final-test reservations. Do not claim a separate final test
unless one is explicitly prepared. Previous GPT-2 partial output is not an input.

Training loads and concatenates saved text shards, applies batched
Dataset.map tokenization with configurable CPU processes, then batched
concatenation/packing and shuffle, as in sparse_embedding/train.py.
No EOS is inserted (matching the old trainer); existing EOS tokens are not masked.
Each map batch drops its short tail. Worker count and map batch size can therefore
affect packing: keep both fixed across arms. Preprocessing cache keys include
source Dataset fingerprint, tokenizer, code, workers, batch size and block size.
HF fingerprints are cache identities, not a fresh cryptographic corpus audit.
New cache names deliberately avoid treating old unverified caches as compatible.

No SQLite deduplication or replacement corpus is generated here. This reuses
the original sampled corpus and its deduplication properties; a new cross-split
duplicate audit is not claimed.

## Train

train.py follows the original HfArgumentParser + TrainingArguments + Trainer
interface. Model construction is isolated in capacity_allocation/modeling.py.
Tokenizer loading is local-only; use the existing Qwen3 tokenizer directory.
The example below illustrates syntax, not tuned settings or an approved job.

```bash
accelerate launch train.py \
  --arm B0 --num_hidden_layers 6 \
  --tokenizer_name /path/to/local/Qwen3-0.6B \
  --data_dir /path/to/Qwen_Qwen3-0.6B/train \
  --eval_data_dir /path/to/Qwen_Qwen3-0.6B/eval \
  --languages en --block_size 2048 \
  --preprocessing_num_workers 16 --preprocessing_batch_size 1000 \
  --preprocessing_cache_dir /path/to/swt_cache \
  --output_dir /path/to/fresh/B0 \
  --num_train_epochs 1 --stop-at-step 10000 \
  --per_device_train_batch_size 4 --gradient_accumulation_steps 16 \
  --per_device_eval_batch_size 1 --bf16 \
  --learning_rate 3e-4 --lr_scheduler_type cosine_with_min_lr \
  --lr_scheduler_kwargs '{"min_lr_rate":0.1}' --warmup_steps 500 \
  --weight_decay 0.1 --adam_beta1 0.9 --adam_beta2 0.95 \
  --max_grad_norm 1 --seed 42 --data_seed 42 \
  --ddp_find_unused_parameters false --ddp_timeout 21600 \
  --save_steps 250 --logging_steps 10 --report_to none
```

Do not set max_steps. Epoch count/data determine the full LR schedule;
--stop_at_step / --stop-at-step is a save-and-stop callback only. A 10k-step
cutoff is not a 10B-token horizon. Freeze the actual effective batch and schedule
before comparing arms. Standard TrainingArguments control BF16, optimizer,
scheduler, logging, checkpointing, DDP and DataLoader workers.

Two narrowly scoped correctness fixes remain in the common CausalTrainer:
count shifted labels for HF 5.9 accumulation normalization, and aggregate
per-example NLL sums/target counts for exact distributed evaluation.
They apply identically to the stock B0 model and every custom arm.
EOS labels are never discarded merely because EOS is also a pad ID.

A checkpoint inside the same output directory resumes automatically (or use
--resume_from_checkpoint). Resume verifies recorded data/config/code.
A successful result.json or a nonempty directory without a checkpoint is refused.
Checkpoints retain the ordinary Trainer optimizer/scheduler/RNG state.
final/ includes HF model files and tokenizer; it is for evaluation, not exact resume.
Success is written only after the expected training step, final save, and optional
evaluation complete. GPU stopping/burns are outside this script.

## Evaluate and test

```bash
python evaluate_capacity.py --checkpoint /path/to/run/final \
  --data_dir /path/to/Qwen_Qwen3-0.6B/eval --languages en \
  --block_size 2048 --preprocessing_num_workers 16 \
  --preprocessing_batch_size 1000 --preprocessing_cache_dir /path/to/swt_cache \
  --device cuda --precision bf16 --batch-size 1 \
  --output /path/to/fresh/eval.json

python -m unittest discover -s tests -v
python -m scripts.smoke_capacity --output temp/qwen_smoke.json
torchrun --standalone --nproc_per_node=2 -m scripts.smoke_capacity \
  --output temp/qwen_ddp_cpu.json
```

Use the same preprocessing settings for training/evaluation. A --split test
label does not construct a new test set: point --data_dir at genuinely held-out
data. Tiny CPU tests and meta-device counts do not establish B200 speed/memory.
GPU smoke tests require separately authorized free GPUs; none is launched here.
commands.sh stays inactive in the development repository.
