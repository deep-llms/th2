# Qwen3 base-model end-token probe — 2026-09-26

The unmodified Qwen/Qwen3-0.6B-Base checkpoint generated `<|endoftext|>`
(token 151643) naturally in both greedy and sampled decoding. This is a base
checkpoint, identified by its official model card as training stage Pretraining.

## Method

- Revision: `ddc928429ed09d9ad603fd762053d0434c15e865`.
- Cached weights were reused; actual SHA256 verified against the pinned HF API:
  `cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba`.
- Local CPU, fp32, SDPA, 8 threads, Transformers 4.57.1. All local GPUs were busy
  with another workload; no GPU processes were changed. B200 was untouched.
- Eight plain-text prompts, no chat template, no special tokens in unmasked input.
- Greedy and sampled decoding, 192-new-token cap each. Sampling: seed 42,
  temperature 0.8, top-p 0.95, top-k 50. Fresh GenerationConfig; no forced EOS,
  forced BOS, minimum-length constraint, or custom logits processor.
- Stop token follows the saved model generation config: 151643. Batched inputs
  were left-padded with that ID and attention-masked. Results inspect only newly
  generated tokens, retaining the first EOS and excluding subsequent padding.

## Results

| Mode | Generations | Naturally emitted endoftext | Emitted im_end | Reached cap |
|---|---:|---:|---:|---:|
| Greedy | 8 | 1 | 0 | 7 |
| Sampled | 8 | 1 | 0 | 7 |

Greedy prompt 5 emitted EOS as generated token 70. Sampled prompt 7 emitted EOS
as generated token 72. Output-ID and completion-count checks passed. This small,
handpicked probe demonstrates capability, not a representative stopping rate.

The tokenizer metadata at this revision names `<|im_end|>` (151645) as EOS,
but config.json and generation_config.json specify `<|endoftext|>` (151643).
Do not assume tokenizer.eos_token_id equals the base model's configured stop ID.

## Interpretation and sources

Qwen's official [control-token documentation](https://qwen.readthedocs.io/en/latest/getting_started/concepts.html#control-tokens)
explains that its pretraining pipeline inserts `<|endoftext|>` between documents;
chat formatting uses `<|im_end|>` at turn ends. A tokenizer call not automatically
appending EOS therefore does not imply that Qwen pretraining omitted end tokens.
This documentation resolves the earlier uncertainty based on the technical report.

[Official base model card](https://huggingface.co/Qwen/Qwen3-0.6B-Base).
Inference alone cannot reconstruct the full pretraining recipe. If a model were
trained from scratch with no end-token targets, that objective would not teach
reliable stopping; having the token in its vocabulary alone is insufficient.
Our continued-training models start from pretrained weights with learned behavior.

The user explicitly accepted current packing. No packing or sampling changes made.

## Local evidence

- Script: `temp/probe_base_eos_20260926.py`.
- Setup, prompts/configs: `temp/qwen-base-eos-20260926-a01/setup.json`.
- Full text/token outputs: `temp/qwen-base-eos-20260926-a01/results.json`.
- Completion: `temp/qwen-base-eos-20260926-a01/complete.json`.
- Execution log: `temp/probe-base-eos-20260926.log`.
- Remote identity metadata: `temp/qwen-base-identity-20260926.json`.
