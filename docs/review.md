# Implementation Review — Round 2

**Reviewed:** commit `c2b75be`. New findings from this pass only. (Round-1 items resolved:
multi-head is now implemented and wired, `num_heads` default 1.)

---

## 1. CRITICAL — Eval never loads the compositional embedding (silently wrong ppl/bench)

`eval/eval_checkpoint.py` → `eval/ppl.py` loads only the backbone via
`AutoModelForCausalLM.from_pretrained(checkpoint)` and calls `model(chunk_input_ids, labels=...)`
— it **never loads `embedding.pt`, never reconstructs the compositional module, and never uses
`inputs_embeds`.** (Verified: no reference to `embedding.pt` / `inputs_embeds` / the compositional
classes anywhere under `eval/`.)

But training saves the two halves **separately**: the backbone (with `embed_tokens = None`) to
`output_dir/backbone`, and the trained compositional embedding to `output_dir/embedding.pt`. So at
eval, `from_pretrained` rebuilds a **fresh random `embed_tokens`** from the config (the saved
state-dict has no such weights), and the trained compositional embedding is **ignored entirely**.

Consequence: for **every** compositional arm (ANT, V0, V1, V2, isolation-control, original_ant),
eval runs the trained backbone + trained `lm_head` on a **random input embedding**. It does not
crash — it produces **plausible-looking but meaningless** perplexity and benchmark numbers. Only
the Standard baseline (`train.py`, which saves a normal full model) evaluates correctly.

This makes the entire measurement pipeline invalid for the arms under test.

**Fix — eval must apply the compositional embedding, one of:**
- **(a) Compositional-aware eval:** load the backbone, set `embed_tokens = None`, rebuild the
  arm via `build_arm(...)`, `load_state_dict(torch.load("embedding.pt"))`, then run
  `e, _ = embed(input_ids); model(inputs_embeds=e, ...)`. Mirror the training forward.
- **(b) Install the embedding as an `embed_tokens` shim** (returns `e` only, stashes `theta`):
  then standard `model(input_ids)` — and `lm_eval` benchmarks, which call the model with
  `input_ids` internally — work unchanged. This is the more robust option because the benchmark
  harness (`eval/benchmarks.py`) cannot easily be switched to `inputs_embeds`.
- Either way: add a sanity check — ANT/V2 eval ppl on a tiny sample must be **far** below the
  ~vocab-size random floor before trusting any number.

## (withdrawn) bf16 embedding — NOT a bug
An earlier draft flagged `embed = embed.to(torch.bfloat16)`. **Verified false alarm:** the Qwen3
config ships `dtype=bfloat16` and `from_config` (transformers 5.9) creates the **backbone in
bf16** too, so the embedding cast simply matches it — no asymmetry, and the entmax `.float()`
guard still works (scores are bf16 under autocast regardless of param dtype). No change.

*Aside (not a bug, not blocking):* consequently the whole setup — baseline and all arms — trains
with **bf16 params + bf16 optimizer states** (no fp32 master weights). Uniform across arms, so it
does not bias the comparison; it is only a deviation from the standard fp32-master AMP recipe that
can slightly affect absolute quality. If you want the standard recipe, set
`config.dtype = torch.float32` before `from_config` in **both** trainers (parity). Optional.

---

## Concerns checked (from the round-2 note)

- **`model.model.embed_tokens = None` (train line ~353)** — correct and intended for *training*
  (forward is driven by `inputs_embeds=e`). But it is **not** merely an eval caveat: eval doesn't
  compensate for it, which is exactly Finding #1. `model.generate()` / any `input_ids` path is
  also broken until the embedding is reinstalled (Finding #1 fix (b) resolves both).
- **`set_format(type="torch", columns=["input_ids"])` drops `labels` (train line ~486)** —
  **confirmed fine, not an issue.** `compute_loss` shifts `input_ids` internally
  (`shift_labels = input_ids[:, 1:]`), so no separate `labels` tensor is needed.

---

**Priority:** #1 (eval) is the one blocker — without it there are no valid results for any
compositional arm; fix before/with the first real run. The bf16 concern is withdrawn (verified
not a bug). The two prior-round operational items (arm launch scripts, `prepare_data` tokenizer
slug) still stand but are unchanged.
