# A2 pilot: implementation spec (Phase 0A and Phase 0B)

Version 2.1, 2026-10-28. This spec is written for the agent that implements and runs the pilot.
- **The "why"** is in `A2_novelty_and_feasibility.md`. This file is the "what" and "how"; it is self-contained for running the pilot.
- **History:**
  - version 1.1 applied an independent implementability review;
  - version 1.2 applied a second review and checks of the public models and of transformers;
  - version 2.0 replaced Phase 1(a), our own narrow WSD run, with Phase 0B, which asks the same question on public checkpoints of the open-sci-ref runs. Phase 1(a) is deferred to `A2_phase1a_deferred_spec.md`;
  - **version 2.1** answers the implementer's review of version 2.0 (`A2_PILOT_REVIEW_20261028.md`) before any result was seen. Each point and its resolution are listed at the end.
  
  The changes are listed at the end.
- **This file is the pre-registration.** Do not change the layer sets, budgets, learning-rate grids, metrics or decision rules after any result is seen. If something cannot be done as written, stop and report; do not improvise.

## 0. Summary

**The questions.** Whole layers are removed from a language model, and the model is then healed by distillation. Two questions follow:
1. **Training length:** does the remaining damage depend on how long the parent was pretrained, with the parent's learning-rate (LR) state held fixed?
2. **LR state:** at the same training length, is it better to prune a checkpoint before or after its LR has been annealed?

**The key fact.** Each open-sci-ref v0.01 dataset has a 300B-token run and a 1T-token run that form one WSD tree:
- the two runs share their weights up to step 58,000;
- the 300B run then anneals (cools its LR down to 0) by step 72,661;
- the 1T run keeps its peak LR until about step 193,763, then anneals by step 242,204;
- the 1T repository publishes checkpoints from that constant-LR stretch.

So, per dataset, four public parents sit on one tree:

| Parent | What it is | Tokens | TPP | LR state |
|---|---|---|---|---|
| **S72** | 1T run, step 72,000 | 297.3B | 184.5 | Stable (LR at its peak) |
| **P300** | 300B run, final | 300.0B | 186.2 | Annealed |
| **S188** | 1T run, step 188,000 | 776.2B | 481.7 | Stable |
| **P1T** | 1T run, final | 1,000.0B | 620.6 | Annealed |

**What to run:**

| Part | Parents | Compares | Cut | Healing | Compute |
|---|---|---|---|---|---|
| **0A** | P300 and P1T, three datasets | Training length at the annealed state (186 vs 621 TPP) | 6 of 24 layers | Pure KD, 0.03–0.8B tokens | About 11.7 node-hours |
| **0B** | S72 and S188, three datasets, plus arm X72 (S72 healed from P300) | LR state at almost equal length (S72 vs P300); training length at the stable state (S72 vs S188, 184 vs 482 TPP) | 6 of 24 layers | Pure KD, 0.03–0.2B tokens | About 11.1 node-hours |

**What to deliver:** the results files and a report (section 7). The decision quantities must be computed mechanically by the rules in sections 5.3 and 6.4; the decision itself is made by the project owner. **Do not run the deferred phases** (section 8) until told.

**What the pilot can and cannot show.**
- It is a screening experiment on public checkpoints. Each comparison is evidence under this parent, teacher and healing recipe; none isolates training length or LR state alone (sections 4.1 and 6.4 list the confounds).
- Its labels are pre-registered screening rules, not significance tests. The 6 or 12 contrasts behind a label share parents and evaluation data, so they are not independent replications.
- A GO label does not establish novelty; that assessment is in the design note.

## 1. Ground rules

1. **Pre-registration.** Record the SHA-256 of this file in every output. Do not tune anything on results: choose LRs only by the grids and rules given, and use only the layer sets listed.
2. **Budgets.**

   | Part | Estimate | Report before launching if the re-estimate exceeds | Hard cap |
   |---|---|---|---|
   | 0A | 11.7 node-hours | 15.0 | 18.0 |
   | 0B | 11.1 node-hours | 14.5 | 17.0 |

   - The estimates rest on the team's older throughput figure, not on a measurement on this node (8 × B200, as the implementer reports). They cover 127 heal runs (56 in Phase 0A, T14's 2, and 69 in Phase 0B), before any divergence reruns.
   - **Re-estimate the whole workload** after the throughput test (T8), not only the KD steps:
     - every heal step;
     - every evaluation (final, tuning, small and secondary sets);
     - the fp32 Gaussian-sensitivity passes and the other measurements of section 3.4;
     - model loading and checkpoint saving.
   - These costs count toward the triggers and caps. Data preparation and downloads do not count; track and report them separately.
   - Stop any run that exceeds 1.5× its own estimate, and report it.
   - Prefer a sharded optimiser state over recomputing activations: recomputation would raise the KD cost by about 23%.
3. **Divergence and reruns.**
   - **Divergence:** a heal run diverges if its loss becomes NaN or infinite, or if its tuning-set val after healing is worse than the same student's unhealed tuning-set val. The final set is never used for this check.
   - **Non-grid run:** rerun it once at half the LR. The rerun replaces the diverged run in every decision quantity.
   - **LR-grid run:** rerun it at half the LR for the record only; that LR is never a candidate (section 5.1).
   - **Second divergence:** record the run as failed and continue.
   - **Failed runs in the decision quantities:** any quantity that needs a failed run is undefined. Never substitute the val of a failed run.
     - A label whose inputs are undefined becomes **INCOMPLETE**, with the failed runs listed (sections 5.3 and 6.4 say when).
     - INCOMPLETE means: report to the project owner, who decides whether to repair the runs. It never triggers a deferred phase.
   - **Run IDs** have the form `P0-<ds>-<arm>-<set>-<budget>-lr<value>`, where `<arm>` is `P300`, `P1T`, `S72`, `X72` or `S188` (section 6.1), and the LR is written as in this spec (for example `-lr3e-4`). A rerun carries its own, halved LR and adds `-rerun` (for example `-lr1.5e-4-rerun`).
4. **Records.** For every run, record:
   - the GPU type and count, and the software versions (including transformers);
   - the code commit;
   - the Hugging Face commit SHA of every downloaded model or dataset;
   - the seeds, wall-clock time and GPU-hours.
5. **Do not:**
   - add an LM (next-token) loss to healing;
   - heal or validate on documents outside the streams defined here;
   - change the order of the heal data between runs (T14(b) is the only exception);
   - patch the RoPE code of any model;
   - use generation or sampling anywhere (all metrics are teacher-forced losses);
   - run a deferred phase.

## 2. Definitions

| Term | Meaning |
|---|---|
| Parent | An unpruned model: P300, P1T, S72 or S188 of one dataset (section 4.1) |
| Stable, annealed | The LR state of a checkpoint. Stable: the LR is still at its peak. Annealed: the run has finished its linear cooldown to 0 |
| Teacher | The frozen model a student is distilled from. It is the student's own parent, except in arm X72, where the student comes from S72 and the teacher is P300 |
| Group g_i | Layers {2i, 2i+1}. The 24-layer parents have g0–g11 |
| Layer set | The groups removed from a parent, fixed in advance (section 4.2) |
| val(m) | Mean token cross-entropy of model m in nats, teacher-forced, on the **final** set (section 3.1) unless another set is named. It is the mean over all T−1 next-token predictions of every sequence, computed by the procedure of section 3.2 ("Precision") |
| val_tune(m) | The same on the tuning set (section 3.1). Every choice in this spec (healing LR, S2, divergence) uses the tuning set or its small subset, never the final set |
| Step s | The state after s optimiser updates |
| Linear ramp | A ramp over k updates from LR a to LR b gives the j-th of those updates (j = 1, …, k) the LR a + (b − a) · j / k. A warm-up from 0 therefore starts at peak/k, and a decay to 0 ends with an update at LR 0 |
| Δ_pre(set) | val(parent with the set removed, unhealed) − val(parent) |
| Δ_post(set, H) | val(the pruned model after H tokens of KD healing) − val(parent) |
| TPP | Pretraining tokens per non-embedding parameter |
| Node-hour | One hour of the whole 8-GPU node |

## 3. Shared components (build once)

### 3.1 Data

**Source.** All heal and validation text comes from the Hugging Face dataset `nht10/cx_sampled`, subset `qwen3_0.6b_base_en_30B`, pinned at one revision.
- It stores raw English document text (CulturaX) in one `text` column.
- It has two splits: `train` (about 30.0B Qwen3 tokens) and `validation` (about 10.0M Qwen3 tokens).
- Validation texts' exact hashes are already removed from train. Record the revision and the exact counts.

**Reserved documents (the heal and tuning source).**
- **Definition:** a uniform random sample of `train` documents. Compute `numpy.random.default_rng(2028).permutation(N)` over the N document indices of the split, and keep the shortest prefix whose Qwen3 token counts sum to at least 1.2B.
  - Counting in Qwen3 tokens keeps the set identical to the one the deferred Phase 1(a) would use.
  - Random sampling, rather than the last documents in stored order, keeps the sample representative if the stored order has structure.
- **Order:** the order of that prefix. This order is used everywhere; there is no second shuffle.
- **Record** the document indices, so that any later pretraining on this dataset can exclude them.

**Heal and tuning documents.** Tokenise the reserved documents, in their order, with the parents' GPT-NeoX tokenizer (`add_special_tokens=False`, no BOS), and append the end-of-text id 0 to each. Then split them, in order, into three disjoint groups:
- **Heal documents:** the shortest prefix that holds at least 0.85B GPT-NeoX tokens.
- **Tuning documents:** the documents that follow, up to the shortest run that holds at least 4,194,304 tokens (1,024 × 4,096).
- **Unused documents:** the rest. Record their count.

If the reserved set is too small for both, extend it by continuing the same permutation, and report it. Record the GPT-NeoX/Qwen3 token ratio.

**Heal stream.** The heal documents, concatenated in order and cut into consecutive 4,096-token sequences, with the final partial sequence dropped. **Every heal run reads the stream from its first sequence, in order.**

**Evaluation sets.**
- **Tuning set:** the tuning documents, concatenated in order and cut into 4,096-token sequences; keep the first 1,024 sequences (4,194,304 tokens). It serves every choice: the healing LRs, the divergence check and, through the small set, S2.
- **Small set:** 256 sequences of the tuning set (1,048,576 tokens), chosen by `numpy.random.default_rng(2029).choice(1024, 256, replace=False)` and kept in ascending index order. Store the indices in a manifest.
  - It serves the measurements of section 3.4 (bypass profiles, update norms, BI, Gaussian sensitivity), `final_kl` and the tests.
  - A random small set matters because the small set chooses S2.
- **Final set:** the validation documents in stored order, tokenised the same way (EOS id 0 after each), concatenated and cut into 4,096-token sequences, with the final partial one dropped. Expect roughly 2,400–2,800 sequences, depending on the token ratio; record the exact count.
  - **It only scores:** the parents' val, Δ_pre, the healed vals and the T15 diagnostics.
  - No choice in this spec uses it, so it stays an untouched evaluation set.
- **Secondary set (no decision attached):** 256 × 4,096 tokens from each dataset's own corpus.
  - Use DCLM-baseline, FineWeb-Edu and the Nemotron-CC high-quality part, taken from a fixed public shard (record which).
  - **Best effort:** if a corpus cannot be obtained within about one hour of effort (Nemotron-CC is distributed outside Hugging Face datasets), skip that one and record it.
  - Label them "may overlap pretraining, more for longer-trained parents".

**Attention.** Plain causal attention over packed sequences, as in the parents' Megatron pretraining. T9 records the effect of this choice.

**Overlap caveat (state in the report).** CulturaX, DCLM, FineWeb-Edu and Nemotron-CC all derive from Common Crawl. The heal and validation text may overlap the parents' pretraining data, and the longer-trained parents saw more of their corpus (up to about 3.4×). The secondary set and the report must address this.

### 3.2 Models and layer removal

**Software.** Pin transformers **4.57.1** (any 4.54–4.57 release works), and record the version. Checked on 2026-10-28:
- **Lower bound:** the `main` branches of all six repositories hold the same `modeling_opensci.py` (git blob `f24e1d0a…`, updated 2025-09-26). It imports `TransformersKwargs`, which exists only from transformers 4.54, although the configs still say 4.49.0.
- **Upper bound:** the same file calls `ROPE_INIT_FUNCTIONS["default"]`, which 5.x removed.
- **Branches:** the `iter_*` branches hold an older `modeling_opensci.py` (git blob `5a0a3a9d…`) that imports `LossKwargs`, which exists only up to 4.53.
- Do not patch RoPE.

**Loading.**
- **P300 and P1T:** `AutoModelForCausalLM.from_pretrained(repo, revision=<pinned main commit>, trust_remote_code=True)`.
- **S72 and S188:** load the weights from the pinned branch commit, but with the code of the same repository's pinned `main` commit: `from_pretrained(repo, revision=<pinned branch commit>, code_revision=<pinned main commit>, trust_remote_code=True)`.
  - The branch's own code cannot be imported on transformers 4.57.
  - **Fallback**, if `code_revision` fails: build the model from `main`, then load the branch's `model.safetensors` with `load_state_dict(strict=False)`. Assert that no key is unexpected and that the missing keys are at most the tied `lm_head.weight`.
- **For every parent:**
  - use `attn_implementation="sdpa"`, `use_cache=False` and no attention mask, so SDPA uses its causal kernel;
  - set the attention dropout to 0 **at load time**, by passing `attention_dropout=0.0` to `from_pretrained`; the configs say 0.1.
    - Each attention module copies the value when it is built, so changing `model.config.attention_dropout` after loading is not enough.
    - Assert that every attention module's `attention_dropout` is 0, in the parents and in every student.

**Removing layers.**
- Rebuild the decoder list from the kept original indices in one step, `model.model.layers = nn.ModuleList([layers[i] for i in keep])`, and set `config.num_hidden_layers` to the new count. The forward loop slices the layer list by this count.
- **Do not delete layers one index at a time.** Each deletion shifts the later indices, so an ascending loop silently removes the wrong layers. After removal, assert the kept original indices; each attention module's `layer_idx` still holds its original value.
- `layer_idx` matters only for the KV cache, which is never used.
- Nothing else is modified: no linear repair, no re-initialisation.

**Precision.** One procedure computes every val, whether for a parent, an unhealed model or a healed model:
- the weights take bf16 values, held in fp32 tensors;
- the forward pass runs under bf16 autocast;
- log-softmax and cross-entropy run in fp32.

How each kind of model gets there:
- **Parents:** their released bf16 weights, upcast once.
- **Healed students:** cast the fp32 master weights to bf16 (this is exactly the saved file), then back to fp32 tensors. **This scored representation determines every decision quantity.**
- **The fp32 master weights:** also evaluated once on the final set and recorded as a diagnostic (`val_post_master`), never used in a decision.
- **For KD,** the teacher is a separate bf16 copy (section 3.3).

The Gaussian-sensitivity measurement (section 3.4) is the one exception: it runs fully in fp32.

### 3.3 KD healing

| Setting | Value |
|---|---|
| Student | The parent with the layer set removed (section 3.2). In arm X72 the student comes from S72 (section 6.1) |
| Teacher | Frozen, eval mode, dropout 0, bf16 under `no_grad`. Keep only its final hidden states and compute logits chunk by chunk |
| Loss | Forward KL(p_teacher ‖ p_student) at each position: summed over the full vocabulary, temperature 1, then averaged over all 262,144 positions of a step, EOT positions included. Log-softmax in fp32, chunked over the sequence. **No LM loss** |
| Trainable | All student parameters, including the tied embedding. Fresh AdamW state for every run |
| Data | The heal stream from its first sequence. A run with budget H uses the first ⌊H / 262,144⌋ steps |
| Batch | 64 × 4,096 = 262,144 tokens per step. Use gradient accumulation as needed |
| Optimiser | AdamW, betas (0.9, 0.95), gradient clip 1.0. Weight decay 0.1 on every weight matrix, including the tied embedding; none on norms and biases. A sharded optimiser state is allowed |
| LR schedule | For a run of n updates: a linear warm-up from 0 to the peak over W = max(10, ⌈0.02 · n⌉) updates, then a linear decay to 0 over the remaining n − W updates (ramps as defined in section 2), so the last update has LR 0. Each budget is a separate run with its own full schedule; never stop a longer run early |
| Precision | bf16 autocast, fp32 master weights and optimiser state |
| Seed | 0 for everything. The data order is fixed by the stream |
| Logging | Every 10 steps: KL loss, gradient norm, LR. At the end, from the scored representation (section 3.2): val_tune(student), val(student) on the final set, and `final_kl`, the mean KD loss on the small set. Also `val_post_master` (section 3.2) |
| Saving | Every healed student as bf16 safetensors: the fp32 master weights cast to bf16, which is the scored representation (about 2.6 GB each, about 330 GB in all). If disk is short, keep at least the students healed with 0.2B and 0.8B tokens (about 85 GB) |

### 3.4 Measurements (for every parent, before any healing)

All 12 parents (four per dataset) get these:
1. **val(parent)** on the final set, the tuning set and the secondary set.
2. **Δ_pre** for each of the 4 layer sets, on the same three sets. The tuning-set values serve the divergence check (rule 1.3); the final-set values enter the decision quantities.
3. **Single-layer bypass profile:** for each layer l, remove only layer l, and record val − val(parent) on the small set.
4. **Relative update norm** of each layer l: the mean over tokens of ‖h_{l+1} − h_l‖ / ‖h_l‖, on the small set.
   - h_l is the residual stream entering decoder layer l. Capture it with forward hooks on decoder-layer inputs and outputs.
   - Do not use the last entry of Hugging Face's `hidden_states`, which comes after the final norm.
5. **Block Influence (BI):**
   - per layer: 1 − mean over tokens of cos(h_l, h_{l+1});
   - per group: 1 − mean cos(input of its first layer, output of its second layer);
   - both on the small set, with the same hooks.
6. **Gaussian sensitivity.** Run fully in fp32, with TF32 disabled.
   - For σ ∈ {0.005, 0.01, 0.02}, add σ · rms(W) · N(0, 1) noise to every linear weight matrix W in the decoder layers (not to embeddings or norms).
   - Measure val on the small set, minus the parent's val on the small set computed the same way (fp32, TF32 off).
   - Use noise seeds 0, 1 and 2; report the mean and spread.

### 3.5 Tests

Run them all before the heals, except T14 and T16, which run during the heals. Record the results, and send the pre-launch results to the project owner before any heal starts.

**Two kinds of test:**
- **Correctness tests** (T1–T8, the formula part of T10, and T11–T13, T16) check the implementation. A failure stops the affected part, which is reported.
- **Diagnostics** (T9, the loss part of T10, T14 and T15) are recorded and, when surprising, flagged in the pre-launch report. They never exclude a parent: a loss value on one evaluation sample is not proof that an implementation or a checkpoint is wrong, and excluding on it would select parents by an outcome related to the effect under study.

| Test | What | Pass |
|---|---|---|
| T1 | Load each of the 12 parents. Check the SHA-256 of the downloaded `model.safetensors` against the LFS SHA-256 that the Hub lists at its pinned commit. After loading, check that `lm_head.weight` shares storage with the embedding, and that every attention module's dropout is 0. Compute small-set val by the section 3.2 procedure and in full fp32, and record the small-set val of the pure-bf16 teacher copy (section 3.3) | The hashes match, the weights are tied and the dropout is 0. The two vals differ by less than 0.002 nats; values are about 2–4 nats. The bf16 teacher's val is recorded, not tested |
| T2 | **Pair identity and shared trunk.** For each dataset, at the pinned `main` commits (section 4.1): diff `config.json` of P300 and P1T, and hash the tokenizer files, `modeling_opensci.py` and `configuration_opensci.py`. Read the Megatron run names in the configs. Compare the LFS SHA-256 of `model.safetensors` on the two repositories' `iter_0058000` branches (metadata only; no download) | Identical except for name or path fields. Run names show the same global batch, peak LR, warm-up, schedule, RoPE θ and data mix, with only the length differing. The `iter_0058000` hashes are equal. Otherwise stop that dataset and report |
| T3 | **Branch identity**, for S72 and S188 of each dataset: diff the branch `config.json` against the 1T `main` one; read its run name; check which code was imported. Compare the weight hashes of the dataset's four parents. For S72 only, also compare its weights' LFS SHA-256 with that of the 300B repository's `iter_0072000` | The configs differ only in `_name_or_path` (and `transformers_version`, if present). The run name is the 1T run's (`samples-1000B`, the same job number as the 1T `main`) at the stated iteration. The imported `modeling_opensci.py` is the `main` blob. The four parents' hashes are distinct, and the S72 hash differs from the 300B one (that run was already cooling down). A failing branch parent is handled as in section 6.1 |
| T4 | **Removal correctness:** (a) removing no layers reproduces the parent's logits; (b) removing layers gives the same logits as replacing the same layers with identities through hooks, for single layers l = 0, 11 and 23 and for every full layer set; (c) save one pruned student, reload it from disk, and recompute its small-set val | (a) bitwise in fp32; (b) max absolute logit difference < 1e-4 in fp32, and the kept original indices are as listed in section 4.2; (c) the reload succeeds and the val matches within 1e-6 |
| T5 | KD loss in fp32, with the student an exact fp32 copy of the teacher | < 1e-6, with finite gradients |
| T6 | Print the LR used by updates 1, W (the end of warm-up), ⌈n/2⌉, n − 1 and n of a 0.2B heal | Matches the ramps of section 2: peak/W at update 1, the peak at update W, and LR 0 at update n |
| T7 | **Disjointness:** heal, tuning and validation documents are pairwise disjoint (by id, and by content hash) | No overlap |
| T8 | **Throughput:** tokens/s for a 1.7B KD step. Record the GPU type | Re-estimate every run (rule 1.2) |
| T9 | **Data diagnostics**, per parent: loss by position bucket (0–511, 512–2,047, 2,048–4,095) on the small set; small-set loss under plain causal attention vs document masking (4D mask with position reset); the GPT-NeoX/Qwen3 token ratio | Recorded. Flag a loss that does not fall across buckets. The masking comparison documents the assumption that the parents did not use document masking |
| T10 | **RoPE**, per parent. Correctness: `inv_freq` equals 1/θ^(2i/64) for the config θ. Diagnostic: the small-set loss at the other θ (10,000 vs 100,000), set by a config override on a separate copy for this test only (not a code patch) | The formula holds. The other-θ loss is recorded, and flagged if it is not higher |
| T11 | **BI and update norm:** computed from hooks (section 3.4); a layer or group replaced by an identity gives 0 | Holds |
| T12 | **KD implementation.** In fp32: the chunked loss and gradients equal the unchunked ones; gradient accumulation of 1 × 64 and 8 × 8 gives the same gradient; the teacher gets no gradient | Within 1e-5 |
| T13 | **Decision dry run:** feed the decision scripts synthetic inputs that hit every label and boundary of sections 5.3 and 6.4. Include abs(E) = 0.008 and 0.003, 10 of 12 and 5 of 6 sign counts, REFRAME and STOP both true, E_pre,d of mixed sign, an E_d of exactly ±0.003, undefined inputs (including exactly 5 of the 6 A_init values defined), and an X72 chosen LR at the top of the grid. Also feed the S2 rule (section 4.2) BI values whose lowest groups equal S1, and a dataset with a failed branch parent | Outputs match hand-computed values |
| T14 | **Run-to-run noise (during Phase 0A)**, on the run `P0-nem-P300-S1-0.05B-lr<chosen>`: (a) rerun it unchanged, as `…-t14a`; (b) rerun it with the same 12,160 heal sequences in an order permuted by `default_rng(1)`, as `…-t14b` | Report both val differences. (a) measures GPU nondeterminism only; (b) measures data-order noise. Neither enters a decision quantity |
| T15 | **LR-state diagnostics**, per dataset: the annealing gain val(S72) − val(P300), and val(S72) − val(S188), on the final set. If the open-sci-ref training logs (Hugging Face dataset `open-sci/open-sci-ref-0.01-logs`) can be read, also record the logged LR of each 1T run at steps 72,000 and 188,000 | Recorded; never used to exclude a parent. Flag, in the pre-launch report: an annealing gain below 0.02 nats, a non-positive second difference, or a logged LR below the peak. Without logs, record the LR history as "documented by the WSD schedule and the iteration, not verified from logs" |
| T16 | **Post-healing round trip (during the heals):** reload the saved bf16 file of the first heal run of each part, and of every 20th run after it; recompute its small-set val by the section 3.2 procedure | It equals the val logged for that student's scored representation within 1e-6 |

## 4. Parents and layer sets

### 4.1 Parents

All 12 parents are **open-sci-ref v0.01** models:
- **Architecture:** 1.7B, 24 layers, width 2,048, 32 query and 32 KV heads of size 64, MLP 8,192, attention and MLP biases, QK-norm, tied embeddings.
- **Tokenizer:** GPT-NeoX, with 50,304 tokens.
- **Training:** Megatron WSD runs with global batch 1,008 × 4,096 (4,128,768 tokens per step), peak LR 4e-3, 25,000 warm-up steps, RoPE θ = 10,000, and a 20% linear cooldown (final LR assumed to be 0).
  - The 300B run takes 72,661 steps; its cooldown starts at about step 58,129.
  - The 1T run takes 242,204 steps; its cooldown starts at about step 193,763.
- **Non-embedding parameters:** 1,611,355,136. Assert the count, and record TPP from it.

**Repositories** (all under `open-sci/`):

| Dataset | Code | 300B repository | 1T repository |
|---|---|---|---|
| DCLM | `dclm` | `open-sci-ref-v0.01-1.7b-dclm-300B-4096` | `open-sci-ref-v0.01-1.7b-dclm-1T-4096` |
| FineWeb-Edu | `fwe` | `open-sci-ref-v0.01-1.7b-fineweb-edu-1.4t-300B-4096` | `open-sci-ref-v0.01-1.7b-fineweb-edu-1.4t-1T-4096` |
| Nemotron-HQ | `nem` | `open-sci-ref-v0.01-1.7b-nemotron-hq-300B-4096` | `open-sci-ref-v0.01-1.7b-nemotron-hq-1T-4096` |

**Pinned commits** (checked on 2026-10-28; load exactly these):

| Dataset | P300: 300B `main` | P1T: 1T `main` | S72: 1T `iter_0072000` | S188: 1T `iter_0188000` |
|---|---|---|---|---|
| `dclm` | `9d02563017b25fd0a1e1849b7cd2bfd02b30d4e0` | `90481ca923a6cba5c8e107e2249e09fcd32ec617` | `2ee702ba4c05cd1494c530d81f4e338bcf70536b` | `0340fcb7b7717498ad8bcb229a653a31a7af0877` |
| `fwe` | `de8d9259fcac769157070eb63cd67fa89d4c2db1` | `ca1795337a73d08235c904609c4b6daa89ecd6ca` | `2acd776f32a524e413b02f9c2c5f1f3a49c8d4e5` | `85e25569105f1c84b0d653f6f8e69df1e30bd3de` |
| `nem` | `5cc94475d50d221ac201909c36aa61855fb2c50f` | `c9ccbca7125d7072cead55a1f0a07e53a3fb6de1` | `a69f22d194ba460ce35e9d29a8d6cc4af919aca1` | `10f3e6ae5e55164ce14709839c230a107d809aef` |

**What was checked on 2026-10-28** (T2 and T3 repeat the checks):
- **Run names.** Within each dataset, all four parents' run names show global batch 1,008, context 4,096, WSD, LR 4e-3, warm-up 25,000 and RoPE θ 10,000.
  - The data fields read `DCLM`, `FineWeb-Edu-1.4T` and `Nemotron-cc-2024-HQ-real-synth-mix` (the Nemotron-CC 2024 high-quality data, real plus synthetic).
  - S72 and S188 carry the 1T run's name and job number.
- **Shared trunk.** In each dataset, the `iter_0058000` branches of the two repositories hold identical weights (same LFS SHA-256). Up to step 58,000 the two runs are one run; the 300B run then cools down, while the 1T run continues at the peak LR.
  - The DCLM 300B repository's `iter_0072000` differs from the 1T one, as expected: by then the 300B run was cooling down.
- **Code and tokenizer.** All six `main` branches hold the same code and tokenizer files.

**Also note:**
- **S72 vs P300:**
  - S72 has seen 0.9% fewer tokens than P300 (297.3B vs 300.0B), which is negligible next to the annealing effect.
  - After step 58,000 the two runs read different parts of the same corpus.
- **Fallback:** if the `nem` pair fails T2, test `open-sci/open-sci-ref-v0.01-1.7b-nemotron-cc-hq-1T-4096` as the 1T partner and record the result. Its commits are not pinned here: pin them at first access and record them. That repository's branches would then have to pass T3 too.
- **Why v0.01 rather than v0.02:** the v0.02 1T repositories are re-uploads of runs with RoPE θ = 100k, and the v0.02 300B runs have no documented recipe.
- **No 100B model:** its RoPE θ (100k) differs from these parents.

### 4.2 Layer sets (24 layers, groups g0–g11)

The same sets are used for all four parents of a dataset.

| Set | Groups | Layers removed | How chosen |
|---|---|---|---|
| S1, deep block | g8, g9, g10 | 16–21 | Fixed; keeps the last group |
| S2, Block Influence | 3 groups | — | The 3 lowest values of the **mean group-BI of the dataset's usable parents** among g1–g10, on the small set. Usable means P300 and P1T, plus each branch parent that passes the identity tests T1 and T3. Ties go to the lower index. **If this equals S1, S3 or S4**, replace its highest-BI group with the lowest-BI group not yet in it, and repeat until it differs from all three. Record the raw choice, the final S2, which parents were used, and each parent's own choice |
| S3, interleaved | g3, g6, g9 | 6–7, 12–13, 18–19 | Fixed |
| S4, random | g1, g2, g6 | 2–5, 12–13 | Already drawn with numpy `default_rng(2027).choice(np.arange(1, 11), 3, replace=False)`; do not redraw |

**Why these rules:**
- S2 is chosen on all usable parents because choosing it on one parent would bias every contrast in that parent's favour. A failing branch parent therefore never holds up Phase 0A.
- S2 must differ from S1: Block Influence is often lowest in the late-middle layers, where S1 sits. Without the replacement rule, S2's runs could duplicate S1's, and the contrasts would rest on one set.

## 5. Phase 0A: training length at the annealed state

### 5.1 Run matrix

Arm codes in run IDs: `P300`, `P1T`. In this part, each arm's parent and teacher are the arm's own model.

| Runs | Parents | Sets | Heal tokens | LR |
|---|---|---|---|---|
| LR grid | P300 and P1T of each dataset (6 parents) | S1 | 0.03B (114 steps) | each of {3e-5, 1e-4, 3e-4} |
| Short heal | the same 6 | S1, S2, S3, S4 | 0.05B (190 steps) | that parent's chosen LR |
| Main heal | the same 6 | S1, S2 | 0.2B (762 steps) | chosen LR |
| Long heal | `nem` P300 and `nem` P1T | S1 | 0.8B (3,051 steps) | chosen LR |

**Chosen LR** (this rule applies to every LR grid in this spec). Take the lowest **tuning-set** val among the non-diverged grid runs. The chosen LR is the smallest grid LR whose tuning-set val is within 0.001 nats of it. If all three grid runs diverge, that parent or arm fails, and its quantities are undefined (rule 1.3).

**Totals:** 6 × (3 + 4 + 2) + 2 = 56 heal runs, plus T14's two reruns: about 5.8B KD tokens.

### 5.2 Compute estimate

- **KD:** about 12.6 GFLOP per token (teacher forward plus student training, with attention at 4,096 context). The team's only throughput figure is about 1.9 PFLOP/s per node, from a 344M pretraining run on GPUs of unrecorded type. At that rate KD runs at about 150K tokens/s, so 5.8B tokens take about 10.7 node-hours.
- **Measurements and secondary validation** for the six parents: about 1 node-hour.
- **Total:** about 11.7 node-hours (rule 1.2).

### 5.3 Decision quantities

Compute these mechanically; the decision is made by the project owner.
1. For each dataset d and set s ∈ {S1, S2}: e_{d,s} = Δ_post(P1T, s, 0.2B) − Δ_post(P300, s, 0.2B).
2. E_d = mean of e_{d,S1} and e_{d,S2}; E = mean of E_d over the three datasets.
3. E_pre,d and E_pre are the same quantities with Δ_pre.
4. **Sign count:** for the 12 contrasts Δ_post(P1T, s, 0.05B) − Δ_post(P300, s, 0.05B) (s = S1–S4, three datasets), count how many have the sign of E. An undefined contrast counts as not having it.
5. **Evaluate the labels in this order**, and report the first that holds together with the truth value of every clause:
   1. **INCOMPLETE:** some E_d is undefined.
   2. **GO:** |E| ≥ 0.008 nats, the three E_d share a sign, and at least 10 of the 12 contrasts have that sign.
   3. **REFRAME:** |E| < 0.003, |E_pre| ≥ 0.05, and the three E_pre,d share the sign of E_pre (healing erases a consistent difference that exists before healing).
   4. **STOP:** |E| < 0.003, and either |E_pre| < 0.03 or the three E_pre,d do not share a sign; or at least one E_d ≥ +0.003 and at least one E_d ≤ −0.003 (the datasets disagree in sign beyond noise).
   5. **GREY:** otherwise.
6. **Also report**, without any decision attached:
   - all of the above on the secondary sets, and val(P1T) − val(P300) on both sets;
   - each E_pre,d next to the mean Δ_pre of its parents, since Δ_pre itself can be tenths of a nat;
   - Δ_post(S1) at 0.05, 0.2 and 0.8B for the `nem` pair;
   - the two T14 differences.

## 6. Phase 0B: LR state, and training length at the stable state

### 6.1 Arms

Phase 0B adds two parents and one cross arm per dataset. Arm P300 comes from Phase 0A: its runs are not repeated. Every run is named in `runs.jsonl` by three fields:

| `arm` | `parent` (the model the student is pruned from) | `teacher` | Role |
|---|---|---|---|
| **S72** | S72 | S72 | Prune the stable checkpoint; heal from it |
| **X72** | S72 | P300 | Prune the stable checkpoint; heal from the annealed checkpoint of almost the same length |
| **S188** | S188 | S188 | The stable state at 2.6× the length of S72 |
| P300 (from 0A) | P300 | P300 | Prune the annealed checkpoint; heal from it |

In Phase 0A, `arm`, `parent` and `teacher` are all P300 or all P1T. **Select runs by `arm`, never by `parent`:** arms S72 and X72 share a parent.

**Baselines for arm X72:** its `val_parent` is val(S72), and its unhealed val (for rule 1.3) is that of S72 with the set removed. Its `val_teacher` is val(P300). Record both.

**If a branch parent fails** the identity tests T1 or T3, report it. Only the arms that use it stop:
- a failing S72 stops arms S72 and X72 of that dataset;
- a failing S188 stops arm S188 of that dataset.

The quantities that need those arms are undefined (rule 1.3). Phase 0A is never affected.

### 6.2 Run matrix

| Runs | Arms | Sets | Heal tokens | LR |
|---|---|---|---|---|
| LR grid | S72, X72 and S188 of each dataset (9 arms) | S1 | 0.03B (114 steps) | each of {3e-5, 1e-4, 3e-4} |
| Short heal | S72 and S188 of each dataset (6 arms) | S1, S2, S3, S4 | 0.05B (190 steps) | that arm's chosen LR (section 5.1) |
| Main heal | the 9 arms | S1, S2 | 0.2B (762 steps) | chosen LR |

The short heals give the stable-state contrast the same 12-contrast sign count as Phase 0A.

**Totals:** 9 × 3 + 6 × 4 + 9 × 2 = 69 heal runs, about 5.6B KD tokens.

### 6.3 Compute estimate

- **KD:** 5.6B tokens at about 150K tokens/s: about 10.3 node-hours.
- **Measurements and secondary validation** for the six new parents: about 0.8 node-hours.
- **Total:** about 11.1 node-hours (rule 1.2).

### 6.4 Decision quantities

Compute these mechanically; the decision is made by the project owner.

**Notation.**
- v(arm, s, H) is the val of that arm's student, with set s, after H tokens of healing; val(parent) is the val of an unpruned parent.
- Δ_post(arm, s, H) = v(arm, s, H) − val(the arm's parent).
- Δ_pre is defined per parent, as in section 2.

**(i) LR state at almost the same length.** For each dataset d and set s ∈ {S1, S2}:
- **A_init,d,s** = v(X72, s, 0.2B) − v(P300, s, 0.2B). The teacher is held fixed (P300). The starting checkpoints differ: in LR state, in the data each run read after step 58,000, and by 0.9% in tokens. Each arm's healing LR is also chosen separately. A_init therefore compares two checkpoint choices under this recipe; it does not isolate the LR state alone.
- **A_teach,d,s** = v(S72, s, 0.2B) − v(X72, s, 0.2B). The starting checkpoint is held fixed (S72); the teacher differs, and the chosen healing LR may differ too.
- **A_d,s** = v(S72, s, 0.2B) − v(P300, s, 0.2B): the whole recipe.

**Label "LR state at 0.2B"**, evaluated on the six A_init,d,s values, in this order:
1. "INCOMPLETE" if fewer than 5 of the 6 are defined;
2. "prune before you anneal" if their mean is ≤ −0.01 nats and at least 5 of the 6 are negative;
3. "anneal first" if their mean is ≥ +0.01 and at least 5 of the 6 are positive;
4. otherwise "no consistent effect meeting the threshold". This label covers small effects and also large but inconsistent ones, so it is not evidence of equivalence; the per-dataset and per-set values must be shown next to it.

When exactly 5 are defined, the mean is taken over those 5, and the undefined value counts as not having the sign that a label needs.

**Flag** the label "LR at grid edge" if X72's chosen LR is the top of its grid (3e-4) for at least one dataset. X72 is the only arm whose student starts behind its teacher by the whole annealing gain. A short heal at a capped LR may not close that gap, so a positive A_init could reflect this rather than the pruning point.

Also report, without a label:
- A_init on S1 at 0.03B, from the grid runs of X72 and P300 at their chosen LRs (scored on the final set), next to the 0.2B value, to show whether the gap is still closing;
- every arm's chosen LR, and whether X72 and P300 chose the same one;
- the means and sign counts of A_teach and A at 0.2B, and of A at 0.05B (S72 vs P300, four sets);
- the Δ-based versions, in which each arm's val is replaced by v − val(its teacher). The Δ-based A_init then equals A_init; the Δ-based A_teach and A each differ from the absolute ones by the annealing gain val(S72) − val(P300);
- the annealing gain itself, per dataset.

**(ii) Training length at the stable state.** This is the Phase 0A contrast with both parents stable. It uses arms S72 and S188, never X72. As in Phase 0A, each arm is healed from its own parent, so the teacher changes with the length too:
- e^S_{d,s} = Δ_post(S188, s, 0.2B) − Δ_post(S72, s, 0.2B), for s ∈ {S1, S2};
- E^S_d is the mean over S1 and S2, and E^S is the mean over the three datasets;
- E^S_pre,d and E^S_pre are the same, with the Δ_pre of parents S188 and S72;
- **sign count:** for the 12 contrasts Δ_post(S188, s, 0.05B) − Δ_post(S72, s, 0.05B) (s = S1–S4, three datasets), count how many have the sign of E^S. An undefined contrast counts as not having it.

Evaluate the labels in this order, exactly as in section 5.3:
1. **INCOMPLETE:** some E^S_d is undefined.
2. **GO:** |E^S| ≥ 0.008 nats, the three E^S_d share a sign, and at least 10 of the 12 contrasts have that sign.
3. **REFRAME:** |E^S| < 0.003, |E^S_pre| ≥ 0.05, and the three E^S_pre,d share the sign of E^S_pre.
4. **STOP:** |E^S| < 0.003, and either |E^S_pre| < 0.03 or the three E^S_pre,d do not share a sign; or at least one E^S_d ≥ +0.003 and at least one ≤ −0.003.
5. **GREY:** otherwise.

Report which clause produced a STOP: a small effect, or datasets that disagree.

**(iii) Report without labels:**
- per dataset, a 2 × 2 table of val, Δ_pre and Δ_post (S1 and S2, 0.2B) for the arms S72, P300, S188 and P1T;
- **the effect per unit of log length**, since the two contrasts span different ranges:
  - E / ln(1,000 / 300.0) = E / 1.204 for the annealed pair;
  - E^S / ln(776.2 / 297.3) = E^S / 0.960 for the stable pair;
  - their difference, as a **descriptive** interaction only. Reading it as a factorial interaction would assume a length effect linear in log length, over ranges that also differ in their endpoints;
- S188 vs P1T, described only, since their lengths differ (776B vs 1,000B tokens).

**How to read the results together** (for the report; the owner decides). Every reading below is evidence under this parent, teacher and healing recipe. The pilot screens whether a controlled experiment is justified; it cannot remove these confounds.
- **The length effect appears only after annealing.** Use this reading only when 0A is GO and 0B's length label is STOP through its first clause (|E^S| < 0.003). It is then consistent with damage that tracks the anneal rather than the length, as Catalan-Tatjer et al. found for quantization.
  - The stable pair ends at 776B tokens, not 1T, so "only after annealing" cannot be fully separated from "only beyond 776B".
  - A STOP through the sign clause means the datasets disagree, not that the effect is absent.
- **The length effect appears in both states, with similar sizes per unit of log length:** this is consistent with a training-length effect under this recipe.
- **A_init:** a large value would be a practical result on its own ("prune before you anneal", or the reverse), for a 0.2B heal under this recipe. See the flag and the 0.03B value above.

## 7. Outputs

Put everything under `results/a2_pilot/`:
- `config/`: this spec's SHA-256, the environment, the model and dataset commits, and the code commit.
- `manifests/`: the reserved, heal, tuning and unused document indices, the small-set indices, and the hashes of the heal stream, tuning set and final set, frozen before any grid run.
- `tests.json`: T1–T16 results, with the diagnostics marked as such.
- `parents.jsonl`: one record per parent (12), holding every measurement in section 3.4, the tokens, TPP and LR state.
- `runs.jsonl`: one record per heal run, with these fields:
  - `run_id`, `part` (0A or 0B), `arm`, `parent`, `teacher` (as defined in section 6.1), `set`, `removed_layers`;
  - `heal_tokens`, `lr`, `steps`;
  - `val_parent`, `val_teacher`, `val_pre`, `val_post`, `delta_pre`, `delta_post` (final set, scored representation), the tuning-set versions `val_tune_pre` and `val_tune_post`, the secondary-set versions, and `val_post_master`;
  - `final_kl`, `diverged`, `rerun_of`;
  - `wallclock_h`, `gpu_hours`.
- `decision_phase0A.json` and `decision_phase0B.json`: the quantities, every clause's truth value, and the labels of sections 5.3 and 6.4.
- **Plots:**
  - Δ_post vs heal tokens per parent;
  - per dataset, the four parents' bypass profiles overlaid;
  - BI and update-norm profiles;
  - the 2 × 2 tables of section 6.4(iii).
- `REPORT.md`, covering:
  - test results, with every flagged diagnostic;
  - deviations from this spec, if any, and why;
  - the decision tables, with the per-dataset and per-set values next to every label;
  - the overlap caveat of section 3.1, and the limits of section 0 ("What the pilot can and cannot show");
  - compute used against budget, with data preparation and downloads listed separately;
  - anything surprising.

## 8. Deferred phases (do not run until told)

Both are specified in `A2_phase1a_deferred_spec.md`:
- **Phase 1(a):** the same stable-vs-annealed test on our own narrow 28-layer Qwen3 model (166M), one WSD run to 6.75B tokens with arms S, P and X. It adds a second architecture with document masking, and exactly matched tokens and data. Estimated at about 3.9 node-hours.
- **Phase 1(b):** our own narrow WSD family to 27B tokens (19–306 TPP), a second trajectory, and a 0.6B arm. It runs only if the owner decides so after this pilot.

## Deviations from the design note, and why

| Change | Reason |
|---|---|
| Phase 0 uses open-sci-ref **v0.01** pairs, not v0.02 | Only the v0.01 pairs have documented matched recipes. v0.02's 1T models are re-uploads of runs with RoPE θ = 100k, and its 300B models have no documented recipe |
| No supplementary 100B point | Its RoPE θ (100k) differs from the v0.01 parents (10k) |
| Phase 0B replaces Phase 1(a); Phase 1(a) is deferred | The public runs are WSD trees with checkpoints at the peak LR, so the stable-vs-annealed test can run at 1.7B scale on three datasets without pretraining |
| Phase 0B adds S188 | It tests the length effect at the stable state, within one run |
| S2 is chosen on the mean BI of a dataset's usable parents, and may not equal another set | Choosing on one parent biases the contrasts in its favour; a duplicate of S1 would leave one set |
| STOP's sign clause counts only datasets with abs(E_d) ≥ 0.003 | A sign difference within noise should not stop the length half; such a result now falls to GREY |
| REFRAME needs the three E_pre,d to share a sign; a small E with an inconsistent E_pre is STOP | Otherwise one dataset with a large pre-healing difference could trigger REFRAME alone |
| A failed decision input gives the label INCOMPLETE | A failed run should lead to repair, not to a deferred phase |
| The final set (all full sequences of the validation split) only scores; LRs, S2 and the divergence check use a tuning set from reserved training documents that no heal run reads | Choices made on the scoring set would bias the small differences that the decisions rest on |
| T9, T10's loss comparison and T15 are diagnostics, not exclusion rules | Excluding a parent on a loss value would select parents by an outcome related to the effect under study |

## Appendix: arithmetic

- **open-sci-ref v0.01 1.7B:**
  - non-embedding parameters per layer = 4·2,048² + 4·2,048 (attention and its biases) + 3·2,048·8,192 + 2·8,192 + 2,048 (MLP and its biases) + 2·2,048 (norms) + 2·64 (QK-norm) = 67,139,712;
  - 24 layers plus the final norm (2,048): 1,611,355,136;
  - embedding: 50,304 × 2,048 = 103,022,592; total 1,714,377,728, which is 3.43 GB in bf16, matching the 3,428,804,400-byte checkpoint files.
- **Tokens and TPP** at 4,128,768 tokens per step:

  | Parent | Steps | Tokens | TPP |
  |---|---|---|---|
  | S72 | 72,000 | 297.3B | 184.5 |
  | P300 | 72,661 | 300.0B | 186.2 |
  | S188 | 188,000 | 776.2B | 481.7 |
  | P1T | 242,204 | 1,000.0B | 620.6 |

- **Cooldown starts:** 0.8 × 72,661 ≈ 58,129, and 0.8 × 242,204 ≈ 193,763.
- **Heal steps at 262,144 tokens per step:** 0.03B → 114; 0.05B → 190; 0.2B → 762; 0.8B → 3,051. The longest heal reads 3,051 × 64 = 195,264 sequences (0.80B tokens), within the 0.85B of heal documents.
- **Data sizes:** tuning set 1,024 × 4,096 = 4,194,304 tokens; small set 256 × 4,096 = 1,048,576 tokens. The reserved set (at least 1.2B Qwen3 tokens, so roughly 1.2–1.3B GPT-NeoX tokens) covers the 0.85B heal documents plus the tuning documents.
- **KD tokens:**
  - Phase 0A: 6 × (3 × 0.03 + 4 × 0.05 + 2 × 0.2) + 2 × 0.8 = 5.74B, plus 0.1B for T14;
  - Phase 0B: 9 × 3 × 0.03 + 6 × 4 × 0.05 + 9 × 2 × 0.2 = 0.81 + 1.2 + 3.6 = 5.61B.
- **KD speed:** 1.9 PFLOP/s ÷ 12.6 GFLOP per token ≈ 151K tokens/s. 5.84B tokens take about 10.7 hours, and 5.61B about 10.3.
- **Log-length ranges:** ln(1,000 / 300.0) = 1.204; ln(776.2 / 297.3) = 0.960.

## Changes in version 2.1 (response to the implementer's review)

The implementer reviewed version 2.0 (SHA-256 `3a8b68d1…`) on 28 October, before any implementation or run, in `A2_PILOT_REVIEW_20261028.md`. Every point was accepted; the review's numbering is kept.

1. **Outcome-dependent eligibility.** Eligibility now rests only on the identity tests T1 and T3.
   - T15 is a diagnostic. It records the annealing gain and, if the training logs can be read, the logged LR at steps 72,000 and 188,000.
   - S2's pool of usable parents no longer depends on any loss value.
   - T9 and the loss part of T10 are diagnostics; T10's formula check remains a correctness test.
   - Section 3.5 separates the two kinds of test, and surprising diagnostics go to the owner in the pre-launch report.
2. **The final set also chose hyperparameters.** A tuning set now serves every choice: the healing LRs, the divergence check and, through the small set, S2.
   - The tuning set comes from reserved training documents that no heal run reads, so the final set keeps its full size and only scores.
   - Documents are split before packing, and the manifests and hashes are frozen before any grid run.
3. **Causal wording.**
   - The descriptions of A_init and A_teach now list what differs besides the factor of interest.
   - "A genuine training-length effect" became "consistent with a training-length effect under this recipe".
   - The difference of normalised slopes is labelled descriptive.
   - Section 0 states that the pilot screens and cannot remove these confounds.
   - No arm was added.
4. **Export precision.** The scored representation is now fixed in advance: the fp32 master weights cast to bf16, which is exactly the saved file, evaluated like the parents. The fp32-master val is recorded as a diagnostic, and the new test T16 checks the round trip after healing.
5. **Label wording.** "No difference" became "no consistent effect meeting the threshold", shown with the per-dataset and per-set values. Section 0 states that labels and sign counts are screening rules, not significance tests, over contrasts that are not independent replications. T14 is listed as a limited diagnostic.
6. **Budget.**
   - The re-estimate after T8 now covers the whole workload: 127 heal runs plus every evaluation, measurement, load and save.
   - Data preparation and downloads are tracked separately and do not count toward the caps.
   - The estimates are marked as not measured on this 8 × B200 node.
   - The gates are unchanged.
7. **Novelty.** Section 0 now states that a GO label does not establish novelty. That assessment lives in the design note, which the implementer did not have.
   - Of the review's three references, Catalan-Tatjer et al. (arXiv 2510.06213) is the note's design template, and LR rewinding (Renda et al., arXiv 2003.02389) is already cited there.
   - Watts et al. (arXiv 2605.02105: flatter pretraining, including shorter annealing, reduces forgetting after post-training and quantization; no pruning) has been added to the note.

**Also from the review's checks:** each attention module copies the dropout rate when it is built. The dropout is therefore now set at load time and asserted in T1.

## Changes in version 2.0

- **Phase 1(a) replaced by Phase 0B.** The open-sci-ref repositories hold checkpoints every 2,000 steps, and each dataset's 300B and 1T runs share weights up to step 58,000. Phase 0B uses the 1T runs' steps 72,000 and 188,000, where the LR is still at its peak. Phase 1(a) moves, unchanged in substance, to `A2_phase1a_deferred_spec.md`.
- **New parents and arm:** S72, S188 and X72 (section 6), with pinned branch commits and loading through the `main` code (`code_revision`).
- **S2** is now chosen on the mean BI of a dataset's usable parents (P300 and P1T, plus each branch parent that passes its tests).
- **Tests renumbered.** New tests: T3 (branch identity) and T15 (LR-state sanity). T1 checks every downloaded weight file against its Hub hash. The narrow-model tests moved to the deferred file.
- **Budgets:** Phase 0A about 11.7 node-hours (T14 included), Phase 0B about 11.1.
- **Run IDs** name the arm (`P300`, `P1T`, `S72`, `S188`, `X72`) instead of `300B` and `1T`.
- **After an independent review of version 2.0** (same day):
  - a failing branch parent stops only its own arms and never Phase 0A;
  - runs carry `arm`, `parent` and `teacher`, and the formulas name arms;
  - the LR-state label states how to handle 5 defined values, is called "at 0.2B", and carries a grid-edge flag;
  - the stable-state contrast gets 0.05B heals on four sets, so its labels have exactly the structure of Phase 0A's;
  - the interaction is reported per unit of log length, with the caveats of reading the labels together.

## Changes in version 1.2 (second check)

The items about the narrow model now live in the deferred file.
- **Public parents verified:**
  - all six repositories and the fallback exist;
  - all three pairs' run names match except for length;
  - the `main` commits are pinned;
  - each pair shares identical weights at `iter_0058000`.
- **Software:** transformers pinned to 4.57.1. The `main` remote code needs at least 4.54 (it imports `TransformersKwargs`), so the earlier "≥ 4.51" was wrong.
- **Layer removal:**
  - rebuild the layer list from the kept indices, because deleting one index at a time removes the wrong layers;
  - for Qwen3, also slice `config.layer_types`;
  - the removal test covers every full set and a save-and-reload round trip.
- **Data:**
  - reserved documents are a seeded random sample, in permutation order;
  - small validation sets are seeded random subsets, stored in manifests;
  - the secondary validation is best effort.
- **Decision rules:**
  - S2 may not equal S1, S3 or S4;
  - REFRAME needs consistent E_pre,d signs, and STOP ignores sign differences within ±0.003;
  - failed inputs give INCOMPLETE;
  - Δ-based quantities are measured against each arm's teacher.
- **Definitions:** step and linear-ramp conventions; val defaults to the final set.
- **KD and tests:**
  - weight decay applies to the tied embedding;
  - the KD-loss test runs in fp32;
  - T1 records the bf16 teacher's val;
  - T14 separates GPU nondeterminism from data-order noise;
  - Gaussian sensitivity runs with TF32 off.

## Changes in version 1.1 (after the implementability review)

- **Data:** reserved heal documents are taken at the document level and removed before packing. Phase 0 tokenises raw text, because the dataset stores text. Validation uses all full sequences.
- **LR grid:** semantics, divergence rules and run IDs defined.
- **Decision rules:** evaluated in a fixed order, with every clause reported.
- **Phase 0 parents:** v0.01 pairs; attention dropout set to 0; no RoPE patching.
- **Numerics and saving:** precision, fresh optimiser state, trainable parameters and saving rules stated.
- **Secondary validation and overlap caveat:** added.
- **Tests:** the data-sanity, RoPE, BI, KD-implementation, decision and noise tests added.
- **Budgets:** triggers and caps restated.


## Approved preflight scope amendment — 28 October 2026 (SGT)

The owner requested that implementation tests not be repeated for all twelve
checkpoints when the architecture, native source and runtime are identical.
This amendment overrides only test repetition in section 3.5; it does not
change the scientific matrix, data, loss, optimizer, budgets or decision rules.

- **Once on the real DCLM/P300 checkpoint:** T4, T5, T11 and T12. After S2 is
  frozen, T4 covers the union of distinct removal patterns from all three
  datasets, including their different S2 choices, plus layers 0, 11 and 23.
  One representative pruned export/reload is sufficient for this code check.
  Sharing is allowed only with identical pinned native source and structural
  configuration, and the same runtime/backend/precision and KD recipe.
- **Once per unchanged recipe/data/runtime:** T6, T7, T8 and T13. Preserve
  existing receipts; a scope amendment alone does not require repeating T8.
  T8 remains a representative measurement, not proof that all workloads cost
  the same. Full-workload estimates still include scoring, loading and saving.
- **Every parent:** T1 loading/hash/tied-weight/dropout/shape checks, finite
  forward output and the cheap T10 formula check. Retain the small-set
  FP32-versus-BF16-AMP comparison: numerical error depends on the weights.
  T2/T3 remain per pair/branch identity checks; identical code does not prove
  that a downloaded weight file or branch is correct.
- **Scientific measurements remain per parent:** BI, update norms, individual
  layer bypasses, Gaussian sensitivity, parent/unhealed losses, T9 and the
  loss part of T10. These measure the checkpoints rather than test shared
  code. T14/T15 and post-healing T16 retain their existing scope.

A shared check receipt must name the representative, exact provenance and
covered removal patterns. Missing/failed evidence is never promoted to passed.
Changing relevant code, model structure, precision, backend or runtime requires
reviewing the affected tests again; new removal patterns require coverage.

Prepared inputs under the preceding spec SHA
`9854e8338cecab6cd6fccd1700e7196c62a329a3b44fa76689d31d11ecbc9afd`
remain valid for this exact amendment. Preserve their original manifests and
receipts and record both original and amended spec identities; do not resample,
retokenize or rewrite historical evidence. Reuse of historical code-test or
throughput receipts requires an explicit reviewed record of unchanged relevant
code/runtime/recipe, not silently relabelling their provenance.


## Approved dependency-based launch amendment — 28 October 2026 (SGT)

The owner approved starting eligible fixed-set arms before unrelated parents
finish their surveys. The first bounded launch is DCLM/P300 S1's three 0.03B
LR-grid fits, with the registered one-half-LR divergence retries only. Each fit
still starts fresh; no budget, LR selection, data, loss or metric is changed.

Before these fits, require the actual representative mechanism checks covering
S1 and single layers 0/11/23, this parent's identity/precision checks, data
integrity/disjointness, recipe/LR/decision tests and measured throughput. Reuse
verified historical results only with an explicit original-receipt checksum and
review of unchanged relevant code/runtime. Score the unhealed S1 baseline first.
Report a bounded workload estimate; charge this work to Phase 0A. No other arm
is eligible from this gate. S2 remains dependent on its full prescribed BI survey;
other parents require their own identity/precision evidence. Extend mechanism
coverage when new removal patterns are introduced. Stop after this grid and
restore verified communicating burns; do not automatically start the full pilot.

This amendment changes execution dependencies only. Previously prepared data
under the original or first preflight-scope spec remains byte-for-byte valid.
