# ADE: Adaptive Dictionary Embeddings — Critical Analysis


## 1. Core Idea

Standard embedding layers assign each word a single vector. Polysemous words like "bank" must encode all meanings into one point. ADE proposes representing each word as a weighted combination of multiple shared anchor vectors from a small codebook, then using a transformer layer to contextualize the result.

The paper claims three contributions: Vocabulary Projection (VP), Grouped Positional Encoding (GPE), and context-aware anchor reweighting via the Segment-Aware Transformer (SAT).

The underlying idea is sound. If implemented correctly — expand anchors, apply GPE, run self-attention over the full anchor sequence, then aggregate — this architecture would genuinely enable context-dependent anchor selection, bridging codebook-based representations with transformer-style contextualization at the sub-word-embedding level.


## 2. Method

### Key dimensions

- N = 128,100 (DeBERTa-v3-base vocabulary size)
- d = 768 (embedding dimension)
- K = codebook size, a hyperparameter. Paper tests K ∈ {100, 200, 300, 500, 700}
- L = input sequence length (after tokenization)
- B = batch size

### Stage 1 — Learn anchors and assignments (knowledge distillation)

The teacher is DeBERTa-v3-base's pretrained embedding matrix **E**_teacher ∈ ℝ^(N×d). This is a standard embedding table: row i is DeBERTa's fixed vector for word i. It is frozen throughout — never updated.

The student has two learnable matrices:

- Anchor matrix **A** ∈ ℝ^(K×d) — K shared anchor vectors, each d-dimensional. Randomly initialized.
- Weight matrix **T** ∈ ℝ^(N×K) — for each of N vocabulary words, K scalar weights determining how much each anchor contributes. Randomly initialized.

For any word i, the student reconstruction is:

    ê_i = Σ_{j=1}^{K} T_{i,j} · a_j

This is matrix multiplication: **Ê** = **T** · **A**, where **Ê** ∈ ℝ^(N×d). Every word uses all K anchors — there is no sparsity at this stage.

The loss minimizes cosine distance between student and teacher embeddings:

    ℒ = 1 − (1/|ℳ|) Σ_{(b,t)∈ℳ} cosine(ê_{b,t}, e^teacher_{b,t})

where ℳ is the set of non-padding token positions in a batch of tokenized text. Cosine similarity measures directional alignment (ignoring magnitude): if two vectors point the same direction, cosine = 1 and loss = 0.

Both **A** and **T** receive gradient updates. Over training, **A** learns reusable semantic building blocks, and **T** learns which combination of blocks best reconstructs each word's DeBERTa embedding.

**Ambiguity about sparsity:** The paper says **T** comes "from A&T" (the Anchor & Transform method of Liang et al., 2020). The original ANT method uses L1 regularization (λ₂‖**T**‖₁) with proximal gradient descent (soft-thresholding) and a non-negativity constraint (**T** ≥ 0) to enforce sparsity — weights below a threshold are set to exactly zero during training. However, ADE's distillation loss as written contains no L1 term. Whether ADE uses ANT's L1 mechanism during distillation or relies solely on hard thresholding in Stage 2 is not clarified.

**Ambiguity about text batches:** The loss indices (b,t) indicate training on batches of real text rather than iterating over vocabulary rows directly. Since both sides of the reconstruction are context-free (word i always maps to the same vectors regardless of sentence), this is fundamentally a matrix factorization problem (**T**·**A** ≈ **E**_teacher) that could be solved by iterating over all N rows. Using text batches introduces frequency-dependent bias — common words receive more gradient updates. The paper does not justify this choice.

### Stage 2 — Vocabulary Projection (VP)

After Stage 1, **T** ∈ ℝ^(N×K) contains weights for all K anchors for every word. Stage 2 makes this sparse.

**Step 2a — Threshold:** Choose threshold τ. For each word i, find active anchors:

    ℐ_i = {j : T_{i,j} ≥ τ}

The surviving weights become β: β_{i,j} = T_{i,j} for j ∈ ℐ_i. The anchor cardinality k_i = |ℐ_i| is the number of surviving anchors for word i. k_i varies per word — different words retain different numbers of anchors.

Example with K=5, τ=0.1:
- Word "bank": T = [0.52, 0.03, 0.41, 0.01, 0.38] → ℐ = {0, 2, 4}, k = 3
- Word "the":  T = [0.01, 0.88, 0.02, 0.03, 0.01] → ℐ = {1}, k = 1

**Step 2b — Build flattened embedding table:** For each word i, concatenate its k_i surviving anchor vectors (from **A**), padded with zero vectors up to length K:

    E[i] = [a_{ℐ_i[1]}; a_{ℐ_i[2]}; ...; a_{ℐ_i[k_i]}; 0; ...; 0] ∈ ℝ^(K·d)

This produces **E** ∈ ℝ^(N × K·d). Each row contains k_i real anchor vectors followed by (K − k_i) zero-padded slots. This table functions as a standard embedding layer: given a token ID, return a single row and reshape it into (K, d).

**What is stored after Stage 2:** The flattened **E**, the sparse weight vectors β_i, the anchor cardinalities k_i, and a mapping from each anchor slot to its original anchor index. The anchor matrix **A** is embedded (duplicated) inside **E** — if two words share an anchor, each holds its own copy.

### Stage 3 — End-to-end fine-tuning on downstream task

SAT and the classifier are randomly initialized. Training uses cross-entropy loss on a classification task (AG News: 4 classes, 120K training samples; DBpedia-14: 14 classes, 560K training samples).

The forward pass, per the algorithm as written (Algorithm 1):

**Step 3a — Anchor lookup:** For each token ID in the input (B, L), retrieve its row from **E**, reshape to (K, d), take the first k_i vectors (the non-padding ones). Also retrieve the stored weights β_{i,1}, ..., β_{i,k_i}.

**Step 3b — Weight each anchor:** Multiply each anchor vector by its scalar weight:

    ã_{i,j} = β_{i,j} · a_{i,j}    for j = 1, ..., k_i

**Step 3c — Aggregate to word level (ScatterAdd):** Sum all weighted anchors belonging to the same word into a single vector:

    x_i = Σ_{j=1}^{k_i} ã_{i,j}

This is implemented as ScatterAdd: flatten all weighted anchors across all words in the batch, then scatter-add them into a (B, L, d) tensor using pre-computed global IDs that map each anchor to its word position. Result: one d-dimensional vector per word — the same shape as a standard embedding output.

**Step 3d — LayerNorm:** Apply layer normalization to **X** ∈ ℝ^(B, L, d).

**Step 3e — Grouped Positional Encoding (GPE):** Construct positional encodings from the stored cardinalities. The paper describes GPE as assigning all anchors of the same word the same positional encoding, while anchors from different words receive different positions.

Example: a 3-word sentence with k = [3, 1, 4] produces a flattened anchor sequence of length 8. GPE assigns positions:

    Anchor index (flattened): [0, 1, 2, 3, 4, 5, 6, 7]
    GPE position assignment:  [0, 0, 0, 1, 2, 2, 2, 2]

Anchors 0–2 (word 1) share PE(0). Anchor 3 (word 2) gets PE(1). Anchors 4–7 (word 3) share PE(2). This preserves word-level grouping while allowing anchor-level variation through embedding values.

However, in the algorithm as written, ScatterAdd has already collapsed anchors to word level before this step. The sequence is (B, L, d) with one vector per word, and GPE reduces to standard positional encoding (position 0 for word 0, position 1 for word 1, etc.). GPE only has a functional effect if applied to the expanded anchor sequence before aggregation. See Section 4 for discussion of this contradiction.

**Step 3f — SAT (Segment-Aware Transformer):** Multi-head self-attention only (no feed-forward sublayer):

    Attention(Q, K_att, V) = softmax(Q · K_att^T / √d_k) V

where Q = XW^Q, K_att = XW^K, V = XW^V (note: K_att is the attention key matrix, not the codebook size K). Each word vector attends to all other word vectors in the sequence. Output: (B, L, d).

**Step 3g — Pooling:** Apply the learned pooler: a linear(d→1) layer produces a scalar score per position (B, L), masked over padding, softmaxed, then used as attention weights to compute a weighted sum of word vectors. Result: pooled (B, d). The pooler has 769 = 768+1 parameters (Table 5).

**Step 3h — Classification:** Apply dropout, then a linear layer mapping d → C (number of classes), then softmax. Loss is cross-entropy between predictions and true labels.

**What is trainable in Stage 3:**

| Component | Trainable? | Notes |
|---|---|---|
| Anchor vectors (inside **E**) | Yes | Gradients flow through ScatterAdd back into anchor vectors |
| Weights β | No | Fixed from Stage 2 |
| Anchor indices ℐ_i | No | Fixed from Stage 2 |
| Cardinalities k_i | No | Fixed from Stage 2 |
| SAT parameters (W^Q, W^K, W^V, W^O) | Yes | Attention-only, no FFN. Trained from random init |
| LayerNorm | Yes | External, 1,536 params |
| Pooler | Yes | Learned linear(d→1) with bias, 769 params |
| Classifier (linear layer) | Yes | Trained from random init |

Total trainable parameters per Table 5: 2,367,749 (SAT 2,362,368 + LayerNorm 1,536 + Pooler 769 + Classifier 3,076). This count **excludes** anchor embeddings — Table 5 notes they are "trained offline via knowledge distillation." Whether anchors are trainable in Stage 3 is contradicted by the paper itself (see Section 6.9). If anchors are trainable (per Section 4.2), add ~77K at K=100 or ~384K at K=500.


## 3. Results

On DBpedia-14 (14-class classification): 98.06% accuracy, surpassing DeBERTa-v3-base (97.80%) with 98.7% fewer trainable parameters (2.37M vs 184.4M). Embedding layer compressed over 40×.

On AG News (4-class classification): 90.64% accuracy, trailing DeBERTa (94.50%).


## 4. The Central Contradiction

The paper's core problem is an internal contradiction between two descriptions of the same architecture.

**Section 3.5** describes anchor-level attention: individual anchors enter SAT, attend to all other anchors across the sequence, and are reweighted dynamically by attention. GPE assigns shared positions to co-anchors so the model knows they belong to the same word. This is the version that makes every claimed contribution meaningful.

**Section 3.6 and Algorithm 1** describe word-level attention: anchors are aggregated into a single vector per word using fixed β weights via ScatterAdd **before** SAT. SAT then processes a standard (B, L, d) word-level tensor. In this version, SAT cannot reweight individual anchors (they no longer exist), and GPE is redundant (one vector per word = standard positional encoding).

These descriptions are mutually exclusive. The algorithm is explicit and matches Section 3.6.

### 4.1 Likely explanation

The idea described in Section 3.5 is probably what the authors intended and possibly implemented. The ablation results are consistent with this: without SAT, accuracy on AG News is erratic (62–74%) regardless of K. With SAT, accuracy scales smoothly from 88% (K=100) to 91% (K=700).

However, this ablation does not definitively distinguish between the two architectures. Even with word-level SAT (pre-collapsed), increasing K produces richer weighted-sum vectors, giving SAT better input to work with. The scaling could be explained by either architecture. The ablation is suggestive but not conclusive.

It is plausible that the correct architecture was implemented but Algorithm 1 was written incorrectly — ScatterAdd placed before SAT instead of after. This single misplacement cascades into making GPE appear redundant and the central claim appear unsupported. This is consistent with AI-assisted writing that understands each component individually but misorders the dataflow.

### 4.2 The corrected algorithm should be

    1. Retrieve active anchors for each token    → flat sequence, length SL = Σ k_i
    2. Add GPE (co-anchors share position)
    3. Apply LayerNorm
    4. Pass through SAT (anchor-level self-attention, attention only, no FFN)
    5. Aggregate to word level using stored β     → (B, L, d)
    6. Learned pooler, classify

### 4.3 Open question: role of β after SAT

If SAT performs anchor-level attention, the attention mechanism produces its own context-dependent weights. This raises the question of what role the stored β from Stage 2 plays:

- **Option A:** β pre-scales anchor vectors before attention (step 2), then SAT's attention produces the final aggregation weights. β biases the input; attention determines the output.
- **Option B:** β is discarded entirely. SAT's attention fully replaces it.
- **Option C:** SAT sees raw (unweighted) anchors, consistent with the paper's Equation 5 (x_t = a_t + PE, no β). β is applied only post-SAT during aggregation to word level. Attention operates on pure anchor semantics; β determines how much each contextualized anchor contributes to the word vector.

The paper does not clarify this. Under any option, the fixed β becomes less important once SAT provides context-dependent weighting. However, the choice affects how much of the distilled knowledge from Stage 1 is preserved versus overwritten during fine-tuning. The implementation spec (Section 8.4) uses Option C.


## 5. Training Efficiency Problem

Even if the architecture is corrected to anchor-level SAT, a significant practical problem exists that the paper does not discuss.

### 5.1 Variable sequence length

Each word i has k_i anchors. A sentence of L words expands to SL = Σ k_i anchor tokens. Different sentences in the same batch produce different SL values even with the same word count, because different words have different anchor cardinalities.

Example:
- Sentence 1: ["The", "bank", "collapsed"] with k = [1, 3, 2] → SL = 6
- Sentence 2: ["He", "ran", "fast"] with k = [1, 1, 1] → SL = 3

These require padding to max SL within a batch, with unpredictable and potentially severe padding waste.

### 5.2 Quadratic attention cost

Self-attention is O(SL²). If average k per word is 5 and L = 128, the anchor sequence length is ~640, making attention cost ~25× higher than word-level attention over the same sentence.

### 5.3 Complex attention masking

Beyond standard padding masks, the expanded sequence may require structured masking decisions: should co-anchors (same word) attend to each other? Should padding anchors be masked per-word or per-sequence? The paper does not address this.

### 5.4 Possible mitigation strategies

- **Cap k_i** to a fixed max_k, truncating anchor counts. Bounds SL ≤ max_k × L but discards information.
- **Intra-word attention only.** Anchors attend within their word group. Cost O(L × k²) instead of O((L×k)²). But removes cross-word context, defeating the purpose.
- **Two-stage attention.** First intra-word attention to reweight anchors, then aggregate to word level, then word-level cross-attention. Architecturally clean but doubles the computation.
- **This efficiency concern may explain the algorithm's ordering.** The authors may have tried anchor-level SAT, encountered the efficiency problem, and moved ScatterAdd before SAT as a practical compromise. Section 3.5 may describe the original design while the algorithm reflects the deployed compromise. If so, the paper should have acknowledged this tradeoff explicitly.


## 6. Other Problems

### 6.1 VP is unnecessary and breaks anchor sharing

The flattened matrix **E** ∈ ℝ^(N × K·d) duplicates anchor vectors across every word that uses them and wastes space on zero-padding. A standard implementation using nn.Embedding for the anchor matrix **A** ∈ ℝ^(K×d) plus stored sparse indices achieves the same result with less memory and correct gradient sharing:

    idx = indices[token_ids]        # (B, L, max_k)
    vecs = anchor_embedding(idx)    # (B, L, max_k, d)
    out = (vecs * weights).sum(2)   # (B, L, d)

The flattened **E** also creates a correctness problem: each word holds its own copy of anchor vectors, so gradient updates cause copies to diverge. The "shared codebook" is no longer shared during Stage 3 fine-tuning. The paper's compression numbers in Table 2 are computed using sparse **A** + index-weight pairs, not the flattened **E** — confirming that the flattened form is not used for the claimed compression.

### 6.2 GPE is not ablated

The paper ablates SAT (with vs. without) but does not ablate GPE separately. Since GPE is one of three claimed contributions, this is a significant gap. The paper acknowledges this: "a dedicated ablation of GPE remains for future work."

### 6.3 Missing details on sparsity mechanism

As noted in Section 2, the distillation loss contains no L1 regularization, and the paper does not clarify whether ANT's L1-based sparsity mechanism is used. Without L1, hard thresholding in Stage 2 would arbitrarily discard non-negligible weights, causing information loss. This is a critical missing detail because the entire VP pipeline depends on **T** being meaningfully sparse.

### 6.4 Text-batch distillation is unjustified

Stage 1 distillation is matrix factorization: **T**·**A** ≈ **E**_teacher. Both sides are context-free. Training on text batches introduces frequency-dependent bias — common words receive more gradient updates than rare words. The paper does not explain this choice.

### 6.5 Narrow evaluation

Only two text classification datasets. No generative tasks, no sequence labeling, no analysis of whether the same word actually receives different representations in different contexts. No visualization of learned anchor assignments. For a paper whose core claim is context-dependent composition, the absence of any polysemy-focused experiment is notable.

### 6.6 Parameter savings attribution is imprecise

The paper compares ADE (2.37M params) against DeBERTa-v3-base (184.4M params) and attributes the 98.7% reduction to the anchor embedding method. From Table 5, the actual savings breakdown is: word embedding elimination saves 98.4M params, encoder reduction (12L→1L SAT) saves 82.7M, positional/type embeddings save 0.4M, and pooler saves 0.6M. The word embedding elimination actually saves more than the encoder reduction. The paper frames the compression as a property of multi-anchor embeddings, but both factors contribute substantially.

### 6.7 Zero-anchor enforcement mechanism unclear

ADE claims 1 ≤ k_i (Section 3.3) and clamps the GPE sub-length map to ≥ 1 (Algorithm 1 line 4), but does not explain how k_i ≥ 1 is enforced during vocabulary projection if all weights for a word fall below τ. The original ANT paper observed that 2,673 out of 59,047 movies received entirely zero rows. ADE's clamping step acknowledges the possibility but provides no mechanism to guarantee at least one anchor survives thresholding.

### 6.8 Upstream cost not accounted

Distillation (Stage 1) and ANT training for sparse **T** represent significant computational costs not included in comparisons. DeBERTa-v3-base must be fully trained first as the teacher.

### 6.9 Frozen vs trainable anchor contradiction

The paper contradicts itself on whether anchors are trainable during Stage 3. Section 4.2 states: "the anchor embeddings within E are not frozen — they continue to be updated during training." Appendix C (Table 5 note) states: "anchor embeddings (76,800 params at K=100) are excluded from this count as they are trained offline via knowledge distillation and frozen during downstream fine-tuning." These are mutually exclusive. This is the same class of internal inconsistency as the Algorithm 1 vs. Section 3.5 contradiction identified in Section 4.

### 6.10 Selective ANT comparison on DBpedia

ADE's Table 1 cites ANT at 97.20% on DBpedia-14. However, ANT's own results (Appendix K.1, Table 11) show multiple configurations: |A|=100 with random init achieves 98.2% (28M embedding params), |A|=80 with cluster init achieves 98.1% (30M params). ADE cites only the most compressed result (|A|=20, 97.2%, 7M params), omitting configurations that exceed ADE's 98.06%. The comparison is further complicated by different classifiers (ANT uses CNN, ADE uses SAT), different vocabulary sizes (ANT: V=563K word-level, ADE: N=128K subword), and different embedding parameter counts. But the selective citation is notable.

### 6.11 Inference latency not cited in analysis

ADE provides actual latency measurements (Table 6, Appendix D) that the analysis sections of this document do not reference: DeBERTa-v3-base runs at 26.3 ms/batch (1217 samples/s), while ADE (K=100) runs at 180.4 ms/batch (177 samples/s) — 6.9× slower despite 98.7% fewer parameters. The paper acknowledges this: "the dominant inference cost arises from the anchor-expanded sequence length processed by SAT." These measurements directly support the efficiency concerns raised in Section 5 of this document.


## 7. Summary

The core idea — shared anchor codebook, GPE for word-level grouping, transformer attention for context-aware anchor selection — is a coherent and potentially valuable contribution. If implemented as described in Section 3.5 (anchor-level self-attention before aggregation), every claimed contribution is meaningful and the ablation results support this design.

However, the paper as written has a fundamental contradiction: Algorithm 1 aggregates anchors before SAT, making GPE redundant and context-aware reweighting impossible. Two interpretations exist:

1. **The algorithm is correct and the prose is wrong.** In this case, ADE is just a codebook embedding + single-layer word-level transformer, and two of three claimed contributions (GPE, context-aware reweighting) are non-functional.

2. **The idea and implementation are correct but the algorithm is written wrong.** In this case, the actual architecture works as Section 3.5 describes, but the paper fails to accurately document it, and an additional efficiency problem (quadratic cost on expanded anchor sequences) exists but is not discussed.

Either way, the paper has significant documentation and methodological gaps: missing sparsity details, unnecessary VP construction, absent GPE ablation, narrow evaluation, and misleading parameter comparisons. A good idea, poorly documented, looks the same as a bad idea from the outside.


## 8. Implementation Specification (Corrected)

This section provides a concrete, implementable pipeline that preserves the paper's core idea while fixing all identified problems. Every ambiguity is resolved with a specific choice and rationale.

### 8.1 Overview of corrected pipeline

The key differences from the paper's Algorithm 1:

- **Stage 1:** Use proximal gradient descent (from ANT) to enforce sparsity in **T** during distillation — NOT L1 in the loss, which does not produce exact zeros. Train over vocabulary rows directly, not text batches.
- **Stage 2:** Store anchor matrix **A** + sparse indices and weights. Do NOT build the flattened **E** matrix.
- **Stage 3:** Expand anchors → apply GPE → run SAT on expanded anchor sequence → aggregate to word level using β weights → pool → classify. This is the opposite order from Algorithm 1, which aggregates before SAT.

### 8.2 Stage 1 — Distillation with sparsity via proximal gradient descent

**Goal:** Learn anchor matrix **A** ∈ ℝ^(K×d) and sparse weight matrix **T** ∈ ℝ^(N×K) such that **T**·**A** ≈ **E**_teacher.

**Setup:**

Load DeBERTa-v3-base (model identifier: "microsoft/deberta-v3-base") and extract its embedding weight matrix. This is a frozen (N, d) matrix where N=128100 and d=768. Each row is DeBERTa's pretrained vector for one vocabulary token. This is E_teacher.

Initialize two trainable matrices:
- **A** ∈ ℝ^(K×d): the anchor matrix. Recommended initialization: run k-means++ on E_teacher to get K cluster centers. Each anchor starts as the centroid of a cluster of related words. Fallback: random initialization with small values (e.g., normal distribution, std=0.02).
- **T** ∈ ℝ^(N×K): the weight matrix. Initialize with small non-negative values (e.g., uniform in [0, 0.02)).

**Loss function — smooth part only:**

Compute the reconstruction: Ê = T @ A, shape (N, d). Compute the cosine similarity between each row of Ê and the corresponding row of E_teacher, then average. Loss = 1 − mean cosine similarity.

The loss contains ONLY the cosine reconstruction error. L1 is NOT included in the loss function. The ANT paper explicitly warns that subgradient descent on L1 does not produce exact zeros. Sparsity is instead enforced by the proximal operator applied after each gradient step.

**Proximal gradient descent — applied AFTER each optimizer step:**

Each training iteration has two steps:
1. Compute gradients of the cosine loss (smooth loss only) and update A and T via the optimizer.
2. Immediately after the optimizer step, apply the proximal operator to T: set every entry of T to max(T − η·λ₂, 0), where η is the learning rate.

This proximal operator (from ANT, Equation 3) combines soft-thresholding and non-negativity in one step. Any entry in T smaller than η·λ₂ after the gradient step is set to exactly zero. Any entry that goes negative (from the gradient update) is also clamped to zero. This produces clean sparsity — entries not useful for reconstruction are driven to exact zero, not just close to zero.

**Why this ordering matters:** The optimizer updates T based on the cosine loss gradient (which pulls T toward better reconstruction). Then the proximal operator pulls T toward sparsity (setting small entries to zero). The balance between these two forces, controlled by λ₂, determines how sparse T becomes. If an anchor is genuinely useful for reconstructing a word, the gradient will keep its weight above the threshold. If it is not useful, the proximal operator will zero it out.

**Note on adaptive optimizers:** With SGD, the learning rate η is the same for all parameters, so `lr * λ₂` is the correct threshold. With Adam/AdamW, the effective learning rate differs per parameter (η / (√v + ε)). ANT uses YOGI (an Adam variant) and applies the proximal operator with the base learning rate. For simplicity, use SGD for Stage 1 (matrix factorization converges well with SGD), or use Adam with the base learning rate in the proximal step as an approximation.

**Training over vocabulary directly:** Since both **T**·**A** and **E**_teacher are context-free (the same word always produces the same vectors regardless of sentence), this is pure matrix factorization. Iterate over all N=128,100 vocabulary rows each epoch. If memory is tight, split into mini-batches of vocabulary rows (e.g., 4096 rows per batch). This avoids the frequency bias of text-batch training — every word receives equal optimization attention.

**Recommended hyperparameters for Stage 1:**

| Parameter | Value | Notes |
|---|---|---|
| Optimizer | SGD (recommended) or Adam | SGD makes proximal step clean; Adam is an approximation |
| Learning rate | 1e-2 (SGD) or 1e-3 (Adam) | |
| λ₂ (sparsity weight) | 1e-3 | Start here, increase if **T** is not sparse enough after training |
| Epochs | 500–1000 | Matrix factorization, cheap per epoch (128K rows × K multiply) |
| Batch size | 4096–8192 | Vocabulary rows per batch, not text sequences |
| Anchor init | k-means++ on E_teacher | Falls back to random if clustering is too slow |

**Convergence check:** After training, inspect **T**. For a good result, most entries per row should be exactly zero, with a few clearly dominant weights. Count the average number of non-zero entries per row — this will become the average k_i after thresholding. If **T** is still dense (most entries non-zero), increase λ₂. If reconstruction quality is poor (mean cosine similarity < 0.9), decrease λ₂ or increase K. With proper proximal gradient descent, the threshold τ in Stage 2 can be set very small (e.g., τ = 1e-6) or even zero — entries that should be inactive are already exactly zero from the proximal operator, unlike the subgradient approach where small-but-nonzero entries require a meaningful threshold to clean up.

### 8.3 Stage 2 — Build sparse lookup table

**Goal:** Convert the dense weight matrix **T** into a sparse set of anchor assignments per word.

**Step 2a — Threshold:** With proper proximal gradient descent in Stage 1, most inactive entries in **T** are already exactly zero. Apply a small threshold τ (e.g., 1e-6) to catch floating point noise. For each word i, keep anchor j if T[i, j] ≥ τ. Record the surviving indices and their weights. If Stage 1 used subgradient instead of proximal gradient, use a larger τ (e.g., 0.05).

**Step 2b — Handle zero-anchor words:** Some words (typically very rare ones) may have all weights below τ, resulting in zero active anchors. Fallback: assign them the single anchor with the highest weight in their row of **T**, with weight 1.0. This guarantees every word has at least one anchor.

**Step 2c — Cap anchor count:** To enable efficient batching in Stage 3, cap the number of active anchors per word at max_k (e.g., 12). If a word has more than max_k active anchors, keep only the max_k with the highest weights. The paper uses no cap (handles variable cardinalities via padding masks), but capping is necessary for efficient batched GPU computation. Setting max_k at the average k_i (≈8) would truncate roughly half the words; 12 captures more of the distribution tail.

**Step 2d — Build padded lookup tensors:** Create two tables, both of shape (N, max_k):
- indices_table (integer): for each word, the anchor indices of its active anchors, padded with zeros. Store as int32 for disk storage; cast to int64 at runtime for PyTorch embedding indexing.
- weights_table (float32): for each word, the corresponding weights, padded with zeros.

Real anchors occupy the first k_i slots. Padding slots contain zero weights, which allows the mask `weights != 0` to distinguish real anchors from padding in Stage 3.

**What is stored after Stage 2:**

| Component | Shape | Type | Description |
|---|---|---|---|
| **A** | (K, d) | nn.Embedding | Shared anchor matrix, trainable in Stage 3 |
| indices_table | (N, max_k) | Buffer (int) | Per-word anchor indices, frozen |
| weights_table | (N, max_k) | Buffer (float) | Per-word anchor weights β, frozen |

Total storage: K×d floats for anchors + N × max_k int32 for indices + N × max_k float32 for weights. For K=500, d=768, N=128100, max_k=12: anchors = 384K floats (~1.5 MB), indices = ~6.1 MB (int32), weights = ~6.1 MB (float32), total ~14 MB. The paper's 9.28 MB at K=500 (Table 2) is lower because it uses variable-length sparse storage (only k_i entries per word, avg k_i ≈ 8, no padding waste), while our implementation pads every word to max_k=12 slots. PyTorch requires int64 at runtime for embedding indexing, but the paper's storage figures use int32. No duplication of anchor vectors.

### 8.4 Stage 3 — Corrected forward pass with anchor-level SAT

This is the corrected pipeline where SAT operates on the expanded anchor sequence before aggregation.

**Model components:**

- anchor_emb: an embedding layer of size (K, d). Initialized by copying Stage 1's trained **A**. Trainable.
- indices_table: buffer of shape (N, max_k), int64. From Stage 2. Frozen.
- weights_table: buffer of shape (N, max_k), float32. From Stage 2. Frozen.
- pos_encoding: positional encoding for word positions. The paper uses fixed (generated) positional encodings — Table 5 shows 0 positional parameters for ADE, and Section 3.4 says "Generate standard positional encodings." For the corrected implementation, either fixed sinusoidal PE (matching the paper, 0 trainable params) or learned PE (adds ~L×d params, e.g. ~197K at max_word_len=256) can be used. max_word_len is the maximum word-level sequence length — NOT the expanded SL = L × max_k.
- sat_layer: multi-head self-attention only (NO feed-forward sublayer) with d_model=d, 8 attention heads, dropout 0.1. This is NOT a standard transformer encoder layer — it has Q/K/V/O projections but no FFN, no internal LayerNorm, no activation function. Total params: 4 × (768×768 + 768) = 2,362,368. Trainable, initialized randomly.
- layer_norm: layer normalization over dimension d. Applied before SAT (see note below). Trainable.
- pooler: a learned linear layer (d → 1) with bias, producing per-position scalar scores. Apply softmax over unmasked positions to get attention weights, then compute weighted sum of word vectors. Total params: 769 = 768 + 1. This is NOT simple mean pooling. Trainable.
- classifier: linear layer from d to C (number of classes). Trainable.
- dropout: dropout with rate 0.1, applied before classifier.
- pad_token_id: the tokenizer's padding token ID, used in the pooler mask.

**Note on LayerNorm placement:** The paper's Algorithm 1 places LayerNorm between ScatterAdd and SAT (line 8, before line 10). In the corrected pipeline where SAT comes before aggregation, a reasonable placement is on the expanded anchor sequence before SAT — normalizing the raw anchor embeddings + GPE before they enter attention. This is a design choice that may affect training stability.

**Forward pass with tensor shapes at every step:**

**Input:** token_ids of shape (B, L) — a batch of tokenized, padded sequences.

**Step 3a — Gather anchor indices and weights for each token:**

Look up each token ID in indices_table and weights_table:
- idx: (B, L, max_k) — for each token, its max_k anchor indices
- w: (B, L, max_k) — for each token, its max_k anchor weights
- mask: (B, L, max_k) — True where w != 0 (real anchors), False where padding

**Step 3b — Retrieve anchor vectors from the shared codebook:**

Look up each index in idx from anchor_emb:
- anchor_vecs: (B, L, max_k, d) — raw anchor embeddings

No weighting by β here. SAT will see the raw anchor vectors. β is applied after SAT during aggregation.

**Step 3c — Reshape to expanded anchor sequence:**

Flatten the L and max_k dimensions into one sequence dimension:
- SL = L × max_k
- expanded: (B, SL, d) — reshape anchor_vecs from (B, L, max_k, d)
- expanded_mask: (B, SL) — reshape mask from (B, L, max_k)

The ordering places all max_k anchors of word 0 first (positions 0 to max_k-1), then word 1 (positions max_k to 2×max_k-1), and so on. This ordering must match the GPE construction in the next step.

**Step 3d — Apply Grouped Positional Encoding (GPE):**

Construct a position index vector of length SL where all max_k slots of the same word share the same position. For word at index l, its anchor slots at flattened positions [l×max_k, ..., (l+1)×max_k−1] all receive position l:

- word_positions: (SL,) — [0,0,...,0, 1,1,...,1, 2,2,...,2, ...] with max_k repeats per word

Example with L=3, max_k=3: word_positions = [0, 0, 0, 1, 1, 1, 2, 2, 2]

Look up word_positions in pos_encoding to get (SL, d), then add to expanded (broadcasting over batch). Result: expanded (B, SL, d), now with positional information.

**Step 3e — SAT (multi-head self-attention on expanded sequence, NO FFN):**

Apply LayerNorm to the expanded sequence. Then pass through multi-head self-attention (Q/K/V/O projections only — no feed-forward sublayer, no activation function). Use a key padding mask derived from expanded_mask: positions where expanded_mask is False (padding anchors) should be ignored — they do not attend to anything and nothing attends to them.

- Input: (B, SL, d) with padding mask (B, SL)
- After LayerNorm: (B, SL, d)
- After attention: (B, SL, d)

Each real anchor attends to all other real anchors across the full sequence. An anchor for "bank" can attend to anchors for "river" elsewhere in the sentence, enabling context-dependent representation.

**Step 3f — Aggregate back to word level using β weights:**

Reshape SAT output back to (B, L, max_k, d). Multiply each anchor's output by its weight w and by the mask (to zero out padding). Sum over the max_k dimension:

- SAT output reshaped: (B, L, max_k, d)
- Multiply by w (unsqueezed to (B, L, max_k, 1)) and mask (unsqueezed to (B, L, max_k, 1))
- Sum over dim 2: word_vectors (B, L, d)

Each word's final representation is the β-weighted sum of its contextualized anchors.

**Step 3g — Learned pooling and classification:**

Apply the learned pooler: pass each word vector through the linear(d→1) layer to get a scalar score per position, shape (B, L). Mask out padding positions (set scores to -∞ where token_ids == pad_token_id). Apply softmax over the remaining positions to get attention weights, shape (B, L). Compute the weighted sum of word_vectors using these weights. Result: pooled (B, d).

Apply dropout, then the linear classifier. Result: logits (B, C). Loss: cross-entropy between logits and labels.

**Summary of tensor shapes through the corrected forward pass:**

    token_ids:          (B, L)
    idx:                (B, L, max_k)
    anchor_vecs:        (B, L, max_k, d)
    expanded:           (B, L*max_k, d)       ← anchor-level sequence
    + GPE:              (B, L*max_k, d)
    LayerNorm:          (B, L*max_k, d)
    SAT output:         (B, L*max_k, d)
    reshape:            (B, L, max_k, d)
    β-weighted sum:     (B, L, d)             ← back to word level
    pooler scores:      (B, L)
    pooled:             (B, d)
    logits:             (B, C)

### 8.5 What is trainable in Stage 3

| Component | Trainable? | Notes |
|---|---|---|
| anchor_emb (shared **A**) | Yes | Initialized from Stage 1, updated by gradients |
| indices_table | No | Frozen from Stage 2 |
| weights_table (β) | No | Frozen from Stage 2 |
| pos_encoding | Depends | Fixed sinusoidal (0 params, matches paper) or learned (~197K params) |
| sat_layer (attention only) | Yes | 2,362,368 params. Trained from random init |
| layer_norm | Yes | 1,536 params. Trained from random init |
| pooler | Yes | 769 params. Learned attention pooling. Trained from random init |
| classifier | Yes | Trained from random init |

Because anchors are stored in a single nn.Embedding(K, d), all words that share an anchor point to the same row. Gradients from different words accumulate on the same anchor vector. The codebook remains shared throughout training — unlike the paper's flattened **E** which duplicates anchors.

### 8.6 Recommended hyperparameters for Stage 3

| Parameter | Value | Notes |
|---|---|---|
| Optimizer | AdamW | |
| Learning rate | 2e-5 | Paper's value (Section 4.2). Our corrected architecture may need tuning |
| Weight decay | 0.01 | |
| Dropout | 0.1 | Applied before classifier and in SAT attention |
| Epochs | 5–10 | Standard for classification fine-tuning |
| Batch size | 32–64 | Adjust for GPU memory (SL = L × max_k is long) |
| Max sequence length L | 128–256 | After tokenization |
| max_k | 12 | Cap on anchors per word. Paper uses no cap (avg k_i ≈ 8); 12 captures most of the distribution tail while bounding SL |
| SAT heads | 8 | |
| K (codebook size) | 500 | Paper's best. Tune between 200–700 |

**Memory note:** With L=128 and max_k=12, the expanded sequence length SL=1536. Self-attention cost is O(SL²) ≈ O(2.4M) per sample. This is comparable to a standard transformer with L=1536, which is manageable. Reduce batch size or max_k if memory is tight.

### 8.7 Data, tokenizer, and preprocessing

**Tokenizer:** Use DeBERTa-v3-base's tokenizer (model identifier: "microsoft/deberta-v3-base"). This is the same tokenizer whose vocabulary defines the N=128,100 token IDs used throughout the pipeline. Note the pad_token_id from this tokenizer — it is needed for the learned pooler's mask in Step 3g.

**Datasets:**

| Dataset | HuggingFace name | Classes | Train | Test | Text field | Label field |
|---|---|---|---|---|---|---|
| AG News | "ag_news" | 4 | 120K | 7.6K | "text" | "label" (0–3) |
| DBpedia-14 | "fancyzhx/dbpedia_14" | 14 | 560K | 70K | "title" + "content" | "label" (0–13) |

Note on DBpedia: The ADE paper describes each sample as "a Wikipedia title and abstract." The HuggingFace dataset has separate "title" and "content" fields. Concatenate both (e.g., "title. content") to match the paper's input format.

**Preprocessing:** Tokenize each text sample using the DeBERTa tokenizer with truncation and padding to a fixed max_length (e.g., 128). This produces a tensor of token IDs of shape (L,) per sample, where L = max_length. This max_length must match the max_word_len used for the positional encoding table in Stage 3.

**Validation split:** Neither dataset has an official validation split. Hold out 5–10% of training data for validation. Use validation performance for hyperparameter tuning and early stopping.

### 8.8 Evaluation

Evaluate on test sets after training. Metric: accuracy for both datasets.

| Dataset | Classes | Train size | Test size | Paper's result | Target |
|---|---|---|---|---|---|
| AG News | 4 | 120K | 7.6K | 90.64% | Match or exceed |
| DBpedia-14 | 14 | 560K | 70K | 98.06% | Match or exceed |

The corrected pipeline (anchor-level SAT) should match or exceed the paper's numbers if the core idea is sound. If results are significantly lower, the first things to check: Stage 1 reconstruction quality (mean cosine similarity should be > 0.9), average k_i after thresholding (should be 3-8 per word, not 1 everywhere), and SAT training convergence (loss should decrease steadily).

