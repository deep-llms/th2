# Anticipatory Deep-KV Pilot
## Minimal Four-Arm From-Scratch Specification

## 1. Goal

Test whether a Transformer can use **predicted mature K/V states early** to obtain useful deep-to-shallow feedback while preserving a normal single-pass, token-parallel training graph.

The core hypothesis is:

\[
\text{shallow state}
\rightarrow
\widehat{\text{deep K/V}}
\rightarrow
\text{future shallow attention}
\]

with the actual deep K/V produced later in the **same forward pass** used only as an optional alignment target.

There is:

- no external teacher,
- no privileged second pass,
- no Jacobi/fixed-point iteration,
- no recurrent token-by-token training,
- no calibration phase before or during training.

The first pilot should change as little as possible.

---

# 2. Backbone and training budget

Use the existing Qwen3-0.6B-style from-scratch training pipeline and keep all ordinary model/data/training settings identical across arms.

For the current Qwen3-0.6B configuration:

- 28 Transformer blocks
- hidden size = 1024
- query heads = 16
- KV heads = 8
- head dimension = 128
- sequence length = 2048

All arms must use:

- the same tokenizer,
- the same training stream,
- the same data order,
- the same packing/preprocessing,
- the same optimizer and LR schedule,
- the same global token batch,
- the same initialization seed for parameters shared across matched modified arms.

## Training budget

Train **each arm for 1.0B non-padding input tokens**.

If the existing global batch remains 32,768 input tokens per optimizer update, use:

\[
30{,}518\ \text{updates}
\]

which corresponds to:

\[
1{,}000{,}013{,}824\ \text{input tokens}.
\]

Use exactly the same number and ordering of training tokens for every arm.

The four-arm pilot therefore uses approximately **4B total training tokens**.

This is a mechanism pilot, not a compute-optimal pretraining run. Do not extend one arm beyond 1B tokens based on intermediate results. If the result is promising, define a separate scaled follow-up.

---

# 3. Exact depth convention

Use **1-based human block numbering** in this document.

The earlier residual-stream pair \( (4,20) \) meant:

- shallow state = output of block 4,
- deep state = output of block 20.

For the K/V formulation, preserve those exact residual coordinates:

\[
r_t^4
=
\text{output of block 4}
=
\text{input residual to block 5}
\]

and:

\[
r_t^{20}
=
\text{output of block 20}
=
\text{input residual to block 21}.
\]

Therefore:

- the **consumer attention** is block 5,
- the **shallow alignment target**, when used, is block-5 native K/V,
- the **deep alignment target**, when used, is block-21 native K/V.

This avoids an off-by-one change relative to the previous \(4 \rightarrow 20\) residual-state interpretation.

---

# 4. Shared auxiliary feedback branch

Arms B, C, and D use the **same forward architecture**.

Let:

\[
u_t
=
\operatorname{RMSNorm}_{\text{block5-input}}(r_t^4)
\]

be the normalized state entering block-5 attention.

## 4.1 Query

Reuse the **native block-5 query**.

Do not add a second query projection for the auxiliary branch.

Conceptually:

\[
Q_t^5
=
\operatorname{RoPE}
\left(
\operatorname{QNorm}
\left(
W_Q^5 u_t
\right)
\right).
\]

The same \(Q_t^5\) is used by both:

- native block-5 self-attention,
- the auxiliary feedback attention.

## 4.2 Predicted K/V

Use two simple bias-free linear projections:

\[
\hat K_t
=
\operatorname{KNorm}_{aux}
\left(
P_K u_t
\right)
\]

\[
\hat V_t
=
P_V u_t.
\]

Shapes must match Qwen's native GQA K/V shapes:

\[
8 \text{ KV heads} \times 128 \text{ dimensions}.
\]

For the first pilot, do **not** add an MLP, Transformer block, router, or multi-depth predictor.

Apply the same RoPE convention and the same position IDs to \(\hat K\) when it is consumed by attention.

## 4.3 Strict-past feedback attention

The auxiliary branch uses only:

\[
j < t.
\]

Define:

\[
m_t^{aux}
=
\operatorname{Attn}
\left(
Q_t^5,
\hat K_{<t},
\hat V_{<t}
\right)
\]

with the auxiliary visibility mask equal to:

\[
(j<t)
\cap
\text{backbone allowed-context mask}.
\]

For a query with no valid strict-past source, the auxiliary output is exactly zero.

## 4.4 Output injection

Give the auxiliary branch its own output projection:

\[
c_t
=
W_O^{aux}m_t^{aux}.
\]

Initialize:

\[
W_O^{aux}=0.
\]

No additional gate is needed in the first pilot.

Block 5 then uses:

\[
r_t^4
+
\text{native-attention-output}_t
+
c_t
\]

before continuing through the normal remainder of block 5.

Thus every modified arm starts exactly at Base behavior.

---

# 5. K/V target convention

Targets must be the **native K/V states actually produced by Qwen attention**, not arbitrary post-block hidden-state projections.

## Key target

Use the native key:

- after the layer's K projection,
- after native per-head K normalization,
- before RoPE.

The predicted key is matched in this same pre-RoPE normalized key space.

RoPE is applied only when the predicted key is used by attention.

## Value target

Use the native value directly after its V projection and reshape into KV heads.

No additional normalization or calibration is introduced.

---

# 6. Alignment loss

For any aligned arm, use direct L1 consistency:

\[
L_K
=
\operatorname{mean}
\left|
\hat K
-
\operatorname{stopgrad}(K^{target})
\right|
\]

\[
L_V
=
\operatorname{mean}
\left|
\hat V
-
\operatorname{stopgrad}(V^{target})
\right|.
\]

Combine them as:

\[
L_{KV}
=
\frac{L_K+L_V}{2}.
\]

The aligned objective is:

\[
L
=
L_{LM}
+
\lambda_{KV}L_{KV}
\]

with the first-run value:

\[
\boxed{\lambda_{KV}=1}.
\]

Do not:

- pre-compute normalization constants,
- pause training for calibration,
- use separate \(\lambda_K,\lambda_V\),
- use adaptive loss balancing,
- tune \(\lambda_{KV}\) during the run.

Log \(L_K\) and \(L_V\) separately for diagnostics, but do not change their weights in this pilot.

`stopgrad` applies only to the target side of the auxiliary loss. LM gradients flow normally through the full model and through the auxiliary use path.

---

# 7. The four arms

## Arm A — Base

Unmodified Qwen Transformer.

No auxiliary branch.

\[
L_A=L_{LM}.
\]

### Question answered

What is the baseline LM quality at the same token budget?

---

## Arm B — ExtraAttn-NoAlign

Enable the auxiliary branch defined in Section 4.

The auxiliary K/V are learned from the shallow block-5 input:

\[
(\hat K,\hat V)
=
(P_K(u),P_V(u)).
\]

There is **no K/V alignment loss**.

\[
L_B=L_{LM}.
\]

### Question answered

Does simply adding a learned strict-past auxiliary attention path help?

This is the clean capacity/architecture control.

---

## Arm C — ShallowKV-Align

Use exactly the same forward architecture and initialization as Arm B.

Add an auxiliary loss that matches the predicted K/V to the **native block-5 K/V** produced from the same shallow state.

Let:

\[
K^{shallow},V^{shallow}
\]

be block-5 native K/V.

Then:

\[
L_C
=
L_{LM}
+
\frac{
L_1(\hat K,\operatorname{sg}(K^{shallow}))
+
L_1(\hat V,\operatorname{sg}(V^{shallow}))
}{2}.
\]

### Question answered

Does adding an easy same-depth K/V consistency objective help merely as regularization?

This arm controls for the possibility that any K/V alignment loss helps, regardless of deep knowledge.

---

## Arm D — DeepKV-Align

Use exactly the same forward architecture and initialization as Arms B and C.

Later in the **same forward pass**, capture the native K/V from block 21, corresponding to the residual state after block 20:

\[
K^{deep},V^{deep}.
\]

Use:

\[
L_D
=
L_{LM}
+
\frac{
L_1(\hat K,\operatorname{sg}(K^{deep}))
+
L_1(\hat V,\operatorname{sg}(V^{deep}))
}{2}.
\]

### Question answered

Does explicitly anticipating mature deep K/V improve the model beyond:

- Base,
- extra attention capacity,
- and generic shallow K/V alignment?

This is the main experimental arm.

---

# 8. Critical comparisons

## B vs A — extra attention effect

\[
\text{ExtraAttn-NoAlign}
\quad \text{vs} \quad
\text{Base}.
\]

If B beats A, the extra attention pathway itself is useful.

This alone is not evidence for deep anticipation.

---

## C vs B — generic alignment effect

\[
\text{ShallowKV-Align}
\quad \text{vs} \quad
\text{ExtraAttn-NoAlign}.
\]

If C beats B, an auxiliary K/V consistency objective provides useful regularization even without deep information.

---

## D vs B — deep alignment vs no alignment

\[
\text{DeepKV-Align}
\quad \text{vs} \quad
\text{ExtraAttn-NoAlign}.
\]

Tests whether deep supervision helps relative to the identical architecture trained only by LM loss.

---

## D vs C — most important depth-specific control

\[
\boxed{
\text{DeepKV-Align}
\quad \text{vs} \quad
\text{ShallowKV-Align}
}
\]

Both arms have:

- the same architecture,
- the same number of parameters,
- the same auxiliary loss form,
- the same coefficient,
- the same token budget.

The only substantive difference is whether the target comes from shallow or mature deep K/V.

If D reliably beats C, that is the cleanest evidence in this pilot that **the value comes specifically from anticipating deeper knowledge**, rather than from an extra attention branch or an arbitrary consistency loss.

---

## D vs A — practical value

\[
\text{DeepKV-Align}
\quad \text{vs} \quad
\text{Base}.
\]

The proposed mechanism is only interesting if it ultimately improves the LM itself.

---

# 9. Fairness requirements

All arms must use:

- identical training examples in identical order,
- identical preprocessing and packing,
- identical token/update budget,
- identical optimizer and LR schedule,
- identical sequence length,
- identical evaluation data.

For Arms B/C/D:

- auxiliary module parameter shapes must be identical,
- initialization must be byte-identical,
- only the presence/type of the auxiliary target may differ.

Do not change architecture, loss weights, layer pair, or training length after observing one arm.

---

# 10. Minimal acceptance tests before training

Only a small set of tests is required.

1. **Base equivalence**  
   With the zero-initialized auxiliary output projection, Arms B/C/D must initially reproduce Base logits within numerical tolerance.

2. **Strict-past causality**  
   Auxiliary attention must have zero mass for \(j \ge t\).

3. **Empty-source safety**  
   The first valid token of each isolated context/segment must receive exactly zero auxiliary output.

4. **Mask equivalence**  
   Auxiliary visibility must never exceed the backbone's allowed-context mask.

5. **Target correctness**  
   Shallow target tensors must exactly equal native block-5 pre-RoPE normalized K and native V.  
   Deep target tensors must exactly equal native block-21 pre-RoPE normalized K and native V.

6. **Stop-gradient correctness**  
   Backpropagating only \(L_{KV}\) must not produce gradients through the target tensors.

No larger acceptance-test suite is required for this pilot.

---

# 11. What not to add yet

Do not add in the first pilot:

- privileged teachers,
- extra forward passes,
- Jacobi/fixed-point training,
- WhiteMatter-style routing,
- multiple source depths,
- multiple predicted channels,
- nonlinear predictor MLPs,
- query-conditioned operators,
- route-KL losses,
- message losses,
- cosine losses,
- adaptive loss weighting,
- confidence gates,
- sparsity gates,
- raw hidden-state prediction,
- deep-minus-shallow residual prediction.

If the four-arm pilot shows a real deep-specific gain, these become follow-up experiments.

---

# 12. Decision rule

The key result is not simply:

\[
D>A.
\]

The strongest evidence for the intended mechanism is:

\[
\boxed{
D>C
}
\]

together with:

\[
D>B
\]

and:

\[
D>A.
\]

That pattern means the gain cannot be explained only by:

- additional attention capacity,
- a learned auxiliary memory,
- or a generic same-depth consistency loss.

It specifically supports the hypothesis that **anticipating mature deep K/V is useful**.

If D does not beat B or C, do not immediately add a more complicated router or predictor. First inspect whether the simple linear predictor is underfitting the deep K/V target; only then define a separate follow-up experiment.
