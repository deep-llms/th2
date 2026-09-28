# Deep Functional-Loss Variants
## Same Arm D Forward, New Auxiliary Loss Only

## 1. Scope

This specification defines two new loss variants of **Arm D — DeepKV-Align**.

**The forward architecture must remain exactly the same as Arm D.**

Do not change:

- the backbone,
- shallow/deep depth pair,
- block numbering,
- auxiliary predictor,
- native shallow query,
- predicted K/V dimensions,
- strict-past mask,
- RoPE treatment,
- GQA layout,
- auxiliary output projection,
- initialization,
- optimizer,
- LR schedule,
- batch size,
- training data/order,
- training duration,
- or evaluation procedure.

The **only change from Arm D is the auxiliary loss**.

The two variants to run are:

1. **Variant F — K routing KL only**
2. **Variant G — K routing KL + message loss**

There is no raw K/V L1 or MSE loss in either variant.

---

# 2. Inherited Arm D forward

Use the exact Arm D forward path.

With the existing \(4 \rightarrow 20\) residual convention:

- shallow residual = output of block 4 / input residual to block 5,
- consumer attention = block 5,
- deep target state = output of block 20 / input residual to block 21,
- actual deep K/V = native block-21 K/V.

Let:

\[
Q_t^s
\]

be the native shallow query already used by the auxiliary branch, and let:

\[
\hat K_t,\hat V_t
\]

be the auxiliary K/V predicted from the shallow state exactly as in Arm D.

The real auxiliary forward remains:

\[
m_t^{pred}
=
\operatorname{Attn}
\left(
Q_t^s,
\hat K_{<t},
\hat V_{<t}
\right),
\]

using the same strict-past condition:

\[
j<t
\]

intersected with the backbone allowed-context mask.

The auxiliary output is injected exactly as in Arm D.

Actual deep \(K^d,V^d\) produced later in the same normal forward pass are **never injected back into the model**. They are used only to construct detached auxiliary targets.

---

# 3. Key representation used by the loss

For both predicted and actual deep keys, construct the keys in the exact form used by attention:

1. K projection,
2. native/per-arm K normalization,
3. RoPE with the exact position IDs,
4. normal GQA expansion/repetition required to pair the KV heads with the shallow query heads.

No new projection is introduced for the loss.

The predicted keys used here must be the same predicted keys used by the real auxiliary forward.

---

# 4. Detach the query inside the auxiliary loss

The real forward continues to use the normal differentiable shallow query:

\[
Q^s.
\]

LM gradients flow through it normally.

For **auxiliary-loss computation only**, define:

\[
Q_{\mathrm{loss}}^s
=
\operatorname{stopgrad}(Q^s).
\]

Use this same detached shallow query to construct both the deep-reference and predicted routing distributions.

This keeps the loss change comparable to the original Arm D alignment loss:

- the auxiliary loss trains the predicted K/V pathway,
- it does not make the shared shallow query move merely to make the auxiliary target easier to match,
- the normal LM objective can still update the shallow query and the entire model.

---

# 5. Deep-reference and predicted routing distributions

For each valid query position \(t\) and shallow query head \(h\), compute:

\[
z_{t,j}^{deep}
=
\frac{
Q_{\mathrm{loss},t}^{s}
{K_j^{d}}^\top
}{
\sqrt{d_h}
}
+
M_{t,j},
\]

and:

\[
z_{t,j}^{pred}
=
\frac{
Q_{\mathrm{loss},t}^{s}
{\hat K_j}^\top
}{
\sqrt{d_h}
}
+
M_{t,j},
\]

where \(M\) is exactly the same mask for both paths:

\[
M
=
\text{strict-past mask}
\cap
\text{backbone allowed-context mask}.
\]

Thus only:

\[
j<t
\]

and backbone-visible source positions may receive probability mass.

Use the normal attention scale:

\[
1/\sqrt{d_h}.
\]

For this experiment, use no extra distillation temperature:

\[
\boxed{T=1}.
\]

Define:

\[
A_t^{deep}
=
\operatorname{softmax}(z_t^{deep}),
\]

\[
A_t^{pred}
=
\operatorname{softmax}(z_t^{pred}).
\]

Rows with no valid strict-past source must be excluded rather than softmaxed as all-masked rows.

---

# 6. K routing loss

The K-side functional target is the routing distribution itself:

\[
\boxed{
L_{\mathrm{route}}
=
\operatorname{mean}_{t,h}
D_{\mathrm{KL}}
\left(
\operatorname{stopgrad}(A_{t,h}^{deep})
\;\Vert\;
A_{t,h}^{pred}
\right)
}
\]

where the mean is over:

- valid query positions,
- shallow query heads.

Use the KL direction exactly as written:

\[
D_{\mathrm{KL}}
\left(
A^{deep}\Vert A^{pred}
\right).
\]

The deep-reference distribution is detached.

The predicted distribution is differentiable with respect to the predicted-key pathway.

Compute the KL/log-softmax reduction in fp32.

This loss does **not** require:

\[
\hat K \approx K^d
\]

coordinate-by-coordinate.

It requires the predicted keys to cause the shallow query to route across past tokens similarly to actual deep keys.

---

# 7. Functional message target for the K+V variant

Variant G additionally matches the **content delivered by attention**.

Expand/repeat the actual deep V and predicted V with the same GQA mapping needed for the shallow query heads.

Using the distributions defined above, compute:

\[
m_t^{deep}
=
A_t^{deep} V_{<t}^{d},
\]

and:

\[
m_t^{pred}
=
A_t^{pred} \hat V_{<t}.
\]

These messages are compared **before the auxiliary output projection**.

Define:

\[
\boxed{
L_{\mathrm{msg}}
=
\operatorname{mean}
\operatorname{SmoothL1}
\left(
m^{pred},
\operatorname{stopgrad}(m^{deep})
\right)
}
\]

over:

- valid query positions,
- query heads,
- head dimensions.

Use the framework's standard SmoothL1 with:

\[
\boxed{\beta=1.0}.
\]

Compute the reduction in fp32.

This is the V/content supervision used in this version.

Do not also add raw:

\[
L_1(\hat V,V^d),
\]

MSE on V, V-V relation loss, or another value loss.

---

# 8. Variant F — K routing KL only

The first new arm uses only the routing target:

\[
\boxed{
L_F
=
L_{\mathrm{LM}}
+
0.3\,L_{\mathrm{route}}
}
\]

There is no auxiliary loss on \(\hat V\).

The predicted values are trained only through the ordinary LM gradient flowing through the real auxiliary attention branch.

### Question

> Does functional deep routing supervision improve over Arm B, which has the same auxiliary forward architecture but no deep-derived auxiliary target?

---

# 9. Variant G — K routing KL + message loss

The second new arm keeps **exactly the same K-routing term and weight as Variant F**, then adds the message/content target:

\[
\boxed{
L_G
=
L_{\mathrm{LM}}
+
0.3\,L_{\mathrm{route}}
+
0.3\,L_{\mathrm{msg}}
}
\]

This is intentional.

Do **not** average the two auxiliary terms into:

\[
0.3(L_{\mathrm{route}}+L_{\mathrm{msg}})/2,
\]

because that would reduce the K-routing coefficient from \(0.3\) in Variant F to \(0.15\) in Variant G and confound the comparison.

Variant G differs from Variant F only by the addition of:

\[
0.3\,L_{\mathrm{msg}}.
\]

### Question

> After keeping deep routing supervision fixed, does adding deep supervision for the content actually delivered by attention provide additional benefit?

---

# 10. The two runs required

Run exactly these two new arms first:

| Variant | Auxiliary objective |
|---|---|
| **F — K routing only** | \(L_{LM}+0.3L_{route}\) |
| **G — K routing + message** | \(L_{LM}+0.3L_{route}+0.3L_{msg}\) |

A V-only arm is **not required** in this round.

Do not add another loss or change a weight after seeing an intermediate checkpoint.

---

# 11. Required comparisons

The main control remains:

**Arm B — ExtraAttn-NoAlign**

Arm B has the same auxiliary forward architecture but no deep-derived auxiliary target.

## F vs B

\[
\boxed{
F<B
}
\]

in validation loss would show that deep-derived **routing** knowledge provides value beyond the auxiliary attention architecture itself.

## G vs F

\[
\boxed{
G<F
}
\]

would show that deep-derived **delivered-content/message** supervision adds value after keeping K-routing supervision fixed.

## G vs B

If G beats B, the combined functional deep supervision improves over the no-deep-supervision control.

Also report Base and the previous raw deep-K/V L1 runs for context, but do not change this experiment based on those earlier results.

---

# 12. Implementation sketch

For the auxiliary loss only:

```python
# Q/K tensors shown after normal attention preparation:
# q_shallow: [batch, q_heads, seq, head_dim]
# k_deep:    [batch, q_heads, seq, head_dim]  # after GQA expansion
# k_pred:    [batch, q_heads, seq, head_dim]  # after GQA expansion
# v_deep:    [batch, q_heads, seq, head_dim]  # after GQA expansion
# v_pred:    [batch, q_heads, seq, head_dim]  # after GQA expansion

q_loss = q_shallow.detach()

deep_logits = matmul(q_loss, k_deep.transpose(-1, -2)) / sqrt(head_dim)
pred_logits = matmul(q_loss, k_pred.transpose(-1, -2)) / sqrt(head_dim)

deep_logits = apply_same_aux_mask(deep_logits)
pred_logits = apply_same_aux_mask(pred_logits)

deep_probs = softmax(deep_logits.float(), dim=-1).detach()
pred_log_probs = log_softmax(pred_logits.float(), dim=-1)

loss_route_rows = (
    deep_probs
    * (log(deep_probs.clamp_min(eps)) - pred_log_probs)
).sum(dim=-1)

loss_route = loss_route_rows[valid_query_rows].mean()
```

For Variant G:

```python
pred_probs = pred_log_probs.exp()

deep_msg = matmul(deep_probs, v_deep.float()).detach()
pred_msg = matmul(pred_probs, v_pred.float())

loss_msg = smooth_l1(
    pred_msg[valid_query_rows],
    deep_msg[valid_query_rows],
    beta=1.0,
    reduction="mean",
)
```

The real model forward is unchanged from Arm D. These computations exist only to form the auxiliary losses.

---

# 13. Summary

This version is deliberately narrow:

- **same Arm D forward**,
- same predicted K/V auxiliary branch,
- same shallow/deep locations,
- same training recipe,
- no external teacher or extra pass,
- raw K/V reconstruction removed,
- functional deep routing introduced through softmax + KL,
- optional functional value/content supervision introduced through delivered-message matching.

The two experiments are:

\[
\boxed{
F:\ L_{LM}+0.3L_{route}
}
\]

and:

\[
\boxed{
G:\ L_{LM}+0.3L_{route}+0.3L_{msg}.
}
\]

Nothing else should change between Arm D and these variants except the auxiliary-loss definition.
