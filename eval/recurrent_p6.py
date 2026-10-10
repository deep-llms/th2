"""Evaluation-only P6 recurrence; real targets are published to *future* tokens.

At each proxy layer we retain the token's pre-injection residual. Once that
token has completed the decoder, its four-MLP target replaces the predictor in
the existing injection formula, and only its cached K/V are rebuilt. We never
replay earlier outputs or feed a token its own real target. This is a changed
inference rule, not an oracle upper bound for the trained P6 model.
"""
import torch
from torch.nn import functional as F
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv

MODES = ('native_proxy', 'past_proxy', 'past_real', 'disabled')


def injected_residual(model, layer_number, hidden, prediction):
    head = model.heads[str(layer_number)]
    delta = model.gated_prediction(prediction, head.alpha, hidden)
    with torch.autocast(hidden.device.type, enabled=False):
        scale = (hidden.float().square().mean(-1, keepdim=True) + 1e-6).sqrt()
    return hidden + (delta * scale).to(hidden.dtype)


def project_kv(layer, normalized, rotary):
    a = layer.self_attn
    shape = (*normalized.shape[:-1], -1, a.head_dim)
    k = a.k_norm(a.k_proj(normalized).view(shape)).transpose(1, 2)
    v = a.v_proj(normalized).view(shape).transpose(1, 2)
    _, k = apply_rotary_pos_emb(k, k, *rotary)
    return k, v


@torch.no_grad()
def recurrent_hidden(model, context, mode, *, attention_fn=None, observer=None, progress=None):
    """Return final hidden states for a packed batch without changing model state.

    native_proxy is the unchanged P6 control. past_proxy and past_real both
    disable current-token injection and differ only in what populates past K/V.
    Separate batch rows run in parallel; tokens within each row run sequentially.
    Explicit document masks isolate caches; position IDs reset per document.
    """
    if model.training or model.arm != 'P6-iso' or mode not in MODES or model.gates_disabled:
        raise ValueError('Requires eval-mode P6-iso, enabled gates and a supported mode')
    if context.segments is None or not bool(context.valid.all()):
        raise ValueError('Requires fully packed document-isolated inputs')
    docs = context.segments
    starts = torch.ones_like(docs, dtype=torch.bool)
    starts[:, 1:] = docs[:, 1:] != docs[:, :-1]
    positions = torch.arange(docs.shape[1], device=docs.device).expand_as(docs)
    expected = positions - positions.masked_fill(~starts, 0).cummax(-1).values
    if not torch.equal(context.position_ids, expected):
        raise ValueError('RoPE positions must reset at each document boundary')
    for row, boundaries in zip(docs, starts):
        ids = row[boundaries]
        if ids.unique().numel() != ids.numel():
            raise ValueError('Document IDs must occupy contiguous spans')
    if model.settings.lookahead != 4:
        raise ValueError('This experiment requires the original four-block P6 target')
    attention_fn = attention_fn or F.scaled_dot_product_attention
    backbone = model.backbone.model
    embeddings = backbone.embed_tokens(context.input_ids)
    rotary_all = backbone.rotary_emb(embeddings, context.position_ids)
    caches, outputs = {}, []
    batch, length = context.input_ids.shape
    for t in range(length):
        hidden = embeddings[:, t:t+1]
        rotary = tuple(x[:, t:t+1] for x in rotary_all)
        allowed = (docs[:, :t+1] == docs[:, t:t+1])[:, None, None, :]
        sources, mlps = {}, []
        for index, layer in enumerate(backbone.layers):
            number = index + 1
            u = layer.input_layernorm(hidden)
            if number in model.layers:
                sources[number] = hidden
                if mode == 'native_proxy':
                    prediction = model.heads[str(number)].estimates(u)[0]
                    hidden = injected_residual(model, number, hidden, prediction)
                    u = layer.input_layernorm(hidden)
            a = layer.self_attn
            q = a.q_norm(a.q_proj(u).view(batch, 1, -1, a.head_dim)).transpose(1, 2)
            q, _ = apply_rotary_pos_emb(q, q, *rotary)
            k, v = project_kv(layer, u, rotary)
            if index not in caches:
                shape = (batch, k.shape[1], length, a.head_dim)
                caches[index] = (k.new_empty(shape), v.new_empty(shape))
            keys, values = caches[index]
            keys[:, :, t:t+1].copy_(k)
            values[:, :, t:t+1].copy_(v)
            groups = q.shape[1] // k.shape[1]
            attended = attention_fn(q.to(v.dtype), repeat_kv(keys[:, :, :t+1].to(v.dtype), groups),
                                    repeat_kv(values[:, :, :t+1], groups), attn_mask=allowed,
                                    dropout_p=0., is_causal=False, scale=a.scaling)
            residual = hidden + a.o_proj(attended.transpose(1, 2).reshape(batch, 1, -1))
            mlp = layer.mlp(layer.post_attention_layernorm(residual))
            hidden = residual + mlp
            if mode == 'past_real':
                mlps.append(mlp.float())
        outputs.append(backbone.norm(hidden))
        # Publish only after all decoder layers have consumed this token's
        # native current entry. No current query can read this replacement.
        if mode in ('past_real', 'past_proxy'):
            for number, source in sources.items():
                layer = backbone.layers[number-1]
                if mode == 'past_real':
                    raw = mlps[number-1].clone()
                    for value in mlps[number:number+3]:
                        raw.add_(value)
                    prediction = model.normalize_target(raw, number)
                else:
                    prediction = model.heads[str(number)].estimates(layer.input_layernorm(source))[0]
                replacement = injected_residual(model, number, source, prediction)
                k, v = project_kv(layer, layer.input_layernorm(replacement), rotary)
                keys, values = caches[number-1]
                keys[:, :, t:t+1].copy_(k)
                values[:, :, t:t+1].copy_(v)
                if observer is not None:
                    observer(t, number, source, prediction, k, v, mlps)
        if progress is not None and ((t+1) % 256 == 0 or t+1 == length):
            progress(t+1, length)
    return torch.cat(outputs, dim=1)
