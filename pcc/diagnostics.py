"""Correctness preflight on synthetic tokens, before consuming probe data.

Passing these checks establishes implementation correctness, not scientific gain.
The same checks can run on a tiny random Qwen for CPU development or on the
exact local pretrained snapshot. Neither mode trains the backbone.
"""
import copy
import hashlib

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from .model import Context, CorrectionAdapter, FrozenQwen
from .objectives import alignment_loss, correction_scale, optimizer
from .protocol import HOOKS, PAIRS, SCREEN_SEED


def require(condition, message):
    """Acceptance checks must remain active under python -O."""
    if not bool(condition):
        raise RuntimeError(message)


def reference_visibility(context):
    """Small, independent loop reference used only on preflight contexts."""
    b, t = context.input_ids.shape
    expected = torch.zeros(b, 1, t, t, device=context.input_ids.device, dtype=torch.bool)
    for b in range(context.input_ids.shape[0]):
        for t in range(context.input_ids.shape[1]):
            for j in range(t + 1):
                same_segment = context.segments is None or context.segments[b, t] == context.segments[b, j]
                expected[b, 0, t, j] = context.valid[b, t] and context.valid[b, j] and same_segment
    return expected


def tiny_backbone():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(7)
        config = Qwen3Config(vocab_size=67, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=28, num_attention_heads=2,
                            num_key_value_heads=1, head_dim=128, rope_theta=1_000_000,
                            attention_dropout=0.0, tie_word_embeddings=True)
        config._attn_implementation = "eager"
        return FrozenQwen(Qwen3ForCausalLM(config), strict_config=False)


def fingerprint(model):
    digest = hashlib.sha256()
    for name, p in model.named_parameters():
        digest.update(name.encode())
        digest.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def paired_adapters(backbone, seed=SCREEN_SEED):
    device = next(backbone.model.parameters()).device
    # Initialize on CPU so paired initialization does not depend on GPU count.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        adapter = CorrectionAdapter(backbone.model.config.hidden_size,
                                    backbone.model.config.rms_norm_eps)
    return adapter.to(device), copy.deepcopy(adapter).to(device)


def run_preflight(backbone):
    model = backbone.model
    device = next(model.parameters()).device
    dtype = model.dtype
    bf16 = dtype == torch.bfloat16
    atol, rtol = (0.02, 0.01) if bf16 else (1e-6, 1e-5)
    checks = {}
    before = fingerprint(model)
    ids = (torch.arange(24, device=device).reshape(2, 12) + 3) % model.config.vocab_size
    valid = torch.ones_like(ids, dtype=torch.bool)
    positions = torch.arange(12, device=device).expand_as(ids)
    ordinary = Context(ids, valid, positions)

    def equal(name, actual, expected, exact=False):
        torch.testing.assert_close(actual, expected, atol=0 if exact else atol,
                                   rtol=0 if exact else rtol)
        checks[name] = {"passed": True, "max_abs_error": float((actual.float() - expected.float()).abs().max())}

    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
        clean = backbone.clean(ordinary, HOOKS)
        vanilla = model(input_ids=ids, attention_mask=valid.long(), position_ids=positions,
                        use_cache=False).logits
        equal("vanilla_equivalence", model.lm_head(clean.final_hidden), vanilla)
        next_inputs, handles = {}, []
        def capture(h):
            def hook(module, args):
                next_inputs[h] = args[0].detach()
            return hook
        try:
            for h in HOOKS:
                handles.append(model.model.layers[h].input_layernorm.register_forward_pre_hook(capture(h)))
            backbone.clean(ordinary, HOOKS)
        finally:
            for h in handles:
                h.remove()
        for h in HOOKS:
            equal(f"post_block_hook_{h}", clean.states[h], next_inputs[h], exact=True)
        for s, d in PAIRS:
            teacher, student = paired_adapters(backbone)
            for label, adapter, source in (("deep", teacher, d), ("shallow", student, s)):
                delta = adapter(clean.states[s], clean.states[source], ordinary.allowed(), clean.rotary)
                equal(f"zero_correction_{label}_{s}_{d}", delta, torch.zeros_like(delta), exact=True)
                hidden = backbone.tail(clean, s, delta)
                equal(f"zero_init_logits_{label}_{s}_{d}", model.lm_head(hidden), vanilla, exact=True)

    # Nonzero, nonuniform output weights make causality tests sensitive to
    # wiring mistakes and avoid cancellations from identical projection rows.
    teacher, student = paired_adapters(backbone)
    with torch.no_grad():
        generator = torch.Generator(device="cpu").manual_seed(81)
        teacher.out.weight.copy_(torch.randn(teacher.out.weight.shape, generator=generator) * 0.02)
        student.out.weight.copy_(teacher.out.weight)
    valid = valid.clone()
    valid[1, -2:] = False
    segments = torch.zeros_like(ids)
    segments[:, 6:] = 1
    positions = positions.clone()
    positions[:, 6:] -= 6
    context = Context(ids, valid, positions, segments)
    expected_visibility = reference_visibility(context)
    equal("independent_context_mask", context.allowed().float(), expected_visibility.float(), exact=True)
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
        clean = backbone.clean(context, HOOKS)
        changed_ids = ids.clone()
        changed_ids[:, 5:] = (changed_ids[:, 5:] + 1) % model.config.vocab_size
        changed = backbone.clean(Context(changed_ids, valid, positions, segments), HOOKS)
        # Change only an earlier isolated segment. Later-segment states and
        # corrections must stay identical, even though the changed tokens are past.
        isolated_ids = ids.clone()
        isolated_ids[:, :6] = (isolated_ids[:, :6] + 7) % model.config.vocab_size
        isolated = backbone.clean(Context(isolated_ids, valid, positions, segments), HOOKS)
        padding_ids = ids.clone()
        padding_ids[~valid] = (padding_ids[~valid] + 9) % model.config.vocab_size
        padded = backbone.clean(Context(padding_ids, valid, positions, segments), HOOKS)
        for h in HOOKS:
            equal(f"isolated_segment_hook_{h}", clean.states[h][:, 6:], isolated.states[h][:, 6:], exact=True)
            equal(f"padding_invariance_hook_{h}", clean.states[h][valid], padded.states[h][valid], exact=True)
        for name, adapter, source in (("teacher", teacher, 20), ("student", student, 4)):
            delta, diag = adapter(clean.states[4], clean.states[source], context.allowed(), clean.rotary, diagnostics=True)
            other = adapter(changed.states[4], changed.states[source], context.allowed(), changed.rotary)
            equal(f"future_perturbation_{name}", delta[:, :5], other[:, :5], exact=True)
            other_segment = adapter(isolated.states[4], isolated.states[source], context.allowed(), isolated.rotary)
            equal(f"isolated_segment_correction_{name}", delta[:, 6:], other_segment[:, 6:], exact=True)
            mask = expected_visibility & torch.ones(12, 12, device=device, dtype=torch.bool).tril(-1)
            require(torch.count_nonzero(diag["attention"].masked_select(~mask.expand_as(diag["attention"]))) == 0,
                    f"{name}: forbidden attention mass")
            empty = ~mask[:, 0].any(-1)
            require(torch.count_nonzero(delta[empty]) == 0, f"{name}: nonzero empty-source correction")
            require(torch.count_nonzero(diag["message"][empty]) == 0, f"{name}: nonzero empty-source message")
            require(torch.isfinite(delta).all() and torch.isfinite(diag["attention"]).all(), f"{name}: nonfinite attention")
            require(torch.count_nonzero(delta[~empty]) > 0, "Causality check must use a nonzero branch")
            checks[f"strict_past_context_and_empty_rows_{name}"] = {"passed": True}

    teacher.requires_grad_(False)
    opt = optimizer(student)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
        target = teacher(clean.states[4], clean.states[20], context.allowed(), clean.rotary).detach()
        prediction = student(clean.states[4], clean.states[4], context.allowed(), clean.rotary)
        scale = correction_scale(target, context.targets())
        hidden = backbone.tail(clean, 4, prediction, checkpoint_layers=True)
        sums, counts = backbone.losses(hidden, context, chunk_size=5)
        lm_loss = sums.sum() / counts.sum()
    require(lm_loss.requires_grad, "LM loss is detached from the adapter through the frozen tail")
    lm_loss.backward()
    require(any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                for p in student.parameters()), "LM-only adapter gradients are missing/nonfinite")
    require(all(not p.requires_grad and p.grad is None for p in model.parameters()), "Backbone is not frozen")
    require(all(p.grad is None for p in teacher.parameters()), "Teacher received student gradients")
    require(all(not state.requires_grad for state in clean.states.values()), "Clean sources have gradients")
    opt.step()
    require(before == fingerprint(model), "Backbone changed after adapter update")
    require(all(state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.float32 for state in opt.state.values()),
            "AdamW moments must be fp32")
    # Test the alignment boundary separately with a deliberately differentiable
    # target, rather than relying on an already-detached/frozen teacher.
    differentiable_target = target.detach().requires_grad_()
    opt.zero_grad(set_to_none=True)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
        prediction = student(clean.states[4], clean.states[4], context.allowed(), clean.rotary)
        alignment_loss(prediction, differentiable_target, context.targets(), scale).backward()
    require(differentiable_target.grad is None, "Alignment loss differentiated its teacher target")
    require(any(p.grad is not None and p.grad.abs().sum() > 0 for p in student.parameters()), "Alignment gradients missing")
    checks["frozen_backbone_and_lm_only_tail_gradients"] = {"passed": True}
    checks["alignment_teacher_detach"] = {"passed": True}
    checks["fp32_optimizer_moments"] = {"passed": True}
    return {"status": "ok", "stage": "correctness_preflight", "checks": checks,
            "atol": atol, "rtol": rtol, "zero_init_tolerance": 0,
            "backbone_sha256_before_and_after": before,
            "scientific_gain_established": False, "test_unlocked": False}
