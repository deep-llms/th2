"""A2 Q: fixed asymmetric grids and true-sequential OpenSci GPTQ.

GPTQ error propagation follows IST-DASLab/gptq (Apache-2.0), pinned in
resources/a2_quant_gptq_reference_20261031.json; license is alongside it.
Changes: the preregistered grid does not expand its range to include zero;
H is normalized over tokens; native OpenSci forwards and BF16 storage rounding.
No model loading, GPU ownership or network operations live in this module.
"""
import hashlib
import torch

GROUPS = (('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'),
          ('self_attn.o_proj',), ('mlp.gate_proj', 'mlp.up_proj'), ('mlp.down_proj',))


def grid(weight, bits):
    w = weight.float()
    lo = w.amin(-1, keepdim=True)
    hi = w.amax(-1, keepdim=True)
    scale = (hi - lo) / (2 ** bits - 1)
    # Degenerate rows have no specified grid. Preserve their constant value;
    # callers record their count rather than silently divide by zero.
    constant = scale == 0
    safe = torch.where(constant, torch.ones_like(scale), scale)
    return safe, torch.round(-lo / safe), constant, lo


def on_grid(weight, params, bits):
    scale, zero, constant, lo = params
    value = (torch.clamp(torch.round(weight.float() / scale) + zero,
                         0, 2 ** bits - 1) - zero) * scale
    return torch.where(constant, lo, value)


def rtn(weight, bits, group_size=128):
    out = torch.empty_like(weight, dtype=torch.float32)
    for start in range(0, weight.shape[1], group_size):
        part = weight[:, start:start + group_size].float()
        out[:, start:start + group_size] = on_grid(part, grid(part, bits), bits)
    return out


def gptq(weight, hessian, bits, group_size=128, damping=.01, audit=False):
    """Block and group size coincide; group grids see preceding block updates."""
    w = weight.float().clone()
    h = hessian.float().clone()
    dead = h.diagonal() == 0
    h[dead, dead] = 1
    w[:, dead] = 0
    h.diagonal().add_(damping * h.diagonal().mean())
    inv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True)
    result = torch.empty_like(w)
    fixed_grid_idempotent = True
    for start in range(0, w.shape[1], group_size):
        stop = min(start + group_size, w.shape[1])
        block = w[:, start:stop].clone()
        params = grid(w[:, start:stop], bits)
        errors = torch.zeros_like(block)
        for col in range(stop - start):
            q = on_grid(block[:, col:col + 1], params, bits).flatten()
            result[:, start + col] = q
            error = (block[:, col] - q) / inv[start + col, start + col]
            errors[:, col] = error
            block[:, col:] -= error[:, None] * inv[start + col, start + col:stop][None, :]
        w[:, stop:] -= errors @ inv[start:stop, stop:]
        fixed_grid_idempotent = fixed_grid_idempotent and torch.equal(
            result[:, start:stop], on_grid(result[:, start:stop], params, bits))
    return (result, dict(fixed_grid_idempotent=fixed_grid_idempotent)) if audit else result


def matrix_manifest(model):
    return [dict(name=f'model.layers.{i}.{name}.weight', shape=list(module.weight.shape),
                 weights=module.weight.numel())
            for i, layer in enumerate(model.model.layers)
            for name, module in layer.named_modules() if isinstance(module, torch.nn.Linear)]


def weight_digest(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def grid_test(weight, bits):
    """QT3 checks actual grid values before the separately specified BF16 cast."""
    q = rtn(weight, bits)
    result = dequant_test(q, bits)
    result['constant_groups'] = sum(int(grid(weight[:, s:s+128], bits)[2].sum())
                                   for s in range(0, weight.shape[1], 128))
    return result


def dequant_test(q, bits):
    """Check an actual dequantized matrix, including GPTQ's error-corrected grid."""
    repeat = rtn(q, bits)
    cardinality = True
    for start in range(0, q.shape[1], 128):
        ordered = q[:, start:start + 128].sort(dim=1).values
        counts = 1 + (ordered[:, 1:] != ordered[:, :-1]).sum(1)
        cardinality = cardinality and bool((counts <= 2 ** bits).all())
    return dict(cardinality=cardinality, idempotent=torch.equal(q, repeat),
                repeat_max_abs=float((q - repeat).abs().max()))


@torch.no_grad()
def quantize_rtn(model, bits, record):
    for i, layer in enumerate(model.model.layers):
        for name, module in layer.named_modules():
            if isinstance(module, torch.nn.Linear):
                test = grid_test(module.weight, bits)
                record('QT3', test['cardinality'] and test['idempotent'],
                       dict(layer=i, matrix=name, bits=bits, **test))
                module.weight.copy_(rtn(module.weight, bits).bfloat16().float())


class _Captured(Exception):
    pass


def _move(value, device):
    if isinstance(value, torch.Tensor): return value.to(device)
    if isinstance(value, tuple): return tuple(_move(x, device) for x in value)
    if isinstance(value, dict): return {k: _move(v, device) for k, v in value.items()}
    return value


@torch.no_grad()
def capture_inputs(model, rows):
    """Capture native embeddings and exact mask/rotary arguments, without a full forward."""
    inputs = []; kwargs = None
    device = next(model.parameters()).device
    def capture(module, args, kw):
        nonlocal kwargs
        inputs.append(args[0].detach().cpu())
        if kwargs is None: kwargs = _move(kw, 'cpu')
        raise _Captured()
    hook = model.model.layers[0].register_forward_pre_hook(capture, with_kwargs=True)
    try:
        for row in rows:
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                try: model.model(input_ids=row['input_ids'][None].to(device), use_cache=False)
                except _Captured: pass
    finally: hook.remove()
    return inputs, kwargs


@torch.no_grad()
def quantize_gptq(model, rows, bits, record):
    model.eval().requires_grad_(False)
    device = next(model.parameters()).device
    inputs, kw = capture_inputs(model, rows)
    kw = _move(kw, device)
    for i, layer in enumerate(model.model.layers):
        modules = dict(layer.named_modules())
        for names in GROUPS:
            hs = {name: torch.zeros((modules[name].in_features,) * 2,
                                   dtype=torch.float32, device=device) for name in names}
            counts = {name: 0 for name in names}
            hooks = []
            for name in names:
                def accumulate(module, args, output, name=name):
                    x = args[0].detach().reshape(-1, args[0].shape[-1]).float()
                    # Disable autocast here: calibration forwards alone use BF16.
                    with torch.autocast(device.type, enabled=False):
                        hs[name].addmm_(x.T, x)
                    counts[name] += x.shape[0]
                hooks.append(modules[name].register_forward_hook(accumulate))
            try:
                for value in inputs:
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                        layer(value.to(device), **kw)
            finally:
                for hook in hooks: hook.remove()
            for name in names:
                h = hs.pop(name) * (2.0 / counts[name])
                weight = modules[name].weight
                try:
                    q, grid_audit = gptq(weight, h, bits, audit=True)
                    test = dequant_test(q, bits)
                    test.update(grid_audit, repeat_definition='Refit min-max on dequantized weights; fixed-grid identity also reported')
                    record('QT3', test['cardinality'] and test['idempotent'],
                           dict(layer=i, matrix=name, bits=bits, **test))
                    weight.copy_(q.bfloat16().float())
                    record('GPTQ_matrix', True, dict(layer=i, matrix=name, tokens=counts[name], bits=bits))
                except Exception as error:
                    # Preserve maximum registered work while identifying an affected model.
                    weight.copy_(rtn(weight, bits).bfloat16().float())
                    record('GPTQ_matrix', False, dict(layer=i, matrix=name, bits=bits,
                        error=repr(error), fallback='RTN for this matrix; model unverified'))
                del h
        outputs = []
        for value in inputs:
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                outputs.append(layer(value.to(device), **kw)[0].cpu())
        inputs = outputs
        record('GPTQ_layer', True, dict(layer=i, bits=bits))
