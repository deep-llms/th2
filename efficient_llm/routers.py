"""Matched pre-attention linear routers and frozen dense-teacher collection."""
import torch
from torch import nn
from efficient_llm.group_selection import contributions, reconstruction_errors, subset_masks, apply_mask


def subset_costs(c, masks):
    """Joint omitted-vector energy versus the sum of independent omitted energies."""
    joint = reconstruction_errors(c, masks)
    independent = c.float().square().sum(-1) @ (~masks).float().T
    return torch.stack((joint, independent), dim=-2)


class TeacherFeatures:
    """Read normalized inputs and same-position costs without changing outputs.

    Use no_grad, not inference_mode: detached features later train routers.
    """
    def __init__(self, model, keep, stride=16):
        if stride < 1 or model.config.model_type != "qwen3":
            raise ValueError("Require Qwen3 and a positive sampling stride")
        self.model, self.stride = model, stride
        self.groups = model.config.num_key_value_heads
        self.masks = subset_masks(self.groups, keep, device=next(model.parameters()).device)
        self.handles, self.inputs, self.costs = [], {}, {}

    def __enter__(self):
        for i, layer in enumerate(self.model.model.layers):
            def inputs(module, args, output, index=i):
                self.inputs[index] = output[:, self.stride-1::self.stride].detach().reshape(-1, output.shape[-1]).clone()
            def outputs(module, args, index=i):
                x = args[0][:, self.stride-1::self.stride].detach().reshape(-1, args[0].shape[-1])
                with torch.autocast(x.device.type, enabled=False):
                    c = contributions(x, module.weight, self.groups)
                    self.costs[index] = subset_costs(c, self.masks).detach()
            self.handles.extend((layer.input_layernorm.register_forward_hook(inputs),
                                 layer.self_attn.o_proj.register_forward_pre_hook(outputs)))
        return self

    def take(self):
        layers = len(self.model.model.layers)
        if set(self.inputs) != set(range(layers)) or set(self.costs) != set(range(layers)):
            raise RuntimeError("Missing teacher layer/position")
        x = torch.stack([self.inputs[i] for i in range(layers)], dim=1)
        costs = torch.stack([self.costs[i] for i in range(layers)], dim=1)
        self.inputs.clear(); self.costs.clear()
        return x, costs

    def __exit__(self, *args):
        for h in self.handles:
            h.remove()
        self.handles.clear(); self.inputs.clear(); self.costs.clear()


class RouterPair(nn.Module):
    """Independent banks, identical initialization; native HF Trainer updates.

    Sum each bank's mean CE. Disable gradient clipping to avoid coupling banks.
    """
    def __init__(self, layers, hidden, groups, keep, scales, temperature=0.5, seed=42):
        super().__init__()
        if temperature <= 0 or tuple(scales.shape) != (layers,) or not torch.isfinite(scales).all() or not (scales > 0).all():
            raise ValueError("Require positive finite training-only scales and temperature")
        masks = subset_masks(groups, keep)
        generator = torch.Generator().manual_seed(seed)
        initial = torch.randn(layers, hidden, len(masks), generator=generator) * .01
        self.weight = nn.Parameter(initial.unsqueeze(0).repeat(2,1,1,1))
        self.bias = nn.Parameter(torch.zeros(2, layers, len(masks)))
        self.register_buffer("scales", scales.float().clone().contiguous())
        self.register_buffer("masks", masks)
        self.temperature = temperature
        self.spec = dict(layers=layers, hidden=hidden, groups=groups, keep=keep,
                         temperature=temperature, seed=seed)

    def scores(self, features):
        with torch.autocast(features.device.type, enabled=False):
            return torch.einsum("blh,klhs->bkls", features.float(), self.weight) + self.bias

    def targets(self, costs):
        values = costs.float().permute(0,2,1,3)
        values = values - values.min(-1, keepdim=True).values
        return torch.softmax(-values / (self.temperature*self.scales[None,None,:,None]), -1)

    def forward(self, features, costs):
        # Equal-sized feature batches; accumulation=1. There is no LM/token loss.
        logits = self.scores(features)
        targets = self.targets(costs).detach()
        per_bank = -(targets*logits.log_softmax(-1)).sum(-1).mean(dim=(0,2))
        return {"loss": per_bank.sum(), "logits": logits}

    def layer_choice(self, x, layer, bank):
        with torch.autocast(x.device.type, enabled=False):
            return (x.float() @ self.weight[bank, layer] + self.bias[bank, layer]).argmax(-1)


class RouterSelection:
    """Decide BEFORE attention; the O-input hook only applies the saved choice.

    Optional regret observes outputs AFTER selection and never changes it.
    Dense execution reference; all native K/V groups are still computed/cached.
    """
    def __init__(self, model, routers, bank, collect=False, stride=16):
        if bank not in (0,1) or stride < 1:
            raise ValueError("Invalid router bank or stride")
        if len(model.model.layers) != routers.spec["layers"] or model.config.num_key_value_heads != routers.spec["groups"]:
            raise ValueError("Router and backbone shapes differ")
        self.model, self.routers, self.bank = model, routers, bank
        self.collect, self.stride = collect, stride
        self.handles, self.pending, self.stats = [], {}, {}

    def __enter__(self):
        for i, layer in enumerate(self.model.model.layers):
            def choose(module, args, x, index=i):
                if index in self.pending:
                    raise RuntimeError("Unconsumed router decision")
                self.pending[index] = self.routers.layer_choice(x, index, self.bank)
            def apply(module, args, index=i):
                choice = self.pending.pop(index)
                x = args[0]
                if choice.shape != x.shape[:2]:
                    raise ValueError("Router/attention position alignment differs")
                if self.collect:
                    sample = x[:, self.stride-1::self.stride].reshape(-1, x.shape[-1])
                    selected = choice[:, self.stride-1::self.stride].reshape(-1)
                    with torch.autocast(x.device.type, enabled=False):
                        c = contributions(sample, module.weight, self.routers.spec["groups"])
                        errors = reconstruction_errors(c, self.routers.masks)
                        actual = errors.gather(-1, selected[:, None]).squeeze(-1)
                        best, oracle = errors.min(-1)
                        totals = torch.stack((actual.double().sum(), best.double().sum(),
                            c.sum(-2).square().sum(-1).double().sum(), (selected==oracle).double().sum(),
                            actual.new_tensor(len(actual), dtype=torch.float64)))
                    row = self.stats.setdefault(index, {"totals": torch.zeros_like(totals)})
                    row["totals"] += totals
                if self.routers.spec["keep"] == self.routers.spec["groups"]:
                    return None
                return (apply_mask(x, self.routers.masks[choice]),)
            self.handles.extend((layer.input_layernorm.register_forward_hook(choose),
                                 layer.self_attn.o_proj.register_forward_pre_hook(apply)))
        return self

    def __exit__(self, *args):
        for h in self.handles:
            h.remove()
        self.handles.clear(); self.pending.clear()


def save_routers(routers, path):
    torch.save({"spec": routers.spec, "state": {k:v.detach().cpu() for k,v in routers.state_dict().items()}}, path)


def load_routers(path, device="cpu"):
    data = torch.load(path, map_location="cpu", weights_only=True)
    result = RouterPair(**data["spec"], scales=data["state"]["scales"])
    result.load_state_dict(data["state"], strict=True)
    return result.to(device).eval().requires_grad_(False)
