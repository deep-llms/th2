"""P1/P3 replaced KV heads and exact document-reset scans (proxy spec revision 4)."""
from contextlib import contextmanager
import copy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, eager_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .model import DeepKV, normalize_code
from . import PROXY_ARMS


@dataclass
class ProxySettings:
    groups: int = 2
    lookahead: int = 4
    width: int = 256
    features: int = 255
    chunk_size: int = 64
    lambda_max: float = .1
    warmup_steps: int = 250
    target_centering: bool = True
    # Explicitly optional confirmation-run schedule; disabled in screening.
    decay_start: int | None = None
    decay_end: int | None = None


def proxy_layers(config, family, lookahead=4):
    last = config.num_hidden_layers + 1 - lookahead if family == 'P1' else config.num_hidden_layers - 4
    return tuple(range(2, last + 1, 2))


def compute_budget(config, settings, sequence_length=2048):
    """Forward architecture MACs; exclude targets, normalization and auxiliary recomputation."""
    if (type(settings.chunk_size) is not int or settings.chunk_size <= 0
            or type(sequence_length) is not int or sequence_length <= 0
            or sequence_length % settings.chunk_size):
        raise ValueError('Positive EMS chunk_size must divide sequence length')
    d, n, h = config.hidden_size, config.num_hidden_layers, config.head_dim
    nq, nk = config.num_attention_heads, config.num_key_value_heads
    p1n = len(proxy_layers(config, 'P1', settings.lookahead))
    p3n = len(proxy_layers(config, 'P3'))
    p1 = p1n * 2 * d * settings.width
    p3 = p3n * (d * settings.width + settings.width * settings.features + settings.features * d)
    chunks = sequence_length // settings.chunk_size
    # Step 1 dense CxC matmul, step 2 KxK matmul / T, step 3 carry multiply.
    scan = p3n * settings.features * (settings.chunk_size + chunks / settings.chunk_size + 1)
    base = n * (2*d*(nq+nk)*h + 3*d*config.intermediate_size + nq*h*(sequence_length+1)) + d*config.vocab_size
    result = {'baseline_macs_per_token': base, 'attention_cost': 'causal full-sequence upper bound; documents can reduce useful attention work',
              'training_only_auxiliary_matched': False}
    for family, mac, params, gates in (('P1', p1, p1, p1n*d), ('P3', p3+scan, p3, p3n*3*d)):
        delta = int(round(mac/(3*d*n)/8)*8)
        widened = 3*d*n*delta
        result[family] = dict(extra_macs_per_token=mac, extra_parameters=params+gates,
            gate_parameters=gates, widening=delta, widened_extra_parameters=widened,
            mismatch_fraction_of_total=abs(widened-mac)/(base+mac))
    return result


class EMSPlan:
    """Three reusable FP32 coefficient sets; no scan loop over tokens or chunks."""
    def __init__(self, documents, gamma, chunk_size):
        b, t = documents.shape
        if t % chunk_size:
            raise ValueError('EMS chunk_size must divide sequence length')
        self.b, self.t, self.c = b, t, chunk_size
        k = t // chunk_size
        docs = documents.reshape(b, k, chunk_size)
        device = documents.device
        with torch.autocast(device_type=device.type, enabled=False):
            g = gamma.float().log()
            offset = torch.arange(chunk_size, device=device)
            delta = offset[:, None] - offset[None, :]
            local = (delta >= 0)[None, None] & (docs[:, :, :, None] == docs[:, :, None, :])
            self.local = ((1-gamma.float())[:, None, None] * (g[:, None, None]*delta.clamp_min(0)).exp())[None, :, None] * local[:, None]
            ends = docs[:, :, -1]
            chunk = torch.arange(k, device=device)
            distance = chunk[:, None] - chunk[None, :]
            allowed = (distance >= 0)[None] & (ends[:, :, None] == ends[:, None, :])
            self.ends = (g[:, None, None]*(distance.clamp_min(0)*chunk_size)).exp()[None] * allowed[:, None]
            previous = torch.cat((ends[:, :1], ends[:, :-1]), dim=1)
            carry = docs == previous[:, :, None]
            carry[:, 0] = False
            self.carry = (g[:, None]*(offset+1)).exp()[None, :, None] * carry[:, None]

    def apply(self, x, scale):
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float().reshape(self.b, -1, self.c, x.shape[-1])
            local = self.local[:, scale] @ x
            ends = self.ends[:, scale] @ local[:, :, -1]
            previous = torch.cat((torch.zeros_like(ends[:, :1]), ends[:, :-1]), dim=1)
            return (local + self.carry[:, scale, :, :, None] * previous[:, :, None]).reshape(self.b, self.t, -1)


class ProxyHead(nn.Module):
    def __init__(self, config, family, settings):
        super().__init__()
        self.family = family
        self.w1 = nn.Linear(config.hidden_size, settings.width, bias=False)
        self.w2 = nn.Linear(settings.width, config.hidden_size if family == 'P1' else settings.features, bias=False)
        self.projections = nn.ModuleList([nn.Linear(settings.features//3, config.hidden_size, bias=False)
                                          for _ in range(3)] if family == 'P3' else [])
        self.alpha = nn.Parameter(torch.zeros(1 if family == 'P1' else 3, config.hidden_size))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=config.initializer_range)

    def estimates(self, u, plan):
        value = self.w2(F.silu(self.w1(u)))
        if self.family == 'P1':
            return (value,)
        features = value.chunk(3, dim=-1)
        return tuple(projection(plan.apply(features[c], c).to(value.dtype))
                     for c, projection in enumerate(self.projections))

    def inject(self, u, estimates, disabled=False):
        gate = torch.zeros_like(self.alpha) if disabled else self.alpha
        return u + sum(gate[c] * normalize_code(value).to(u.dtype) for c, value in enumerate(estimates))


class ProxyModel(DeepKV):
    """Shared Qwen backbone, no added attention branch. Targets never mutate during checkpoint replay."""
    def __init__(self, backbone, arm, *, proxy_settings=None, channel_mask=None,
                 baseline_attention='sdpa', **kwargs):
        settings = proxy_settings or ProxySettings()
        cfg = backbone.config
        if arm not in ('A',) + PROXY_ARMS:
            raise ValueError('Unknown proxy screen arm')
        if baseline_attention not in ('sdpa', 'fa4') or (baseline_attention == 'fa4' and arm != 'A'):
            raise ValueError('FA4 isolation is supported only for vanilla arm A')
        if cfg.attention_bias or cfg.hidden_act != 'silu' or cfg.num_attention_heads % cfg.num_key_value_heads:
            raise ValueError('Proxy heads require bias-free Qwen SwiGLU and contiguous integer GQA groups')
        if settings.groups not in (1,2,4) or settings.groups >= cfg.num_key_value_heads:
            raise ValueError('Proxy groups must be 1, 2 or 4 and leave native KV groups')
        if settings.features % 3 or min(settings.width, settings.features, settings.chunk_size) <= 0:
            raise ValueError('Invalid proxy width/features/chunk size')
        if settings.lookahead not in (1,2,4,8) or settings.lambda_max < 0 or settings.warmup_steps <= 0:
            raise ValueError('Invalid proxy lookahead/lambda schedule')
        if (settings.decay_start is None) != (settings.decay_end is None) or (settings.decay_start is not None and
                not settings.warmup_steps <= settings.decay_start < settings.decay_end):
            raise ValueError('Invalid optional lambda decay interval')
        if kwargs.get('causal_attention', False):
            raise ValueError('Proxy screen requires dense document isolation')
        super().__init__(backbone, 'A', **kwargs)
        self.arm, self.proxy_screen, self.settings = arm, True, settings
        self.baseline_attention = baseline_attention
        self.attention_runtime = {'backend': baseline_attention}
        if baseline_attention == 'fa4':
            from .fa4 import load_kernel
            self.fa4_kernel, metadata = load_kernel()
            self.attention_runtime.update(metadata)
        self._fa4_observer = None
        self.family = arm[:2] if arm.startswith(('P1','P3')) else None
        self.routing = 'flow' if arm.endswith('flow') else 'block'
        self.layers = proxy_layers(cfg, self.family, settings.lookahead) if self.family else ()
        if self.family and not self.layers:
            raise ValueError('Model has no eligible proxy layers')
        self.heads = nn.ModuleDict()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(kwargs.get('seed',42)+1)
            for layer in self.layers:
                self.heads[str(layer)] = ProxyHead(cfg, self.family, settings)
        self.mean_layers = self.layers if self.family == 'P1' else tuple(range(4,cfg.num_hidden_layers+1)) if self.family else ()
        self.mean_index = {layer:i for i,layer in enumerate(self.mean_layers)}
        self.register_buffer('mu', torch.zeros(len(self.mean_layers), cfg.hidden_size))
        self.register_buffer('mu_initialized', torch.tensor(not bool(self.family and settings.target_centering)))
        self.register_buffer('gamma', torch.tensor([.5,.9,.99], dtype=torch.float32))
        mask = torch.ones(cfg.hidden_size, dtype=torch.bool) if channel_mask is None else torch.as_tensor(channel_mask, dtype=torch.bool)
        if mask.shape != (cfg.hidden_size,) or not bool(mask.any()):
            raise ValueError('Channel mask must retain at least one channel and match hidden size')
        self.register_buffer('channel_mask', mask)
        self.gates_disabled = False

    @classmethod
    def from_scratch(cls, config, arm, *, proxy_settings=None, sequence_length=2048, **kwargs):
        settings = proxy_settings or ProxySettings()
        config = copy.deepcopy(config)
        budget = compute_budget(config, settings, sequence_length)
        if arm in ('V1','V3'):
            config.intermediate_size += budget['P'+arm[1]]['widening']
        return super().from_scratch(config, arm, proxy_settings=settings, **kwargs)

    def auxiliary_weight(self, step):
        if not self.family or self.arm.endswith('lambda0'):
            return 0.
        result = self.settings.lambda_max * min(max(step,0)/self.settings.warmup_steps,1.)
        if self.settings.decay_start is not None and step > self.settings.decay_start:
            result *= max(0.,(self.settings.decay_end-step)/(self.settings.decay_end-self.settings.decay_start))
        return result

    @torch.no_grad()
    def update_mu(self, sums, counts, *, initialize=False):
        if not self.family or not self.settings.target_centering:
            return
        if bool((counts <= 0).any()):
            raise ValueError('Missing target-centering samples')
        mean = sums / counts[:, None]
        if initialize:
            self.mu.copy_(mean)
            self.mu_initialized.fill_(True)
        else:
            self.mu.mul_(.99).add_(mean, alpha=.01)

    def normalize_target(self, value, layer):
        with torch.no_grad(), torch.autocast(device_type=value.device.type, enabled=False):
            centered = value.float() - (self.mu[self.mean_index[layer]] if self.settings.target_centering else 0.)
            masked = centered * self.channel_mask
            return masked * torch.rsqrt(masked.square().sum(-1, keepdim=True)/self.channel_mask.sum() + 1e-6)

    def attention(self, layer, u, z, mask, rotary):
        a = layer.self_attn
        shape = (*u.shape[:-1], -1, a.head_dim)
        q = a.q_norm(a.q_proj(u).view(shape)).transpose(1,2)
        if z is None:
            k, v = a.k_proj(u), a.v_proj(u)
        else:
            split = (self.backbone.config.num_key_value_heads-self.settings.groups)*a.head_dim
            k = torch.cat((F.linear(u,a.k_proj.weight[:split]), F.linear(z,a.k_proj.weight[split:])),dim=-1)
            v = torch.cat((F.linear(u,a.v_proj.weight[:split]), F.linear(z,a.v_proj.weight[split:])),dim=-1)
        k = a.k_norm(k.view(shape)).transpose(1,2)
        v = v.view(shape).transpose(1,2)
        q, k = apply_rotary_pos_emb(q,k,*rotary)
        if self.baseline_attention == 'fa4':
            from .fa4 import attention
            output = attention(q,k,v,mask,a.scaling,self.fa4_kernel)
            if self._fa4_observer is not None:
                self._fa4_observer(q,k,v,mask)
            return output
        interface = ALL_ATTENTION_FUNCTIONS.get_interface('sdpa', eager_attention_forward)
        output, _ = interface(a,q,k,v,mask,dropout=0.,scaling=a.scaling,is_causal=False)
        return output

    def block(self, index, hidden, mask, rotary, plan):
        layer = self.backbone.model.layers[index]
        u = layer.input_layernorm(hidden)
        head = self.heads[str(index+1)] if str(index+1) in self.heads else None
        estimates = head.estimates(u,plan) if head is not None else ()
        z = head.inject(u,estimates,self.gates_disabled) if head is not None else None
        attended = self.attention(layer,u,z,mask,rotary)
        residual = hidden + layer.self_attn.o_proj(attended.reshape(*u.shape[:-1],-1).contiguous())
        mlp = layer.mlp(layer.post_attention_layernorm(residual))
        return residual+mlp, mlp.detach(), u, *estimates

    def layer_cosines(self, layer, u, estimates, raw_target, plan, valid, auxiliary_grad):
        head = self.heads[str(layer)]
        def compute(source, target, *forward_estimates):
            if self.routing == 'block':
                predictions = head.estimates(source.detach(),plan)
            else:
                predictions = forward_estimates
            with torch.autocast(device_type=source.device.type, enabled=False):
                target = target.detach().float()
                if self.family == 'P1':
                    targets = (self.normalize_target(target,layer),)
                else:
                    targets = tuple(plan.apply(target,c).detach() for c in range(3))
                return torch.stack([(F.cosine_similarity(pred.float()*self.channel_mask, truth, dim=-1, eps=1e-6)*valid).sum(-1)
                                    for pred,truth in zip(predictions,targets)],dim=-1)
        with torch.set_grad_enabled(torch.is_grad_enabled() and auxiliary_grad):
            if self.training and self.checkpoint_aux and torch.is_grad_enabled():
                return checkpoint(compute,u,raw_target,*estimates,use_reentrant=False)
            return compute(u,raw_target,*estimates)

    def _run_backbone(self, context, *, compute_auxiliary_losses, auxiliary_grad, collect_target_statistics,
                      block_observer=None):
        if context.segments is None or not bool(context.valid.all()):
            raise ValueError('Proxy screen requires packed, document-isolated inputs')
        if self.baseline_attention == 'fa4':
            from .fa4 import document_layout
            mask = document_layout(context)  # One shared varlen layout; no dense mask.
        else:
            mask = context.allowed()  # Boolean causal + same-document; one shared mask.
        plan = EMSPlan(context.segments,self.gamma,self.settings.chunk_size) if self.family == 'P3' else None
        hidden = self.backbone.model.embed_tokens(context.input_ids)
        rotary = self.backbone.model.rotary_emb(hidden,context.position_ids)
        sources, windows, cosines = {}, {}, {}
        sums = self.mu.new_zeros(self.mu.shape)
        counts = self.mu.new_zeros(len(self.mean_layers))
        need_targets = bool(self.family and (compute_auxiliary_losses or collect_target_statistics))
        def record(layer, raw):
            if collect_target_statistics:
                with torch.no_grad():
                    sums[self.mean_index[layer]] = (raw.float()*context.valid[:,:,None]).sum((0,1))
                    counts[self.mean_index[layer]] = context.valid.sum()
        for index in range(len(self.backbone.model.layers)):
            layer = index+1
            def call(x, i=index):
                return self.block(i,x,mask,rotary,plan)
            result = (checkpoint(call,hidden,use_reentrant=False)
                      if self.training and self.checkpoint_layers and torch.is_grad_enabled() else call(hidden))
            hidden, mlp, u, *estimates = result
            if block_observer is not None:
                block_observer(layer, hidden)
            if layer in self.layers and need_targets and (self.family == 'P1' or compute_auxiliary_losses):
                sources[layer] = (u,tuple(estimates)) if compute_auxiliary_losses else None
            if not need_targets:
                continue
            if self.family == 'P1':
                for proxy in tuple(sources):
                    with torch.no_grad():
                        windows[proxy] = windows[proxy] + mlp.float() if proxy in windows else mlp.float()
                    if layer == proxy+self.settings.lookahead-1:
                        raw = windows.pop(proxy)
                        record(proxy,raw)
                        source = sources.pop(proxy)
                        if compute_auxiliary_losses:
                            cosines[proxy] = self.layer_cosines(proxy,*source,raw,plan,context.valid,auxiliary_grad)
            elif layer in self.mean_index:
                record(layer,hidden.detach())
                if not compute_auxiliary_losses:
                    continue
                normalized = self.normalize_target(hidden.detach(),layer)
                for proxy in tuple(sources):
                    if proxy+2 <= layer <= min(proxy+6,len(self.backbone.model.layers)):
                        with torch.no_grad():
                            windows[proxy] = windows[proxy] + normalized if proxy in windows else normalized
                        if layer == min(proxy+6,len(self.backbone.model.layers)):
                            raw = windows.pop(proxy)/(layer-proxy-1)
                            source = sources.pop(proxy)
                            if compute_auxiliary_losses:
                                cosines[proxy] = self.layer_cosines(proxy,*source,raw,plan,context.valid,auxiliary_grad)
        if sources or windows:
            raise RuntimeError('Incomplete proxy target windows')
        return self.backbone.model.norm(hidden), cosines, sums, counts

    def hidden_states(self, context):
        hidden, _, _, _ = self._run_backbone(context, compute_auxiliary_losses=False,
                                            auxiliary_grad=False, collect_target_statistics=False)
        return hidden, None, None

    def forward(self, context, *, compute_auxiliary_losses=True, auxiliary_grad=True, collect_target_statistics=False):
        hidden, cosines, sums, counts = self._run_backbone(context,
            compute_auxiliary_losses=compute_auxiliary_losses, auxiliary_grad=auxiliary_grad,
            collect_target_statistics=collect_target_statistics)
        lm_rows, target_counts, tokens = self.lm_statistics(context,hidden)
        width = 3 if self.family == 'P3' else 1
        values = (torch.stack([cosines[layer] for layer in self.layers],dim=1).flatten(1)
                  if cosines else lm_rows.new_zeros((len(lm_rows),len(self.layers)*width)))
        auxiliary_counts = tokens if cosines else torch.zeros_like(tokens)
        auxiliary = (tokens-values.mean(-1)).sum() if cosines else lm_rows.sum()*0
        statistics = torch.cat((lm_rows.detach()[:,None],target_counts[:,None],auxiliary_counts[:,None],
                                values.detach(),tokens[:,None]),dim=1).double()
        return dict(lm_sum=lm_rows.sum(),lm_count=target_counts.sum(),aux_sum=auxiliary,
                    aux_count=auxiliary_counts.sum(),statistics=statistics,
                    center_sums=sums,center_counts=counts)

    @contextmanager
    def without_proxy(self):
        previous = self.gates_disabled
        self.gates_disabled = True
        try:
            yield
        finally:
            self.gates_disabled = previous
