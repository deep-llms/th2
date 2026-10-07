"""Qwen proxy arms with document isolation and detached, standardized targets."""
from contextlib import contextmanager
import copy
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, eager_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .model import DeepKV, normalize_code
from . import ALL_PROXY_ARMS as PROXY_ARMS, ANTICIPATORY_ARMS, MEMORY_ARMS, P6_VARIANTS, anticipatory_layout


@dataclass
class ProxySettings:
    groups: int = 2
    lookahead: int | None = None
    width: int = 256
    features: int = 255
    chunk_size: int = 64
    lambda_max: float = .1
    warmup_steps: int = 250
    alpha_init: float | None = None
    kv_mode: str = 'kv'
    layers: list[int] | None = None
    isolate_estimator: bool | None = None
    module_seed: int | None = None
    aux_recompute: bool = False
    compile_estimator: bool = False
    target_version: str | None = None
    variance_floor: float = .01
    target_clip: float = 10.
    momentum: float = .99
    loss_form: str = 'cosine'
    # Explicitly optional confirmation-run schedule; disabled in screening.
    decay_start: int | None = None
    decay_end: int | None = None


def resolve_proxy_settings(arm, settings=None):
    settings = copy.deepcopy(settings or ProxySettings())
    new = arm in ANTICIPATORY_ARMS
    memory = arm in MEMORY_ARMS
    expected_lookahead = 2 if arm == 'P6-iso-short' else 4
    if settings.lookahead is None:settings.lookahead = expected_lookahead
    if memory:
        version = 'p4p6-r1' if arm=='P7-simple' else 'p7-r1'
        if settings.target_version is None:settings.target_version = version
        if settings.isolate_estimator is None:settings.isolate_estimator = True
        if settings.alpha_init is None:settings.alpha_init = 0.  # No gate parameter in P7.
        if (settings.target_version != version or settings.lookahead != 4 or settings.groups != 2
                or settings.loss_form != 'cosine' or settings.kv_mode != 'kv'
                or settings.layers is not None or not settings.isolate_estimator
                or settings.alpha_init != 0 or settings.aux_recompute or settings.compile_estimator):
            raise ValueError('P7 requires its specified target, two groups, isolated estimator and no gate')
        if settings.module_seed is None:settings.module_seed = 43
    if settings.alpha_init is None:settings.alpha_init = (.1 if arm.startswith('P6') else 1.) if new else 0.
    if settings.target_version is None:settings.target_version = 'p4p6-r1' if new else 'r7'
    isolation = arm in ('P4-iso','P5','P6-iso','P4-iso-4h') + P6_VARIANTS
    if settings.isolate_estimator is None:settings.isolate_estimator = isolation
    if new:
        if (settings.target_version != 'p4p6-r1' or settings.lookahead != expected_lookahead
                or settings.loss_form != 'cosine' or settings.kv_mode != 'kv'
                or settings.layers is not None or settings.isolate_estimator != isolation
                or settings.alpha_init != (.1 if arm.startswith('P6') else 1.)):
            raise ValueError('P4/P5/P6 require their specified target, placement, isolation and gate initialization')
        if settings.module_seed is None:settings.module_seed = 43
    elif not memory and (settings.isolate_estimator or settings.aux_recompute or settings.compile_estimator):
        raise ValueError('Estimator isolation/routing/compilation options require P4/P5/P6')
    if settings.module_seed is not None and (type(settings.module_seed) is not int or settings.module_seed < 0):
        raise ValueError('module_seed must be a nonnegative integer')
    return settings


def proxy_layers(config, family, lookahead=4, layers=None):
    last = config.num_hidden_layers + 1 - lookahead if family == 'P1' else config.num_hidden_layers - 4
    if layers is not None:
        if (family != 'P1' or not layers or any(type(i) is not int or i < 2 or i > last or i % 2 for i in layers)
                or list(layers) != sorted(set(layers))):
            raise ValueError('Explicit proxy layers must be increasing unique eligible even P1 blocks')
        return tuple(layers)
    return tuple(range(2, last + 1, 2))


def compute_budget(config, settings, sequence_length=2048):
    """Forward architecture MACs; exclude targets, normalization and auxiliary recomputation."""
    if (type(settings.chunk_size) is not int or settings.chunk_size <= 0
            or type(sequence_length) is not int or sequence_length <= 0
            or sequence_length % settings.chunk_size):
        raise ValueError('Positive EMS chunk_size must divide sequence length')
    d, n, h = config.hidden_size, config.num_hidden_layers, config.head_dim
    nq, nk = config.num_attention_heads, config.num_key_value_heads
    p1n = len(proxy_layers(config, 'P1', settings.lookahead if settings.lookahead is not None else 4, settings.layers))
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
    # Fixed original placement, independent of the legacy lookahead ablations.
    layers = len(proxy_layers(config,'P1',4))
    plain = layers*2*d*settings.width
    stream = plain+max(0,layers-1)*3*settings.width**2
    for family,mac in [('P4',plain),('P5',stream),('P6',plain)]:
        result[family] = dict(extra_macs_per_token=mac,extra_parameters=mac+layers*d,gate_parameters=layers*d)
    for arm in P6_VARIANTS:
        count = len(anticipatory_layout(arm, n)[0])
        mac = count*2*d*settings.width
        result[arm] = dict(extra_macs_per_token=mac,extra_parameters=mac+count*d,gate_parameters=count*d)
    estimator = 2*d*settings.width+4*settings.width
    projections = 4*h*d
    extra_attention = (2*nq//nk)*h*(sequence_length+1)
    result['P7'] = dict(extra_parameters=layers*(estimator+projections+h),
        extra_macs_per_token=layers*(estimator+projections+extra_attention),
        fa4_extra_macs_per_token=layers*(estimator+projections+(2*nq//nk)*h*(3*sequence_length+1)),
        # Both target and prediction similarity matmuls; excludes backward/softmax.
        relational_forward_macs_per_token=layers*2*min(256,sequence_length-1)*d,
        fa4_note='Duplicated queries add selected-head work; measure actual training throughput')
    result['P7-kq'] = {**result['P7'],
        'extra_parameters':result['P7']['extra_parameters']-layers*2*h*d,
        'extra_macs_per_token':result['P7']['extra_macs_per_token']-layers*2*h*d,
        'fa4_extra_macs_per_token':result['P7']['fa4_extra_macs_per_token']-layers*2*h*d}
    result['P7-mlp'] = dict(result['P7'])
    result['P7-simple'] = {**result['P7'], **{key:result['P7'][key]-layers*4*settings.width
        for key in ('extra_parameters','extra_macs_per_token','fa4_extra_macs_per_token')},
        'relational_forward_macs_per_token':0}
    result['P7-ems'] = {**result['P7'], 'extra_macs_per_token':result['P7']['extra_macs_per_token']+
        layers*2*settings.width*(settings.chunk_size+chunks/settings.chunk_size+1),
        'fa4_extra_macs_per_token':result['P7']['fa4_extra_macs_per_token']+
        layers*2*settings.width*(settings.chunk_size+chunks/settings.chunk_size+1)}
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
        self.alpha = nn.Parameter(torch.full((1 if family == 'P1' else 3, config.hidden_size),
                                            float(settings.alpha_init)))
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
    """Shared Qwen backbone. Targets never mutate during checkpoint replay."""
    def __init__(self, backbone, arm, *, proxy_settings=None, channel_mask=None,
                 attention_backend='sdpa', **kwargs):
        settings = resolve_proxy_settings(arm,proxy_settings)
        new = arm in ANTICIPATORY_ARMS
        memory = arm in MEMORY_ARMS
        cfg = backbone.config
        if channel_mask is not None:
            raise ValueError('Revision r7 does not use a static channel mask')
        if (settings.target_version != ('p4p6-r1' if new or arm=='P7-simple' else 'p7-r1' if memory else 'r7') or settings.loss_form not in ('cosine','smooth_l1')
                or not 0 < settings.variance_floor <= 1 or not 0 < settings.target_clip < float('inf')
                or not 0 <= settings.momentum < 1):
            raise ValueError('Invalid r7 target normalization/loss settings')
        if arm not in ('A',) + PROXY_ARMS:
            raise ValueError('Unknown proxy screen arm')
        if not math.isfinite(settings.alpha_init) or (settings.alpha_init != 0 and not arm.startswith(('P1','P3')) and not new):
            raise ValueError('Finite alpha_init requires a proxy arm when nonzero')
        if settings.kv_mode not in ('kv', 'v') or (settings.kv_mode != 'kv' and not arm.startswith('P1')):
            raise ValueError('proxy_kv_mode must be kv, or v for P1 arms')
        if settings.layers is not None and not arm.startswith('P1'):
            raise ValueError('Explicit proxy layers require a P1 arm')
        if attention_backend not in ('sdpa', 'fa4'):
            raise ValueError('Unknown document-isolated attention backend')
        if cfg.attention_bias or cfg.attention_dropout != 0 or cfg.hidden_act != 'silu' or cfg.num_attention_heads % cfg.num_key_value_heads:
            raise ValueError('Proxy heads require bias-free, zero-dropout Qwen SwiGLU and contiguous integer GQA groups')
        if not new and (settings.groups not in (1,2,4) or settings.groups >= cfg.num_key_value_heads):
            raise ValueError('Proxy groups must be 1, 2 or 4 and leave native KV groups')
        if settings.features % 3 or min(settings.width, settings.features, settings.chunk_size) <= 0:
            raise ValueError('Invalid proxy width/features/chunk size')
        if (type(settings.lookahead) is not int or settings.lookahead not in (1,2,3,4,8)
                or not math.isfinite(settings.lambda_max) or settings.lambda_max < 0 or settings.warmup_steps <= 0):
            raise ValueError('Invalid proxy lookahead/lambda schedule')
        if (settings.decay_start is None) != (settings.decay_end is None) or (settings.decay_start is not None and
                not settings.warmup_steps <= settings.decay_start < settings.decay_end):
            raise ValueError('Invalid optional lambda decay interval')
        if kwargs.get('causal_attention', False):
            raise ValueError('Proxy screen requires document isolation')
        super().__init__(backbone, 'A', **kwargs)
        self.arm, self.proxy_screen, self.settings = arm, True, settings
        self.attention_backend = attention_backend
        self.attention_runtime = {'backend': attention_backend}
        if attention_backend == 'fa4':
            from .fa4 import load_kernel
            self.fa4_kernel, metadata = load_kernel()
            self.attention_runtime.update(metadata)
        self._fa4_observer = None
        self.anticipatory = new
        self.memory_proxy = memory
        self.relational_proxy = memory and arm != 'P7-simple'
        self.increment_target = memory and arm not in ('P7-mlp','P7-simple')
        if arm=='P7-simple' and cfg.num_attention_heads//cfg.num_key_value_heads != 2:
            raise ValueError('P7-simple requires four query heads in two complete KV groups')
        self._proxy_mass = None
        self.value_groups = cfg.num_key_value_heads
        if arm in ('P4-4h','P4-iso-4h'):
            queries_per_kv = cfg.num_attention_heads // cfg.num_key_value_heads
            if cfg.num_attention_heads < 4 or 4 % queries_per_kv:
                raise ValueError('Four proxy query heads must fit complete GQA groups')
            self.value_groups = 4 // queries_per_kv
        self.family = arm[:2] if new or memory or arm.startswith(('P1','P3')) else None
        self.routing = 'flow' if arm.endswith('flow') else 'block'
        self.layers = proxy_layers(cfg, 'P1' if new or memory else self.family, settings.lookahead, settings.layers) if self.family else ()
        if new:self.layers = anticipatory_layout(arm, cfg.num_hidden_layers)[0]
        if self.family and not self.layers:
            raise ValueError('Model has no eligible proxy layers')
        self.heads = nn.ModuleDict()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(settings.module_seed if settings.module_seed is not None else kwargs.get('seed',42)+1)
            # Preserve the parent's initial weights at retained locations. The
            # discarded heads exist only during sparse-arm initialization.
            initialization_layers = anticipatory_layout('P6-iso', cfg.num_hidden_layers)[0] if arm=='P6-iso-sparse' else self.layers
            for layer in initialization_layers:
                if arm=='P7-simple':
                    from .proxy_memory import SimpleMemoryHead
                    self.heads[str(layer)] = SimpleMemoryHead(cfg,settings)
                elif memory:
                    from .proxy_memory import MemoryHead
                    self.heads[str(layer)] = MemoryHead(cfg,settings,native_values=arm=='P7-kq')
                elif new:
                    from .proxy_estimators import AnticipatoryHead
                    head = AnticipatoryHead(cfg,settings,stream=self.family=='P5',first=layer==self.layers[0])
                    if layer in self.layers:self.heads[str(layer)] = head
                else:self.heads[str(layer)] = ProxyHead(cfg, self.family, settings)
            if arm=='P7-simple':
                for head in self.heads.values():head.initialize_projections(cfg)
        if new or arm=='P7-simple':
            from .proxy_estimators import gated_prediction, cosine_loss
            self.gated_prediction,self.cosine_loss = gated_prediction,cosine_loss
            if settings.compile_estimator:
                # Compile functions, not modules: state_dict names stay identical.
                self.gated_prediction=torch.compile(gated_prediction)
                self.cosine_loss=torch.compile(cosine_loss)
                for head in self.heads.values():
                    if self.family=='P5':head.stream_step=torch.compile(head.stream_step)
                    else:head.routed=torch.compile(head.routed)
        self.mean_layers = (self.layers if self.family and self.family != 'P3' else
            tuple((layer, deep) for layer in self.layers
                  for deep in range(layer+2,min(layer+6,cfg.num_hidden_layers)+1)))
        self.mean_index = {key:i for i,key in enumerate(self.mean_layers)}
        self.register_buffer('mu', torch.zeros(len(self.mean_layers), cfg.hidden_size))
        self.register_buffer('mu_initialized', torch.tensor(not bool(self.family)))
        if self.family:
            self.register_buffer('sigma2', torch.ones_like(self.mu))
        else:
            # Preserve the state layout of completed A/V checkpoints; unused by r7 targets.
            self.register_buffer('channel_mask', torch.ones(cfg.hidden_size,dtype=torch.bool))
        self.register_buffer('gamma', torch.tensor([.5,.9,.99], dtype=torch.float32))
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
    def update_statistics(self, sums, squares, counts, *, initialize=None):
        """Sums are raw only in initialization pass 1; otherwise shifted by frozen mu."""
        if not self.family:
            return {}
        if bool((counts <= 0).any()) or not all(bool(torch.isfinite(x).all()) for x in (sums,squares,counts)):
            raise ValueError('Invalid target-normalization samples')
        delta = sums / counts[:,None]
        if initialize == 'mean':
            self.mu.copy_(delta)
            return {}
        if initialize == 'variance':
            self.sigma2.copy_((squares/counts[:,None]).clamp_min(0))
            self.mu_initialized.fill_(True)
            return {}
        if initialize is not None:
            raise ValueError('Unknown normalization initialization pass')
        variance = (squares/counts[:,None]-delta.square()).clamp_min(0)
        # Diagnostics describe pre-update statistics, the ones used by this step.
        # Diagnostic ratios retain every positive variance; guard only exact zeros.
        # FP64 here avoids overflow when a dead channel becomes active.
        scale = self.sigma2.double().clamp_min(torch.finfo(self.sigma2.dtype).tiny)
        diagnostics = dict(mean_lag=(delta.double().abs()/scale.sqrt()).median(-1).values,
                           variance_lag=(variance.double()/scale).median(-1).values,
                           median_variance=self.sigma2.median(-1).values,
                           floored_channels=(self.sigma2 < self.settings.variance_floor*
                               self.sigma2.median(-1,keepdim=True).values).sum(-1))
        momentum = self.settings.momentum
        self.mu.add_(delta, alpha=1-momentum)
        self.sigma2.mul_(momentum).add_(variance,alpha=1-momentum)
        return diagnostics

    def normalize_target(self, value, key, *, return_clipped=False):
        with torch.no_grad(), torch.autocast(device_type=value.device.type, enabled=False):
            index = self.mean_index[key]
            variance = self.sigma2[index]
            floor = self.settings.variance_floor*variance.median()
            standardized = (value.detach().float()-self.mu[index])*torch.rsqrt(variance.clamp_min(floor)+1e-6)
            normalized = standardized.clamp(-self.settings.target_clip,self.settings.target_clip)
            if return_clipped:
                return normalized, (standardized.abs()>self.settings.target_clip).sum()
            return normalized

    def attention(self, layer, u, z, mask, rotary):
        a = layer.self_attn
        shape = (*u.shape[:-1], -1, a.head_dim)
        q = a.q_norm(a.q_proj(u).view(shape)).transpose(1,2)
        if z is None:
            k, v = a.k_proj(u), a.v_proj(u)
        elif self.anticipatory:
            k = a.k_proj(u)
            split = (self.backbone.config.num_key_value_heads-self.value_groups)*a.head_dim
            v = (torch.cat((F.linear(u,a.v_proj.weight[:split]),F.linear(z,a.v_proj.weight[split:])),dim=-1)
                 if split else a.v_proj(z))
        else:
            split = (self.backbone.config.num_key_value_heads-self.settings.groups)*a.head_dim
            k = (a.k_proj(u) if self.settings.kv_mode == 'v' else
                 torch.cat((F.linear(u,a.k_proj.weight[:split]), F.linear(z,a.k_proj.weight[split:])),dim=-1))
            v = torch.cat((F.linear(u,a.v_proj.weight[:split]), F.linear(z,a.v_proj.weight[split:])),dim=-1)
        k = a.k_norm(k.view(shape)).transpose(1,2)
        v = v.view(shape).transpose(1,2)
        q, k = apply_rotary_pos_emb(q,k,*rotary)
        if self.attention_backend == 'fa4':
            from .fa4 import attention
            output = attention(q,k,v,mask,a.scaling,self.fa4_kernel)
            if self._fa4_observer is not None:
                self._fa4_observer(q,k,v,mask)
            return output
        interface = ALL_ATTENTION_FUNCTIONS.get_interface('sdpa', eager_attention_forward)
        output, _ = interface(a,q,k,v,mask,dropout=0.,scaling=a.scaling,is_causal=False)
        return output

    def memory_attention(self, layer, u, head, prediction, mask, rotary, plan):
        from .proxy_memory import rectangular_mask, flash_joint_attention, proxy_mass
        if self.gates_disabled:
            # Equivalent to masking the entire proxy half; retain the original
            # native kernel shape for exact baseline comparison.
            return self.attention(layer,u,None,mask,rotary)
        a = layer.self_attn
        shape = (*u.shape[:-1],-1,a.head_dim)
        q = a.q_norm(a.q_proj(u).view(shape)).transpose(1,2)
        k = a.k_norm(a.k_proj(u).view(shape)).transpose(1,2)
        v = a.v_proj(u).view(shape).transpose(1,2)
        q,k = apply_rotary_pos_emb(q,k,*rotary)
        groups = self.backbone.config.num_key_value_heads
        split = (groups-2)*self.backbone.config.num_attention_heads//groups
        kp,vp = head.entries(prediction,rotary,v[:,-2:])
        if self._proxy_mass is not None:
            name = str(a.layer_idx+1)
            self._proxy_mass[name] = proxy_mass(q[:,split:],k[:,-2:],kp,plan.documents,a.scaling)
        if self.attention_backend == 'fa4':
            from .fa4 import attention
            qn,kn,vn = q[:,:split],k[:,:-2],v[:,:-2]
            native = attention(qn,kn,vn,mask,a.scaling,self.fa4_kernel)
            if self._fa4_observer is not None:
                self._fa4_observer(qn,kn,vn,mask)
            proxy = flash_joint_attention(q[:,split:],k[:,-2:],v[:,-2:],kp,vp,mask,
                                          a.scaling,self.fa4_kernel,self._fa4_observer)
        else:
            interface = ALL_ATTENTION_FUNCTIONS.get_interface('sdpa',eager_attention_forward)
            if not hasattr(plan,'joint_mask'):
                plan.joint_mask = rectangular_mask(mask)  # Shared by all P7 blocks and replay.
            native,_ = interface(a,q[:,:split],k[:,:-2],v[:,:-2],mask,
                                  dropout=0.,scaling=a.scaling,is_causal=False)
            proxy,_ = interface(a,q[:,split:],torch.cat((k[:,-2:],kp),dim=2),
                                 torch.cat((v[:,-2:],vp),dim=2),plan.joint_mask,
                                 dropout=0.,scaling=a.scaling,is_causal=False)
        return torch.cat((native,proxy),dim=2)

    @torch.no_grad()
    def proxy_attention_mass(self, context):
        if not self.memory_proxy or self.gates_disabled or self._proxy_mass is not None:
            raise ValueError('Attention mass requires an enabled P7 model')
        self._proxy_mass = {}
        try:
            self.hidden_states(context)
            return torch.stack([self._proxy_mass[str(layer)] for layer in self.layers])
        finally:
            self._proxy_mass = None

    def block(self, index, hidden, mask, rotary, plan, stream=None):
        layer = self.backbone.model.layers[index]
        u = layer.input_layernorm(hidden)
        head = self.heads[str(index+1)] if str(index+1) in self.heads else None
        if self.memory_proxy and head is not None:
            prediction = head.estimate(u,plan)
            attended = self.memory_attention(layer,u,head,prediction,mask,rotary,plan)
            residual = hidden+layer.self_attn.o_proj(attended.reshape(*u.shape[:-1],-1).contiguous())
            mlp = layer.mlp(layer.post_attention_layernorm(residual))
            return residual+mlp,mlp.detach(),u,prediction
        if self.anticipatory and head is not None:
            if self.family=='P5':
                aux,stream=head.stream_step(u,stream)
                injected=aux.detach()
            else:injected,aux=head.routed(u,self.settings)
            gate=torch.zeros_like(head.alpha) if self.gates_disabled else head.alpha
            delta=self.gated_prediction(injected,gate,u)
            if self.family=='P6':
                with torch.autocast(hidden.device.type,enabled=False):
                    scale=(hidden.detach().float().square().mean(-1,keepdim=True)+1e-6).sqrt()
                hidden=hidden+(delta*scale).to(hidden.dtype)
                actual_u=layer.input_layernorm(hidden)
                attended=self.attention(layer,actual_u,None,mask,rotary)
            else:attended=self.attention(layer,u,u+delta,mask,rotary)
            residual=hidden+layer.self_attn.o_proj(attended.reshape(*u.shape[:-1],-1).contiguous())
            mlp=layer.mlp(layer.post_attention_layernorm(residual))
            return residual+mlp,mlp.detach(),u,aux,stream
        estimates = head.estimates(u,plan) if head is not None else ()
        z = head.inject(u,estimates,self.gates_disabled) if head is not None else None
        attended = self.attention(layer,u,z,mask,rotary)
        residual = hidden + layer.self_attn.o_proj(attended.reshape(*u.shape[:-1],-1).contiguous())
        mlp = layer.mlp(layer.post_attention_layernorm(residual))
        return (residual+mlp,mlp.detach(),u,stream) if self.anticipatory else (residual+mlp, mlp.detach(), u, *estimates)

    def layer_cosines(self, layer, u, estimates, raw_target, plan, valid, auxiliary_grad):
        head = self.heads[str(layer)]
        def compute(source, target, *forward_estimates):
            if self.relational_proxy:
                from .proxy_memory import relational_losses
                prediction = forward_estimates[0]
                with torch.autocast(source.device.type,enabled=False):
                    cosines = (F.cosine_similarity(prediction.float(),target.detach().float(),dim=-1,eps=1e-6)*valid).sum(-1)
                    rel_sum, rel_count = relational_losses(prediction,target,plan)
                    return cosines[:,None],(valid.sum(-1)-cosines)[:,None],rel_sum,rel_count
            elif self.anticipatory or self.memory_proxy:
                return self.cosine_loss(forward_estimates[0],target,valid)
            elif self.routing == 'block':
                predictions = head.estimates(source.detach(),plan)
            else:
                predictions = forward_estimates
            with torch.autocast(device_type=source.device.type, enabled=False):
                target = target.detach().float()
                if self.family != 'P3':
                    targets = (target,)
                else:
                    targets = tuple(plan.apply(target,c).detach() for c in range(3))
                cosines = torch.stack([(F.cosine_similarity(pred.float(),truth,dim=-1,eps=1e-6)*valid).sum(-1)
                                       for pred,truth in zip(predictions,targets)],dim=-1)
                losses = (valid.sum(-1)[:,None]-cosines if self.settings.loss_form == 'cosine' else
                    torch.stack([(F.smooth_l1_loss(pred.float(),truth,reduction='none',beta=1.).mean(-1)*valid).sum(-1)
                                 for pred,truth in zip(predictions,targets)],dim=-1))
                return cosines, losses
        with torch.set_grad_enabled(torch.is_grad_enabled() and auxiliary_grad):
            if self.training and self.checkpoint_aux and torch.is_grad_enabled():
                return checkpoint(compute,u,raw_target,*estimates,use_reentrant=False)
            return compute(u,raw_target,*estimates)

    def _run_backbone(self, context, *, compute_auxiliary_losses, auxiliary_grad, collect_target_statistics,
                      block_observer=None, statistics_mode=None, relational_step=0, sequence_indices=None):
        if context.segments is None or not bool(context.valid.all()):
            raise ValueError('Proxy screen requires packed, document-isolated inputs')
        if self.attention_backend == 'fa4':
            from .fa4 import document_layout
            mask = document_layout(context)  # One shared varlen layout; no dense mask.
        else:
            mask = context.allowed()  # Boolean causal + same-document; one shared mask.
        plan = EMSPlan(context.segments,self.gamma,self.settings.chunk_size) if self.family == 'P3' else None
        if self.memory_proxy:
            from .proxy_memory import MemoryPlan
            ems = EMSPlan(context.segments,self.gamma,self.settings.chunk_size) if self.arm=='P7-ems' else None
            plan = MemoryPlan(context.segments,step=relational_step,sequence_indices=sequence_indices,ems=ems,
                              sample_queries=compute_auxiliary_losses and self.relational_proxy,
                              convolution=self.relational_proxy)
        hidden = self.backbone.model.embed_tokens(context.input_ids)
        rotary = self.backbone.model.rotary_emb(hidden,context.position_ids)
        sources, windows, cosines, bases = {}, {}, {}, {}
        stream,aux_stream=None,None
        sums = self.mu.new_zeros(self.mu.shape)
        squares = torch.zeros_like(sums)
        counts = self.mu.new_zeros(len(self.mean_layers))
        clipped = torch.zeros_like(counts)
        need_targets = bool(self.family and (compute_auxiliary_losses or collect_target_statistics))
        def record(key, raw):
            with torch.no_grad(), torch.autocast(device_type=raw.device.type,enabled=False):
                index = self.mean_index[key]
                if collect_target_statistics:
                    shifted = raw.detach().float() if statistics_mode == 'mean' else raw.detach().float()-self.mu[index]
                    sums[index] = shifted.sum((0,1))
                    squares[index] = shifted.square().sum((0,1)) if statistics_mode != 'mean' else 0
                    counts[index] = context.valid.sum()
                if compute_auxiliary_losses or (collect_target_statistics and statistics_mode is None):
                    normalized, entries = self.normalize_target(raw,key,return_clipped=True)
                    if collect_target_statistics:clipped[index] = entries
                    return normalized
                return None
        for index in range(len(self.backbone.model.layers)):
            layer = index+1
            def call(x, previous=None, i=index):
                return self.block(i,x,mask,rotary,plan,previous)
            if layer in self.layers and need_targets:
                sources[layer] = None
                if self.family == 'P3' or self.increment_target:
                    bases[layer] = hidden.detach().float()
            result = (checkpoint(call,hidden,stream,use_reentrant=False)
                      if self.training and self.checkpoint_layers and torch.is_grad_enabled() else call(hidden,stream))
            hidden, mlp, u, *estimates = result
            if self.anticipatory:
                stream=estimates.pop()
                if self.family=='P5' and self.settings.aux_recompute and layer in self.layers and compute_auxiliary_losses:
                    prediction,aux_stream=self.heads[str(layer)].stream_step(u,aux_stream)
                    estimates=[prediction]
            if block_observer is not None:
                block_observer(layer, hidden)
            if layer in self.layers and need_targets:
                sources[layer] = (u,tuple(estimates)) if compute_auxiliary_losses else None
            if not need_targets:
                continue
            if self.family != 'P3':
                for proxy in tuple(sources):
                    if not self.increment_target:
                        with torch.no_grad():
                            windows[proxy] = windows[proxy] + mlp.float() if proxy in windows else mlp.float()
                    if layer == proxy+self.settings.lookahead-1:
                        raw = (hidden.detach().float()-bases.pop(proxy) if self.increment_target
                               else windows.pop(proxy))
                        normalized = record(proxy,raw)
                        source = sources.pop(proxy)
                        if compute_auxiliary_losses:
                            cosines[proxy] = self.layer_cosines(proxy,*source,normalized,plan,context.valid,auxiliary_grad)
            else:
                for proxy in tuple(sources):
                    if proxy+2 <= layer <= min(proxy+6,len(self.backbone.model.layers)):
                        with torch.no_grad(), torch.autocast(device_type=hidden.device.type,enabled=False):
                            increment = hidden.detach().float()-bases[proxy]
                            normalized = record((proxy,layer),increment)
                            if compute_auxiliary_losses:
                                windows[proxy] = windows[proxy]+normalized if proxy in windows else normalized
                        if layer == min(proxy+6,len(self.backbone.model.layers)):
                            source = sources.pop(proxy)
                            bases.pop(proxy)
                            if compute_auxiliary_losses:
                                raw = windows.pop(proxy)/(layer-proxy-1)
                                cosines[proxy] = self.layer_cosines(proxy,*source,raw,plan,context.valid,auxiliary_grad)
        if sources or windows:
            raise RuntimeError('Incomplete proxy target windows')
        return self.backbone.model.norm(hidden), cosines, sums, squares, counts, clipped

    def hidden_states(self, context):
        hidden, *_ = self._run_backbone(context, compute_auxiliary_losses=False,
                                            auxiliary_grad=False, collect_target_statistics=False)
        return hidden, None, None

    def forward(self, context, *, compute_auxiliary_losses=True, auxiliary_grad=True, collect_target_statistics=False,
                statistics_mode=None, relational_step=0, sequence_indices=None):
        hidden, cosines, sums, squares, counts, clipped = self._run_backbone(context,
            compute_auxiliary_losses=compute_auxiliary_losses, auxiliary_grad=auxiliary_grad,
            collect_target_statistics=collect_target_statistics,statistics_mode=statistics_mode,
            relational_step=relational_step,sequence_indices=sequence_indices)
        if statistics_mode is not None:
            return dict(center_sums=sums,center_squares=squares,center_counts=counts,clip_counts=clipped)
        lm_rows, target_counts, tokens = self.lm_statistics(context,hidden)
        width = 3 if self.family == 'P3' else 1
        values = (torch.stack([cosines[layer][0] for layer in self.layers],dim=1).flatten(1)
                  if cosines else lm_rows.new_zeros((len(lm_rows),len(self.layers)*width)))
        auxiliary_counts = tokens if cosines else torch.zeros_like(tokens)
        losses = (torch.stack([cosines[layer][1] for layer in self.layers],dim=1).flatten(1)
                  if cosines else torch.zeros_like(values))
        auxiliary = losses.mean(-1).sum() if cosines else lm_rows.sum()*0
        relational = {}
        extra = ()
        if self.relational_proxy:
            rel_rows = (torch.stack([cosines[layer][2] for layer in self.layers],dim=1) if cosines else torch.zeros_like(values))
            rel_counts = cosines[self.layers[0]][3] if cosines else torch.zeros_like(tokens)
            relational = dict(rel_sum=rel_rows.mean(-1).sum(),rel_count=rel_counts.sum())
            extra = (rel_rows.detach(),rel_counts[:,None])
        statistics = torch.cat((lm_rows.detach()[:,None],target_counts[:,None],auxiliary_counts[:,None],
                                values.detach(),losses.detach(),*extra,tokens[:,None]),dim=1).double()
        return dict(lm_sum=lm_rows.sum(),lm_count=target_counts.sum(),aux_sum=auxiliary,
                    aux_count=auxiliary_counts.sum(),statistics=statistics,
                    center_sums=sums,center_squares=squares,center_counts=counts,clip_counts=clipped,**relational)

    @contextmanager
    def without_proxy(self):
        previous = self.gates_disabled
        self.gates_disabled = True
        try:
            yield
        finally:
            self.gates_disabled = previous
