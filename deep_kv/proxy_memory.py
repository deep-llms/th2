"""P7: isolated causal estimators, joint native/proxy attention and relational KL.

No model/data RNG is consumed by query sampling. Attention has one softmax;
the FA4 adapter changes entry order, never the mathematical attention problem.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, rotate_half


def eligible_queries(documents):
    eligible = torch.zeros_like(documents, dtype=torch.bool)
    eligible[:, 1:] = documents[:, 1:] == documents[:, :-1]
    return eligible


class MemoryPlan:
    def __init__(self, documents, *, step=0, sequence_indices=None, ems=None, sample_queries=True, convolution=True):
        self.documents, self.ems = documents, ems
        self.conv_masks = tuple((documents[:, lag:] == documents[:, :-lag]).unsqueeze(-1)
                                for lag in range(1, min(4, documents.shape[1]))) if convolution else ()
        self._typed_conv_masks = {}
        self._relational_layout = None
        self.queries = []
        if not sample_queries:
            return  # No GPU-to-CPU sampling synchronization during target bootstrap/inference.
        if sequence_indices is None:
            sequence_indices = range(len(documents))
        if len(sequence_indices) != len(documents):
            raise ValueError('P7 requires one sampling index per sequence')
        # Small metadata copied once per forward, reused by all proxy layers.
        eligible = eligible_queries(documents).cpu()
        selections = []
        for positions, index in zip(eligible, sequence_indices):
            indices = positions.nonzero().flatten()
            generator = torch.Generator(device='cpu')
            generator.manual_seed((int(step)*6364136223846793005 + int(index)*1442695040888963407) % (2**63-1))
            if len(indices) > 256:
                indices = indices[torch.randperm(len(indices), generator=generator)[:256]]
            selections.append(indices)
        # One transfer per microbatch, with views for the individual sequences.
        self.queries = list(torch.cat(selections).to(documents.device).split([len(q) for q in selections]))

    def convolution_masks(self, dtype):
        # One conversion per forward/dtype, shared by layers and checkpoint replay.
        if dtype not in self._typed_conv_masks:
            self._typed_conv_masks[dtype] = tuple(mask.to(dtype=dtype) for mask in self.conv_masks)
        return self._typed_conv_masks[dtype]

    def relational_layout(self):
        """Immutable padded indices/masks shared by every proxy layer and replay."""
        if self._relational_layout is None:
            if len(self.queries) != len(self.documents):
                raise ValueError('Relational loss requires sampled queries')
            counts = self.documents.new_tensor([len(q) for q in self.queries])
            size = max(1, max(len(q) for q in self.queries))
            indices = torch.stack([F.pad(q, (0,size-len(q))) for q in self.queries])
            valid = torch.arange(size, device=indices.device)[None] < counts[:,None]
            positions = torch.arange(self.documents.shape[1], device=indices.device)
            allowed = ((positions[None,None] < indices[:,:,None]) &
                       (self.documents.gather(1,indices)[:,:,None] == self.documents[:,None]))
            # Padding gets one harmless candidate to avoid all-masked softmax.
            # Its KL is explicitly removed below, including its gradient.
            allowed = allowed | ((~valid)[:,:,None] & (positions == 0)[None,None])
            # Cache the actual masked_fill mask and FP32 denominator too;
            # autograd can share this mask instead of saving copies per layer.
            self._relational_layout = indices, valid, ~allowed, counts.float()
        return self._relational_layout


class MemoryHead(nn.Module):
    def __init__(self, config, settings, *, native_values=False):
        super().__init__()
        self.w_in = nn.Linear(config.hidden_size, settings.width, bias=False)
        self.conv = nn.Parameter(torch.empty(4, settings.width))
        self.w_out = nn.Linear(settings.width, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, 2*config.head_dim, bias=False)
        self.v_proj = None if native_values else nn.Linear(config.hidden_size, 2*config.head_dim, bias=False)
        self.k_norm = Qwen3RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=config.initializer_range)
        # Depthwise Conv1d's default fan-in is the kernel length, four.
        nn.init.uniform_(self.conv, -1/math.sqrt(4), 1/math.sqrt(4))

    def estimate(self, u, plan):
        x = self.w_in(u.detach())  # Structural isolation, not a zero upstream VJP.
        c = x*self.conv[0]
        for lag, same in enumerate(plan.convolution_masks(c.dtype), start=1):
            # c owns storage and is not saved by the multiply's backward.
            # Preserve lag/addition order without padded full-length temporaries.
            c[:, lag:].add_(x[:, :-lag]*self.conv[lag]*same)
        if plan.ems is not None:
            c = c + (plan.ems.apply(x, 1)+plan.ems.apply(x, 2)).to(c.dtype)
        return self.w_out(F.silu(c))

    def entries(self, prediction, rotary, native_values, *, input_grad=False, native_keys=None):
        with torch.autocast(prediction.device.type, enabled=False):
            # Opt-in supervised adaptation keeps identical forward values while
            # allowing task gradients into the pretrained estimator. Pretraining
            # and likelihood evaluation retain the original routing by default.
            value = (prediction if input_grad or getattr(self, 'task_finetuning', False) or getattr(self,'lm_predictor',False) else prediction.detach()).float()
            p = (value*torch.rsqrt(value.square().mean(-1, keepdim=True)+1e-6)).to(prediction.dtype)
        shape = (*p.shape[:-1], getattr(self, 'groups', 2), -1)
        key_input = p if not hasattr(self, 'key_transform') else p+self.key_transform(p)
        value_input = p if not hasattr(self, 'value_transform') else p+self.value_transform(p)
        if self.k_proj is None:
            if native_keys is None:raise ValueError('Value-only memory requires native rotated keys')
            k = native_keys
        else:
            k = self.k_norm(self.k_proj(key_input).view(shape)).transpose(1, 2)
            # Only proxy keys need rotating; native keys are already rotated.
            cos,sin = (x.unsqueeze(1) for x in rotary)
            k = k*cos+rotate_half(k)*sin
        v = native_values if self.v_proj is None else self.v_proj(value_input).view(shape).transpose(1, 2)
        return k, v


class SimpleMemoryHead(MemoryHead):
    """P4-iso's tokenwise estimator with P7's detached memory projections."""
    def __init__(self, config, settings):
        nn.Module.__init__(self)
        self.w1 = nn.Linear(config.hidden_size, settings.width, bias=False)
        self.w2 = nn.Linear(settings.width, config.hidden_size, bias=False)
        for module in (self.w1, self.w2):
            nn.init.normal_(module.weight, std=config.initializer_range)

    def initialize_projections(self, config, groups=2):
        # Called after ALL estimators: preserve P4-iso's estimator RNG sequence.
        self.groups = groups
        self.k_proj = nn.Linear(config.hidden_size, groups*config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, groups*config.head_dim, bias=False)
        self.k_norm = Qwen3RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        for module in (self.k_proj, self.v_proj):
            nn.init.normal_(module.weight, std=config.initializer_range)

    def initialize_tuning(self, config, spec):
        if 'transform_width' in spec:
            # Input is already RMS-normalized. Separate low-rank residuals;
            # zero up-projections reproduce the parent's initial forward exactly.
            for name in ('key_transform', 'value_transform'):
                down = nn.Linear(config.hidden_size, spec['transform_width'], bias=False)
                up = nn.Linear(spec['transform_width'], config.hidden_size, bias=False)
                nn.init.normal_(down.weight, std=config.initializer_range)
                nn.init.zeros_(up.weight)
                setattr(self, name, nn.Sequential(down, nn.SiLU(), up))
        if 'mix_init' in spec:
            count = self.groups*config.num_attention_heads//config.num_key_value_heads
            p = spec['mix_init']
            self.value_gate = nn.Parameter(torch.full((count,), math.log(p/(1-p))))

    def estimate(self, u, plan):
        source = u if getattr(self, 'task_finetuning', False) else u.detach()
        return self.w2(F.silu(self.w1(source)))


def next_token_mask(context):
    """One fixed packed position in eight, with the main LM's label exclusions."""
    return context.targets()[:, ::8]


def next_token_loss(prediction, context, norm_weight, lm_weight, epsilon, chunk_size,
                    checkpoint_loss=False):
    """Last-site prediction -> frozen current readout; never train the readout.

    Sampling is fixed by packed offset, not RNG or microbatch size. No future
    hidden state enters the predictor. Only labels are shifted to t+1.
    """
    from torch.utils.checkpoint import checkpoint
    valid = next_token_mask(context)
    labels = (context.input_ids if context.labels is None else context.labels)[:, 1::8]
    labels = labels.masked_fill(~valid, -100)
    selected = prediction[:, :-1:8]
    rows = prediction.new_zeros(len(prediction), dtype=torch.float32)
    def ce(value, targets):
        with torch.autocast(value.device.type, enabled=False):
            h = value.float()
            h = h*torch.rsqrt(h.square().mean(-1, keepdim=True)+epsilon)
            h = (h*norm_weight.detach().float()).to(value.dtype)
        logits = F.linear(h, lm_weight.detach()).float()
        return F.cross_entropy(logits.flatten(0, 1), targets.flatten(),
            ignore_index=-100, reduction='none').view_as(targets).sum(-1)
    for start in range(0, labels.shape[1], chunk_size):
        y = labels[:, start:start+chunk_size]
        x = selected[:, start:start+y.shape[1]]
        rows = rows+(checkpoint(ce, x, y, use_reentrant=False) if checkpoint_loss else ce(x, y))
    return rows


def rectangular_mask(mask, disabled=False):
    return torch.cat((mask, torch.zeros_like(mask) if disabled else mask), dim=-1)


class _MemoryConsumer(nn.Module):
    """Functional view of the existing memory projections; no copied parameters."""
    def __init__(self, head, backend, kernel, scaling, observer):
        super().__init__()
        self.head, self.backend, self.kernel, self.scaling = head, backend, kernel, scaling
        self.observer=observer

    def forward(self, prediction, q, k, v, rotary, mask, plan, attention_module):
        kp,vp=self.head.entries(prediction,rotary,v,input_grad=True)
        if self.backend=='fa4':
            return flash_joint_attention(q,k,v,kp,vp,mask,self.scaling,self.kernel,
                self.observer,doubled_layout=plan.doubled_layout)
        # Identical selected-head joint softmax to the production SDPA consumer.
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        output,_=ALL_ATTENTION_FUNCTIONS['sdpa'](attention_module,q,torch.cat((k,kp),dim=2),
            torch.cat((v,vp),dim=2),plan.joint_mask,dropout=0.,
            scaling=self.scaling,is_causal=False)
        return output


def memory_consumer_loss(layer, head, source, prediction, target, mask, rotary,
                         plan, valid, backend, kernel=None, observer=None):
    """Distill four joint-attention head outputs into the predictor only.

    Native queries/keys/values and projection parameters are frozen. The teacher
    substitutes a detached standardized deep target for the proxy. It is a local
    training target, not an oracle recurrent rollout or an extra inference path.
    Native QKV are shared by teacher and student; no decoder/MLP is replayed.
    """
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    a=layer.self_attn
    with torch.no_grad():
        u=source.detach();shape=(*u.shape[:-1],-1,a.head_dim)
        q=a.q_norm(a.q_proj(u).view(shape)).transpose(1,2)
        k=a.k_norm(a.k_proj(u).view(shape)).transpose(1,2)
        v=a.v_proj(u).view(shape).transpose(1,2)
        rotary=tuple(x.detach() for x in rotary)
        q,k=apply_rotary_pos_emb(q,k,*rotary)
        q,k,v=q[:,-4:],k[:,-2:],v[:,-2:]
    if backend=='fa4' and not hasattr(plan,'doubled_layout'):
        plan.doubled_layout=(mask[0]*2,mask[1]*2)
    if backend=='sdpa' and not hasattr(plan,'joint_mask'):
        plan.joint_mask=rectangular_mask(mask)
    consumer=_MemoryConsumer(head,backend,kernel,a.scaling,observer)
    frozen={name:p.detach() for name,p in consumer.named_parameters()}
    frozen.update({name:b.detach() for name,b in consumer.named_buffers()})
    args=(q,k,v,rotary,mask,plan,a)
    with torch.no_grad():
        teacher=torch.func.functional_call(consumer,frozen,(target.detach().to(prediction.dtype),*args))
    student=torch.func.functional_call(consumer,frozen,(prediction,*args))
    with torch.autocast(source.device.type,enabled=False):
        truth=teacher.float()
        error=(student.float()-truth).square().mean((-1,-2))
        energy=truth.square().mean((-1,-2))+1e-6
        return ((error/energy)*valid).sum(-1)[:,None]


def merge_attention(native, proxy, native_lse, proxy_lse):
    """Reconstruct one joint softmax from its two partitions, with live LSEs.

    Both partitions contain the current token, so neither normalizer is -inf.
    Keep the first native heads and merge only the selected four proxy heads.
    """
    with torch.autocast(native.device.type, enabled=False):
        split = native.shape[2]-proxy.shape[2]
        weight = torch.sigmoid(native_lse[:, :, split:].float()-proxy_lse.float())[..., None]
        selected = weight*native[:, :, split:].float()+(1-weight)*proxy.float()
        return torch.cat((native[:, :, :split], selected.to(native.dtype)), dim=2)


def flash_joint_attention(q, k, v, kp, vp, layout, scaling, kernel, observer=None, *, doubled_layout=None):
    """Interleave [native_j, proxy_j], duplicate queries, retain odd outputs.

    Query 2*t+1 sees exactly both entries of tokens j <= t. Even query outputs
    are discarded (zero adjoints). Document lengths/boundaries double. This
    uses the pinned equal-length causal varlen API, without arbitrary masks,
    unequal-length causal alignment assumptions, or separate softmaxes.
    """
    from .fa4 import attention
    query = q.repeat_interleave(2, dim=2)
    key = torch.stack((k, kp), dim=3).flatten(2, 3)
    value = torch.stack((v, vp), dim=3).flatten(2, 3)
    doubled = (layout[0]*2, layout[1]*2) if doubled_layout is None else doubled_layout
    output = attention(query, key, value, doubled, scaling, kernel)
    if observer is not None:
        observer(query, key, value, doubled)
    return output[:, 1::2]


def relational_losses(prediction, target, plan):
    """Return per-sequence KL sums and query counts; empty rows stay differentiable."""
    with torch.autocast(prediction.device.type, enabled=False):
        pred = F.normalize(prediction.float(), dim=-1, eps=1e-6)
        truth = F.normalize(target.detach().float(), dim=-1, eps=1e-6)
        indices, valid, blocked, counts = plan.relational_layout()
        gather = indices[:,:,None].expand(-1,-1,pred.shape[-1])
        target_logits = torch.bmm(truth.gather(1,gather),truth.transpose(1,2))/.1
        pred_logits = torch.bmm(pred.gather(1,gather),pred.transpose(1,2))/.1
        target_logp = target_logits.masked_fill(blocked, -torch.inf).log_softmax(-1)
        pred_logp = pred_logits.masked_fill(blocked, -torch.inf).log_softmax(-1)
        # Avoid 0*(-inf - -inf), including in backward.
        difference = target_logp.masked_fill(blocked, 0)-pred_logp.masked_fill(blocked, 0)
        per_query = (target_logp.exp()*difference).sum(-1)
        return per_query.masked_fill(~valid,0).sum(-1), counts


@torch.no_grad()
def proxy_mass(q, k, kp, documents, scaling, chunk_size=128):
    """Eager FP32 probabilities on bounded query chunks, diagnostics only."""
    repeats = q.shape[1]//k.shape[1]
    keys = torch.cat((k, kp), dim=2).repeat_interleave(repeats, dim=1).float()
    length = q.shape[2]
    positions = torch.arange(length, device=q.device)
    total = q.new_zeros((), dtype=torch.float32)
    with torch.autocast(q.device.type, enabled=False):
        for start in range(0, length, chunk_size):
            end = min(start+chunk_size, length)
            mask = ((positions[None, :] <= positions[start:end, None])[None, None] &
                    (documents[:, None, start:end, None] == documents[:, None, None, :]))
            logits = (q[:, :, start:end].float() @ keys.transpose(-1, -2))*scaling
            probabilities = logits.masked_fill(~rectangular_mask(mask), -torch.inf).softmax(-1)
            total += probabilities[..., length:].sum()
    return torch.stack((total, total.new_tensor(q.shape[0]*q.shape[1]*length)))
