"""B-DOT's routed middle, reusing native HF Qwen3 stem/coda, norms and attention.

Only Phase A is implemented. Unsupported scientific variants fail explicitly.
The native Qwen3 forward retains masks, RoPE, cache ownership and LM loss.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import Qwen3ForCausalLM
from transformers import initialization as hf_init
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm, apply_rotary_pos_emb, ALL_ATTENTION_FUNCTIONS, eager_attention_forward,
)

from .routing import RoutingConfig, quota_table


class Expert(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.q_proj = layer.self_attn.q_proj
        self.q_norm = layer.self_attn.q_norm
        self.o_proj = layer.self_attn.o_proj
        self.mlp_norm = layer.post_attention_layernorm
        self.mlp = layer.mlp


class Router(nn.Module):
    def __init__(self, config, options, step):
        super().__init__()
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.weight = nn.Parameter(torch.empty(options.S, config.hidden_size))
        self.bias = nn.Parameter(torch.zeros(options.S))
        self.visits = nn.Parameter(torch.zeros(options.S, options.S))
        nn.init.normal_(self.weight, std=options.init_std_scale / math.sqrt(config.hidden_size))
        with torch.no_grad():
            self.bias[step] = options.default_prior

    def forward(self, h, visits, allowed):
        # Disable autocast, including matmuls; casting inputs alone does not do so.
        with torch.autocast(device_type=h.device.type, enabled=False):
            x = h.float()
            x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm.variance_epsilon)
            x = x * self.norm.weight.float()
            z = nn.functional.linear(x, self.weight.float(), self.bias.float())
            z = z + nn.functional.linear(visits.float(), self.visits.float())
            return z.masked_fill(~allowed, -torch.inf)


@dataclass
class RouteContext:
    visits: torch.Tensor
    state: torch.Tensor
    generator: torch.Generator
    reach: int
    soft: bool
    explore: float
    beta: torch.Tensor
    z_losses: list
    routes: list
    counts: list
    entropies: list
    forced: torch.Tensor | None = None
    routing: str | None = None


class RoutedStep(nn.Module):
    def __init__(self, layer, experts, config, options, step):
        super().__init__()
        self.config, self.options, self.step = config, options, step
        # Experts are registered exactly once on Qwen3Model, not at each step.
        object.__setattr__(self, 'experts', experts)
        self.input_layernorm = layer.input_layernorm
        self.k_proj, self.v_proj = layer.self_attn.k_proj, layer.self_attn.v_proj
        self.k_norm = layer.self_attn.k_norm
        self.layer_idx = options.n_stem + step
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.sliding_window = None
        self.is_causal = True
        self.attention_dropout = 0.
        self.gradient_checkpointing = False
        self.router = Router(config, options, step) if options.routing == 'learned' else None
        if options.step_film:
            self.film_gain = nn.Parameter(torch.zeros(config.hidden_size))
            self.film_bias = nn.Parameter(torch.zeros(config.hidden_size))
        if not options.reuse:
            for reach in range(min(options.r, options.S-1)+1):
                allowed, transitions = quota_table(options.S, reach)
                self.register_buffer(f'allowed_{reach}', allowed, persistent=False)
                self.register_buffer(f'transitions_{reach}', transitions, persistent=False)

    def select(self, h, context):
        opt, n = self.options, h.shape[0] * h.shape[1]
        routing = context.routing or opt.routing
        if opt.reuse:
            modules = torch.arange(opt.S, device=h.device)
            allowed = ((modules - self.step).abs() <= context.reach).expand(n, -1)
        else:
            allowed = getattr(self, f'allowed_{min(context.reach, opt.S-1)}')[context.state]
        z = self.router(h.flatten(0, 1), context.visits, allowed) if self.router is not None else None
        if z is not None:
            context.z_losses.append(torch.logsumexp(z, -1).square().sum())
            if self.training:
                with torch.no_grad():
                    logp = z.log_softmax(-1)
                    context.entropies.append(-(logp.exp() * logp.masked_fill(~allowed, 0)).sum())
        if context.forced is not None:
            choice = context.forced[self.step].reshape(n, 1)
            if not bool(allowed.gather(1, choice).all()):
                raise ValueError('Forced diagnostic route violates candidate/quota mask')
        elif routing == 'fixed':
            choice = torch.full((n, 1), self.step, device=h.device, dtype=torch.long)
        elif routing == 'random':
            choice = torch.multinomial(allowed.float(), 1, generator=context.generator)
        else:
            # Stable sorting implements the specified lowest-id tie break.
            k = min(2 if context.soft else 1, opt.S)
            choice = torch.argsort(z + context.beta, dim=-1, descending=True, stable=True)[:, :k]
            if self.training and context.explore:
                replace = torch.rand(n, 1, device=h.device, generator=context.generator) < context.explore
                random_choice = torch.multinomial(allowed.float(), 1, generator=context.generator)
                choice = torch.where(replace, random_choice, choice)
        if z is None or routing != 'learned':
            weights = h.new_ones(n, 1)
        elif context.soft and context.forced is None:
            # A single feasible candidate contributes once, never execute a masked expert.
            valid = allowed.gather(1, choice)
            chosen_z = z.gather(1, choice).masked_fill(~valid, -torch.inf)
            weights = chosen_z.softmax(-1)
            choice = torch.where(valid, choice, choice[:, :1])
        else:
            logp = z.log_softmax(-1).gather(1, choice)
            weights = (logp - logp.detach()).exp() if opt.gate == 'st' else logp.exp()
        top = choice[:, 0]
        context.visits = context.visits.scatter_add(1, top[:, None], torch.ones(n, 1, device=h.device))
        if not opt.reuse:
            context.state = getattr(self, f'transitions_{min(context.reach, opt.S-1)}')[context.state, top]
        context.routes.append(top.reshape(h.shape[:2]).detach())
        context.counts.append(torch.bincount(top, minlength=opt.S))
        return choice, weights

    def execute(self, h, choice, weights, attention_mask, position_embeddings, past_key_values, kwargs):
        b, t, d = h.shape
        n, branches = b*t, choice.shape[1]
        x = self.input_layernorm(h)
        k = self.k_norm(self.k_proj(x).view(b, t, -1, self.head_dim)).transpose(1, 2)
        v = self.v_proj(x).view(b, t, -1, self.head_dim).transpose(1, 2)
        cos, sin = position_embeddings
        # Rotate K once. Q is produced per candidate; do not duplicate K/V projections.
        _, k = apply_rotary_pos_emb(k, k, cos, sin)
        if past_key_values is not None:
            if branches != 1 or self.training:
                raise ValueError('KV caching supports hard evaluation only')
            k, v = past_key_values.update(k, v, self.layer_idx)
        # Branch-major dispatch: only execute selected tokens for each expert.
        flat_choices = choice.T.reshape(-1)
        executed = weights.T.reshape(-1).detach().ne(0).nonzero().flatten()
        order = executed[torch.argsort(flat_choices[executed], stable=True)]
        sizes = torch.bincount(flat_choices[executed], minlength=self.options.S).tolist()
        groups, start = [], 0
        for m, size in enumerate(sizes):
            if size:
                groups.append((m, order[start:start+size]))
            start += size
        q = x.new_zeros(branches*n, self.config.num_attention_heads*self.head_dim)
        for m, idx in groups:
            expert = self.experts[m]
            selected = x.reshape(n, d)[idx % n]
            projected = expert.q_proj(selected).view(-1, self.config.num_attention_heads, self.head_dim)
            q = q.index_copy(0, idx, expert.q_norm(projected).flatten(1))
        interface = ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation, eager_attention_forward)
        attention = []
        for branch in range(branches):
            qb = q[branch*n:(branch+1)*n].view(b, t, -1, self.head_dim).transpose(1, 2)
            qb, _ = apply_rotary_pos_emb(qb, qb, cos, sin)
            a, _ = interface(self, qb, k, v, attention_mask, dropout=0., scaling=self.scaling,
                             sliding_window=None, **kwargs)
            attention.append(a.reshape(n, -1))
        attention = torch.cat(attention)
        result = h.new_zeros(branches*n, d)
        for m, idx in groups:
            expert = self.experts[m]
            h1 = h.reshape(n, d)[idx % n] + expert.o_proj(attention[idx])
            u = expert.mlp_norm(h1)
            if self.options.step_film:
                u = u * (1 + self.film_gain.to(u.dtype)) + self.film_bias.to(u.dtype)
            result = result.index_copy(0, idx, h1 + expert.mlp(u))
        result = result.view(branches, b, t, d)
        weights = weights.T.reshape(branches, b, t, 1).to(h.dtype)
        if branches == 1:
            # Algebraically h + gate*(result-h), but preserves the native forward
            # exactly when gate=1 (avoids subtract/add cancellation in BF16).
            return result[0] + (weights[0]-1) * (result[0]-h)
        return (weights * result).sum(0)

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None,
                past_key_values=None, bdot_context=None, **kwargs):
        if bdot_context is None:
            raise ValueError('Routed steps require BDOTForCausalLM.forward')
        choice, weights = self.select(hidden_states, bdot_context)
        # Route/RNG/count bookkeeping occurs once outside the checkpoint. Gates
        # remain attached to autograd; nonreentrant replay handles unused experts.
        def run(h, gate):
            return self.execute(h, choice, gate, attention_mask, position_embeddings, past_key_values, kwargs)
        if self.gradient_checkpointing and self.training:
            if past_key_values is not None:
                raise ValueError('Training checkpointing cannot mutate a KV cache')
            return checkpoint(run, hidden_states, weights, use_reentrant=False)
        return run(hidden_states, weights)


class BDOTForCausalLM(Qwen3ForCausalLM):
    """Qwen3 interface and native loss; BDOTTrainer adds normalized router loss."""
    def __init__(self, config):
        options = RoutingConfig(**config.bdot)
        config.bdot = options.to_dict()
        if config.num_hidden_layers != options.n_stem + options.S + options.n_coda:
            raise ValueError('Qwen3 layer count must equal stem + S + coda')
        if config.attention_dropout or any(t != 'full_attention' for t in config.layer_types):
            raise ValueError('B-DOT requires zero dropout and full-attention Qwen3')
        super().__init__(config)
        self.options = options
        original = list(self.model.layers[options.n_stem:options.n_stem+options.S])
        self.model.bdot_experts = nn.ModuleList([Expert(layer) for layer in original])
        # Initialize the entire native backbone before adding routers. Their
        # parameters follow the model-init RNG; routing draws use a separate seed.
        for s, layer in enumerate(original):
            self.model.layers[options.n_stem+s] = RoutedStep(
                layer, self.model.bdot_experts, config, options, s)
        self.register_buffer('balance_bias', torch.zeros(options.S))
        self.register_buffer('routing_counter', torch.zeros((), dtype=torch.long))
        self.register_buffer('completed_steps', torch.zeros((), dtype=torch.long))
        self.register_buffer('load_counts', torch.zeros(options.S, options.S, dtype=torch.long), persistent=False)
        self.register_buffer('metric_totals', torch.zeros(9, dtype=torch.float64), persistent=False)
        self.last_update_metrics = {}
        self.progress = 0.
        self.eval_seed = options.seed
        self.eval_counter = 0
        self.routing_override = None
        self.capture_routes = False
        self.last_routes = None

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        # HF materializes nonpersistent buffers with empty_like when loading
        # from meta. Constructor values do not survive from_pretrained, so
        # deterministic tables and update accumulators must be rebuilt here.
        if isinstance(module, BDOTForCausalLM) and hasattr(module, 'load_counts'):
            hf_init.zeros_(module.load_counts)
            hf_init.zeros_(module.metric_totals)
            # RoutedStep is inside a native Qwen3Model: HF dispatches that
            # child's initializer there, so restore our tables at this root.
            if not module.options.reuse:
                for reach in range(min(module.options.r, module.options.S-1)+1):
                    allowed, transitions = quota_table(module.options.S, reach)
                    for step in module.routed_steps:
                        hf_init.copy_(getattr(step, f'allowed_{reach}'), allowed)
                        hf_init.copy_(getattr(step, f'transitions_{reach}'), transitions)

    @property
    def routed_steps(self):
        return self.model.layers[self.options.n_stem:self.options.n_stem+self.options.S]

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None, use_cache=None,
                bdot_num_decisions=None, bdot_forced_routes=None, **kwargs):
        shape = input_ids.shape if input_ids is not None else inputs_embeds.shape[:2]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        n = math.prod(shape)
        reach, soft, eps = self.options.phase(self.progress, self.training)
        # A rank-local, stateless substream indexed by a checkpointed invocation
        # counter. Equal-batch DDP keeps counters equal; rank selects disjoint draws.
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if self.training:
            counter = int(self.routing_counter)
            self.routing_counter.add_(1)
            seed = (self.options.seed + 1000003*counter + 1000000007*rank) % (2**63-1)
        else:
            seed = (self.eval_seed + 1000003*self.eval_counter + 1000000007*rank) % (2**63-1)
            self.eval_counter += 1
        if self.routing_override is not None:
            if self.training or self.routing_override not in ('fixed', 'random'):
                raise ValueError('Routing override is an evaluation-only fixed/random intervention')
        generator = torch.Generator(device=device).manual_seed(seed)
        context = RouteContext(torch.zeros(n, self.options.S, device=device),
            torch.zeros(n, dtype=torch.long, device=device), generator, reach, soft, eps,
            self.balance_bias.detach().clone(), [], [], [], [], bdot_forced_routes, self.routing_override)
        if self.training and (use_cache or past_key_values is not None):
            raise ValueError('B-DOT training requires use_cache=False')
        output = super().forward(input_ids=input_ids, inputs_embeds=inputs_embeds,
            attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values,
            labels=labels, use_cache=use_cache, bdot_context=context, **kwargs)
        if self.training:
            self.load_counts.add_(torch.stack(context.counts))
        if self.capture_routes:
            self.last_routes = torch.stack(context.routes)
        if labels is not None and self.training:
            z_sum = torch.stack(context.z_losses).sum() if context.z_losses else output.loss.new_zeros(())
            denominator = bdot_num_decisions if bdot_num_decisions is not None else n*self.options.S
            with torch.no_grad():
                lm_count = labels[:, 1:].ne(-100).sum()
                lm_denominator = kwargs.get('num_items_in_batch', lm_count)
                routes = torch.stack(context.routes)
                off_default = (routes != torch.arange(self.options.S, device=device)[:, None, None]).sum()
                entropy = torch.stack(context.entropies).sum() if context.entropies else z_sum.new_zeros(())
                self.metric_totals.add_(torch.stack((output.loss.detach()*lm_denominator,
                    lm_count, z_sum.detach(), lm_count.new_tensor(n*self.options.S),
                    off_default, lm_count.new_tensor(n*self.options.S), entropy,
                    context.visits.gt(1).any(-1).sum(), lm_count.new_tensor(n))).double())
            output.loss = output.loss + self.options.z_loss_coef*z_sum / denominator
        return output

    @torch.no_grad()
    def finish_update(self):
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(self.load_counts)
            torch.distributed.all_reduce(self.metric_totals)
        totals = self.load_counts.sum(-1)
        active = totals > 0
        if self.options.routing == 'learned' and bool(active.any()):
            shares = self.load_counts[active].float() / totals[active, None]
            target = torch.eye(self.options.S, device=shares.device)[active].mean(0)
            self.balance_bias.add_(self.options.balance_rate * (target - shares.mean(0)).sign())
        values = self.metric_totals.tolist()
        self.last_update_metrics = dict(lm_loss_last_update=values[0]/max(values[1],1),
            router_z_loss_last_update=values[2]/max(values[3],1),
            off_default_share_last_update=values[4]/max(values[5],1),
            router_entropy_last_update=values[6]/max(values[3],1),
            reuse_rate_last_update=values[7]/max(values[8],1))
        # All ranks have the global counts here. A zero-load module is reported
        # explicitly; do not hide it with an epsilon in a max/min ratio.
        module_loads = self.load_counts.sum(0).tolist()
        min_load, max_load = min(module_loads), max(module_loads)
        self.last_update_metrics['module_load_max_min_ratio_last_update'] = max_load/min_load if min_load else None
        for s, (total, default) in enumerate(zip(totals.tolist(), self.load_counts.diag().tolist())):
            self.last_update_metrics[f'off_default_step_{s}_last_update'] = 1-default/total if total else 0.
        for m, load in enumerate(module_loads):
            self.last_update_metrics[f'module_{m}_decisions_last_update'] = load
        self.metric_totals.zero_()
        self.load_counts.zero_()
        self.completed_steps.add_(1)

    def native_reference(self):
        """Plain Qwen3 with identical backbone weights, for T1 (FiLM must be off)."""
        if self.options.step_film:
            raise ValueError('Native equivalence requires step_film=False')
        from copy import deepcopy
        config = deepcopy(self.config)
        del config.bdot
        ref = Qwen3ForCausalLM(config).to(device=self.device, dtype=self.dtype)
        ref.model.embed_tokens.load_state_dict(self.model.embed_tokens.state_dict())
        ref.lm_head.load_state_dict(self.lm_head.state_dict())
        ref.model.norm.load_state_dict(self.model.norm.state_dict())
        for i, layer in enumerate(self.model.layers):
            target = ref.model.layers[i]
            if not isinstance(layer, RoutedStep):
                target.load_state_dict(layer.state_dict())
                continue
            expert = self.model.bdot_experts[layer.step]
            for name in ('q_proj', 'q_norm', 'o_proj'):
                getattr(target.self_attn, name).load_state_dict(getattr(expert, name).state_dict())
            for name in ('k_proj', 'k_norm', 'v_proj'):
                getattr(target.self_attn, name).load_state_dict(getattr(layer, name).state_dict())
            target.input_layernorm.load_state_dict(layer.input_layernorm.state_dict())
            target.post_attention_layernorm.load_state_dict(expert.mlp_norm.state_dict())
            target.mlp.load_state_dict(expert.mlp.state_dict())
        return ref
