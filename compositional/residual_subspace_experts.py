"""Exactly tied residual subspace experts for token embeddings.

The effective embedding of token ``i`` is

    e_i = B_0(z_i) + sum_{g in TopK(i)} pi_ig V_g(U_g(z_i))

where ``z_i`` is a compact token factor, ``B_0`` is a global projection, and
each expert is a low-rank residual projection.  Routing depends only on the
token factor, so a token has the same deterministic vector on the input and
output sides.  The tied output head in :mod:`compositional.tied_head` exploits
this structure without materializing the full vocabulary-by-hidden table.

The residual down projections are initialized to zero while the up
projections are random.  Consequently the module starts as an exact global
low-rank embedding, but the down projections receive nonzero gradients on the
first optimization step (the standard one-zero-factor residual initialization).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def residual_subspace_experts_parameter_count(
    vocab_size,
    embed_dim,
    base_rank,
    expert_rank,
    num_experts,
    router_dim,
):
    """Return the exact trainable-parameter count, including every bias."""
    values = {
        "vocab_size": vocab_size,
        "embed_dim": embed_dim,
        "base_rank": base_rank,
        "expert_rank": expert_rank,
        "num_experts": num_experts,
        "router_dim": router_dim,
    }
    values = {name: int(value) for name, value in values.items()}
    if any(value <= 0 for value in values.values()):
        raise ValueError("all dimensions must be positive")

    v = values["vocab_size"]
    d = values["embed_dim"]
    r = values["base_rank"]
    q = values["expert_rank"]
    g = values["num_experts"]
    k = values["router_dim"]
    token_factors = v * r
    base_projection = d * r + d
    # Each residual expert has Linear(r, q) and Linear(q, d), both with bias.
    experts = g * (q * r + q + d * q + d)
    router = k * r + k + g * k
    return token_factors + base_projection + experts + router


class ResidualSubspaceExpertsEmbed(nn.Module):
    """Global low-rank embedding plus learned top-k residual subspaces.

    Routing is vocabulary-static for a fixed parameter state: it is computed
    from a token's own factor and never from sequence context.  Therefore this
    module is valid for exact input/output weight tying.
    """

    def __init__(
        self,
        vocab_size,
        embed_dim,
        *,
        base_rank=120,
        expert_rank=80,
        num_experts=12,
        router_dim=32,
        top_k=2,
        router_temperature=1.0,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)
        self.base_rank = int(base_rank)
        self.expert_rank = int(expert_rank)
        self.num_experts = int(num_experts)
        self.router_dim = int(router_dim)
        self.top_k = int(top_k)
        self.router_temperature = float(router_temperature)
        self._validate_hyperparameters()

        # Persist non-shape routing structure so a checkpoint remains fully
        # reconstructable even if train_config.json is lost.
        self.register_buffer(
            "routing_top_k", torch.tensor(self.top_k, dtype=torch.long)
        )
        self.register_buffer(
            "routing_temperature",
            torch.tensor(self.router_temperature, dtype=torch.float64),
        )

        self.token_factors = nn.Parameter(
            torch.empty(self.vocab_size, self.base_rank)
        )
        self.base_proj = nn.Linear(
            self.base_rank, self.embed_dim, bias=True
        )

        # Batched Linear weights. Keeping each expert family in one Parameter
        # makes checkpoint structure explicit and avoids Python ModuleList
        # topology depending on a saved hyperparameter.
        self.expert_down_weight = nn.Parameter(torch.empty(
            self.num_experts, self.expert_rank, self.base_rank
        ))
        self.expert_down_bias = nn.Parameter(torch.empty(
            self.num_experts, self.expert_rank
        ))
        self.expert_up_weight = nn.Parameter(torch.empty(
            self.num_experts, self.embed_dim, self.expert_rank
        ))
        self.expert_up_bias = nn.Parameter(torch.empty(
            self.num_experts, self.embed_dim
        ))

        self.router_proj = nn.Linear(
            self.base_rank, self.router_dim, bias=True
        )
        self.expert_keys = nn.Parameter(torch.empty(
            self.num_experts, self.router_dim
        ))
        self.reset_parameters()
        # Transient graph-bearing router data consumed exactly once by the
        # trainer. It is never serialized and is not populated by output-head
        # routing or materialization.
        self._router_aux = None
        # Training-only graph-bearing route shared by input lookup and the
        # subsequent tied head in the same model forward.
        self._output_route_cache = None
        # No-grad route reused across autoregressive evaluation forwards.
        self._inference_route_cache = None
        self._inference_expert_cache = None
        self.register_load_state_dict_post_hook(self._clear_cache_after_load)

    @staticmethod
    def _clear_cache_after_load(module, incompatible_keys):
        # These non-shape settings are checkpoint-authoritative buffers. Keep
        # the fast Python scalars used in forward synchronized even for callers
        # that load a state directly rather than rebuilding through loading.py.
        module.top_k = int(module.routing_top_k.item())
        module.router_temperature = float(
            module.routing_temperature.item()
        )
        module._validate_hyperparameters()
        module._router_aux = None
        module._output_route_cache = None
        module.clear_inference_cache()

    def clear_inference_cache(self):
        """Discard derived routing data after any possible parameter change."""
        self._output_route_cache = None
        self._inference_route_cache = None
        self._inference_expert_cache = None

    def train(self, mode=True):
        # Both directions matter: eval->train invalidates derived inference
        # data after future optimizer updates, while train->eval releases an
        # unconsumed auxiliary graph held by callers such as finetune/train.py.
        if mode != self.training:
            self._router_aux = None
            self._output_route_cache = None
            self.clear_inference_cache()
        return super().train(mode)

    def _apply(self, fn):
        # Cached tensors are device/dtype specific and are not Parameters or
        # buffers, so they must never survive a module conversion.
        self._router_aux = None
        self._output_route_cache = None
        self.clear_inference_cache()
        result = super()._apply(fn)
        # Module.to(dtype=...) normally casts every floating buffer. Preserve
        # the checkpoint-authoritative scalar exactly; arbitrary temperatures
        # such as 0.7 must not round to BF16 and then fail strict config checks.
        self.routing_temperature = self.routing_temperature.to(
            dtype=torch.float64
        )
        return result

    def _validate_hyperparameters(self):
        dimensions = (
            self.vocab_size,
            self.embed_dim,
            self.base_rank,
            self.expert_rank,
            self.num_experts,
            self.router_dim,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("all dimensions must be positive")
        if not 1 <= self.top_k <= self.num_experts:
            raise ValueError(
                "top_k must be in [1, num_experts], got "
                f"top_k={self.top_k}, num_experts={self.num_experts}"
            )
        if not torch.isfinite(torch.tensor(self.router_temperature)):
            raise ValueError("router_temperature must be finite")
        if self.router_temperature < 1e-4:
            raise ValueError(
                "router_temperature must be at least 1e-4 for stable BF16 routing"
            )

    def reset_parameters(self):
        nn.init.normal_(self.token_factors, std=0.02)
        nn.init.normal_(self.base_proj.weight, std=0.02)
        nn.init.zeros_(self.base_proj.bias)

        # Exact global-low-rank function at initialization. Initializing only
        # the first factor to zero is important: zeroing both factors would
        # also zero both factors' first-step gradients.
        nn.init.zeros_(self.expert_down_weight)
        nn.init.zeros_(self.expert_down_bias)
        nn.init.normal_(self.expert_up_weight, std=0.02)
        nn.init.zeros_(self.expert_up_bias)

        nn.init.normal_(self.router_proj.weight, std=0.02)
        nn.init.zeros_(self.router_proj.bias)
        nn.init.normal_(self.expert_keys, std=0.02)

    @property
    def parameter_count(self):
        return residual_subspace_experts_parameter_count(
            self.vocab_size,
            self.embed_dim,
            self.base_rank,
            self.expert_rank,
            self.num_experts,
            self.router_dim,
        )

    @staticmethod
    def no_decay_parameters():
        """Parameters whose scale is removed exactly by the forward map."""
        # AdamW decay cannot regularize a key's direction because route()
        # normalizes it. It can only collapse the norm toward the numerical
        # epsilon, so keep keys out of decay while retaining ordinary decay on
        # token factors and every learned projection weight.
        return ("expert_keys",)

    @classmethod
    def structure_from_state(cls, state):
        """Validate a checkpoint state and recover all structural dimensions."""
        if not isinstance(state, dict):
            raise TypeError("Residual-expert state must be a dictionary")
        required = {
            "token_factors",
            "base_proj.weight",
            "base_proj.bias",
            "expert_down_weight",
            "expert_down_bias",
            "expert_up_weight",
            "expert_up_bias",
            "router_proj.weight",
            "router_proj.bias",
            "expert_keys",
            "routing_top_k",
            "routing_temperature",
        }
        missing = required - set(state)
        extra = set(state) - required
        if missing or extra:
            raise ValueError(
                "Residual-expert state has an invalid schema "
                f"(missing={sorted(missing)}, extra={sorted(extra)})"
            )

        z = state["token_factors"]
        base_weight = state["base_proj.weight"]
        base_bias = state["base_proj.bias"]
        down_weight = state["expert_down_weight"]
        down_bias = state["expert_down_bias"]
        up_weight = state["expert_up_weight"]
        up_bias = state["expert_up_bias"]
        router_weight = state["router_proj.weight"]
        router_bias = state["router_proj.bias"]
        keys = state["expert_keys"]
        routing_top_k = state["routing_top_k"]
        routing_temperature = state["routing_temperature"]
        if any(not torch.is_tensor(value) for value in state.values()):
            raise ValueError("Residual-expert state values must all be tensors")
        if z.ndim != 2 or base_weight.ndim != 2 or base_bias.ndim != 1:
            raise ValueError("Residual-expert base factors have invalid ranks")
        vocab_size, base_rank = z.shape
        embed_dim, projected_rank = base_weight.shape
        if projected_rank != base_rank or base_bias.shape != (embed_dim,):
            raise ValueError("Residual-expert base projection shapes do not match")
        if down_weight.ndim != 3:
            raise ValueError("expert_down_weight must be three-dimensional")
        num_experts, expert_rank, down_rank = down_weight.shape
        if down_rank != base_rank or down_bias.shape != (
            num_experts, expert_rank
        ):
            raise ValueError("Residual-expert down-projection shapes do not match")
        if up_weight.shape != (num_experts, embed_dim, expert_rank):
            raise ValueError("Residual-expert up-projection weight shape is invalid")
        if up_bias.shape != (num_experts, embed_dim):
            raise ValueError("Residual-expert up-projection bias shape is invalid")
        if router_weight.ndim != 2:
            raise ValueError("router_proj.weight must be two-dimensional")
        router_dim, router_rank = router_weight.shape
        if router_rank != base_rank or router_bias.shape != (router_dim,):
            raise ValueError("Residual-expert router projection shapes do not match")
        if keys.shape != (num_experts, router_dim):
            raise ValueError("Residual-expert key shape is invalid")
        if routing_top_k.shape != () or routing_top_k.dtype != torch.long:
            raise ValueError("routing_top_k must be a scalar int64 tensor")
        top_k = int(routing_top_k.item())
        if not 1 <= top_k <= num_experts:
            raise ValueError("routing_top_k is outside the valid expert range")
        if routing_temperature.shape != ():
            raise ValueError("routing_temperature must be a scalar tensor")
        temperature = float(routing_temperature.item())
        if not torch.isfinite(torch.tensor(temperature)) or temperature < 1e-4:
            raise ValueError(
                "routing_temperature must be finite and at least 1e-4"
            )
        return {
            "vocab_size": int(vocab_size),
            "embed_dim": int(embed_dim),
            "base_rank": int(base_rank),
            "expert_rank": int(expert_rank),
            "num_experts": int(num_experts),
            "router_dim": int(router_dim),
            "top_k": top_k,
            "router_temperature": temperature,
        }

    def route(self, factors, *, return_dense=True, return_router_probs=False):
        """Return top-k indices/weights, optionally with a dense sparse tensor.

        ``factors`` may have any leading shape and must end in ``base_rank``.
        The dense tensor is used only by training diagnostics and the optional
        load-balancing loss; the embedding and tied head compute from the
        compact top-k representation.
        """
        if factors.shape[-1] != self.base_rank:
            raise ValueError(
                f"expected factor width {self.base_rank}, got "
                f"{factors.shape[-1]}"
            )
        # An explicit epsilon prevents a nearly collapsed BF16 query/key from
        # producing a 1e12-scale normalization derivative (the PyTorch default
        # eps is tuned for FP32 rather than this mixed-precision training path).
        queries = F.normalize(self.router_proj(factors), dim=-1, eps=1e-6)
        keys = F.normalize(self.expert_keys, dim=-1, eps=1e-6)
        scores = (queries @ keys.T) / self.router_temperature
        top_scores, top_indices = torch.topk(
            scores, self.top_k, dim=-1, sorted=True
        )
        top_weights = F.softmax(top_scores, dim=-1)
        theta = None
        if return_dense:
            theta = torch.zeros_like(scores).scatter(
                -1, top_indices, top_weights
            )
        router_probs = F.softmax(scores, dim=-1) if return_router_probs else None
        return theta, top_indices, top_weights, router_probs

    def _embed_factors(
        self,
        factors,
        *,
        top_indices=None,
        top_weights=None,
        return_theta,
    ):
        leading_shape = factors.shape[:-1]
        flat_factors = factors.reshape(-1, self.base_rank)
        if (top_indices is None) != (top_weights is None):
            raise ValueError(
                "top_indices and top_weights must be supplied together"
            )
        if top_indices is None:
            _, top_indices, top_weights, _ = self.route(
                flat_factors, return_dense=False
            )
        else:
            top_indices = top_indices.reshape(-1, self.top_k)
            top_weights = top_weights.reshape(-1, self.top_k)
            expected = (flat_factors.size(0), self.top_k)
            if top_indices.shape != expected or top_weights.shape != expected:
                raise ValueError(
                    "routing tensors must both have shape "
                    f"{expected}, got {tuple(top_indices.shape)} and "
                    f"{tuple(top_weights.shape)}"
                )
        residual = flat_factors.new_zeros(
            flat_factors.size(0), self.embed_dim
        )

        # The loop is over a small fixed number of experts. Each selected token
        # is evaluated exactly top_k times in total. Empty selections still run
        # through zero-row Linears, keeping every expert in the autograd graph
        # when DDP unused-parameter detection is disabled.
        for expert in range(self.num_experts):
            positions = (top_indices == expert).nonzero(as_tuple=False)
            token_positions = positions[:, 0]
            slots = positions[:, 1]
            selected = flat_factors.index_select(0, token_positions)
            latent = F.linear(
                selected,
                self.expert_down_weight[expert],
                self.expert_down_bias[expert],
            )
            contribution = F.linear(
                latent,
                self.expert_up_weight[expert],
                self.expert_up_bias[expert],
            )
            weights = top_weights[token_positions, slots].unsqueeze(-1)
            residual.index_add_(
                0, token_positions, contribution * weights
            )

        output = self.base_proj(flat_factors) + residual
        output = output.view(*leading_shape, self.embed_dim)
        theta = None
        if return_theta:
            theta = flat_factors.new_zeros(
                flat_factors.size(0), self.num_experts
            ).scatter(-1, top_indices, top_weights)
            theta = theta.view(*leading_shape, self.num_experts)
        return output, theta

    def _vocabulary_route(self, *, return_router_probs):
        """Route the full table once for exact finite-precision weight tying."""
        can_cache = not self.training and not torch.is_grad_enabled()
        if (can_cache and not return_router_probs
                and self._inference_route_cache is not None):
            top_indices, top_weights = self._inference_route_cache
            return top_indices, top_weights, None

        _, top_indices, top_weights, router_probs = self.route(
            self.token_factors,
            return_dense=False,
            return_router_probs=return_router_probs,
        )
        if can_cache:
            self._inference_route_cache = (
                top_indices.detach(), top_weights.detach()
            )
        return top_indices, top_weights, router_probs

    def pop_router_aux_loss(self):
        """Return a Switch-style load-balancing loss and clear its graph.

        Hard top-k load is detached, while importance uses the full softmax
        over router scores. Unlike a loss formed from sparse post-top-k
        weights, this gives every expert key a gradient and can recover an
        expert that currently receives no token.
        """
        if self._router_aux is None:
            return None
        router_probs, top_indices = self._router_aux
        self._router_aux = None
        flat_probs = router_probs.reshape(-1, self.num_experts)
        flat_indices = top_indices.reshape(-1, self.top_k)
        hard_load = F.one_hot(
            flat_indices, num_classes=self.num_experts
        ).to(flat_probs.dtype).mean(dim=(0, 1)).detach()
        importance = flat_probs.mean(dim=0)
        return self.num_experts * (hard_load * importance).sum()

    def expert_output_data(self):
        """Return grouped token ids, gates, and down-projected token factors.

        During gradient-enabled training these tensors are rebuilt with their
        full graph. During no-grad evaluation they depend only on fixed model
        parameters, so caching them avoids a vocabulary-wide router and down
        projection at every autoregressive decoding step.
        """
        can_cache = not self.training and not torch.is_grad_enabled()
        if can_cache and self._inference_expert_cache is not None:
            # A normal model forward populated this with the same persistent
            # inference route; the expert cache already incorporates it.
            self._output_route_cache = None
            return self._inference_expert_cache

        if self._output_route_cache is not None:
            top_indices, top_weights = self._output_route_cache
            # The graph is needed for one tied head only. Releasing the module
            # reference here prevents it surviving past backward if a caller
            # retains the model but not the returned logits.
            self._output_route_cache = None
        else:
            top_indices, top_weights, _ = self._vocabulary_route(
                return_router_probs=False
            )
        grouped = []
        for expert in range(self.num_experts):
            positions = (top_indices == expert).nonzero(as_tuple=False)
            token_ids = positions[:, 0]
            slots = positions[:, 1]
            selected_factors = self.token_factors.index_select(0, token_ids)
            token_latent = F.linear(
                selected_factors,
                self.expert_down_weight[expert],
                self.expert_down_bias[expert],
            )
            weights = top_weights[token_ids, slots]
            grouped.append((token_ids, weights, token_latent))

        if can_cache:
            # no_grad already makes these graph-free. Explicit detach documents
            # the invariant and protects it if call context changes later.
            grouped = tuple(
                tuple(tensor.detach() for tensor in expert_data)
                for expert_data in grouped
            )
            self._inference_expert_cache = grouped
        return grouped

    def forward(self, input_ids, doc_mask=None):
        # Route all vocabulary rows once. The exact same top-k indices and
        # weights are gathered for the input and later consumed by the tied
        # output head, eliminating finite-precision route drift between a
        # small batch GEMM and a full-vocabulary GEMM on GPU.
        top_indices, top_weights, router_probs = self._vocabulary_route(
            return_router_probs=torch.is_grad_enabled()
        )
        self._output_route_cache = (top_indices, top_weights)
        flat_ids = input_ids.reshape(-1)
        input_indices = top_indices.index_select(0, flat_ids).view(
            *input_ids.shape, self.top_k
        )
        input_weights = top_weights.index_select(0, flat_ids).view(
            *input_ids.shape, self.top_k
        )
        if router_probs is not None:
            self._router_aux = (
                router_probs.index_select(0, flat_ids),
                input_indices,
            )
        else:
            self._router_aux = None
        factors = self.token_factors[input_ids]
        return self._embed_factors(
            factors,
            top_indices=input_indices,
            top_weights=input_weights,
            return_theta=True,
        )

    def materialize(self, token_ids=None):
        """Materialize selected effective embeddings for tests/analysis only."""
        if token_ids is None:
            token_ids = torch.arange(
                self.vocab_size, device=self.token_factors.device
            )
        flat_ids = token_ids.reshape(-1)
        all_indices, all_weights, _ = self._vocabulary_route(
            return_router_probs=False
        )
        factors = self.token_factors[token_ids]
        embeddings, _ = self._embed_factors(
            factors,
            top_indices=all_indices.index_select(0, flat_ids).view(
                *token_ids.shape, self.top_k
            ),
            top_weights=all_weights.index_select(0, flat_ids).view(
                *token_ids.shape, self.top_k
            ),
            return_theta=False,
        )
        return embeddings
