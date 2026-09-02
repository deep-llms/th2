"""Mixture-of-Softmaxes wrapper for exact tied output heads.

The wrapped head remains the sole implementation of the token classifier.
This module only produces multiple context vectors and mixes their normalized
distributions, so it adds no vocabulary-indexed parameters and preserves exact
input/output tying.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class MixtureOfSoftmaxesHead(nn.Module):
    """Wrap ``inner(hidden) -> logits`` in a K-way mixture of softmaxes.

    The result is an fp32 tensor of normalized log probabilities with shape
    ``(*hidden_states.shape[:-1], vocab_size)``. Component zero is the input
    hidden state itself; each additional component uses a factored nonlinear
    context transform.
    """

    def __init__(self, inner, embed_dim, num_components=3,
                 context_rank=256, chunk_size=2048):
        super().__init__()
        self.inner = inner
        self.embed_dim = int(embed_dim)
        self.num_components = int(num_components)
        self.context_rank = int(context_rank)
        self.chunk_size = int(chunk_size)

        if self.embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if self.num_components <= 0:
            raise ValueError("num_components must be positive")
        if self.context_rank <= 0:
            raise ValueError("context_rank must be positive")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        # K=1 is a literal control: no context or prior parameters are created.
        if self.num_components == 1:
            self.register_parameter("context_down", None)
            self.register_parameter("context_up", None)
            self.register_parameter("context_bias", None)
            self.prior = None
        else:
            extra = self.num_components - 1
            self.context_down = nn.Parameter(torch.empty(
                extra, self.context_rank, self.embed_dim
            ))
            self.context_up = nn.Parameter(torch.empty(
                extra, self.embed_dim, self.context_rank
            ))
            self.context_bias = nn.Parameter(torch.zeros(
                extra, self.embed_dim
            ))
            self.prior = nn.Linear(
                self.embed_dim, self.num_components, bias=True
            )
            nn.init.normal_(self.context_down, mean=0.0, std=0.02)
            nn.init.normal_(self.context_up, mean=0.0, std=0.02)
            nn.init.normal_(self.prior.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.prior.bias)

        # Non-persistent, device-following accumulators. Recording outside the
        # checkpointed function prevents backward recomputation from counting a
        # microbatch twice.
        self.register_buffer(
            "_prior_usage_sum",
            torch.zeros(self.num_components, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_prior_entropy_sum", torch.zeros((), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_prior_metric_count", torch.zeros((), dtype=torch.float32),
            persistent=False,
        )
        self._has_step_metrics = False

    @property
    def embed(self):
        """The exact input embedder used by every mixture component."""
        return self.inner.embed

    def _contexts(self, flat_h):
        """Return K contexts with shape ``(K, n, d)``."""
        if flat_h.ndim != 2 or flat_h.shape[1] != self.embed_dim:
            raise ValueError(
                f"expected hidden states shaped (n, {self.embed_dim}), "
                f"got {tuple(flat_h.shape)}"
            )
        if self.num_components == 1:
            return flat_h.unsqueeze(0)
        down = torch.einsum("nd,kcd->knc", flat_h, self.context_down)
        transformed = torch.einsum("knc,kdc->knd", down, self.context_up)
        transformed = torch.tanh(transformed + self.context_bias[:, None, :])
        return torch.cat((flat_h.unsqueeze(0), transformed), dim=0)

    def _chunk_outputs(self, flat_h):
        """Return log probabilities plus detached prior statistics."""
        contexts = self._contexts(flat_h)
        component_logits = self.inner(
            contexts.reshape(-1, self.embed_dim)
        ).reshape(self.num_components, flat_h.shape[0], -1).float()
        component_log_probs = F.log_softmax(component_logits, dim=-1)
        if self.num_components == 1:
            empty = component_log_probs.new_zeros((1,))
            return component_log_probs.squeeze(0), empty, empty.squeeze(0)
        log_prior = F.log_softmax(self.prior(flat_h).float(), dim=-1)
        log_probs = torch.logsumexp(
            component_log_probs + log_prior.T.unsqueeze(-1), dim=0
        )
        with torch.no_grad():
            probabilities = log_prior.detach().exp()
            usage_sum = probabilities.sum(dim=0)
            entropy_sum = -(
                probabilities * log_prior.detach()
            ).sum()
        return log_probs, usage_sum, entropy_sum

    def _chunk_log_probs(self, flat_h):
        """Unchunked helper used by independent references and library callers."""
        return self._chunk_outputs(flat_h)[0]

    @torch.no_grad()
    def _record_prior_metrics(self, usage_sum, entropy_sum, count):
        if self.num_components == 1 or count == 0:
            return
        self._prior_usage_sum.add_(
            usage_sum.to(self._prior_usage_sum.dtype)
        )
        self._prior_entropy_sum.add_(
            entropy_sum.to(self._prior_entropy_sum.dtype)
        )
        self._prior_metric_count.add_(float(count))
        self._has_step_metrics = True

    def forward(self, hidden_states):
        if hidden_states.ndim < 2 or hidden_states.shape[-1] != self.embed_dim:
            raise ValueError(
                f"expected final hidden dimension {self.embed_dim}, got "
                f"{tuple(hidden_states.shape)}"
            )
        leading_shape = hidden_states.shape[:-1]
        flat_h = hidden_states.reshape(-1, self.embed_dim)

        outputs = []
        use_checkpoint = torch.is_grad_enabled() and (
            flat_h.requires_grad
            or any(parameter.requires_grad for parameter in self.parameters())
            or any(parameter.requires_grad for parameter in self.embed.parameters())
        )
        for start in range(0, flat_h.shape[0], self.chunk_size):
            chunk = flat_h[start:start + self.chunk_size]
            if use_checkpoint:
                output, usage_sum, entropy_sum = checkpoint(
                    self._chunk_outputs, chunk, use_reentrant=False
                )
            else:
                output, usage_sum, entropy_sum = self._chunk_outputs(chunk)
            outputs.append(output)
            if self.training:
                self._record_prior_metrics(
                    usage_sum, entropy_sum, chunk.shape[0]
                )

        # A language-model batch cannot normally be empty, but preserving a
        # well-defined empty shape makes this module usable in unit/library code.
        if outputs:
            result = torch.cat(outputs, dim=0)
            vocab_size = result.shape[-1]
        else:
            probe = hidden_states.new_empty((0, self.embed_dim))
            result = self._chunk_outputs(probe)[0]
            vocab_size = result.shape[-1]
        return result.reshape(*leading_shape, vocab_size)

    def pop_step_metrics(self):
        """Return and reset device-local prior sums for trainer aggregation."""
        if self.num_components == 1 or not self._has_step_metrics:
            return None
        count = self._prior_metric_count.detach().clone()
        metrics = {
            f"mos_prior_usage_{index}": (
                self._prior_usage_sum[index].detach().clone(), count.clone()
            )
            for index in range(self.num_components)
        }
        metrics["mos_prior_entropy"] = (
            self._prior_entropy_sum.detach().clone(), count.clone()
        )
        self._prior_usage_sum.zero_()
        self._prior_entropy_sum.zero_()
        self._prior_metric_count.zero_()
        self._has_step_metrics = False
        return metrics
