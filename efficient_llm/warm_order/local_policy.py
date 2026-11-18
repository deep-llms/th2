"""Local-swap follow-up primitives; not part of the completed frozen pilot.

The backbone receives only sampled-route CE. Router costs and prefix features
are detached. Policy gradient minimizes expected NLL under the *actual*
exploration mixture, including its derivative, with an action-independent
canonical-cost baseline. No optimizer step may occur between labels and CE.
"""
import torch
from torch import nn

LOCAL_ORDERS = ((0, 1, 2, 3), (1, 0, 2, 3), (0, 2, 1, 3), (0, 1, 3, 2))


class LocalRouter(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 128),
                                 nn.SiLU(), nn.Linear(128, 4))
        # Identical, explicit prior for both trainable heads: 85% canonical.
        with torch.no_grad():
            self.net[-1].weight.zero_()
            self.net[-1].bias.copy_(torch.tensor([.85, .05, .05, .05]).log())

    def forward(self, features):
        return self.net(features.detach().float())


def policy_probabilities(logits, exploration):
    if logits.ndim != 2 or logits.shape[1] != 4:
        raise ValueError('Four actions required')
    if not 0 <= exploration <= 1:
        raise ValueError('Exploration must be in [0,1]')
    if not torch.isfinite(logits).all():
        raise ValueError('Nonfinite logits')
    # Retain float64 for analytic CPU checks; FP32 for lower precision logits.
    z = logits if logits.dtype == torch.float64 else logits.float()
    return (1 - exploration) * z.softmax(-1) + exploration / 4


def sample_actions(probabilities, generator):
    return torch.multinomial(probabilities, 1, generator=generator).squeeze(1)


def policy_gradient_loss(logits, choices, sampled_cost, canonical_cost,
                         exploration):
    """Unbiased score-function gradient of expected cost, not negative reward."""
    probs = policy_probabilities(logits, exploration)
    if choices.dtype != torch.long or choices.shape != (len(logits),):
        raise ValueError('One integer action per example required')
    if (choices < 0).any() or (choices >= 4).any():
        raise ValueError('Invalid action')
    if sampled_cost.shape != choices.shape or canonical_cost.shape != choices.shape:
        raise ValueError('One sampled and baseline cost per example required')
    if not torch.isfinite(sampled_cost).all() or not torch.isfinite(canonical_cost).all():
        raise ValueError('Nonfinite costs')
    advantage = (sampled_cost - canonical_cost).detach()
    logp = probs.gather(1, choices[:, None]).squeeze(1).log()
    return (advantage * logp).mean()


def full_information_loss(logits, costs, temperature=.02):
    if costs.shape != logits.shape or logits.ndim != 2 or logits.shape[1] != 4:
        raise ValueError('Expected batch by four counterfactual costs')
    if temperature <= 0 or not torch.isfinite(costs).all() or not torch.isfinite(logits).all():
        raise ValueError('Invalid labels or temperature')
    target = (-(costs.detach() - costs.detach().amin(-1, keepdim=True)) / temperature).softmax(-1)
    return -(target * logits.log_softmax(-1)).sum(-1).mean()


def routed_nll(model, inputs, choices):
    if choices.dtype != torch.long or choices.shape != (len(inputs),):
        raise ValueError('One integer action per example required')
    if (choices < 0).any() or (choices >= len(LOCAL_ORDERS)).any():
        raise ValueError('Invalid action')
    result = None
    for k, order in enumerate(LOCAL_ORDERS):
        indices = (choices == k).nonzero().flatten()
        if len(indices):
            values = model.nll(inputs[indices], order)
            if result is None:
                result = values.new_zeros(len(inputs))
            result = result.index_copy(0, indices, values)
    if result is None:
        raise ValueError('Empty batch')
    return result
