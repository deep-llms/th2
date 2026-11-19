"""Scheduled mixture exposure; uniform flooring itself preserves hard argmax."""
import torch
from .soft_route import Model as SoftModel


def epsilon_at(update):
    if isinstance(update, bool) or not isinstance(update, int) or update < 0:
        raise ValueError('Expected a nonnegative integer update index')
    return .4 * max(0., min(update / 500., 1., (1500 - update) / 500.))


def floor_probabilities(probabilities, epsilon):
    if not 0 <= epsilon <= .4 or probabilities.shape[-1] != 4:
        raise ValueError('Invalid exploration floor')
    return probabilities * (1. - epsilon) + epsilon / 4.


class Model(SoftModel):
    def __init__(self, config, mode='dynamic', canonical_probability=.999):
        super().__init__(config, mode, canonical_probability)
        self.register_buffer('exploration_update', torch.tensor(0, dtype=torch.int64))
        self.use_exploration = False
        self.epsilon = 0.
        self.gate_observations = {}

    def set_training_update(self, update):
        self.epsilon = epsilon_at(update)
        self.exploration_update.fill_(update)
        self.use_exploration = True

    def disable_exploration(self):
        self.use_exploration = False

    def gates(self, h, depth):
        p = super().gates(h, depth)
        if self.training and self.use_exploration:
            q = floor_probabilities(p, self.epsilon)
        else:
            q = p
        self.gate_observations[depth] = (p.detach(), q.detach())
        return q


def load_initial(config):
    from .soft_route import load_initial as parent_loader, ARMS
    from .model import Config
    parent = parent_loader(config)
    model = Model(Config(**config['model']), ARMS[config['arm']], config['canonical_probability'])
    missing, extra = model.load_state_dict(parent.state_dict(), strict=False)
    if missing != ['exploration_update'] or extra:
        raise ValueError('Unexpected exploration initialization schema')
    return model
