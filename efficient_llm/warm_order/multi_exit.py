"""Full-depth objective with explicitly accounted shared short-exit supervision."""
import torch


def objective(model, tokens, path, auxiliary_weight):
    """All arms evaluate all exits; zero-weight controls retain the same graph."""
    if len(path) != 4 or sorted(path) != list(range(4)):
        raise ValueError('A complete nonrepeating four-layer order is required')
    if auxiliary_weight not in (0., .2):
        raise ValueError('Unregistered auxiliary weight')
    exits = torch.stack([model.nll(tokens, path[:depth]) for depth in range(1, 5)], 1)
    final = exits[:, -1].mean()
    short = exits[:, :3].mean()
    return final + auxiliary_weight * short, final, short


def block_tokens(tokens):
    """Logical transformer-block token forwards; excludes projection and backward FLOPs."""
    return 18 * tokens.numel()


def path_for(arm, step, total=None):
    from .model import ORDERS
    if arm not in ('C', 'R', 'CD', 'RD'):
        raise ValueError('Unknown registered arm')
    return list(ORDERS[step % len(ORDERS)] if arm in ('R', 'RD') else ORDERS[0])
