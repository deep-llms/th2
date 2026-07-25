import math

import torch
from torch.optim import Optimizer


class Yogi(Optimizer):
    """YOGI optimizer with optional per-coordinate L1 proximal step.

    Second-moment update uses an additive sign rule instead of Adam's EMA,
    providing better convergence for sparse gradients.

    The proximal step (soft-threshold + non-negativity clamp) is applied to
    parameters in groups that have ``apply_proximal=True``. Set ``l1_penalty``
    on the optimizer instance before calling ``step()`` to control the
    penalty strength.

    Args:
        params: Iterable of parameter groups. Use ``apply_proximal=True``
            in a group dict to mark parameters for the proximal step.
        lr: Learning rate (default: 1e-2).
        betas: Coefficients for running averages (default: (0.9, 0.999)).
        eps: Term added to denominator for numerical stability (default: 1e-3).
        v_init: Initial value for second-moment estimate (default: 1e-6).
    """

    def __init__(self, params, lr=1e-2, betas=(0.9, 0.999), eps=1e-3, v_init=1e-6):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)
        self.v_init = v_init
        self.l1_penalty = 0.0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            b1, b2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            apply_proximal = group.get("apply_proximal", False)

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.full_like(p, self.v_init)

                m, v = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1

                m.mul_(b1).add_(grad, alpha=1 - b1)

                g2 = grad * grad
                v.add_((v - g2).sign_() * g2, alpha=b2 - 1)

                denom = v.sqrt().add_(eps)
                bc1 = 1 - b1 ** state["step"]
                bc2 = 1 - b2 ** state["step"]
                step_size = lr * math.sqrt(bc2) / bc1

                p.addcdiv_(m, denom, value=-step_size)

                if self.l1_penalty > 0 and apply_proximal:
                    thr = self.l1_penalty * (step_size / denom)
                    p.data.sub_(thr)
                    p.data.clamp_min_(0)

        return loss
