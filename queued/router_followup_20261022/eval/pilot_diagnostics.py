"""Read-only router instrumentation. No training, optimizer or checkpoint writes.

The temporary wrappers call the production select method unchanged. They are
removed even on failure; evaluation RNG is owned by the existing route study.
"""
from contextlib import contextmanager
from dataclasses import replace
from types import MethodType

import numpy as np
import torch


class RouterObserver:
    def __init__(self, model, accelerator, sequences):
        self.model, self.accelerator = model, accelerator
        self.local_sequences = len(range(accelerator.process_index, sequences, accelerator.num_processes))
        self.sequences = sequences
        self.rows = [dict(seen=0, tokens=0, decisions=0, violations=0, gate_errors=0,
                         lse_sum=0., lse_square_sum=0., lse_count=0, gaps=[], biased_gaps=[],
                         saturated=0, saturated_eligible=0, eligible=0)
                     for _ in model.routed_steps]
        self.restore, self.handles = [], []
        self.latest = {}

    def __enter__(self):
        for index, step in enumerate(self.model.routed_steps):
            if step.router is not None:
                def hook(module, inputs, output, index=index):
                    self.latest[index] = output
                self.handles.append(step.router.register_forward_hook(hook))
            original = step.select
            previous = step.__dict__.get('select')
            self.restore.append((step, previous))
            def select(layer, h, context, index=index, original=original):
                row = self.rows[index]
                keep = min(h.shape[0], max(0, self.local_sequences-row['seen']))
                observe = self.model.routing_override is None and keep > 0
                if observe:
                    if layer.options.reuse:
                        allowed = ((torch.arange(layer.options.S, device=h.device)-index).abs()
                                   <= context.reach).expand(h.shape[0]*h.shape[1], -1)
                    else:
                        allowed = getattr(layer, f'allowed_{min(context.reach, layer.options.S-1)}')[context.state]
                self.latest.pop(index, None)
                choice, gate = original(h, context)
                if observe:
                    count = keep*h.shape[1]
                    valid = allowed[:count]
                    row['seen'] += keep
                    row['tokens'] += count
                    row['decisions'] += choice[:count].numel()
                    row['violations'] += int((~valid.gather(1, choice[:count])).sum())
                    row['gate_errors'] += int(gate[:count].ne(1).sum())
                    z = self.latest.get(index)
                    probs = (z[:count].detach().softmax(-1) if z is not None
                             else valid.float()/valid.sum(-1, keepdim=True))
                    selected_p = probs.gather(1, choice[:count, :1]).flatten()
                    eligible = valid.sum(-1) > 1
                    saturated = selected_p > .99
                    row['saturated'] += int(saturated.sum())
                    row['saturated_eligible'] += int((saturated & eligible).sum())
                    row['eligible'] += int(eligible.sum())
                    if z is not None:
                        z = z[:count].detach()
                        multiple = valid.sum(-1) > 1
                        lse = torch.logsumexp(z[multiple], -1).double()
                        row['lse_sum'] += float(lse.sum())
                        row['lse_square_sum'] += float(lse.square().sum())
                        row['lse_count'] += len(lse)
                        selected_scores = (probs if layer.options.select_on == 'probs' else z) + context.beta
                        selected_scores = selected_scores.masked_fill(~valid, -torch.inf)
                        for key, values in [('gaps', z), ('biased_gaps', selected_scores)]:
                            top = values[multiple].topk(2, dim=-1).values
                            row[key].append((top[:, 0]-top[:, 1]).cpu().numpy())
                return choice, gate
            step.select = MethodType(select, step)
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        for step, previous in self.restore:
            if previous is None:
                del step.select
            else:
                step.select = previous
        self.latest.clear()

    def finish(self):
        parts = [self.rows]
        acc = self.accelerator
        if acc.num_processes > 1:
            parts = [None]*acc.num_processes if acc.is_main_process else None
            torch.distributed.gather_object(self.rows, parts, dst=0)
            if not acc.is_main_process:
                return None
        beta = self.model.balance_bias.detach().float().cpu().tolist()
        report = dict(sequences=self.sequences, beta=beta, beta_range=max(beta)-min(beta), steps=[])
        raw_gaps = []
        for step in range(self.model.options.S):
            rows = [part[step] for part in parts]
            sums = {k: sum(r[k] for r in rows) for k in
                    ('seen', 'tokens', 'decisions', 'violations', 'gate_errors', 'lse_sum', 'lse_square_sum', 'lse_count',
                     'saturated', 'saturated_eligible', 'eligible')}
            if sums['seen'] != self.sequences:
                raise RuntimeError('Router diagnostic prefix was not visited exactly once')
            out = dict(step=step, **sums,
                       mean_logsumexp=sums['lse_sum']/sums['lse_count'] if sums['lse_count'] else None,
                       mean_square_logsumexp=sums['lse_square_sum']/sums['lse_count'] if sums['lse_count'] else None)
            for key in ('gaps', 'biased_gaps'):
                arrays = [a for r in rows for a in r[key] if len(a)]
                values = np.concatenate(arrays) if arrays else np.array([])
                out[key] = dict(count=len(values), median=float(np.median(values)) if len(values) else None,
                                p90=float(np.quantile(values, .9)) if len(values) else None)
                if key == 'gaps' and len(values):
                    raw_gaps.append(values)
            report['steps'].append(out)
        values = np.concatenate(raw_gaps) if raw_gaps else np.array([])
        report['raw_gap'] = dict(count=len(values), median=float(np.median(values)) if len(values) else None,
                                 p90=float(np.quantile(values, .9)) if len(values) else None)
        report['window_violations'] = sum(r['violations'] for r in report['steps'])
        report['saturated_share'] = sum(r['saturated'] for r in report['steps']) / max(
            sum(r['tokens'] for r in report['steps']), 1)
        report['saturated_eligible_share'] = sum(r['saturated_eligible'] for r in report['steps']) / max(
            sum(r['eligible'] for r in report['steps']), 1)
        report['select_on'] = self.model.options.select_on
        report['logit_cap'] = self.model.options.logit_cap
        report['nonunit_gates'] = sum(r['gate_errors'] for r in report['steps'])
        report['passed'] = report['window_violations'] == report['nonunit_gates'] == 0
        report['scope'] = 'First frozen validation sequences; only as-trained hard routes. Logit gaps exclude sole-candidate decisions.'
        return report


def router_gradient_check(model, row, *, isolate_documents=True, reset_positions=True, length=128,
                          center_logits=False):
    """Full LM finite difference on one prefix, with routes/denominators frozen.

    Production ST autograd is compared with p(z+eps)/p(z0), including changes
    to downstream router logits. FP32/math SDPA and reference projections avoid
    BF16 quantization defeating finite differences. No weight is updated.
    Also check every router's local gate VJP in FP64 using the actual LM adjoint.
    Optional centering freezes offsets from the original forward, then tests one
    router at a time at uniform allowed logits; others retain original logits.
    ST gates remain one and routes/weights stay fixed;
    this tests the same machinery without saturated, numerically invisible gates.
    It does not measure the checkpoint's original router-gradient magnitudes.
    """
    from packed_training import PackedCollator, model_inputs
    from packed_evaluation import preserve_rng_and_mode
    from torch.nn.attention import sdpa_kernel, SDPBackend
    if model.options.routing != 'learned' or model.options.gate != 'st':
        raise ValueError('Gradient check requires learned ST routing')
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('Finite difference expects original FP32 checkpoint parameters')
    old_options, old_backend = model.options, model.config._attn_implementation
    old_flags = [(p, p.requires_grad) for p in model.parameters()]
    old_fields = {k: getattr(model, k) for k in
                  ('capture_routes', 'last_routes', 'eval_counter', 'eval_seed', 'routing_override', 'evaluation_policy')}
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_select, handles = [], []
    state = dict(mode='baseline', perturb_step=None, epsilon=0.)
    z_values, gates, denominators, offsets = {}, {}, {}, {}
    row = {k: v[:length] for k, v in row.items() if k in ('input_ids','labels','segments','attention_mask')}
    batch = model_inputs(PackedCollator(isolate_documents, reset_positions, 'sdpa')([row]))
    batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
    if not int(batch['labels'][:, 1:].ne(-100).sum()):
        raise ValueError('Gradient-check prefix has no targets')
    try:
        model.options = replace(old_options, expert_backend='reference')
        for layer in model.routed_steps:
            layer.options = model.options
        model.set_attn_implementation('sdpa')
        torch.backends.cuda.matmul.allow_tf32 = False
        for p in model.parameters():
            p.requires_grad_(False)
        for layer in model.routed_steps:
            for p in layer.router.parameters():
                p.requires_grad_(True)
        for index, layer in enumerate(model.routed_steps):
            def hook(module, inputs, z, index=index):
                if state['mode'] == 'baseline':
                    offsets[index] = z.detach().masked_fill(~torch.isfinite(z), 0)
                elif center_logits and state.get('center_step') == index:
                    z = z - offsets[index]
                if state['mode'] == 'numeric' and state['perturb_step'] == index:
                    z = z + state['epsilon']*state['direction']
                z_values[index] = z
                return z
            handles.append(layer.router.register_forward_hook(hook))
            original, previous = layer.select, layer.__dict__.get('select')
            old_select.append((layer, previous))
            def select(this, h, context, index=index, original=original):
                choice, weight = original(h, context)
                z = z_values[index]
                selected = z.log_softmax(-1).gather(1, choice)
                if state['mode'] == 'numeric':
                    weight = (selected-denominators[index]).exp()
                else:
                    denominators[index] = selected.detach().clone()
                gates[index] = weight
                return choice, weight
            layer.select = MethodType(select, layer)
        with preserve_rng_and_mode(model), sdpa_kernel(SDPBackend.MATH), torch.autocast(model.device.type, enabled=False):
            model.eval(); model.capture_routes = True; model.routing_override = None
            model.evaluation_policy = 'target'; model.eval_counter = 0
            with torch.no_grad():
                original_loss = float(model(**batch).loss)
                routes = model.last_routes.clone()
            size = model.options.S
            if not center_logits:
                state['mode'] = 'analytic'
                loss = model(**batch, bdot_forced_routes=routes).loss
                zs, ws = [z_values[i] for i in range(size)], [gates[i] for i in range(size)]
                grads = torch.autograd.grad(loss, zs+ws)
            results = []
            for index in range(size):
                if center_logits:
                    state.update(mode='analytic', center_step=index)
                    loss = model(**batch, bdot_forced_routes=routes).loss
                    z = z_values[index]
                    grad, gate_grad = torch.autograd.grad(loss, (z, gates[index]))
                else:
                    z, grad, gate_grad = zs[index], grads[index], grads[size+index]
                baseline = float(loss.detach())
                if baseline != original_loss:
                    raise RuntimeError('Diagnostic changed the forward LM loss')
                direction = grad.detach().sign().masked_fill(~torch.isfinite(z), 0)
                analytic = float((grad*direction).sum())
                if not analytic > 0 or not torch.isfinite(grad).all():
                    raise RuntimeError('Missing/nonfinite router gradient')
                chosen = routes[index].reshape(-1,1)
                zd, dd = z.detach().double(), direction.double()
                def selected_logp(values):
                    # Unlike subtracting logsumexp, this retains tiny log p
                    # when the chosen probability is almost one.
                    selected = values.gather(1, chosen)
                    other = values.scatter(1, chosen, -torch.inf).logsumexp(-1, keepdim=True)
                    return -torch.nn.functional.softplus(other-selected)
                denom = selected_logp(zd)
                def local(eps):
                    change = torch.expm1(selected_logp(zd+eps*dd)-denom)
                    return float((change*gate_grad.detach().double()).sum())
                local_numeric = (local(1e-4)-local(-1e-4))/2e-4
                numeric = []
                state.update(mode='numeric', perturb_step=index, direction=direction)
                for eps in ((.002, .001) if center_logits else (.02, .01)):
                    values = []
                    for sign in (1, -1):
                        state['epsilon'] = sign*eps
                        with torch.no_grad():
                            values.append(float(model(**batch, bdot_forced_routes=routes).loss))
                        if not torch.equal(routes, model.last_routes):
                            raise RuntimeError('Routes changed in finite difference')
                    value = (values[0]-values[1])/(2*eps)
                    numeric.append(dict(epsilon=eps, derivative=value,
                                        absolute_error=abs(value-analytic), relative_error=abs(value-analytic)/analytic))
                local_pass = local_numeric > 0 and abs(local_numeric-analytic) <= 2e-6 + 2e-4*analytic
                full_pass = all(n['derivative'] > 0 and n['absolute_error'] <= 2e-6+.03*analytic for n in numeric)
                results.append(dict(step=index, analytic=analytic, local_numeric=local_numeric,
                                    local_pass=local_pass, full_model=numeric, passed=local_pass and full_pass))
            return dict(passed=all(r['passed'] for r in results), loss=baseline, steps=results,
                        sequences=1, tokens=len(row['input_ids']), scored_tokens=int(batch['labels'][:,1:].ne(-100).sum()),
                        backend='FP32/math SDPA/reference experts, same checkpoint weights',
                        centered_logits=center_logits,
                        centering_scope='one router at a time; other routers retain original logits' if center_logits else None,
                        direction='sign of analytic logit gradient; all routes and base log-prob denominators fixed',
                        tolerances=dict(full_absolute=2e-6,full_relative=.03,local_absolute=2e-6,local_relative=2e-4))
    finally:
        for h in handles:
            h.remove()
        for layer, previous in old_select:
            if previous is None: del layer.select
            else: layer.select = previous
        for p, flag in old_flags:
            p.requires_grad_(flag)
        model.options = old_options
        for layer in model.routed_steps: layer.options = old_options
        model.set_attn_implementation(old_backend)
        for k,v in old_fields.items(): setattr(model,k,v)
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
