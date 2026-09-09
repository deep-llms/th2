"""Fixed-probe gradients; no optimizer steps, clipping, or training-path edits."""
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from eval.diagnostic_data import BUCKETS
from eval.diagnostic_metrics import interfaces, target_losses


def selected_parameters(model):
    return {name: p for name, p in model.named_parameters() if (
        name.startswith(('model.embed_tokens.', 'lm_head.')) or 'norm' in name or
        any(part in name for part in ('.transition.', '.self_attn.q_proj.', '.self_attn.o_proj.',
                                       '.mlp.up_proj.', '.mlp.down_proj.', '.down.', '.up.')))}


def interface_gradients(model):
    """Return row-aligned gradients for each complete vocabulary interface."""
    inp, out = model.get_input_embeddings(), model.get_output_embeddings()
    if getattr(model.config, 'model_type', None) != 'capacity_allocation_qwen3':
        return dict(input=inp.weight.grad, output=out.weight.grad)
    if model.config.interface_type == 'independent':
        return dict(input=inp.embedding.weight.grad, output=out.head.weight.grad)
    input_grads = [inp.shared.weight.grad]
    if inp.input_private is not None:
        input_grads.append(inp.input_private.weight.grad)
    output_grads = [out.shared_head.weight.grad]
    if out.output_private_head is not None:
        output_grads.append(out.output_private_head.weight.grad)
    # The shared parameter's total gradient appears on both sides. These row
    # summaries are descriptive for projected/partial tying; B0's separate-path
    # decomposition below remains the only causal input/output split.
    return dict(input=input_grads[0] if len(input_grads) == 1 else torch.cat(input_grads, dim=1),
                output=output_grads[0] if len(output_grads) == 1 else torch.cat(output_grads, dim=1))


@torch.no_grad()
def tensor_statistics(weight, gradient, eps=1e-12):
    if gradient is None:
        return dict(status='no_gradient', shape=list(weight.shape), elements=weight.numel())
    w, g = weight.detach().float(), gradient.detach().float()
    if g.is_sparse:
        g = g.to_dense()
    if not torch.isfinite(w).all() or not torch.isfinite(g).all():
        raise ValueError('Nonfinite parameter/gradient')
    wn, gn = torch.linalg.vector_norm(w).item(), torch.linalg.vector_norm(g).item()
    wr, gr = wn/math.sqrt(w.numel()), gn/math.sqrt(g.numel())
    return dict(status='ok', shape=list(weight.shape), elements=weight.numel(),
                weight_l2=wn, grad_l2=gn, weight_rms=wr, grad_rms=gr, grad_to_weight=gr/(wr+eps))


@torch.no_grad()
def row_statistics(gradient, mapping):
    if gradient.ndim != 2 or len(mapping) != gradient.shape[0]:
        raise ValueError('Vocabulary-row gradient/mapping mismatch')
    g = gradient.detach().float()
    if g.is_sparse:
        g = g.to_dense()
    rms = (torch.linalg.vector_norm(g, dim=1)/math.sqrt(g.shape[1])).cpu()
    if not torch.isfinite(rms).all():
        raise ValueError('Nonfinite row gradient')
    result = {}
    for i, name in enumerate(BUCKETS):
        values = rms[torch.as_tensor(np.flatnonzero(mapping == i))]
        result[name] = dict(rows=values.numel(),
            mean_row_grad_rms=values.mean().item() if values.numel() else None,
            median_row_grad_rms=torch.quantile(values, .5).item() if values.numel() else None,
            p90_row_grad_rms=torch.quantile(values, .9).item() if values.numel() else None,
            zero_fraction=values.eq(0).float().mean().item() if values.numel() else None)
    return result


@torch.no_grad()
def path_comparison(total, input_grad, output_grad, mapping, tolerance):
    # Chunk float64 dot-products/residuals; never allocate multiple Vxd doubles.
    a, b, t = input_grad.cpu(), output_grad.cpu(), total.cpu()
    accum = np.zeros((len(BUCKETS), 8), dtype=np.float64)
    for start in range(0, len(mapping), 512):
        aa, bb, tt = (x[start:start+512].double() for x in (a, b, t))
        active = aa.ne(0).any(dim=1)
        row_values = torch.stack(((aa*aa).sum(1), (bb*bb).sum(1), (aa*bb).sum(1),
            ((tt-aa-bb)**2).sum(1), (tt*tt).sum(1), active.double(),
            (bb[...]*bb[...]).sum(1)*active, (aa*bb).sum(1)*active), dim=1)
        for i in range(len(BUCKETS)):
            mask = torch.as_tensor(mapping[start:start+512] == i)
            accum[i] += row_values[mask].sum(0).numpy()
    def describe(values, rows):
        ai, bo, dot, residual, total2, active, active_bo, active_dot = values
        denom = math.sqrt(ai*bo)
        active_denom = math.sqrt(ai*active_bo)
        size = rows*a.shape[1]
        return dict(rows=int(rows), active_input_rows=int(active),
            active_input_fraction=float(active/rows) if rows else None,
            input_grad_rms=math.sqrt(ai/size) if size else None,
            output_grad_rms=math.sqrt(bo/size) if size else None,
            cosine=float(np.clip(dot/denom, -1, 1)) if denom else None,
            active_rows_cosine=float(np.clip(active_dot/active_denom, -1, 1)) if active_denom else None,
            additivity_relative_l2=math.sqrt(residual)/max(math.sqrt(total2), 1e-30))
    overall = describe(accum.sum(0), len(mapping))
    if overall['additivity_relative_l2'] > tolerance:
        raise ValueError(f'Tied-gradient additivity failed: {overall}')
    return dict(overall=overall, tolerance=tolerance,
        buckets={name: describe(accum[i], int(np.sum(mapping == i))) for i, name in enumerate(BUCKETS)})


def _logits(model, batch, path):
    inputs = {k: v for k, v in batch.items() if k != 'labels'}
    if path == 'total':
        return model(**inputs, use_cache=False).logits
    embedding = model.get_input_embeddings()
    table = embedding.weight
    if path == 'output':
        ids = inputs.pop('input_ids')
        inputs['inputs_embeds'] = F.embedding(ids, table.detach(), padding_idx=embedding.padding_idx)
    hidden = model.model(**inputs, use_cache=False).last_hidden_state
    return F.linear(hidden, table.detach() if path == 'input' else table)


def probe_gradients(model, datasets, probe_ids, mapping, *, device='cpu', precision='fp32', batch_size=1):
    if precision not in ('fp32', 'bf16') or batch_size < 1:
        raise ValueError('Supported probe precision and positive batch required')
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError('Probe requires a standalone model without existing gradients')
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
        raise ValueError('Offline probe supports single unsharded model only')
    selected = {lang: data.select(probe_ids[lang]) for lang, data in datasets.items()}
    targets = 0
    for data in selected.values():
        for row in data:
            valid = row['labels'][1:].ne(-100)
            if 'attention_mask' in row:
                valid &= row['attention_mask'][1:].bool()
            targets += valid.sum().item()
    if not targets:
        raise ValueError('Empty probe targets')
    was_training = model.training
    table = model.get_input_embeddings().weight
    direct_tied = (getattr(model.config, 'model_type', None) == 'qwen3'
                   and table is model.get_output_embeddings().weight)
    paths = ('total', 'input', 'output') if direct_tied else ('total',)
    grads, report, losses_by_path = {}, {}, {}
    try:
        model.eval()  # Dropout off for the same function on every path.
        for path in paths:
            model.zero_grad(set_to_none=True)
            nll, count = 0., 0
            for data in selected.values():
                for batch in DataLoader(data, batch_size=batch_size, shuffle=False):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    with torch.autocast(device, dtype=torch.bfloat16, enabled=precision == 'bf16'):
                        logits = _logits(model, batch, path)
                    values, _ = target_losses(logits, batch)
                    if not torch.isfinite(values).all():
                        raise ValueError('Nonfinite probe loss')
                    loss = values.sum()/targets  # Aggregate, not average batch gradients.
                    if path == 'total':
                        loss.backward()
                    else:
                        gradient, = torch.autograd.grad(loss, table)
                        if table.grad is None:
                            table.grad = gradient
                        else:
                            table.grad.add_(gradient)
                    nll += values.detach().double().sum().item()
                    count += values.numel()
            if count != targets:
                raise ValueError('Probe target coverage mismatch')
            losses_by_path[path] = nll/targets
            if path == 'total':
                report['parameters'] = {name: tensor_statistics(p, p.grad)
                                        for name, p in selected_parameters(model).items()}
                report['rows'] = {side: row_statistics(gradient, mapping)
                                  for side, gradient in interface_gradients(model).items()}
            if direct_tied:
                grads[path] = table.grad.detach().float().cpu().clone()
        if direct_tied:
            if max(losses_by_path.values())-min(losses_by_path.values()) > (1e-4 if precision == 'fp32' else .02):
                raise ValueError('Detached diagnostic paths changed forward loss')
            report['tied_paths'] = path_comparison(grads['total'], grads['input'], grads['output'], mapping,
                                                  5e-5 if precision == 'fp32' else .03)
            report['tied_path_rows'] = {path: row_statistics(g, mapping) for path, g in grads.items()}
        else:
            status = ('not_applicable_projected_or_partial_tying'
                      if getattr(model.config, 'tie_word_embeddings', False)
                      else 'not_applicable_untied')
            report['tied_paths'] = dict(status=status)
        report.update(probe_ids=probe_ids, scored_targets=targets, nll=losses_by_path,
            precision=precision, master_weights=str(next(model.parameters()).dtype),
            gradient_scaling=False, clipping=False, optimizer_steps=0,
            reduction='single_process_token_mean_over_all_fixed_probe_batches',
            model_mode='eval', interpretation='checkpoint_probe_not_historical_training_gradient')
        return report
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)
