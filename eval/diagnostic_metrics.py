"""Token-weighted bucket NLL and centered interface spectra (no model updates)."""
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from eval.diagnostic_data import BUCKETS


def nll_summary(sums, counts):
    def row(total, count):
        loss = float(total/count) if count else None
        return dict(nll_sum=float(total), scored_targets=int(count), nll=loss,
                    ppl=math.exp(loss) if loss is not None and loss < 700 else None)
    return dict(overall=row(sum(sums), sum(counts)),
                buckets={name: row(sums[i], counts[i]) for i, name in enumerate(BUCKETS)})


def target_losses(logits, batch, chunk_tokens=256):
    """Shift once, respect labels/-100 and padding, retain real EOS targets.

    FP32 CE in chunks avoids an additional full FP32 [B,T,V] allocation.
    The returned vector remains differentiable for fixed-probe gradients.
    """
    if chunk_tokens < 1:
        raise ValueError('Positive CE chunk size required')
    labels = batch.get('labels', batch['input_ids'])[:, 1:]
    valid = labels.ne(-100)
    if 'attention_mask' in batch:
        valid = valid & batch['attention_mask'][:, 1:].bool()
    if not valid.any():
        raise ValueError('No valid shifted targets')
    values, targets = [], []
    for b in range(labels.shape[0]):
        for start in range(0, labels.shape[1], chunk_tokens):
            stop = min(start+chunk_tokens, labels.shape[1])
            mask = valid[b, start:stop]
            if mask.any():
                target = labels[b, start:stop][mask]
                losses = F.cross_entropy(logits[b, start:stop][mask].float(), target, reduction='none')
                values.append(losses)
                targets.append(target)
    return torch.cat(values), torch.cat(targets)


@torch.no_grad()
def frequency_nll(model, datasets, mapping, *, device='cpu', precision='fp32', batch_size=1):
    if batch_size < 1 or precision not in ('fp32', 'bf16'):
        raise ValueError('Positive batch and supported precision required')
    lookup = torch.as_tensor(mapping.astype(np.int64), device=device)
    by_language = {}
    aggregate_sum = np.zeros(len(BUCKETS), dtype=np.float64)
    aggregate_count = np.zeros(len(BUCKETS), dtype=np.int64)
    model.eval()
    for lang, data in datasets.items():
        sums, counts = np.zeros(len(BUCKETS)), np.zeros(len(BUCKETS), dtype=np.int64)
        for batch in DataLoader(data, batch_size=batch_size, shuffle=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            inputs = {k: v for k, v in batch.items() if k != 'labels'}
            with torch.autocast(device, dtype=torch.bfloat16, enabled=precision == 'bf16'):
                logits = model(**inputs, use_cache=False).logits
            losses, targets = target_losses(logits, batch)
            if not torch.isfinite(losses).all():
                raise ValueError('Nonfinite target NLL')
            groups = lookup[targets]
            for i in range(len(BUCKETS)):
                mask = groups == i
                sums[i] += losses[mask].sum(dtype=torch.float64).item()
                counts[i] += mask.sum().item()
        if counts.sum() <= 0:
            raise ValueError('No diagnostic evaluation targets')
        by_language[lang] = nll_summary(sums, counts)
        aggregate_sum += sums
        aggregate_count += counts
    return dict(by_language=by_language, token_weighted=nll_summary(aggregate_sum, aggregate_count))


def interfaces(model):
    """Return tables and row-vector adapters in actual body coordinates.

    PyTorch Linear stores [out,in]: input uses weight.T; output uses weight.
    C/D can have different input/output body widths; never pretend both are1024.
    """
    inp, out = model.get_input_embeddings(), model.get_output_embeddings()
    if getattr(model.config, 'model_type', None) == 'capacity_allocation_qwen3':
        return dict(input=(inp.embedding.weight, inp.projection.weight.T),
                    output=(out.head.weight, out.projection.weight))
    if getattr(model.config, 'model_type', None) != 'qwen3' or inp.weight is not out.weight:
        raise ValueError('Expected Stagewise interfaces or truly tied Qwen B0')
    return dict(input=(inp.weight, None), output=(out.weight, None))


@torch.no_grad()
def centered_covariance(table, rows, chunk_rows=2048):
    if table.ndim != 2 or chunk_rows < 1:
        raise ValueError('Matrix and positive chunk size required')
    rows = np.asarray(rows, dtype=np.int64)
    if len(rows) == 0:
        return None
    # Float64 CPU, blockwise centered Welford merge: stable even for large means.
    n, mean = 0, torch.zeros(table.shape[1], dtype=torch.float64)
    moment = torch.zeros((table.shape[1], table.shape[1]), dtype=torch.float64)
    for start in range(0, len(rows), chunk_rows):
        index = torch.as_tensor(rows[start:start+chunk_rows], device=table.device)
        x = table[index].detach().to(device='cpu', dtype=torch.float64)
        if not torch.isfinite(x).all():
            raise ValueError('Nonfinite embedding rows')
        k, center = x.shape[0], x.mean(dim=0)
        delta = center-mean
        moment += (x-center).T @ (x-center) + torch.outer(delta, delta)*(n*k/(n+k))
        mean += delta*(k/(n+k))
        n += k
    return moment/n


def spectrum_report(covariance, rows, rank_ceiling):
    if covariance is None:
        return dict(rows=0, status='empty_subset', rank_ceiling=rank_ceiling)
    covariance = (covariance+covariance.T)/2
    eig = torch.linalg.eigvalsh(covariance.double()).flip(0)
    scale = eig.abs().max().item()
    if eig.min().item() < -max(scale*1e-10, 1e-14):
        raise ValueError('Materially negative covariance eigenvalue')
    eig = eig.clamp_min(0)
    # Remove roundoff-only directions, using float64 dimension-scaled tolerance.
    threshold = scale * torch.finfo(torch.float64).eps * covariance.shape[0] * 10
    eig[eig <= threshold] = 0
    energy = eig.sum().item()
    p = eig/energy if energy else torch.zeros_like(eig)
    cumulative = p.cumsum(0)
    positive = p[p > 0]
    return dict(status='ok' if energy else 'zero_variance', rows=int(rows),
        dimensions=covariance.shape[0], rank_ceiling=int(rank_ceiling),
        centered_rank_ceiling=min(int(rank_ceiling), max(0, int(rows)-1)),
        numerical_rank=int((eig > 0).sum()), covariance_eigenvalues=eig.tolist(),
        covariance_scaled_singular_values=eig.sqrt().tolist(),
        singular_values=(eig*rows).sqrt().tolist(), energy_fraction=p.tolist(),
        cumulative_energy=cumulative.tolist(), total_variance=energy,
        effective_rank=math.exp(-(positive*positive.log()).sum().item()) if energy else 0.,
        stable_rank=energy/eig[0].item() if energy else 0.,
        k90=int(torch.searchsorted(cumulative, .9).item()+1) if energy else 0,
        k95=int(torch.searchsorted(cumulative, .95).item()+1) if energy else 0)


@torch.no_grad()
def embedding_spectra(model, counts, mapping, *, include_buckets=False, chunk_rows=2048):
    subsets = {'seen': np.flatnonzero(counts > 0), 'all': np.arange(len(counts))}
    if include_buckets:
        subsets.update({name: np.flatnonzero(mapping == i) for i, name in enumerate(BUCKETS)})
    reports, reused = {}, {}
    for side, (table, adapter) in interfaces(model).items():
        raw, effective = {}, {}
        for name, rows in subsets.items():
            key = (id(table), name)
            if key not in reused:
                cov = centered_covariance(table, rows, chunk_rows)
                reused[key] = (cov, spectrum_report(cov, len(rows), table.shape[1]))
            cov, raw[name] = reused[key]
            if adapter is None:
                effective[name] = raw[name]
            else:
                a = adapter.detach().to(device='cpu', dtype=torch.float64)
                transformed = a.T @ cov @ a if cov is not None else None
                effective[name] = spectrum_report(transformed, len(rows), min(table.shape[1], a.shape[1]))
        reports[side] = dict(raw=raw, effective=effective, table_shape=list(table.shape),
            body_width=adapter.shape[1] if adapter is not None else table.shape[1])
    return dict(centering='uniform_selected_rows', accumulation='float64_cpu_centered_block_merge',
        effective_method='adapter_transformed_raw_covariance_exact_no_V_by_body_materialization',
        primary_subset='seen', seen_definition='training_count_positive_including_observed_special_tokens',
        interfaces=reports)
