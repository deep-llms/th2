"""Local CPU analysis harness. No model loading, training, deployment or network.

Reservation expects upstream whole-document content hashes, not row hashes.
The caller must later prove the production catalogue/exclusion inventory complete.
This prototype refuses CUDA: hardware admission and numerical gates are pending.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch

from .layer_dependence import Plan, PERMUTATIONS, execute, common_input, paired_scores


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _keys(keys):
    if any(not isinstance(k, str) or re.fullmatch('[0-9a-f]{64}', k) is None for k in keys):
        raise ValueError('Require canonical whole-document SHA256 keys')


def reserve(row_ids, doc_keys, excluded_keys, screen_count, confirm_count, seed):
    """Choose one row per content-unique document, lock both splits together."""
    if len(row_ids) != len(doc_keys) or not len(row_ids):
        raise ValueError('Invalid catalogue shape')
    if any(type(i) is not int or i < 0 for i in row_ids) or len(set(row_ids)) != len(row_ids):
        raise ValueError('Rows must be unique nonnegative integer IDs')
    _keys(doc_keys); _keys(excluded_keys)
    if any(type(n) is not int or n <= 0 for n in (screen_count, confirm_count)):
        raise ValueError('Positive document counts required')
    catalogue = sorted(zip(row_ids, doc_keys))
    groups = {}
    excluded = set(excluded_keys)
    for row, key in catalogue:
        if key not in excluded:
            groups.setdefault(key, []).append(row)
    if len(groups) < screen_count+confirm_count:
        raise ValueError('Insufficient independent documents; no silent reuse or shrinking')
    rng = np.random.default_rng(seed)
    keys = sorted(groups)
    chosen = rng.permutation(len(keys))[:screen_count+confirm_count]
    selected = [dict(doc_key=keys[i], row_id=int(rng.choice(groups[keys[i]]))) for i in chosen]
    return dict(schema=1, seed=seed, screen=selected[:screen_count], confirm=selected[screen_count:],
                catalogue_sha256=hashlib.sha256(json_bytes(catalogue)).hexdigest(),
                excluded_sha256=hashlib.sha256(json_bytes(sorted(excluded))).hexdigest(),
                excluded_documents=len(excluded), eligible_documents=len(groups),
                limitation='Hash keys supplied by caller; upstream provenance/completeness not established here')


def cases():
    result = [dict(name='perm-'+''.join(str(a) for a, _ in p.slots), kind='plan',
                   slots=p.slots, family='permutation') for p in PERMUTATIONS]
    for component in ('attention', 'mlp'):
        for j in range(3):
            path = list(range(4)); path[j], path[j+1] = path[j+1], path[j]
            slots = tuple((path[i], i) if component=='attention' else (i, path[i]) for i in range(4))
            result.append(dict(name=f'{component}-swap-{j}-{j+1}', kind='plan', slots=slots, family=component))
    for slot in range(4):
        for candidate in (None, 0, 1, 2, 3):
            result.append(dict(name=f'input-{slot}-block-{candidate}', kind='common',
                               slot=slot, candidate=candidate, family='common_input'))
    return result


def model_sha(model):
    h = hashlib.sha256(json_bytes(asdict(model.c)))
    for name, value in sorted(list(model.named_parameters())+list(model.named_buffers())):
        a = value.detach().contiguous()
        h.update(json_bytes([name, str(a.dtype), list(a.shape)]))
        h.update(a.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def summarize(data, protocol):
    n = data['token_count']
    result = []
    for i, case in enumerate(protocol['cases']):
        count = int(n[i].sum())
        if count <= 0:
            raise ValueError('No scored tokens')
        result.append(dict(name=case['name'], family=case['family'], tokens=count,
                           nll=float(data['nll_sum'][i].sum()/count),
                           reference_nll=float(data['reference_nll_sum'][i].sum()/count),
                           delta_nll=float((data['nll_sum'][i]-data['reference_nll_sum'][i]).sum()/count),
                           kl_reference_to_intervention=float(data['kl_reference_to_intervention_sum'][i].sum()/count),
                           agreement=float(data['agreement_sum'][i].sum()/count),
                           update_rms=float(data['update_rms_sum'][i].sum()/count) if case['kind']=='common' else None))
    return dict(scope='CPU fixture/instrumentation only; no production scientific conclusions', rows=result)


@torch.no_grad()
def run_cpu(model, tokens, row_ids, doc_keys, out, batch=8):
    """Execute the fixed 50-case panel; preserve raw per-document sums locally."""
    if tokens.device.type != 'cpu' or model.embed.weight.device.type != 'cpu':
        raise ValueError('Local CPU prototype only; production GPU admission is not implemented')
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('CPU instrumentation currently requires float32 parameters')
    if type(batch) is not int or batch <= 0 or tokens.ndim != 2 or tokens.dtype != torch.long:
        raise ValueError('Invalid CPU batch')
    if len(tokens) != len(row_ids) or len(row_ids) != len(doc_keys) or len(set(doc_keys)) != len(doc_keys):
        raise ValueError('Exactly one selected row per document required')
    if any(type(i) is not int or i < 0 for i in row_ids) or len(set(row_ids)) != len(row_ids):
        raise ValueError('Invalid selected row IDs')
    _keys(doc_keys)
    # Probe the actual supplied model before creating any output directory.
    reference, _ = execute(model, tokens[:1])
    del reference
    out = Path(out)
    if out.exists():
        raise ValueError('Fresh output required; never overwrite or resume implicitly')
    initial = model_sha(model)
    root = Path(__file__).resolve().parents[2]
    sources = ['warm_order/layer_dependence_pipeline.py', 'warm_order/layer_dependence.py',
               'warm_order/model.py', 'dot_pool/model.py']
    protocol = dict(schema=1, cases=cases(), config=asdict(model.c), model_sha256=initial,
                    token_sha256=hashlib.sha256(tokens.contiguous().numpy().tobytes()).hexdigest(),
                    input_shape=list(tokens.shape),
                    batch=batch, precision='CPU supplied model; float32 logits and float64 metric reduction',
                    torch=torch.__version__, source={p:file_sha(root/'efficient_llm'/p) for p in sources})
    out.mkdir(parents=True)
    (out/'protocol.json').write_bytes(json_bytes(protocol))
    arrays = {name: np.zeros((50, len(tokens)), dtype=np.float64) for name in
              ('nll_sum', 'reference_nll_sum', 'kl_reference_to_intervention_sum', 'agreement_sum', 'update_rms_sum')}
    arrays['token_count'] = np.zeros((50, len(tokens)), dtype=np.int64)
    for start in range(0, len(tokens), batch):
        x = tokens[start:start+batch]
        ref, _ = execute(model, x)
        for i, case in enumerate(protocol['cases']):
            if case['kind']=='plan':
                hidden, _ = execute(model, x, Plan(case['slots']))
            else:
                hidden, info = common_input(model, x, case['slot'], case['candidate'])
                # Same positions whose logits predict scored suffix targets.
                rms = info['update'].double().square().mean(-1).sqrt()
                arrays['update_rms_sum'][i, start:start+len(x)] = rms[:, model.c.prefix-1:-1].sum(1).numpy()
            metrics = paired_scores(model, x, hidden, ref)
            for key, values in metrics.items():
                arrays[key][i, start:start+len(x)] = values.numpy()
    if model_sha(model) != initial:
        raise ValueError('Model changed during frozen analysis')
    arrays['row_ids'] = np.array(row_ids, dtype=np.int64)
    arrays['doc_keys'] = np.array(doc_keys, dtype='<U64')
    np.savez_compressed(out/'metrics.npz', **arrays)
    (out/'report.json').write_bytes(json_bytes(summarize(arrays, protocol)))
    manifest = {p: file_sha(out/p) for p in ('protocol.json', 'metrics.npz', 'report.json')}
    (out/'manifest.json').write_bytes(json_bytes(manifest))
    expected = file_sha(out/'manifest.json')
    replay(out, expected)
    return expected


def replay(out, expected_manifest_sha256):
    out = Path(out)
    if file_sha(out/'manifest.json') != expected_manifest_sha256:
        raise ValueError('Manifest identity mismatch')
    manifest = json.loads((out/'manifest.json').read_text())
    if set(manifest) != {'protocol.json', 'metrics.npz', 'report.json'}:
        raise ValueError('Unexpected members')
    for name, digest in manifest.items():
        if file_sha(out/name) != digest:
            raise ValueError('Changed member: '+name)
    protocol = json.loads((out/'protocol.json').read_text())
    if json_bytes(protocol['cases']) != json_bytes(cases()):
        raise ValueError('Protocol cases mismatch')
    with np.load(out/'metrics.npz', allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    if set(data) != {'nll_sum', 'reference_nll_sum', 'kl_reference_to_intervention_sum', 'agreement_sum', 'update_rms_sum', 'token_count', 'row_ids', 'doc_keys'}:
        raise ValueError('Unexpected metrics schema')
    size = len(data['row_ids'])
    if data['doc_keys'].shape != (size,) or len(set(data['doc_keys'])) != size or len(set(data['row_ids'])) != size:
        raise ValueError('Duplicate/malformed document or row identity')
    _keys(data['doc_keys'].tolist())
    for key in set(data)-{'row_ids', 'doc_keys'}:
        if data[key].shape != (50, size) or not np.isfinite(data[key]).all():
            raise ValueError('Malformed numeric metric')
    if (data['token_count'] <= 0).any() or not np.array_equal(data['token_count'], np.broadcast_to(data['token_count'][0], (50, size))):
        raise ValueError('Token denominators changed across interventions')
    if protocol['input_shape'][0] != size or data['token_count'].dtype != np.int64 or not np.all(data['token_count'] == protocol['input_shape'][1]-protocol['config']['prefix']):
        raise ValueError('Token count disagrees with recorded suffix')
    if (data['nll_sum'] < 0).any() or (data['reference_nll_sum'] < 0).any() or (data['update_rms_sum'] < 0).any():
        raise ValueError('Invalid negative NLL or magnitude')
    if (data['agreement_sum'] < 0).any() or (data['agreement_sum'] > data['token_count']).any():
        raise ValueError('Invalid agreement count')
    if (data['kl_reference_to_intervention_sum'] < -1e-6*data['token_count']).any():
        raise ValueError('Negative KL beyond float32 rounding tolerance')
    if not np.allclose(data['reference_nll_sum'], data['reference_nll_sum'][0], atol=0, rtol=0):
        raise ValueError('Reference changed across interventions')
    np.testing.assert_allclose(data['nll_sum'][0], data['reference_nll_sum'][0], atol=0, rtol=0)
    np.testing.assert_allclose(data['kl_reference_to_intervention_sum'][0], 0, atol=0, rtol=0)
    for i, case in enumerate(protocol['cases']):
        if case['kind']=='common' and case['candidate'] is None:
            np.testing.assert_array_equal(data['update_rms_sum'][i], 0)
    if json_bytes(summarize(data, protocol)) != (out/'report.json').read_bytes():
        raise ValueError('Report aggregation mismatch')
    return dict(status='PASS', cases=50, documents=size, manifest_sha256=expected_manifest_sha256)
