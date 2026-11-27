"""Map authenticated router row IDs to packed-heal rows, never to documents."""
import io
import json
from pathlib import Path

import numpy as np

from scripts.audit_pilot_exclusion_inventory import audit, read_export, array_ids
from scripts.audit_horizon_exclusion_inventory import describe, sha, ROOT


def main():
    audit()  # Existing pilot and Horizon archive/source/actual-ID gates.
    recovery_path = ROOT/'results/layer_dependency_20261120/e2_order_recovered.json'
    recovery = json.loads(recovery_path.read_text())
    order_path = ROOT/recovery['output']
    ready_path = ROOT/'results/a2_dot_data_20261118/ready_verified.json'
    assert sha(ready_path) == recovery['ready_receipt_sha256']
    ready = json.loads(ready_path.read_text())['ready']['dot_e2']
    stream = ready['streams']['train']
    assert sha(order_path) == stream['order_sha256'] == recovery['actual_sha256']
    order = np.load(order_path, mmap_mode='r', allow_pickle=False)
    assert order.shape == (stream['rows'],) and order.dtype == np.dtype('<i8')
    stem = ready['path']+'/train'
    plan = json.loads((ROOT/'resources/warm_order_data_20261118.json').read_text())
    hp = ROOT/'results/horizon_20261119/export_a01/verified/inputs/complete.json'
    hprep = json.loads(hp.read_text())
    hc = json.loads((hp.parent.parent/'runs/D-s0/config.json').read_text())
    assert plan['stems']['fit'] == hc['eval']['stems']['fit'] == stem
    for pins in (plan['files'], hprep['hashes']):
        assert pins[stem+'.bin'] == stream['token_sha256']
        assert pins[stem+'.starts'] == stream['starts_sha256']
    payloads, pilot_proof = read_export(ROOT/'results/warm_order_20261118')
    selections = {}
    for seed in (0, 1):
        selections[f'pilot_fit_s{seed}'] = array_ids(
            payloads[f'eval/R-s{seed}/fit_dev_costs.npz'], 'fit_ids')
    ep = ROOT/'results/adaptive_refinement_20261119/export_a01/verified/inputs/eval.npz'
    assert sha(ep) == hc['hashes'][hc['eval']['indices']]
    with np.load(ep, allow_pickle=False) as z:
        selections['horizon_fit'] = z['fit'].copy()
        selections['horizon_valid'] = z['valid'].copy()
    arrays, descriptions = {}, {}
    for name, ids in selections.items():
        original = describe(ids)
        if ids.max() >= len(order):
            raise ValueError('Out-of-range destination row')
        packed = np.asarray(order[ids])
        arrays[name+'_e2_row'] = ids
        arrays[name+'_packed_heal_row'] = packed
        descriptions[name] = dict(destination=original, packed=describe(packed))
    arrays['horizon_monitor_e2_row'] = selections['horizon_valid'][:128]
    arrays['horizon_monitor_packed_heal_row'] = order[selections['horizon_valid'][:128]]
    # The monitor is already contained in validation, so do not double-count it.
    destination_union = np.unique(np.concatenate(list(selections.values())))
    packed_union = np.unique(np.concatenate([arrays[k+'_packed_heal_row'] for k in selections]))
    assert len(destination_union) == len(packed_union)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    out = ROOT/'temp/layer_dependency_review/recovered_indices/router_packed_rows.npz'
    if out.exists():
        with np.load(out, allow_pickle=False) as z:
            assert set(z.files) == set(arrays)
            for k, v in arrays.items():
                np.testing.assert_array_equal(z[k], v)
    else:
        with out.open('xb') as f:
            f.write(buf.getvalue())
    print(json.dumps(dict(scope='Authenticated E2 router row -> packed-heal row only',
                          production_ready=False, document_mapping_verified=False,
                          source_sha256=sha(Path(__file__)), order_sha256=sha(order_path),
                          recovery_proof_sha256=sha(recovery_path),
                          pilot_archive_sha256=pilot_proof['archive_sha256'],
                          horizon_eval_sha256=sha(ep), stream=stem,
                          selections=descriptions, monitor_rows=128,
                          unique_selected_packed_rows=len(packed_union),
                          local_output=str(out.relative_to(ROOT)), output_sha256=sha(out),
                          bytes=out.stat().st_size,
                          excluded_from_this_mapping=['Pilot dev uses direct packed tuning, not E2 train.',
                                                      'Pilot/Horizon final sets use direct final streams, not E2 train.',
                                                      'Missing selected training arrays and other historical exposures.'],
                          remaining='Join original starts and49byte text-hash records, then complete historical document union.',
                          flags=['No document independence or overlap inference from unique row IDs.',
                                 'No token/model/remote access; originals unchanged.']), indent=2))


if __name__ == '__main__':
    main()
