"""Recreate only the E2 row permutation locally; require the recorded byte hash.

No corpus tokens, training, checkpoint loading, network or remote writes.
The ignored output is an index array, not a verified document catalogue.
"""
import hashlib
import io
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    evidence = ROOT/'results/layer_dependency_20261120/provenance_audit.json'
    prior = json.loads(evidence.read_text())
    receipt = prior['prior_data_receipt']
    ready = ROOT/receipt['path']
    assert digest(ready) == receipt['sha256']
    recipe = ROOT/'scripts/a2_dot_rebuild_data.py'
    assert digest(recipe) == prior['local_source_sha256']['scripts/a2_dot_rebuild_data.py']
    train = json.loads(ready.read_text())['ready']['dot_e2']['streams']['train']
    assert train['rows'] == 2_000_000_000//512 == 3_906_250
    assert train['order_sha256'] == prior['e2_original_document_mapping']['order_sha256']
    # Same constructor/seed and np.save serialization as repack_e2; the expected
    # complete-file hash decides whether this runtime actually reproduces it.
    order = np.random.default_rng(0).permutation(train['rows'])
    buffer = io.BytesIO()
    np.save(buffer, order, allow_pickle=False)
    payload = buffer.getvalue()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != train['order_sha256']:
        raise ValueError('Reconstructed permutation does not match; no output written')
    out = ROOT/'temp/layer_dependency_review/recovered_indices/train_order.npy'
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        if digest(out) != actual:
            raise ValueError('Existing output differs; no overwrite')
    else:
        with out.open('xb') as f:
            f.write(payload)
    assert digest(out) == actual
    print(json.dumps(dict(status='LOCAL_E2_ORDER_BYTE_HASH_MATCH',
                          output=str(out.relative_to(ROOT)), bytes=len(payload), rows=len(order),
                          expected_sha256=train['order_sha256'], actual_sha256=actual,
                          numpy_version=np.__version__, ready_receipt_sha256=digest(ready),
                          provenance_record_sha256=digest(evidence), recipe_sha256=digest(recipe),
                          recovery_script_sha256=digest(Path(__file__)),
                          mapping='E2 destination r -> original packed-heal row order[r]',
                          production_ready=False, original_documents_verified=False,
                          missing=['Original packed starts and49byte source-document records',
                                   'Actual Horizon/pilot selected training rows or exact eligible-pool values',
                                   'Complete historical exclusion union and reservation'],
                          limits=['Matches a historical receipt, not fresh node state.',
                                  'No token data reconstructed; no original-document disjointness claim.',
                                  'Index stays in ignored local temp; only this proof is committed.']), indent=2))


if __name__ == '__main__':
    main()
