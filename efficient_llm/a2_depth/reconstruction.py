"""Bind freshly reconstructed bytes without rewriting historical provenance."""
import json
from pathlib import Path
from efficient_llm.diagnostic_data import sha256
from .common import SPEC_SHA


def rebuilt_manifest(root, historical, receipt_lock):
    """Return the new manifest hash only after the content-equality contract.

    DT6 must still hash physical tokens/boundaries and the actual reader view.
    This validates provenance, not a substitute for that physical verification.
    """
    path = Path(receipt_lock['path'])
    if sha256(path) != receipt_lock['sha256']:
        raise ValueError('Reconstruction receipt changed')
    receipt = json.loads(path.read_text())
    root = Path(root)
    expected = historical['data_manifest']
    actual = json.loads((root/'manifest.json').read_text())
    identity = sha256(root/'manifest.json')
    required = (
        receipt.get('status') == 'verified',
        receipt.get('spec_sha256') == SPEC_SHA,
        receipt.get('original_manifest_sha256') == historical['provenance']['data_sha256'],
        receipt.get('new_manifest_sha256') == identity,
        Path(receipt.get('stream', '')).resolve() == root.resolve(),
        receipt.get('full_and_prefix_content_equal') is True,
        receipt.get('document_order_equal') is True,
        receipt.get('selected_sequences') == 3124992,
        receipt.get('prefix_sequences') == 781248,
        receipt['tokens']['full_sha256'] == expected['streams']['heal']['token_sha256'],
        receipt['boundaries']['full_sha256'] == expected['streams']['heal']['starts_sha256'],
    )
    if not all(required):
        raise ValueError('Reconstruction receipt does not bind the historical stream')
    for key in ('status', 'streams', 'tokenizer_sha256', 'eos_id', 'order',
                'excluded_original_tail_tokens', 'dropped_after_shuffle', 'document_count',
                'documents_sha256', 'document_order_sha256', 'heldout_documents_sha256',
                'document_dtype', 'T7', 'T18'):
        if actual[key] != expected[key]:
            raise ValueError('Reconstructed manifest content differs: ' + key)
    return identity


def arm_order(cfg):
    """Preserve the original default; allow only the explicitly requested rerun order."""
    order = tuple(cfg.get('arm_order', ('P300', 'P1T', 'X72')))
    if order not in (('P300', 'P1T', 'X72'), ('X72', 'P300', 'P1T')):
        raise ValueError('Unregistered C arm coverage/order')
    if order[0] == 'X72' and not cfg.get('arm_order_authorization'):
        raise ValueError('X72-first order requires recorded user authorization')
    return order
