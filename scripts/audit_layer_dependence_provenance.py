"""Local evidence inventory only; never opens remote paths or loads model weights.

This audits the selected Horizon export, not the complete historical exclusion
union. It deliberately cannot publish a production reservation or READY marker.
"""
import hashlib
import json
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def audit():
    base = ROOT / 'results/horizon_20261119'
    export = base / 'export_a01/verified'
    handoff = json.loads((base / 'node_handoff_verified.json').read_text())
    archive = base / 'export_a01/results.tar.gz'
    if sha(archive) != handoff['archive_sha256']:
        raise ValueError('Historical archive identity mismatch')
    with tarfile.open(archive, 'r:gz') as tar:
        entries = [m for m in tar.getmembers()
                   if m.name.removeprefix('./') == 'export-members.json']
        if len(entries) != 1 or not entries[0].isfile():
            raise ValueError('Missing or ambiguous pinned member manifest')
        manifest = json.load(tar.extractfile(entries[0]))
    checked = {}

    def read_member(name):
        p = export / name
        digest = sha(p)
        if digest != manifest[name]:
            raise ValueError('Historical member identity mismatch: ' + name)
        checked[str(p.relative_to(ROOT))] = digest
        return json.loads(p.read_text())

    prepared = read_member('inputs/complete.json')
    runs = []
    for arm in ('D', 'WL'):
        for seed in (0, 1):
            name = f'{arm}-s{seed}'
            cfg = read_member(f'runs/{name}/config.json')
            recipe = read_member(f'runs/{name}/train/recipe.json')
            if recipe['config'] != cfg['train']:
                raise ValueError('Actual training recipe differs')
            if prepared['files'][name + '.json'] != checked[str(
                    (export / f'runs/{name}/config.json').relative_to(ROOT))]:
                raise ValueError('Prepared config differs')
            rowpath = cfg['train']['indices']
            rowsha = prepared['files'][Path(rowpath).name]
            if recipe['inputs'][rowpath] != rowsha:
                raise ValueError('Training row identity differs')
            runs.append(dict(
                name=name, checkpoint_candidates_only=True,
                training_stem=cfg['train']['data'],
                training_rows=dict(node_path=rowpath, sha256=rowsha,
                                   present_in_this_local_export=False),
                training_steps=cfg['train']['steps'],
                training_batch=cfg['train']['batch'],
                router_and_monitor_indices=cfg['eval']['indices'],
                index_sha256=cfg['hashes'][cfg['eval']['indices']],
                validation_curve_uses_first_n_valid=cfg['train']['probe_size'],
                evaluation_stems=cfg['eval']['stems']))
            if any(Path(x).name == Path(rowpath).name for x in manifest):
                raise ValueError('Update audit: row array now present in export')

    sources = ['scripts/a2_prepare_corpora.py', 'scripts/a2_dot_rebuild_data.py',
               'efficient_llm/warm_order/data.py',
               'efficient_llm/warm_order/evaluate.py',
               'efficient_llm/warm_order/horizon_prepare.py',
               'efficient_llm/warm_order/horizon_train.py',
               'efficient_llm/warm_order/horizon_eval.py']
    ready = ROOT / 'results/a2_dot_data_20261118/ready_verified.json'
    ready_data = json.loads(ready.read_text())
    return dict(
        status='LOCAL_EVIDENCE_AUDIT_ONLY', production_ready=False,
        real_content_catalogue_verified=False, fresh_reservation_created=False,
        scope='Four candidate Horizon D/WL checkpoints; historical exclusion union incomplete',
        archive_sha256=handoff['archive_sha256'], verified_members=checked,
        local_source_sha256={p: sha(ROOT / p) for p in sources},
        prior_data_receipt=dict(path=str(ready.relative_to(ROOT)), sha256=sha(ready),
            original_checked_utc=ready_data['checked_utc'], fresh_node_check=False),
        e2_original_document_mapping=dict(
            node_order_path=ready_data['ready']['dot_e2']['path']+'/train_order.npy',
            order_sha256=ready_data['ready']['dot_e2']['streams']['train']['order_sha256'],
            mapping='destination row r came from packed-heal row order[r]',
            boundary_reset='repack_e2 forcibly sets starts[r,0]=1; cannot recover original document clusters from these starts',
            original_starts_and_document_index_required=True),
        candidates=runs,
        document_index_schema=dict(file='parts/<corpus>/<shard>/documents.bin',
            struct='<Q32sQB', record_bytes=49,
            fields=['source_row', 'exact_utf8_text_sha256', 'tokens_including_EOS', 'group_index'],
            groups=['heal', 'tuning', 'final'],
            partition='int.from_bytes(sha256[:8], big) % 10000: tuning <2, final <10, heal otherwise'),
        mapping_required=[
            'Verify each documents.bin against its original complete.json and pinned source receipt.',
            'Reconstruct each group stream in shard/source-row order from document token lengths.',
            'Apply packed source_prefixes including truncated final documents; never hash a window as a whole document.',
            'Map DOT prefix and any shuffled/extracted windows through their exact provenance; verify starts parity.',
            'Map historical training/router/validation/final/diagnostic row IDs to whole-text hashes.',
            'Union exclusions across corpora and histories, then lock screen and confirmation before scientific inspection.'
        ],
        limitations=[
            'Ordinal cluster_ids are stream-local positions, not content hashes; E2 resets every row start, so these clusters cannot certify original-document separation even within the E2 stream.',
            'The same exact-text hash rule routes exact duplicates to the same group across corpora, but does not deduplicate within a group.',
            'Exact UTF8 hashes do not detect normalized, near-duplicate or semantic overlap.',
            'Previously verified node receipts are historical evidence, not current availability or ownership checks.',
            'No remote access, checkpoint loading, inference, fitting or training was performed.'
        ])


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2, allow_nan=False))
