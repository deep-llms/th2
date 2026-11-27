"""CPU-only synthetic counterexample to the historical document-split claim.

Runs the actual E2 repacker and Horizon input preparer on temporary data.
Does not run a model, use a GPU, alter historical data or infer actual overlap.
"""
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from scripts.a2_dot_rebuild_data import repack_e2
from efficient_llm.warm_order.data import Rows, sha
from efficient_llm.warm_order.evaluate import cluster_ids
from efficient_llm.warm_order.horizon_prepare import prepare


def reproduce():
    with tempfile.TemporaryDirectory(prefix='layer-doc-boundary-') as directory:
        root = Path(directory)
        source = root / 'source'
        source.mkdir()
        # One genuine document containing 84 whole 512-token windows.
        tokens = (np.arange(84*512, dtype='<u4') % 31) + 1
        tokens[-1] = 0
        starts = np.zeros(84*512, dtype='u1')
        starts[0] = 1
        tokens.tofile(source / 'heal.bin')
        starts.tofile(source / 'heal.starts')
        destination = root / 'e2'
        repack_e2(source, destination, [('train', 'heal', 84*512)])
        order = np.load(destination / 'train_order.npy', allow_pickle=False)
        rows = Rows(destination / 'train')
        np.testing.assert_array_equal(rows.ids, tokens.reshape(84, 512)[order])
        labels = cluster_ids(rows, np.arange(84))
        assert np.unique(labels).size == 84
        np.testing.assert_array_equal(rows.eligible(), np.arange(84))
        indices = root / 'eval.npz'
        np.savez(indices, fit=np.array([0]), valid=np.array([1]))
        stem = str(destination / 'train')
        training = dict(data=stem, model=dict(length=512), steps=80, batch=1)
        pinned = {p: sha(p) for p in (stem+'.bin', stem+'.starts', str(indices))}
        plan = dict(hashes=pinned, eval=dict(indices=str(indices)),
                    training={'0': training, '1': training})
        output = root / 'horizon-inputs'
        prepare(plan, output)
        done = json.loads((output / 'complete.json').read_text())
        assert done['training_router_document_overlap'] == 0
        selections = {}
        for seed in (0, 1):
            chosen = np.load(output / f'train-s{seed}.npy', allow_pickle=False)
            assert len(chosen) == 80 and int(chosen.min()) >= 2 and int(chosen.max()) < 84
            # Ground truth comes from the original document, not repacked starts.
            source_rows = order[chosen]
            np.testing.assert_array_equal(starts.reshape(84, 512).sum(axis=1), [1]+[0]*83)
            selections[str(seed)] = dict(repacked_rows=chosen.tolist(),
                                          original_rows=source_rows.tolist(),
                                          true_document_ids=[0])
        files = ['scripts/a2_dot_rebuild_data.py', 'scripts/a2_prepare_corpora.py',
                 'efficient_llm/warm_order/data.py', 'efficient_llm/warm_order/evaluate.py',
                 'efficient_llm/warm_order/horizon_prepare.py']
        repo = Path(__file__).resolve().parents[1]
        return dict(
            status='COUNTEREXAMPLE_REPRODUCED', scope='Synthetic CPU fixture only',
            source_sha256={p: hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in files},
            actual_model_loaded=False, training_run=False, remote_access=False,
            original_documents=1, repacked_rows=84,
            original_row_for_repacked_row=order.tolist(),
            cluster_ids_from_repacked_starts=labels.tolist(),
            router_fit_rows=[0], router_valid_rows=[1],
            horizon_training_selections=selections,
            emitted_training_router_document_overlap=done['training_router_document_overlap'],
            true_shared_document_ids_between_train_fit_valid=[0],
            conclusion='Row-start reset destroys original-document identity in this clustering path. Zero reported ordinal-cluster overlap does not prove whole-document exclusion.',
            real_corpus_overlap_count=None,
            required_action='Recover original rows via train_order.npy, then join source whole-text hashes; do not infer original document IDs from repacked starts.'
        )


if __name__ == '__main__':
    print('COUNTEREXAMPLE_JSON ' + json.dumps(reproduce(), sort_keys=True))
