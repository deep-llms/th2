import tempfile
import unittest
import json
from pathlib import Path

import numpy as np

from efficient_llm.dot_pool.data import sha
from scripts.dot_pool_sr_recheck import verify_accuracy, run, check_inputs


class SRCorrectionContract(unittest.TestCase):
    def test_changed_physical_data_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td);data = p / 'data.npy';data.write_bytes(b'original')
            manifest = p / 'manifest.json'
            manifest.write_text(json.dumps({'files': {'data.npy': {'bytes': 8, 'sha256': sha(data)}}}))
            expected = {'inputs': {}, 'data_manifest': str(manifest), 'data_manifest_sha256': sha(manifest)}
            check_inputs(expected)
            data.write_bytes(b'changed!')
            with self.assertRaisesRegex(ValueError, 'member changed'):
                check_inputs(expected)

    def test_all_four_split_masks_must_match_headline_accuracy(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'routes.npz'
            splits = ['id_dev', 'ood_dev', 'id_test', 'ood_test']
            final = {'metrics': {s: {'targets': 4, 'accuracy': .5} for s in splits}}
            values = {s + '_correct': np.array([True, False, True, False]) for s in splits}
            np.savez(p, **values)
            self.assertEqual(len(verify_accuracy(final, p)), 4)
            values['ood_test_correct'][0] = False
            np.savez(p, **values)
            with self.assertRaisesRegex(ValueError, 'ood_test'):
                verify_accuracy(final, p)

    def test_existing_output_refused_before_loading_or_mutating(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td);sentinel = p / 'original';sentinel.write_text('keep')
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                run({}, p)
            self.assertEqual(sentinel.read_text(), 'keep')
