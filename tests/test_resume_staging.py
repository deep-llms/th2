"""Exercise actual B/F/G continuation after moving a checkpoint to a new queue."""
import json
from pathlib import Path
import tempfile
import unittest
import torch
from safetensors.torch import load_file
from tests.test_train import fixture, invoke
from scripts.stage_deep_kv_resume import stage, digest


class ResumeStagingTests(unittest.TestCase):
    def test_relocated_resume_matches_uninterrupted_and_preserves_original(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = fixture(root)
            sources = {}
            originals = {}
            for arm in 'BFG':
                source = root / ('original-' + arm)
                invoke(root, {**config, 'arm': arm, 'output_dir': str(source), 'stop_after': 2})
                sources[arm] = str(source)
                originals[arm] = {str(p.relative_to(source)): digest(p) for p in source.rglob('*') if p.is_file()}
            destination = root / 'continuation'
            receipt = stage(sources, destination, 2)
            self.assertEqual(set(receipt['arms']), set('BFG'))
            for arm in 'BFG':
                output = destination / arm
                self.assertFalse((output / 'result.json').exists())
                invoke(root, {**config, 'arm': arm, 'output_dir': str(output), 'stop_after': 3,
                              'resume_from_checkpoint': str(output / 'checkpoint-2')})
                full = root / ('full-' + arm)
                invoke(root, {**config, 'arm': arm, 'output_dir': str(full), 'stop_after': 3})
                expected = load_file(full / 'model.safetensors')
                for name, value in load_file(output / 'model.safetensors').items():
                    torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
                source = Path(sources[arm])
                self.assertEqual(originals[arm], {str(p.relative_to(source)): digest(p) for p in source.rglob('*') if p.is_file()})
            with self.assertRaisesRegex(ValueError, 'already exists'):
                stage(sources, destination, 2)
            (Path(sources['G']) / 'checkpoint-2/optimizer.pt').unlink()
            with self.assertRaisesRegex(ValueError, 'Incomplete checkpoint'):
                stage(sources, root / 'broken', 2)
            self.assertFalse((root / 'broken').exists())


if __name__ == '__main__':
    unittest.main()
