import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from interim_evaluation import stable_checkpoint, state_digest, validate
from interim_watch import checkpoint_for, BAND, ORIGINAL, should_stop


class InterimTests(unittest.TestCase):
    def test_snapshot_is_independent_and_requires_identity(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'last.pt'
            ck={'model':{'w':torch.tensor([2.])},'epoch':3,'args':{
                'model':'band_stem','epochs':450,'train_split':'train','val_frac':.1,
                'split_seed':42,'seed':0,'data_seed':42,'norm':'imagenet'}}
            torch.save(ck,p)
            loaded,raw=stable_checkpoint(p)
            validate(loaded,'band_stem')
            ck['model']['w'].add_(1);torch.save(ck,p)
            self.assertEqual(loaded['model']['w'].item(),2.)
            self.assertNotEqual(raw,p.read_bytes())
            with self.assertRaises(ValueError):validate(loaded,'srresnet')
    def test_state_hash_covers_bn_and_weights(self):
        m=torch.nn.BatchNorm1d(3).eval();before=state_digest(m)
        with torch.no_grad():m(torch.ones(2,3))
        self.assertEqual(before,state_digest(m))
        m.running_mean.add_(1)
        self.assertNotEqual(before,state_digest(m))
    def test_corrupt_snapshot_fails(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'last.pt';p.write_bytes(b'')
            with self.assertRaises(RuntimeError):stable_checkpoint(p,attempts=1)
    def test_stop_marker_and_owner_failure(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            with patch('interim_watch.original_running',return_value=True):
                self.assertFalse(should_stop(p,{},float('inf')))
                (p/'STOP').touch()
                self.assertTrue(should_stop(p,{},float('inf')))
            (p/'STOP').unlink()
            with patch('interim_watch.original_running',return_value=False):
                self.assertTrue(should_stop(p,{},float('inf')))
    def test_paths(self):
        self.assertEqual(checkpoint_for('band_strong',BAND/'study'),BAND/'study/band_strong/last.pt')
        with patch('pathlib.Path.exists',return_value=False):
            self.assertEqual(checkpoint_for('grip_nodown',ORIGINAL/'supervised/run/study').parent.name,'grip_nodown_base')


if __name__=='__main__':unittest.main()
