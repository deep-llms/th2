import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.native_baseline_pipeline import wait_for_download


class DownloadGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root/'data'
        self.model = self.root/'model'
        subset = self.data/'subsets/test'
        subset.mkdir(parents=True)
        self.model.mkdir()
        manifest = subset/'manifest.json'
        manifest.write_text(json.dumps({'shards': {
            'train': [{'files': {'payload': {'bytes': 4}}}], 'validation': []}}))
        marker = self.data/'UPLOAD_COMPLETE.json'
        marker.write_text('{}')
        self.contract = dict(subset='test', files=1,
            manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
            completion_sha256=hashlib.sha256(marker.read_bytes()).hexdigest(),
            tokenizer_files={'tokenizer.json': {'bytes': 2}})
        (self.model/'tokenizer.json').write_text('{}')

    def test_missing_then_partial_then_stable(self):
        ticks = iter([0, 1, 2, 3])
        calls = []
        def advance(seconds):
            calls.append(seconds)
            if len(calls) < 3:
                (self.data/'payload').write_bytes(b'a' if len(calls) == 1 else b'data')
        with patch('scripts.native_baseline_pipeline.time.monotonic', side_effect=lambda: next(ticks)), \
             patch('scripts.native_baseline_pipeline.time.sleep', side_effect=advance), patch('builtins.print'):
            wait_for_download(self.data, self.model, self.contract, 100)
        self.assertEqual(len(calls), 3)

    def test_timeout_refuses_missing_input(self):
        with patch('scripts.native_baseline_pipeline.time.monotonic', side_effect=[0, 2]), \
             patch('scripts.native_baseline_pipeline.time.sleep') as sleep, patch('builtins.print'):
            with self.assertRaises(TimeoutError):
                wait_for_download(self.data, self.model, self.contract, 1)
            sleep.assert_not_called()

    def test_untrusted_manifest_rejected_before_wait(self):
        (self.data/'subsets/test/manifest.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'metadata'):
            wait_for_download(self.data, self.model, self.contract, 100)

    def test_same_size_rewrite_requires_stability(self):
        target = self.data/'payload'
        target.write_bytes(b'data')
        calls = []
        def rewrite_once(seconds):
            calls.append(seconds)
            if len(calls) == 1:
                import os
                stat = target.stat()
                os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        with patch('scripts.native_baseline_pipeline.time.sleep', side_effect=rewrite_once), patch('builtins.print'):
            wait_for_download(self.data, self.model, self.contract, 100)
        self.assertEqual(len(calls), 2)


if __name__ == '__main__':
    unittest.main()
