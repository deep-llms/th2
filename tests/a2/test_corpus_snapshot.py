import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scripts.a2_corpus_snapshot import snapshot
from scripts.a2_prepare_corpora import sha


class SnapshotTests(unittest.TestCase):
    def test_completed_prefix_crosses_shards_and_rejects_tamper_or_gaps(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            tokenizer = root / 'tokenizer'; tokenizer.mkdir()
            (tokenizer / 'tokenizer.json').write_text('fixture identity only')
            token_sha = sha(tokenizer / 'tokenizer.json')
            records = [dict(index=i, local=f'unused-{i}', sha256=str(i)) for i in range(3)]
            plan = dict(root=str(root), length=8, eos_id=0, tokenizer_path=str(tokenizer),
                tokenizer_files={'tokenizer.json':token_sha}, corpora={'dclm':{'files':records}})
            path = root / 'plan.json'; path.write_text(json.dumps(plan))
            for i in range(2):
                part = root / 'parts/dclm' / f'{i:05d}'; part.mkdir(parents=True)
                np.arange(i*10, (i+1)*10, dtype='<u4').tofile(part / 'heal.bin')
                np.ones(10, dtype='u1').tofile(part / 'heal.starts')
                saved = dict(status='complete', source=records[i], tokenizer_sha256=token_sha,
                    streams={'heal':{'tokens':10}}, files={p.name:dict(bytes=p.stat().st_size, sha256=sha(p)) for p in part.iterdir()})
                (part / 'complete.json').write_text(json.dumps(saved))
            before = {str(p):sha(p) for p in (root / 'parts').rglob('*') if p.is_file()}
            result = snapshot(path, 'dclm', root / 'snapshot', 19)
            self.assertEqual(result['streams']['heal']['tokens'], 16)
            np.testing.assert_array_equal(np.fromfile(root / 'snapshot/heal.bin',dtype='<u4'),np.arange(16))
            self.assertTrue(json.loads((root / 'snapshot/complete.json').read_text())['native_reader_passed'])
            snapshot(path, 'dclm', root / 'short', 8)
            for name in ('heal.bin', 'heal.starts'):
                short = (root/'short'/name).read_bytes()
                self.assertEqual(short, (root/'snapshot'/name).read_bytes()[:len(short)])
            self.assertEqual(before,{str(p):sha(p) for p in (root / 'parts').rglob('*') if p.is_file()})
            with self.assertRaises(ValueError):snapshot(path,'dclm',root/'gap',24)
            self.assertFalse((root/'gap').exists())
            (root/'parts/dclm/00000/heal.bin').write_bytes(b'bad')
            with self.assertRaises(ValueError):snapshot(path,'dclm',root/'corrupt',16)
            self.assertFalse((root/'corrupt').exists())
