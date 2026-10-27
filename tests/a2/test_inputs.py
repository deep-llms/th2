import contextlib
import hashlib
import io
import tempfile
import unittest
from pathlib import Path

from scripts.a2_prepare_inputs import verify_inputs


class InputTests(unittest.TestCase):
    def test_pinned_bytes_before_preparation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            model=root/'model'; model.write_bytes(b'weights')
            tokenizer=root/'tokenizer.json'; tokenizer.write_bytes(b'tokenizer')
            manifest=root/'subsets/en/manifest.json'
            manifest.parent.mkdir(parents=True);manifest.write_bytes(b'manifest')
            digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            plan=dict(downloads=[dict(path=str(model),sha256=digest(model))],
                qwen_path=d,tokenizer_path=d,pool_path=d,
                tokenizer_provenance={k:dict(files={'tokenizer.json':digest(tokenizer)}) for k in ('qwen','gpt_neox')},
                pool_contract=dict(subset='en',manifest_sha256=digest(manifest)))
            with contextlib.redirect_stdout(io.StringIO()):
                verified=verify_inputs(plan)
                self.assertEqual(verified[str(model)]['bytes'],7)
                model.write_bytes(b'badfile')
                with self.assertRaisesRegex(ValueError,'input missing or digest mismatch'):
                    verify_inputs(plan)
                model.write_bytes(b'weights');tokenizer.write_bytes(b'bad tokenizer')
                with self.assertRaisesRegex(ValueError,'Tokenizer mismatch'):
                    verify_inputs(plan)
