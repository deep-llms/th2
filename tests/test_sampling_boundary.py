"""Exercise the existing sampler only on temporary local synthetic parquet data."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import load_from_disk
    import prepare_data
    from pcc.data import PreparedContexts
    HAS_SAMPLING = True
except ImportError:
    HAS_SAMPLING = False


@unittest.skipUnless(HAS_SAMPLING, "Requires the project's local sampling dependencies")
class SamplingBoundaryTests(unittest.TestCase):
    def test_saved_outputs_are_raw_text_and_eval_budget_is_not_exact(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw" / "en"
            raw.mkdir(parents=True)
            parquet = raw / "synthetic.parquet"
            texts = [f"synthetic document {i}" for i in range(5)]
            pq.write_table(pa.table({"text": texts}), parquet)
            calls = []
            def tokenizer(batch, **kwargs):
                calls.append(kwargs)
                return {"input_ids": [[1] * 7 for _ in batch]}
            args = SimpleNamespace(tokenizer_path="unused-local-tokenizer", tokenizer_name="synthetic/tokenizer",
                local_files_only=True, data_dir=str(root / "sampled"), raw_dir=str(root / "raw"),
                flush_every=1, tokenize_batch_size=2)
            with patch.dict(prepare_data.LANG_CONFIG, {"en": {"target_tokens": 10, "eval_tokens": 10}}), \
                    patch.object(prepare_data.AutoTokenizer, "from_pretrained", return_value=tokenizer) as load, \
                    patch.dict(prepare_data.os.environ), redirect_stdout(io.StringIO()):
                prepare_data.sample_by_token_count(args, {"en": [str(parquet)]})
            load.assert_called_once_with("unused-local-tokenizer", local_files_only=True)
            self.assertTrue(calls)
            self.assertTrue(all(call == {"add_special_tokens": False} for call in calls))
            base = root / "sampled" / "synthetic_tokenizer"
            train = load_from_disk(str(base / "train" / "en" / "shard_0000"))
            evaluation = load_from_disk(str(base / "eval" / "en"))
            self.assertEqual(train.column_names, ["text"])
            self.assertEqual(evaluation.column_names, ["text"])
            self.assertEqual(len(train), 2)  # 14 train tokens cross the 10-token boundary.
            self.assertEqual(len(evaluation), 1)  # Only 7 eval tokens reach the 20-token total.
            self.assertTrue(set(train["text"]).isdisjoint(evaluation["text"]))
            self.assertTrue(set(train["text"]) | set(evaluation["text"]) <= set(texts))
            self.assertFalse((base / "dev").exists())
            self.assertFalse((base / "test").exists())
            with self.assertRaisesRegex(ValueError, "raw-text Arrow"):
                PreparedContexts(base / "train" / "en", "train", 10)


if __name__ == "__main__":
    unittest.main()
