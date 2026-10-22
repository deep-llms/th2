"""Real HF cache/packing tests on local synthetic text, entirely CPU/offline."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                  TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from datasets import Dataset, DatasetDict, load_from_disk
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, TrainingArguments
from train import DataArguments
from training_data import (document_end, load_text, prepare_training_data, tokenize_batch,
                           require_resume_data, save_data_recipe)


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.assets = self.root / "tokenizer"
        vocab = {"[UNK]": 0, "[PAD]": 1, "<|im_end|>": 2, "<|endoftext|>": 3,
                 "one": 4, "two": 5, "three": 6, "four": 7}
        backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
            pad_token="[PAD]", eos_token="<|im_end|>", additional_special_tokens=["<|endoftext|>"],
            model_max_length=64)
        self.tokenizer.save_pretrained(self.assets)
        self.texts = ["one two three", "four", "", "one two", "three four one", "two three", "four one"]
        self.saved = self.root / "saved"
        Dataset.from_dict({"text": self.texts, "extra": list(range(len(self.texts)))}).save_to_disk(self.saved)
        self.args = DataArguments(data_dir=str(self.saved), block_size=4, map_batch_size=3,
                                  document_end_token="<|endoftext|>")
        self.training = TrainingArguments(output_dir=str(self.root / "output"), use_cpu=True, report_to="none", seed=42)

    def prepare(self):
        return prepare_training_data(self.args, self.training, self.tokenizer, self.assets, 4)

    def test_boundary_exact_tokens_empty_literal_and_no_truncation(self):
        self.assertEqual(document_end(self.tokenizer)["id"], 2)
        self.assertEqual(document_end(self.tokenizer, "<|endoftext|>")["id"], 3)
        data = tokenize_batch({"text": ["one two", "", "<|endoftext|>", "one " * 70]}, self.tokenizer, 3)
        self.assertEqual(data["input_ids"][:3], [[4, 5, 3], [3], [3, 3]])
        self.assertEqual(len(data["input_ids"][3]), 71)
        self.assertEqual(data["attention_mask"][2], [1, 1])
        with self.assertRaises(ValueError):
            document_end(self.tokenizer, "unknown")
        with self.assertRaises(ValueError):
            tokenize_batch({"text": [None]}, self.tokenizer, 3)

    def test_saved_and_parquet_order_single_file_and_metadata_columns(self):
        parquet = self.root / "parquet"
        parquet.mkdir()
        Dataset.from_dict({"text": self.texts[:3], "extra": [1, 2, 3]}).to_parquet(parquet / "a.parquet")
        Dataset.from_dict({"text": self.texts[3:], "other": [True] * 4}).to_parquet(parquet / "b.parquet")
        raw, records = load_text(parquet, str(self.root / "cache"))
        self.assertEqual(raw["text"], self.texts)
        self.assertEqual(raw.column_names, ["text"])
        self.assertEqual([Path(p).name for p in records[0]["sources"]], ["a.parquet", "b.parquet"])
        one, _ = load_text(parquet / "a.parquet", str(self.root / "cache"))
        self.assertEqual(one["text"], self.texts[:3])
        saved, _ = load_text(self.saved)
        self.assertEqual(saved[:], raw[:])
        packed_saved, _ = self.prepare()
        self.args.data_dir = str(parquet)
        self.args.data_cache_dir = str(self.root / "cache")
        packed_parquet, _ = self.prepare()
        self.assertEqual(packed_saved[:], packed_parquet[:])
        self.args.data_cache_dir = str(self.root / "another-cache")
        rebuilt_parquet, _ = self.prepare()
        self.assertEqual(packed_parquet[:], rebuilt_parquet[:])

    def test_language_shard_order_and_split_rejection(self):
        root = self.root / "languages"
        for path, text in [("vi/shard_0001", "four"), ("en", "one"), ("vi/shard_0000", "two")]:
            Dataset.from_dict({"text": [text]}).save_to_disk(root / path)
        raw, _ = load_text(root)
        self.assertEqual(raw["text"], ["one", "two", "four"])
        ds = Dataset.from_dict({"text": ["one"]})
        DatasetDict(train=ds, validation=ds).save_to_disk(self.root / "dict")
        with self.assertRaisesRegex(ValueError, "split"):
            load_text(self.root / "dict")
        split_root = self.root / "splits"
        ds.save_to_disk(split_root / "train")
        ds.save_to_disk(split_root / "validation")
        with self.assertRaisesRegex(ValueError, "split"):
            load_text(split_root)
        files = self.root / "mixed"
        files.mkdir()
        ds.to_parquet(files / "train-000.parquet")
        ds.to_parquet(files / "validation-000.parquet")
        with self.assertRaisesRegex(ValueError, "split"):
            load_text(files)

    def test_cache_hit_and_forced_recompute_give_identical_order(self):
        first, recipe = self.prepare()
        with patch("training_data.tokenize_batch", side_effect=AssertionError("cache miss")), \
             patch("training_data.group_texts", side_effect=AssertionError("cache miss")):
            cached, cached_recipe = self.prepare()
        self.assertEqual(first[:], cached[:])
        self.assertEqual(recipe, cached_recipe)
        self.args.overwrite_cache = True
        rebuilt, rebuilt_recipe = self.prepare()
        self.assertEqual(first[:], rebuilt[:])
        self.assertEqual(recipe, rebuilt_recipe)
        self.assertEqual(recipe["scored_target_tokens"], sum(label != -100 for row in first for label in row["labels"]))
        for row in first:
            self.assertEqual(row["labels"][0], -100)
            for i in range(1, 4):
                expected = row["input_ids"][i] if row["segments"][i] == row["segments"][i - 1] else -100
                self.assertEqual(row["labels"][i], expected)

    def test_backend_reuses_data_but_resume_requires_same_backend(self):
        first, sdpa = self.prepare()
        with patch("training_data.tokenize_batch", side_effect=AssertionError("cache miss")), \
             patch("training_data.group_texts", side_effect=AssertionError("cache miss")), \
             patch("importlib.metadata.version", return_value="4.0.0b33"):
            other, fa4 = prepare_training_data(self.args, self.training, self.tokenizer,
                                               self.assets, 4, attention_backend="fa4")
        self.assertEqual(first[:], other[:])
        self.assertEqual(sdpa["shuffled_fingerprint"], fa4["shuffled_fingerprint"])
        self.assertEqual(fa4["attention_backend"], "fa4")
        self.assertEqual(fa4["fa4_runtime"]["version"], "4.0.0b33")
        save_data_recipe(self.root, sdpa)
        with self.assertRaisesRegex(ValueError, "differs"):
            require_resume_data(self.root, fa4)

    def test_workers_and_batch_sizes_match_independent_reference(self):
        fingerprints = set()
        for workers, batch in [(1, 3), (2, 3), (2, 2)]:
            self.args.preprocessing_num_workers = workers
            self.args.map_batch_size = batch
            actual, recipe = self.prepare()
            fingerprints.add(recipe["packed_fingerprint"])
            cached, cached_recipe = self.prepare()
            self.assertEqual(actual[:], cached[:])
            self.assertEqual(recipe, cached_recipe)
            expected, offset = [], 0
            for rank in range(workers):
                count = len(self.texts) // workers + (rank < len(self.texts) % workers)
                texts = self.texts[offset:offset + count]
                offset += count
                for start in range(0, len(texts), batch):
                    flat = []
                    for text in texts[start:start + batch]:
                        flat.extend(self.tokenizer.encode(text, add_special_tokens=False) + [3])
                    expected.extend([flat[i:i + 4] for i in range(0, len(flat) // 4 * 4, 4)])
            # HF shuffle is verified separately; compare packed rows as a multiset.
            self.assertEqual(sorted(actual["input_ids"]), sorted(expected))
        self.assertEqual(len(fingerprints), 3)

    def test_data_seed_and_boundary_change_invalidate_correct_stage(self):
        first, recipe = self.prepare()
        self.training.seed = 123
        self.training.data_seed = 42
        same, same_recipe = self.prepare()
        self.assertEqual(first[:], same[:])
        self.assertEqual(recipe, same_recipe)
        self.training.data_seed = 123
        _, different = self.prepare()
        self.assertEqual(recipe["packed_fingerprint"], different["packed_fingerprint"])
        self.assertNotEqual(recipe["shuffled_fingerprint"], different["shuffled_fingerprint"])
        self.args.document_end_token = "<|im_end|>"
        _, boundary = self.prepare()
        self.assertNotEqual(different["tokenized_fingerprint"], boundary["tokenized_fingerprint"])

    def test_resume_requires_matching_data_receipt(self):
        _, recipe = self.prepare()
        checkpoint = self.root / "checkpoint"
        checkpoint.mkdir()
        with self.assertRaisesRegex(ValueError, "lacks"):
            require_resume_data(checkpoint, recipe)
        save_data_recipe(checkpoint, recipe)
        require_resume_data(checkpoint, recipe)
        changed = dict(recipe, block_size=8)
        with self.assertRaisesRegex(ValueError, "differs"):
            require_resume_data(checkpoint, changed)
        # Input stat changes invalidate even a saved dataset's unchanged HF fingerprint.
        arrow = Path(load_from_disk(self.saved).cache_files[0]["filename"])
        old = arrow.stat()
        os.utime(arrow, ns=(old.st_atime_ns, old.st_mtime_ns + 1_000_000))
        _, altered = self.prepare()
        self.assertNotEqual(recipe["tokenized_fingerprint"], altered["tokenized_fingerprint"])


if __name__ == "__main__":
    unittest.main()
