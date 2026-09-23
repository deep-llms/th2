"""Direct sampler loading, legacy packing equivalence, and matched prefixes."""
from itertools import chain
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_sampling_boundary import HAS_SAMPLING

if HAS_SAMPLING:
    import numpy as np
    from datasets import Dataset, concatenate_datasets, load_from_disk
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast
    from pcc.data import SampledDataLoader, validate_matched_data
    from pcc.probe import verify_slice
    from pcc.protocol import DATA_SEED


def sampled_fixture(root, sharded=True, documents=2003):
    vocab = {"[UNK]": 0, **{str(i): i for i in range(1, 65)}, "[BOS]": 65, "[EOS]": 66}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(single="[BOS] $A [EOS]",
                                                 special_tokens=[("[BOS]", 65), ("[EOS]", 66)])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
                                        bos_token="[BOS]", eos_token="[EOS]")
    texts = [" ".join(str((i * 7 + j) % 60 + 1) for j in range(3 + i % 9)) for i in range(documents)]
    if sharded:
        # Publish out of order to exercise sorted shard loading.
        Dataset.from_dict({"text": texts[701:]}).save_to_disk(root / "shard_0001")
        Dataset.from_dict({"text": texts[:701]}).save_to_disk(root / "shard_0000")
    else:
        Dataset.from_dict({"text": texts}).save_to_disk(root)
    return root, tokenizer, texts


@unittest.skipUnless(HAS_SAMPLING, "Requires the project's local data environment")
class SampledDataTests(unittest.TestCase):
    def test_bounded_training_pool_is_seeded_unique_and_does_not_change_validation(self):
        import hashlib
        with tempfile.TemporaryDirectory() as temp, patch("pcc.data.CONTEXT", 12):
            root = Path(temp)
            path, tokenizer, texts = sampled_fixture(root / "train")
            selected = np.sort(np.random.default_rng(DATA_SEED).choice(len(texts), size=1000, replace=False))
            expected_path = root / "expected"
            Dataset.from_dict({"text": [texts[i] for i in selected]}).save_to_disk(expected_path)
            expected = SampledDataLoader(tokenizer, root / "expected-cache")(expected_path, "train", 120)
            loader = SampledDataLoader(tokenizer, root / "cache", train_documents=1000)
            actual = loader(path, "train", 120)
            np.testing.assert_array_equal(actual.ids, expected.ids)
            self.assertEqual(actual.metadata["document_selection"]["indices_sha256"],
                             hashlib.sha256(selected.astype('<i8').tobytes()).hexdigest())
            again = SampledDataLoader(tokenizer, root / "cache2", train_documents=1000)(path, "train", 120)
            np.testing.assert_array_equal(actual.ids, again.ids)
            dev = loader(path, "dev", 120)
            full = SampledDataLoader(tokenizer, root / "full-cache")(path, "dev", 120)
            np.testing.assert_array_equal(dev.ids, full.ids)
            self.assertNotIn("document_selection", dev.metadata)
            with self.assertRaisesRegex(ValueError, "no resampling"):
                SampledDataLoader(tokenizer, root / "short", train_documents=3000)(path, "train", 120)
            for bad in (True, 0, -1, 1.5):
                with self.assertRaises(ValueError):
                    SampledDataLoader(tokenizer, root / "bad", train_documents=bad)

    def test_matches_legacy_map_shuffle_and_reuses_prefix_without_export(self):
        with tempfile.TemporaryDirectory() as temp, patch("pcc.data.CONTEXT", 12):
            root = Path(temp)
            path, tokenizer, texts = sampled_fixture(root / "train")
            before = {str(p): p.read_bytes() for p in path.rglob("*") if p.is_file()}
            loader = SampledDataLoader(tokenizer, root / "cache")
            train = loader(path, "train", 120)
            screen = loader(path, "train", 48)
            verify_slice(screen, train)
            # Independent implementation of train.py's default map recipe.
            raw = concatenate_datasets([load_from_disk(str(p)) for p in sorted(path.glob("shard_*"))])
            tokenized = raw.map(lambda batch: tokenizer(batch["text"], add_special_tokens=False),
                                batched=True, remove_columns=["text"], keep_in_memory=True)
            def legacy_group(batch):
                concatenated = {key: list(chain(*rows)) for key, rows in batch.items()}
                size = len(concatenated["input_ids"]) // 12 * 12
                return {key: [values[i:i+12] for i in range(0, size, 12)]
                        for key, values in concatenated.items()}
            expected = tokenized.map(legacy_group, batched=True, keep_in_memory=True).shuffle(seed=DATA_SEED)
            np.testing.assert_array_equal(train.ids, expected[:10]["input_ids"])
            self.assertFalse(np.isin(train.ids, [65, 66]).any())
            self.assertEqual(len(loader.packed), 1)
            self.assertEqual(len(list((root / "cache").glob("split-*"))), 1)
            self.assertEqual(before, {str(p): p.read_bytes() for p in path.rglob("*") if p.is_file()})
            again = SampledDataLoader(tokenizer, root / "another-cache")(path, "train", 120)
            np.testing.assert_array_equal(train.ids, again.ids)
            self.assertEqual(train.metadata, again.metadata)
            # A caller mutating a batch/prefix cannot change the shared source.
            screen.ids[:] = 0
            np.testing.assert_array_equal(loader(path, "train", 48).ids, train.ids[:4])
            self.assertFalse(list(root.rglob("*.npz")))

    def test_validation_padding_shortfall_and_split_aliases(self):
        with tempfile.TemporaryDirectory() as temp, patch("pcc.data.CONTEXT", 12):
            root = Path(temp)
            train_path, tokenizer, _ = sampled_fixture(root / "train")
            val_path, _, _ = sampled_fixture(root / "val", sharded=False, documents=10)
            loader = SampledDataLoader(tokenizer, root / "cache")
            train = loader(train_path, "train", 48)
            val = loader(val_path, "dev", 26)
            self.assertEqual(val.valid.sum(), 26)
            self.assertEqual(val.valid[-1].sum(), 2)
            self.assertTrue((val.ids[~val.valid] == 0).all())
            validate_matched_data(train, val)
            with self.assertRaisesRegex(ValueError, "no resampling"):
                loader(val_path, "dev", 120)
            with self.assertRaisesRegex(ValueError, "causal target"):
                loader(val_path, "dev", 25)
            with self.assertRaisesRegex(ValueError, "complete contexts"):
                loader(train_path, "train", 26)
            alias = root / "alias"
            alias.symlink_to(train_path, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "distinct fixed split"):
                validate_matched_data(train, loader(alias, "dev", 48))
            with self.assertRaisesRegex(ValueError, "share dataset shards"):
                validate_matched_data(train, loader(train_path / "shard_0000", "dev", 48))

    def test_insufficient_packed_data_and_wrong_schema_fail(self):
        with tempfile.TemporaryDirectory() as temp, patch("pcc.data.CONTEXT", 12):
            root = Path(temp)
            path, tokenizer, _ = sampled_fixture(root / "val", sharded=False, documents=1)
            loader = SampledDataLoader(tokenizer, root / "cache")
            with self.assertRaisesRegex(ValueError, "no resampling"):
                loader(path, "dev", 24)
            Dataset.from_dict({"input_ids": [[1, 2]]}).save_to_disk(root / "wrong")
            with self.assertRaisesRegex(ValueError, "text-only"):
                loader(root / "wrong", "dev", 24)


if __name__ == "__main__":
    unittest.main()
