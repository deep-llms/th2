import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from transformers import AutoModelForCausalLM, TrainingArguments

from capacity_allocation.data import PackedTokens, prepare, sha256, document_keys, hash_fraction
from train_capacity import OrderedTrainer, main as train_main, schedule_budget
from capacity_allocation.modeling import build_model, experiment_config


class ToyTokenizer:
    eos_token_id = 96

    def __len__(self):
        return 97

    def encode(self, text, *, add_special_tokens, truncation):
        assert not add_special_tokens and not truncation
        return [ord(c) % 95 for c in text]


def fixture(root):
    rows = [dict(text=f"Document {i}: test words for a local CPU smoke run.",
                 source="synthetic", url=f"https://example.test/{i}") for i in range(100)]
    rows.append(dict(rows[0], url="https://example.test/duplicate"))
    source = root / "source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    inventory = root / "source.json"
    inventory.write_text(json.dumps(dict(dataset="uonlp/CulturaX", language="en", revision="a"*40,
                        selection_rule="SYNTHETIC unit fixture, not actual CulturaX",
                        shards=[dict(path=source.name, sha256=sha256(source))])))
    data = root / "packs"
    prepare(inventory, ToyTokenizer(), data, sequence_length=8,
            validation_fraction=.2, test_fraction=.2, seed=11)
    return inventory, data


class DataAndTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_hash_sampling_and_split_assignment(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            inventory, _ = fixture(root)
            output = root / "sampled"
            report = prepare(inventory, ToyTokenizer(), output, sequence_length=8,
                             seed=0, sample_fraction=.3,
                             validation_fraction=.2, test_fraction=.2)
            expected = {}
            for line in (root / "source.jsonl").read_text().splitlines():
                _, content, key = document_keys(json.loads(line))
                if hash_fraction(key, "sample:0") < .3:
                    value = hash_fraction(key, "split:0")
                    expected[content] = ("validation" if value < .2 else
                                         "test" if value < .4 else "train")
            con = sqlite3.connect(output / "documents.sqlite")
            actual = dict(con.execute("SELECT content_sha256,split FROM documents"))
            con.close()
            self.assertEqual(actual, expected)
            self.assertEqual(sum(s["documents"] for s in report["splits"].values()), len(expected))
            self.assertLess(len(expected), 100)

    def test_preparation_failure_has_no_completion_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            inventory, _ = fixture(root)
            output = root / "too_small"
            with self.assertRaisesRegex(ValueError, "insufficient packed tokens"):
                prepare(inventory, ToyTokenizer(), output,
                        minimum_tokens={"train": 10**12})
            self.assertFalse((output / "manifest.json").exists())
            source = root / "source.jsonl"
            source.write_text(source.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, "Source checksum mismatch"):
                prepare(inventory, ToyTokenizer(), root / "bad_hash")
            self.assertFalse((root / "bad_hash").exists())

    def test_frozen_packing_and_exact_dedup(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            inventory, data = fixture(root)
            report = json.loads((data / "manifest.json").read_text())
            self.assertEqual(report["exact_duplicates_removed"], 1)
            con = sqlite3.connect(data / "documents.sqlite")
            self.assertEqual(con.execute("SELECT count(*) FROM documents").fetchone()[0], 100)
            for split in ("train", "validation", "test"):
                dataset = PackedTokens(data, split, verify_hash=True)
                values = np.fromfile(data / f"{split}.bin", dtype="<u4")
                self.assertEqual((values == 96).sum(), report["splits"][split]["documents"])
                torch.testing.assert_close(dataset[0]["input_ids"], dataset[0]["labels"])
                for row, offset, length in con.execute(
                        "SELECT row_number,token_offset,tokens FROM documents WHERE split=? ORDER BY token_offset", (split,)):
                    text = json.loads((root / "source.jsonl").read_text().splitlines()[row])["text"]
                    expected = ToyTokenizer().encode(text, add_special_tokens=False, truncation=False) + [96]
                    self.assertEqual(values[offset:offset+length].tolist(), expected)
            con.close()
            second = root / "packs2"
            prepare(inventory, ToyTokenizer(), second, sequence_length=8,
                    validation_fraction=.2, test_fraction=.2, seed=11)
            for split in ("train", "validation", "test"):
                self.assertEqual(sha256(data / f"{split}.bin"), sha256(second / f"{split}.bin"))
            with self.assertRaises(FileExistsError):
                prepare(inventory, ToyTokenizer(), data)
            with (data / "train.bin").open("ab") as handle:
                handle.write(b"bad")
            with self.assertRaises(ValueError):
                PackedTokens(data, "train")

    def test_schedule_is_independent_of_stop(self):
        budget = schedule_budget(1_000_000_000, 4, 8, 8, 2048)
        self.assertEqual(budget["tokens_per_step"], 524288)
        self.assertEqual(budget["steps"], 1908)
        self.assertLess(budget["rounding_extra_tokens"], budget["tokens_per_step"])
        self.assertEqual(budget, schedule_budget(1_000_000_000, 8, 4, 8, 2048))

    def test_trainer_counts_shifted_targets_and_accumulation(self):
        with tempfile.TemporaryDirectory() as folder:
            model = build_model(experiment_config("A128", tiny=True))
            trainer = OrderedTrainer(model=model, args=TrainingArguments(
                output_dir=folder, use_cpu=True, report_to="none", gradient_accumulation_steps=2))
            x = torch.randint(0, 97, (4, 8))
            batches = [dict(input_ids=a, labels=a.clone()) for a in x.chunk(2)]
            count = trainer._get_num_items_in_batch(batches, torch.device("cpu"))
            self.assertEqual(count.item(), 28)
            total_loss = sum(trainer.compute_loss(model, dict(batch), num_items_in_batch=count)
                             for batch in batches)
            reference = model(x, labels=x).loss
            torch.testing.assert_close(total_loss, reference)

    def test_tiny_training_and_output_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, data = fixture(root)
            for arm in ("B0", "A128", "C", "D"):
                output = root / arm
                argv = ["train_capacity.py", "--arm", arm, "--data", str(data), "--output", str(output),
                        "--train-tokens", "128", "--stop-at-step", "2", "--batch-per-device", "2",
                        "--gradient-accumulation", "2", "--learning-rate", ".001", "--warmup-ratio", "0",
                        "--weight-decay", ".1", "--beta2", ".95", "--precision", "fp32",
                        "--eval-steps", "2", "--save-steps", "2", "--logging-steps", "1",
                        "--eval-batch-size", "17", "--cpu", "--tiny", "--gradient-checkpointing"]
                with self.subTest(arm=arm), patch.object(sys, "argv", argv):
                    train_main()
                result = json.loads((output / "result.json").read_text())
                self.assertEqual(result["global_step"], 2)
                self.assertEqual(result["budget"]["steps"], 4)
                self.assertEqual(result["processed_tokens"], 64)
                self.assertEqual(result["scored_targets"], 56)
                self.assertEqual(result["status"], "stopped_at_step")
                self.assertTrue((output / "checkpoint-2" / "optimizer.pt").is_file())
                self.assertTrue((output / "initial_scales.json").is_file())
                restored = AutoModelForCausalLM.from_pretrained(output / "final", local_files_only=True).eval()
                heldout = PackedTokens(data, "validation")
                direct, targets = 0., 0
                with torch.no_grad():
                    for batch in torch.utils.data.DataLoader(heldout, batch_size=17):
                        n = batch["labels"][:, 1:].numel()
                        direct += restored(**batch).loss.item() * n
                        targets += n
                self.assertEqual(result["validation_scored_targets"], targets)
                self.assertAlmostEqual(result["validation_nll"], direct/targets, places=6)
                with patch.object(sys, "argv", argv), self.assertRaises(ValueError):
                    train_main()  # Never overwrite an existing run.
                if arm == "A128":
                    # Simulate an interruption after a checkpoint: preserve the
                    # successful short-run report separately, then exercise the
                    # interrupted-run resume path against an uninterrupted run.
                    (output / "result.json").rename(output / "short_run_result.json")
                    resumed = argv.copy()
                    resumed[resumed.index("--stop-at-step")+1] = "4"
                    resumed += ["--resume", str(output / "checkpoint-2")]
                    with patch.object(sys, "argv", resumed):
                        train_main()
                    reference = argv.copy()
                    reference[reference.index("--stop-at-step")+1] = "4"
                    reference[reference.index("--output")+1] = str(root / "uninterrupted")
                    with patch.object(sys, "argv", reference):
                        train_main()
                    a = AutoModelForCausalLM.from_pretrained(output / "final", local_files_only=True)
                    b = AutoModelForCausalLM.from_pretrained(root / "uninterrupted" / "final", local_files_only=True)
                    for key, value in a.state_dict().items():
                        torch.testing.assert_close(value, b.state_dict()[key], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
