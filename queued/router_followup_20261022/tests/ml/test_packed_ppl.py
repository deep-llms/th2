"""Packed held-out scoring, checkpoint CLI, and exact distributed coverage."""
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from test_document_isolation import model, row, torch
from test_fa4 import reference_kernel
from accelerate import Accelerator
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoTokenizer, PreTrainedTokenizerFast, TrainingArguments
from train import DataArguments
from training_data import prepare_training_data, save_data_recipe
from packed_training import PackedCollator
from eval.packed_ppl import (evaluate_packed, summed_nll, data_arguments,
                             require_held_out, write_result)
import packed_fa4

ROOT = Path(__file__).resolve().parents[2]


class PackedPplTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = [row([3, 4, 31, 8, 9, 10, 31], [0, 0, 0, 1, 1, 1, 1]),
                     row([31] * 7, list(range(7))),
                     row([3, 4, 5, 6, 7, 8, 31], [0] * 7)]
        self.accelerator = Accelerator(cpu=True, mixed_precision="no")

    def fixture(self):
        checkpoint = self.root / "checkpoint"
        checkpoint.mkdir()
        vocab = {"[PAD]": 0, "[UNK]": 1, **{f"t{i}": i for i in range(2, 31)}, "[EOS]": 31}
        backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="[EOS]", pad_token="[PAD]",
            unk_token="[UNK]", model_max_length=32).save_pretrained(checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        model().save_pretrained(checkpoint)
        train = self.root / "train"
        valid = self.root / "validation"
        Dataset.from_dict({"text": ["t3 t4 t5 t6 t7 t8"] * 4}).save_to_disk(train)
        Dataset.from_dict({"text": ["t3 t4", "t6 " * 6, "t8 " * 10, ""]}).save_to_disk(valid)
        prep = TrainingArguments(output_dir=str(self.root / "unused"), use_cpu=True, report_to="none", seed=0)
        data = DataArguments(data_dir=str(train), block_size=7)
        _, recipe = prepare_training_data(data, prep, tokenizer, checkpoint, 7)
        save_data_recipe(checkpoint, recipe)
        # A saved tokenizer has passed through the same preprocessing as training.
        tokenizer.save_pretrained(checkpoint)
        val_data, val_recipe = prepare_training_data(data_arguments(recipe, str(valid)), prep,
                                                     tokenizer, checkpoint, 7)
        self.assertEqual(len(val_data), 3)
        self.accelerator = Accelerator(cpu=True, mixed_precision="no")
        return checkpoint, valid, val_data, recipe, val_recipe

    def test_token_weighted_loss_matches_independent_documents_and_batch_sizes(self):
        m = model().eval()
        reference = 0.
        with torch.inference_mode():
            for ids in ([3, 4, 31], [8, 9, 10, 31], [3, 4, 5, 6, 7, 8, 31]):
                inputs = torch.tensor([ids])
                reference += torch.nn.functional.cross_entropy(m(inputs).logits[0, :-1],
                    inputs[0, 1:], reduction="none").double().sum().item()
        for batch in (1, 2, 3):
            actual = evaluate_packed(m, self.rows, PackedCollator(), self.accelerator, batch)
            self.assertEqual(actual["num_tokens"], 11)
            self.assertEqual(actual["num_sequences"], 3)
            self.assertAlmostEqual(actual["nll_sum"], reference, places=5)
            self.assertAlmostEqual(actual["loss"], reference / 11, places=6)
            self.assertAlmostEqual(actual["perplexity"], math.exp(reference / 11), places=5)
        with self.assertRaisesRegex(ValueError, "no scored targets"):
            evaluate_packed(m, self.rows[1:2], PackedCollator(), self.accelerator)
        with self.assertRaisesRegex(ValueError, "no scored targets"):
            evaluate_packed(m, [], PackedCollator(), self.accelerator)

    def test_fa4_adapter_uses_identical_scored_targets_and_loss(self):
        m = model()
        sdpa = evaluate_packed(m, self.rows, PackedCollator(), self.accelerator, 2)
        with patch.object(packed_fa4, "load_kernel", return_value=reference_kernel), \
             patch.object(packed_fa4, "validate_kernel_inputs"):
            packed_fa4.register_backend()
            m.set_attn_implementation(packed_fa4.BACKEND)
            fa4 = evaluate_packed(m, self.rows, PackedCollator(attention_backend="fa4"), self.accelerator, 2)
        self.assertEqual(sdpa["num_tokens"], fa4["num_tokens"])
        self.assertAlmostEqual(sdpa["loss"], fa4["loss"], places=6)

    def test_chunked_ce_ignores_boundaries_and_nonfinite_results_fail(self):
        torch.manual_seed(10)
        logits = torch.randn(2, 11, 32)
        labels = torch.randint(0, 32, (2, 11))
        labels[:, ::3] = -100
        expected = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 32),
            labels[:, 1:].reshape(-1), ignore_index=-100, reduction="none").double().sum()
        for chunk in (1, 3, 20):
            torch.testing.assert_close(summed_nll(logits, labels, chunk), expected, rtol=0, atol=0)
        m = model()
        with torch.no_grad():
            m.lm_head.weight.fill_(float("nan"))
        with self.assertRaisesRegex(ValueError, "nonfinite loss"):
            evaluate_packed(m, self.rows, PackedCollator(), self.accelerator)

    def test_recipe_boundaries_overlap_and_output_guards(self):
        _, valid, _, train_recipe, val_recipe = self.fixture()
        require_held_out(train_recipe, val_recipe)
        with self.assertRaisesRegex(ValueError, "overlaps training"):
            require_held_out(train_recipe, train_recipe)
        bad = copy.deepcopy(train_recipe)
        bad["positions"] = "unsupported"
        with self.assertRaisesRegex(ValueError, "Unsupported checkpoint"):
            data_arguments(bad, str(valid))
        output = self.root / "result.json"
        write_result(output, {"status": "ok", "loss": 1.})
        with self.assertRaises(FileExistsError):
            write_result(output, {"status": "wrong"})
        self.assertEqual(json.loads(output.read_text())["status"], "ok")
        with self.assertRaises(ValueError):
            write_result(self.root / "nan.json", {"loss": float("nan")})
        self.assertFalse((self.root / "nan.json").exists())

    def run_cli(self, checkpoint, valid, output, *, distributed=False, limit=None, success=True):
        command = [sys.executable]
        if distributed:
            command += ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2"]
        command += [str(ROOT / "eval/packed_ppl.py"), "--checkpoint", str(checkpoint),
            "--eval_data_dir", str(valid), "--output_file", str(output),
            "--use_cpu", "true", "--batch_size", "2"]
        if limit:
            command += ["--max_sequences", str(limit)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120,
                                cwd=self.root, env=os.environ.copy())
        self.assertEqual(result.returncode == 0, success, result.stdout + result.stderr)
        return json.loads(output.read_text()) if success else result

    def test_checkpoint_cli_distributed_uneven_shards_and_empty_rank(self):
        checkpoint, valid, dataset, _, val_recipe = self.fixture()
        single = self.run_cli(checkpoint, valid, self.root / "single.json")
        multi = self.run_cli(checkpoint, valid, self.root / "multi.json", distributed=True)
        self.assertEqual(single["num_sequences"], 3)
        self.assertEqual(single["num_tokens"], val_recipe["scored_target_tokens"])
        self.assertEqual(multi["num_tokens"], single["num_tokens"])
        self.assertEqual(multi["world_size"], 2)
        self.assertEqual(multi["data_recipe"]["shuffled_fingerprint"], single["data_recipe"]["shuffled_fingerprint"])
        self.assertAlmostEqual(single["loss"], multi["loss"], places=6)
        limited = self.run_cli(checkpoint, valid, self.root / "limited.json", distributed=True, limit=1)
        expected = evaluate_packed(model(), dataset.select([0]), PackedCollator(), self.accelerator)
        self.assertEqual(limited["num_sequences"], 1)
        self.assertEqual(limited["num_tokens"], expected["num_tokens"])
        self.assertAlmostEqual(limited["loss"], expected["loss"], places=6)


if __name__ == "__main__":
    unittest.main()
