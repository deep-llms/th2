"""Optional offline CPU integration tests; require the project's ML environment."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                  TRANSFORMERS_OFFLINE="1", WANDB_MODE="offline", OMP_NUM_THREADS="1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen3Config, set_seed
from train import ModelArguments, build_model

torch.set_num_threads(1)


class BaselineTests(unittest.TestCase):
    def test_offline_hub_requests_cannot_reach_network(self):
        from unittest.mock import patch
        import httpx
        from huggingface_hub import HfApi
        from huggingface_hub.errors import OfflineModeIsEnabled
        from train import BaselineTrainingArguments
        # Ordinary HF defaults; this offline entry point has no publication path.
        args = BaselineTrainingArguments(output_dir=str(self.root/'offline'),
                                         use_cpu=True, report_to='none')
        self.assertFalse(args.push_to_hub)
        self.assertEqual(os.environ['HF_HUB_OFFLINE'], '1')
        self.assertEqual(os.environ['HF_HUB_DISABLE_TELEMETRY'], '1')
        with patch.object(httpx.HTTPTransport, 'handle_request',
                          side_effect=AssertionError('Network transport must not run')) as transport:
            with self.assertRaises(OfflineModeIsEnabled):
                HfApi(token='synthetic-not-a-credential').create_repo('offline-test', repo_type='model')
            transport.assert_not_called()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.assets = self.root / "tiny model"
        self.assets.mkdir()
        tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "[EOS]": 2,
                                        "one": 3, "two": 4, "three": 5, "four": 6}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]",
                                pad_token="[PAD]", eos_token="[EOS]",
                                model_max_length=32).save_pretrained(self.assets)
        Qwen3Config(vocab_size=7, hidden_size=16, intermediate_size=32,
                    num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                    head_dim=8, max_position_embeddings=32, bos_token_id=None,
                    eos_token_id=2, pad_token_id=1).save_pretrained(self.assets)
        self.data = self.root / "data"
        for lang in ("en", "vi/shard_0000"):
            Dataset.from_dict({"text": ["one two three four " * 8] * 4}).save_to_disk(self.data / lang)

    def args(self, out):
        return dict(config_name=str(self.assets), tokenizer_name=str(self.assets),
                    data_dir=str(self.data), output_dir=str(out), use_cpu=True,
                    report_to="none", block_size=8, max_steps=2, save_steps=1,
                    logging_steps=1, per_device_train_batch_size=2,
                    gradient_accumulation_steps=2, learning_rate=0.001,
                    disable_tqdm=True, dataloader_pin_memory=False, seed=42)

    def run_train(self, args, *, cli=False, success=True, accelerate=False):
        if cli:
            argv = [item for k, v in args.items() for item in ("--" + k, str(v))]
        else:
            cfg = self.root / "arguments.json"
            cfg.write_text(json.dumps(args))
            argv = [str(cfg)]
        command = [sys.executable]
        if accelerate:
            command += ["-m", "accelerate.commands.launch", "--config_file",
                        str(ROOT / "resources/accelerate_config.example.yaml"),
                        "--cpu", "--mixed_precision", "no", "--num_processes", "1",
                        "--num_cpu_threads_per_process", "1"]
        result = subprocess.run([*command, str(ROOT / "train.py"), *argv],
                                cwd=self.root, env=os.environ.copy(), capture_output=True,
                                text=True, timeout=120)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def test_scratch_and_pretrained_real_training_save_reload(self):
        set_seed(42)
        initial = build_model(ModelArguments(config_name=str(self.assets)))
        original = {k: v.clone() for k, v in initial.state_dict().items()}
        out = self.root / "scratch"
        self.run_train(self.args(out), accelerate=True)
        state = json.loads((out / "trainer_state.json").read_text())
        self.assertEqual(state["global_step"], 2)
        self.assertTrue(math.isfinite(json.loads((out / "train_results.json").read_text())["train_loss"]))
        for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json", "train_config.json", "data_recipe.json", "training_recipe.json"):
            self.assertTrue((out / "checkpoint-2" / name).is_file(), name)
        recipe = json.loads((out / "data_recipe.json").read_text())
        self.assertEqual(recipe, json.loads((out / "checkpoint-2" / "data_recipe.json").read_text()))
        self.assertEqual(recipe["document_end"]["id"], 2)
        self.assertEqual(recipe["policy"], "document_segments_map_v2")
        loaded = AutoModelForCausalLM.from_pretrained(out, local_files_only=True)
        self.assertTrue(any(not torch.equal(original[k], v) for k, v in loaded.state_dict().items()))
        # Pretrained construction must preserve every saved weight exactly.
        reloaded = build_model(ModelArguments(model_name_or_path=str(out)))
        for k, v in loaded.state_dict().items():
            self.assertTrue(torch.equal(v, reloaded.state_dict()[k]), k)
        loaded.eval()
        reloaded.eval()
        ids = torch.tensor([[3, 4, 5, 6]])
        with torch.no_grad():
            self.assertTrue(torch.equal(loaded(ids).logits, reloaded(ids).logits))
        second = self.root / "continued"
        args = self.args(second)
        del args["config_name"]
        del args["tokenizer_name"]
        args["model_name_or_path"] = str(out)
        args["gradient_checkpointing"] = True
        parquet = self.root / "training.parquet"
        Dataset.from_dict({"text": ["one two three four " * 8] * 8}).to_parquet(parquet)
        args["data_dir"] = str(parquet)
        args["data_cache_dir"] = str(self.root / "hf-cache")
        self.run_train(args, cli=True)
        updated = AutoModelForCausalLM.from_pretrained(second, local_files_only=True)
        self.assertTrue(any(not torch.equal(v, updated.state_dict()[k]) for k, v in loaded.state_dict().items()))
        # Score an actual trained checkpoint on a separate local Parquet split.
        validation = self.root / "validation.parquet"
        Dataset.from_dict({"text": ["four three two one " * 8] * 2}).to_parquet(validation)
        result_file = self.root / "held-out.json"
        result = subprocess.run([sys.executable, str(ROOT / "eval/packed_ppl.py"),
            "--checkpoint", str(second / "checkpoint-2"), "--eval_data_dir", str(validation),
            "--output_file", str(result_file), "--use_cpu", "true", "--batch_size", "2"],
            cwd=self.root, env=os.environ.copy(), capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        held_out = json.loads(result_file.read_text())
        self.assertEqual(held_out["global_step"], 2)
        self.assertGreater(held_out["num_tokens"], 0)
        self.assertTrue(math.isfinite(held_out["perplexity"]))

    def test_json_model_config_and_missing_assets(self):
        model = build_model(ModelArguments(config_name=str(self.assets / "config.json")))
        self.assertEqual(model.config.num_hidden_layers, 2)
        with self.assertRaisesRegex(ValueError, "local"):
            build_model(ModelArguments(config_name="missing-org/missing-model"))

    def test_periodic_validation_cli_resume_receipt_and_training_parity(self):
        validation = self.root / "validation.parquet"
        Dataset.from_dict({"text": ["four three two one " * 8] * 2}).to_parquet(validation)
        plain = self.root / "plain"
        self.run_train(self.args(plain))
        out = self.root / "periodic"
        args = dict(self.args(out), eval_data_dir=str(validation), eval_strategy="steps",
                    eval_steps=1, per_device_eval_batch_size=2, eval_max_sequences=3, stop_after=1)
        self.run_train(args)
        checkpoint = out / "checkpoint-1"
        receipt = json.loads((checkpoint / "training_recipe.json").read_text())
        self.assertEqual(receipt["validation"]["selection"]["selected_sequences"], 3)
        self.assertEqual(receipt["validation"]["data_recipe"]["shuffle_seed"], 0)
        old_receipt = (out / "training_recipe.json").read_bytes()
        wrong = dict(args, stop_after=2, eval_max_sequences=2)
        result = self.run_train(wrong, success=False)
        self.assertIn("validation.selection", result.stderr)
        self.assertEqual(old_receipt, (out / "training_recipe.json").read_bytes())
        args["stop_after"] = 2
        self.run_train(args)
        state = json.loads((out / "trainer_state.json").read_text())
        entries = [x for x in state["log_history"] if "eval_loss" in x]
        self.assertEqual([x["step"] for x in entries], [1, 2])
        self.assertTrue(all(x["eval_num_sequences"] == 3 for x in entries))
        full = AutoModelForCausalLM.from_pretrained(plain, local_files_only=True)
        monitored = AutoModelForCausalLM.from_pretrained(out, local_files_only=True)
        for name, value in full.state_dict().items():
            torch.testing.assert_close(value, monitored.state_dict()[name], atol=0, rtol=0)
        result = self.run_train(dict(self.args(self.root / "overlap"),
            eval_data_dir=str(self.data), eval_strategy="steps", eval_steps=1), success=False)
        self.assertIn("overlaps training", result.stderr)

    def test_cutoff_cli_json_and_automatic_resume_keep_full_schedule(self):
        out = self.root / "cutoff"
        args = self.args(out)
        args.update(max_steps=4, stop_after=2, warmup_ratio=.75, lr_scheduler_type="cosine",
                    save_strategy="no")
        self.run_train(args, cli=True)
        checkpoint = out / "checkpoint-2"
        self.assertTrue((checkpoint / "optimizer.pt").is_file())
        self.assertEqual(json.loads((checkpoint / "trainer_state.json").read_text())["max_steps"], 4)
        config = json.loads((checkpoint / "train_config.json").read_text())
        self.assertEqual(config["training"]["stop_after"], 2)
        self.assertEqual(config["training"]["max_steps"], 4)
        # Rejected resumes must not overwrite the saved recipe or checkpoint.
        receipt = (checkpoint / "training_recipe.json").read_bytes()
        root_receipt = (out / "training_recipe.json").read_bytes()
        wrong = dict(args, stop_after=4, learning_rate=.002)
        result = self.run_train(wrong, success=False)
        self.assertIn("training.learning_rate", result.stderr)
        self.assertEqual(receipt, (checkpoint / "training_recipe.json").read_bytes())
        self.assertEqual(root_receipt, (out / "training_recipe.json").read_bytes())
        optimizer = checkpoint / "optimizer.pt"
        optimizer.rename(checkpoint / "optimizer.backup")
        try:
            result = self.run_train(dict(args, stop_after=4), success=False)
            self.assertIn("Incomplete resume checkpoint", result.stderr)
            self.assertFalse((out / "checkpoint-4").exists())
        finally:
            (checkpoint / "optimizer.backup").rename(optimizer)
        # Same output selects last checkpoint automatically; operational changes allowed.
        args.update(stop_after=4, save_total_limit=3, logging_steps=2)
        self.run_train(args)
        self.assertTrue((out / "checkpoint-4/scheduler.pt").is_file())
        metrics = json.loads((out / "train_results.json").read_text())
        self.assertEqual((metrics["global_step"], metrics["schedule_steps"]), (4, 4))
        reference = self.root / "reference"
        args["output_dir"] = str(reference)
        self.run_train(args)
        resumed = AutoModelForCausalLM.from_pretrained(out, local_files_only=True)
        full = AutoModelForCausalLM.from_pretrained(reference, local_files_only=True)
        for name, value in full.state_dict().items():
            torch.testing.assert_close(value, resumed.state_dict()[name], atol=0, rtol=0, msg=name)

    def test_unvalidated_attention_and_rope_are_rejected(self):
        config = json.loads((self.assets / "config.json").read_text())
        config["rope_parameters"] = {"rope_type": "linear", "factor": 2., "rope_theta": 10000.}
        changed = self.root / "changed.json"
        changed.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "RoPE"):
            build_model(ModelArguments(config_name=str(changed)))
        config["rope_parameters"]["rope_type"] = "default"
        config["layer_types"] = ["full_attention", "sliding_attention"]
        changed.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "full-attention"):
            build_model(ModelArguments(config_name=str(changed)))

    def test_nonempty_output_is_preserved(self):
        out = self.root / "old"
        out.mkdir()
        marker = out / "keep.txt"
        marker.write_text("keep")
        result = self.run_train(self.args(out), success=False)
        self.assertIn("nonempty without a checkpoint", result.stderr)
        self.assertEqual(marker.read_text(), "keep")

    def test_launcher_quotes_paths_and_forwards_arguments(self):
        capture = self.root / "capture.json"
        fake = self.root / "fake-python"
        fake.write_text("#!/usr/bin/env python3\nimport json, os, sys\n"
                        "open(os.environ['CAPTURE'], 'w').write(json.dumps(sys.argv[1:]))\n")
        fake.chmod(0o700)
        script = ROOT / "scripts/train_qwen3_0.6b_baseline.sh"
        subprocess.run(["bash", "-n", str(script)], check=True)
        subprocess.run(["bash", str(script), str(ROOT / "resources/accelerate_config.example.yaml"),
                        str(self.assets), str(self.assets), str(self.data), str(self.root / "out space"),
                        "--max_steps", "2"], check=True,
                       env={**os.environ, "PYTHON_BIN": str(fake), "CAPTURE": str(capture)})
        argv = json.loads(capture.read_text())
        self.assertIn(str(ROOT / "train.py"), argv)
        self.assertEqual(argv[argv.index("--config_name") + 1], str(self.assets))
        self.assertEqual(argv[-2:], ["--max_steps", "2"])
        self.assertNotIn("--no_embhub", argv)
        self.assertEqual(argv[argv.index("--document_end_token") + 1], "<|endoftext|>")


if __name__ == "__main__":
    unittest.main()
