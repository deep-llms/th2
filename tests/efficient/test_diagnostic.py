"""End-to-end offline scorer and uneven two-rank inference checks."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", HF_HUB_OFFLINE="1",
                  TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from efficient_llm.diagnostic import next_token_sums, paired_summary
from efficient_llm.diagnostic_data import sha256, prepare_probe, choose_documents


class ScoringTests(unittest.TestCase):
    def test_shift_and_chunked_nll(self):
        torch.manual_seed(1)
        logits, tokens = torch.randn(3, 9, 17), torch.randint(0, 17, (3, 10))
        expected = torch.nn.functional.cross_entropy(logits.transpose(1,2), tokens[:, 1:], reduction="none").double().sum(1)
        torch.testing.assert_close(next_token_sums(logits, tokens[:, 1:], chunk=4), expected)
        self.assertEqual(paired_summary([2., 4., 6.], [3., 5., 7.])["paired_document_bootstrap_95ci"], [-1., -1.])

    def test_eos_never_inserted_inside_truncated_prefix(self):
        class Tokenizer:
            def __call__(self, text, **kwargs):
                return {"input_ids": list(map(int, text.split()))[:kwargs["max_length"]]}
        rows, _ = choose_documents([{"text": "1 2 3 4 5 6"}, {"text": "7 8 9 10"}], Tokenizer(),
            count=2, length=4, seed=4, excluded=set(), source="fixture", eod_id=99)
        self.assertEqual(sorted(rows), [[1,2,3,4,5], [7,8,9,10,99]])

    def test_saved_shard_preparation_hash_checks_and_disjointness(self):
        from datasets import Dataset
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); data = root/"data"; model = root/"model"; model.mkdir()
            (model/"weights.safetensors").write_bytes(b"test-only")
            (data/"subsets/test").mkdir(parents=True)
            shards = {}
            for split in ("train", "validation"):
                path = data/"subsets/test"/split/"shard"
                offset = 0 if split == "train" else 20
                Dataset.from_dict({"text": [str(i) for i in range(offset,offset+10)]}).save_to_disk(path)
                files = {str(p.relative_to(data)): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in path.iterdir()}
                shards[split] = [{"files": files}]
            manifest = data/"subsets/test/manifest.json"
            manifest.write_text(json.dumps({"shards": shards}))
            (data/"UPLOAD_COMPLETE.json").write_text('{"state":"complete"}')
            contract = root/"contract.json"
            contract.write_text(json.dumps({"subset": "test", "manifest_sha256": sha256(manifest),
                "completion_sha256": sha256(data/"UPLOAD_COMPLETE.json"), "tokenizer_files": {},
                "model_weights": {"weights.safetensors": {"bytes": 9, "sha256": sha256(model/"weights.safetensors")}},
                "revision": "fixture", "tokenizer_revision": "fixture"}))
            class Tokenizer:
                def __call__(self, text, **kwargs): return {"input_ids": [int(text)+1]*5}
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=Tokenizer()), \
                 patch("training_data.document_end", return_value={"id": 99}):
                report = prepare_probe(data, model, root/"probe", length=4, calibration=3, evaluation=4, contract_path=contract)
                self.assertEqual(report["status"], "passed")
                a, b = [report["documents"][s] for s in ("calibration", "evaluation")]
                self.assertFalse({x["text_sha256"] for x in a} & {x["text_sha256"] for x in b})
                fresh = prepare_probe(data, model, root/"fresh", length=4, calibration=3, evaluation=4,
                    contract_path=contract, exclude_manifest=root/"probe/manifest.json")
                self.assertEqual(fresh['excluded_evaluation_manifest_sha256'], sha256(root/'probe/manifest.json'))
                for key in ('text_sha256', 'tokens_sha256'):
                    self.assertFalse({x[key] for x in b} & {x[key] for x in fresh['documents']['evaluation']})
                recovered = prepare_probe(data, model, root/'recovery', length=4,
                    calibration=3, evaluation=4, contract_path=contract, seed=19,
                    recovery_from=root/'probe')
                self.assertTrue(recovered['evaluation_reused'])
                self.assertEqual(recovered['documents']['evaluation'], b)
                before=torch.load(root/'probe/tokens.pt',weights_only=True)
                after=torch.load(root/'recovery/tokens.pt',weights_only=True)
                self.assertTrue(torch.equal(before['evaluation'],after['evaluation']))
                for key in ('text_sha256','tokens_sha256'):
                    used={x[key] for x in a+b}
                    self.assertFalse(used & {x[key] for x in recovered['documents']['calibration']})
                with self.assertRaisesRegex(ValueError,'mutually exclusive'):
                    prepare_probe(data,model,root/'both',length=4,calibration=3,evaluation=4,
                        contract_path=contract,recovery_from=root/'probe',
                        exclude_manifest=root/'probe/manifest.json')
                next((data/"subsets/test/train/shard").glob("*.arrow")).write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "checksum"):
                    prepare_probe(data, model, root/"corrupt", length=4, calibration=3, evaluation=4, contract_path=contract)

    def test_cli_serial_and_two_rank_agree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); model = root/"model"; probe = root/"probe"; probe.mkdir()
            torch.manual_seed(9)
            config = Qwen3Config(vocab_size=47, hidden_size=32, intermediate_size=64,
                num_hidden_layers=2, num_attention_heads=8, num_key_value_heads=4, head_dim=8,
                max_position_embeddings=64, attention_dropout=0., attention_bias=True)
            Qwen3ForCausalLM(config).save_pretrained(model)
            tokens = {"calibration": torch.randint(0,47,(5,9)), "evaluation": torch.randint(0,47,(7,9))}
            torch.save(tokens, probe/"tokens.pt")
            (probe/"manifest.json").write_text(json.dumps({"status":"passed", "length":8, "seed":42,
                "calibration_documents":5, "evaluation_documents":7, "tokens_sha256":sha256(probe/"tokens.pt"),
                "model_files":{p.name:sha256(p) for p in model.iterdir() if p.is_file()}}))
            common = ["-m", "efficient_llm.diagnostic", "run", "--model", str(model), "--probe", str(probe),
                "--device", "cpu", "--dtype", "float32", "--stride", "2", "--keep", "3", "2"]
            for parallel in (False, True):
                prefix = [sys.executable]
                if parallel:
                    prefix += ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2"]
                path = root/("parallel.json" if parallel else "serial.json")
                result = subprocess.run(prefix+common+["--output", str(path)], capture_output=True, text=True,
                    env=os.environ.copy(), timeout=180)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            serial, parallel = [json.loads((root/p).read_text()) for p in ("serial.json","parallel.json")]
            for keep in ("3", "2"):
                s, p = serial["budgets"][keep], parallel["budgets"][keep]
                self.assertEqual(s["static_groups_per_layer"], p["static_groups_per_layer"])
                self.assertEqual(p["heldout_dense_reconstruction"][0]["positions"], 28)
                for policy in s["methods"]:
                    self.assertEqual(p["methods"][policy]["scored_tokens"], 56)
                    torch.testing.assert_close(torch.tensor(s["methods"][policy]["per_document_nll"]),
                                               torch.tensor(p["methods"][policy]["per_document_nll"]), rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
