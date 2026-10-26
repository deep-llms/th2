"""Native Qwen3 SDPA vs independent per-document and loss references, CPU only."""
import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                  TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from torch.nn import functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM, TrainingArguments, set_seed
from packed_training import PackedCollator, PackedTrainer, model_inputs
from training_data import tokenize_with_segments, group_texts

torch.set_num_threads(1)


def model():
    set_seed(42)
    config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                        head_dim=8, max_position_embeddings=32, attention_dropout=0.,
                        use_cache=False, tie_word_embeddings=False)
    return Qwen3ForCausalLM._from_config(config, attn_implementation="sdpa")


def row(ids, segments):
    return dict(input_ids=ids, segments=segments, attention_mask=[1] * len(ids), labels=list(ids))


def objective(model, batch):
    output = model(**model_inputs(batch))
    loss_sum = F.cross_entropy(output.logits[:, :-1].flatten(0, 1), batch["labels"][:, 1:].flatten(),
                               ignore_index=-100, reduction="sum")
    return output, loss_sum


class IsolationTests(unittest.TestCase):
    def setUp(self):
        self.collate = PackedCollator()
        self.row = row([3, 4, 31, 8, 9, 10, 31], [10, 10, 10, 11, 11, 11, 11])

    def test_mask_positions_labels_and_single_document_identity(self):
        batch = self.collate([self.row])
        self.assertEqual(batch["position_ids"].tolist(), [[0, 1, 2, 0, 1, 2, 3]])
        self.assertEqual(batch["labels"].tolist(), [[-100, 4, 31, -100, 9, 10, 31]])
        mask = model_inputs(batch)["attention_mask"]
        for q in range(7):
            for k in range(7):
                self.assertEqual(bool(mask[0, 0, q, k]), k <= q and self.row["segments"][k] == self.row["segments"][q])
        m = model().eval()
        ordinary = torch.tensor([[3, 4, 5, 31]])
        with torch.no_grad():
            expected = m(ordinary).logits
            actual = m(**model_inputs(self.collate([row(ordinary[0].tolist(), [0] * 4)]))).logits
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_packed_equals_separate_documents_loss_and_parameter_gradients(self):
        m = model()
        packed = self.collate([self.row])
        outputs, summed = objective(m, packed)
        parts, total = [], 0
        for a, b in ((0, 3), (3, 7)):
            batch = self.collate([row(self.row["input_ids"][a:b], [0] * (b - a))])
            output, loss = objective(m, batch)
            parts.append(output.logits)
            total = total + loss
        torch.testing.assert_close(outputs.logits, torch.cat(parts, 1), atol=2e-7, rtol=2e-5)
        torch.testing.assert_close(summed, total, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(outputs.loss, summed / 5)
        summed.backward()
        gradients = {k: p.grad.clone() for k, p in m.named_parameters()}
        m.zero_grad(set_to_none=True)
        total.backward()
        for name, p in m.named_parameters():
            torch.testing.assert_close(p.grad, gradients[name], atol=2e-6, rtol=2e-5, msg=name)

    def test_no_cross_document_or_future_output_or_input_gradient_leakage(self):
        for checkpoint in (False, True):
            m = model().train()
            if checkpoint:
                m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            original = self.collate([self.row])
            changed = self.collate([row([15, 16, 31, 8, 9, 10, 31], self.row["segments"])])
            inputs = model_inputs(original)
            embeddings = m.get_input_embeddings()(inputs.pop("input_ids")).detach().requires_grad_()
            before = m(inputs_embeds=embeddings, **inputs).logits
            after = m(**model_inputs(changed)).logits
            torch.testing.assert_close(before[:, 3:], after[:, 3:], atol=0, rtol=0)
            before[:, 4].square().sum().backward()
            self.assertEqual(embeddings.grad[:, :3].count_nonzero().item(), 0)
            self.assertEqual(embeddings.grad[:, 5:].count_nonzero().item(), 0)
            self.assertGreater(embeddings.grad[:, 3:5].abs().sum().item(), 0)
            future = self.collate([row([3, 4, 31, 8, 9, 17, 18], self.row["segments"])])
            out = m(**model_inputs(future)).logits
            torch.testing.assert_close(before[:, :5], out[:, :5], atol=0, rtol=0)
        # Ensure the perturbation really matters without isolation.
        control = PackedCollator(False, False)
        m = model().eval()
        before = m(**model_inputs(control([self.row]))).logits
        after = m(**model_inputs(control([row([15, 16, 31, 8, 9, 10, 31], self.row["segments"])]))).logits
        self.assertGreater((before[:, 3:] - after[:, 3:]).abs().max().item(), 1e-5)

    def test_literal_eos_empty_documents_continuations_and_worker_indices(self):
        class Tokenizer:
            def __call__(self, texts, **kwargs):
                return {"input_ids": [[int(t) for t in text.split()] for text in texts]}
        tokens = tokenize_with_segments({"text": ["", "3 31 4 5 6 7", "", "8 9 10"]},
                                        [100, 101, 102, 103], Tokenizer(), 31)
        packed = group_texts(tokens, 6)
        rows = [{key: value[i] for key, value in packed.items() if key != "target_count"} for i in range(2)]
        batch = self.collate(rows)
        self.assertEqual(batch["segments"].tolist(), [[100, 101, 101, 101, 101, 101], [101, 101, 102, 103, 103, 103]])
        self.assertEqual(batch["position_ids"].tolist(), [[0, 0, 1, 2, 3, 4], [0, 1, 0, 0, 1, 2]])
        self.assertEqual(batch["labels"].tolist(), [[-100, -100, 31, 4, 5, 6], [-100, 31, -100, -100, 9, 10]])
        with self.assertRaises(KeyError):
            self.collate([{k: v for k, v in self.row.items() if k != "segments"}])
        with self.assertRaisesRegex(ValueError, "nondecreasing"):
            self.collate([row([3, 4, 5], [1, 2, 1])])

    def test_trainer_global_token_normalization_with_unequal_microbatches(self):
        # Second microbatch has zero targets; first has five. It must contribute
        # zero (not NaN), without changing the first microbatch's normalization.
        batches = [self.collate([self.row]), self.collate([row([31] * 7, list(range(7)))])]
        m = model()
        reference = copy.deepcopy(m)
        with tempfile.TemporaryDirectory() as folder:
            trainer = PackedTrainer(model=m, args=TrainingArguments(output_dir=folder, use_cpu=True,
                report_to="none", remove_unused_columns=False, gradient_accumulation_steps=2), data_collator=self.collate)
            count = trainer._get_num_items_in_batch(batches, torch.device("cpu"))
            self.assertEqual(count.item(), 5)
            for batch in batches:
                trainer.compute_loss(m, trainer._prepare_inputs(batch), num_items_in_batch=count).backward()
            _, summed = objective(reference, batches[0])
            (summed / 5).backward()
            for (name, p), (_, ref) in zip(m.named_parameters(), reference.named_parameters()):
                torch.testing.assert_close(p.grad, ref.grad, atol=1e-7, rtol=1e-6, msg=name)
            with self.assertRaisesRegex(ValueError, "no within-document"):
                trainer._get_num_items_in_batch([batches[1]], torch.device("cpu"))
