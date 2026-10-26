"""CPU tests of FA4 plumbing with an explicit per-document reference kernel.

These test integration and gradients, NOT the real CUDA kernel numerics.
Production code has no CPU fallback. Actual GPU comparison is a separate probe.
"""
import copy
import tempfile
import unittest
from unittest.mock import patch

from test_document_isolation import model, row, torch
from transformers import TrainingArguments
import packed_fa4
from packed_training import PackedCollator, PackedTrainer, model_inputs
from train import build_model, ModelArguments


def reference_kernel(q, k, v, *, cu_seqlens_q, cu_seqlens_k,
                     max_seqlen_q, max_seqlen_k, causal, softmax_scale):
    assert torch.equal(cu_seqlens_q, cu_seqlens_k)
    assert cu_seqlens_q.dtype == torch.int32
    assert max_seqlen_q == max_seqlen_k and causal
    parts = []
    for a, b in zip(cu_seqlens_q[:-1].tolist(), cu_seqlens_q[1:].tolist()):
        assert b - a <= max_seqlen_q
        out = torch.nn.functional.scaled_dot_product_attention(
            q[a:b].transpose(0, 1)[None], k[a:b].transpose(0, 1)[None],
            v[a:b].transpose(0, 1)[None], is_causal=True, enable_gqa=True, scale=softmax_scale)
        parts.append(out[0].transpose(0, 1))
    return torch.cat(parts), None


class FA4Tests(unittest.TestCase):
    def setUp(self):
        self.rows = [row([3, 4, 31, 8, 9, 10, 31], [0, 0, 0, 1, 1, 1, 1]),
                     row([31, 8, 9, 10, 11, 12, 31], [1, 2, 2, 2, 2, 2, 2])]
        # Same segment ID across row boundary must still start a new fragment.

    def fake_kernel(self):
        self.enterContext(patch.object(packed_fa4, "validate_kernel_inputs"))
        kernel = self.enterContext(patch.object(packed_fa4, "load_kernel", return_value=reference_kernel))
        packed_fa4.register_backend()
        return kernel

    def test_layout_positions_labels_and_no_dense_mask(self):
        for reset in (True, False):
            dense = PackedCollator(True, reset)(self.rows)
            varlen = PackedCollator(True, reset, "fa4")(self.rows)
            self.assertEqual(varlen["fa4_cu_seqlens"].tolist(), [0, 3, 7, 8, 14])
            self.assertEqual(varlen["fa4_max_seqlen"], 6)
            for key in ("input_ids", "labels", "position_ids"):
                self.assertTrue(torch.equal(varlen[key], dense[key]), key)
            prepared = model_inputs(varlen)
            self.assertEqual(prepared["attention_mask"], {"full_attention": None})
            self.assertFalse(prepared["use_cache"])
            self.assertNotIn("segments", prepared)

    def test_native_model_logits_loss_gradients_checkpointing_and_leakage(self):
        kernel = self.fake_kernel()
        for checkpoint in (False, True):
            for reset in (False, True):
                dense = model().train()
                varlen = copy.deepcopy(dense)
                varlen.set_attn_implementation(packed_fa4.BACKEND)
                if checkpoint:
                    varlen.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                sdpa_batch = model_inputs(PackedCollator(True, reset)(self.rows))
                collate = PackedCollator(True, reset, "fa4")
                fa4_batch = model_inputs(collate(self.rows))
                a, b = dense(**sdpa_batch), varlen(**fa4_batch)
                torch.testing.assert_close(a.logits, b.logits, atol=3e-7, rtol=3e-5)
                torch.testing.assert_close(a.loss, b.loss)
                a.loss.backward()
                b.loss.backward()
                for (name, p), (_, ref) in zip(varlen.named_parameters(), dense.named_parameters()):
                    torch.testing.assert_close(p.grad, ref.grad, atol=2e-7, rtol=3e-5, msg=name)
                changed = copy.deepcopy(self.rows)
                changed[0]["input_ids"][:3] = [17, 18, 19]
                out = varlen(**model_inputs(collate(changed)))
                torch.testing.assert_close(b.logits[0, 3:], out.logits[0, 3:], atol=0, rtol=0)
                torch.testing.assert_close(b.logits[1], out.logits[1], atol=0, rtol=0)
                embeds = varlen.get_input_embeddings()(fa4_batch.pop("input_ids")).detach().requires_grad_()
                varlen(inputs_embeds=embeds, **fa4_batch).logits[0, 4].square().sum().backward()
                self.assertEqual(embeds.grad[0, :3].count_nonzero().item(), 0)
                self.assertEqual(embeds.grad[0, 5:].count_nonzero().item(), 0)
                self.assertEqual(embeds.grad[1].count_nonzero().item(), 0)
        self.assertGreater(kernel.call_count, 16)

    def test_trainer_accumulation_and_native_checkpoint_reload(self):
        self.fake_kernel()
        m = model()
        reference = copy.deepcopy(m)
        m.set_attn_implementation(packed_fa4.BACKEND)
        rows = [self.rows[0], row([31] * 7, list(range(7)))]
        collate = PackedCollator(attention_backend="fa4")
        with tempfile.TemporaryDirectory() as folder:
            args = TrainingArguments(output_dir=folder, use_cpu=True, report_to="none",
                remove_unused_columns=False, gradient_accumulation_steps=2,
                gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
                per_device_train_batch_size=1, max_steps=1, save_strategy="no",
                max_grad_norm=0., dataloader_pin_memory=False, disable_tqdm=True)
            opt = torch.optim.SGD(m.parameters(), lr=.03)
            trainer = PackedTrainer(model=m, args=args, train_dataset=rows, data_collator=collate,
                optimizers=(opt, torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.)))
            trainer.train()
            ref_opt = torch.optim.SGD(reference.parameters(), lr=.03)
            reference(**model_inputs(PackedCollator()(rows))).loss.backward()
            ref_opt.step()
            for (name, p), (_, ref) in zip(m.named_parameters(), reference.named_parameters()):
                torch.testing.assert_close(p, ref, atol=2e-7, rtol=2e-5, msg=name)
            trainer.save_model(folder)
            # Pretrained construction selects backend explicitly; weights stay native HF.
            for backend in ("sdpa", "fa4"):
                loaded = build_model(ModelArguments(model_name_or_path=folder, attention_backend=backend))
                for name, value in m.state_dict().items():
                    torch.testing.assert_close(value, loaded.state_dict()[name], atol=0, rtol=0)

    def test_missing_wrong_kernel_and_unsupported_inputs_fail_closed(self):
        packed_fa4.load_kernel.cache_clear()
        with patch.object(packed_fa4, "version", return_value="wrong"):
            with self.assertRaisesRegex(RuntimeError, "requires flash-attn-4"):
                packed_fa4.load_kernel()
        from importlib.metadata import PackageNotFoundError
        with patch.object(packed_fa4, "version", side_effect=PackageNotFoundError):
            with self.assertRaisesRegex(RuntimeError, "no SDPA fallback"):
                packed_fa4.load_kernel()
        with self.assertRaisesRegex(ValueError, "requires document isolation"):
            PackedCollator(False, False, "fa4")
        q = torch.randn(1, 2, 3, 8)
        cu = torch.tensor([0, 3], dtype=torch.int32)
        for extra, message in (({}, "metadata"),
                ({"fa4_cu_seqlens": cu, "fa4_max_seqlen": 3}, "CUDA"),
                ({"fa4_cu_seqlens": cu, "fa4_max_seqlen": 3, "dropout": .1}, "zero dropout"),
                ({"fa4_cu_seqlens": cu, "fa4_max_seqlen": 3, "use_cache": True}, "use_cache=False")):
            with self.assertRaisesRegex(ValueError, message):
                packed_fa4.attention_forward(None, q, q, q, None, **extra)
        with tempfile.TemporaryDirectory() as folder:
            m = model()
            m.config.attention_dropout = .1
            m.config.save_pretrained(folder)
            with self.assertRaisesRegex(ValueError, "attention_dropout=0"):
                build_model(ModelArguments(config_name=folder, attention_backend="fa4"))
            with self.assertRaisesRegex(ValueError, "sdpa or fa4"):
                build_model(ModelArguments(config_name=folder, attention_backend="typo"))

    def test_trainer_rejects_metadata_splitting_and_backend_mismatch(self):
        self.fake_kernel()
        with tempfile.TemporaryDirectory() as folder:
            for setting in ("split_batches", "dispatch_batches"):
                m = model()
                m.set_attn_implementation(packed_fa4.BACKEND)
                args = TrainingArguments(output_dir=folder, use_cpu=True, report_to="none",
                    remove_unused_columns=False, accelerator_config={setting: True})
                with self.assertRaisesRegex(ValueError, "per-rank collation"):
                    PackedTrainer(model=m, args=args, data_collator=PackedCollator(attention_backend="fa4"))
            args = TrainingArguments(output_dir=folder, use_cpu=True, report_to="none", remove_unused_columns=False)
            with self.assertRaisesRegex(ValueError, "must match"):
                PackedTrainer(model=model(), args=args, data_collator=PackedCollator(attention_backend="fa4"))


if __name__ == "__main__":
    unittest.main()
