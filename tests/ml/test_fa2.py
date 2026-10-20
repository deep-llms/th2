"""CPU FA2 integration checks using an explicit reference kernel, not CUDA claims."""
import copy
import tempfile
import unittest
from unittest.mock import patch

import torch
import packed_fa2
import packed_fa4
from packed_training import PackedCollator, PackedTrainer, model_inputs
from test_fa4 import reference_kernel
from test_document_isolation import model, row
from train import ModelArguments, build_model
from transformers import TrainingArguments


class FA2Tests(unittest.TestCase):
    def test_shared_layout_gradients_update_and_reload(self):
        with patch.object(packed_fa2, 'load_kernel', return_value=reference_kernel), \
             patch.object(packed_fa4, 'validate_kernel_inputs'):
            packed_fa2.register_backend()
            rows = [row([3, 4, 31, 8, 9, 10, 31], [0, 0, 0, 1, 1, 1, 1]),
                    row([31] * 7, list(range(7)))]
            dense = model().train()
            fa2 = copy.deepcopy(dense)
            fa2.set_attn_implementation(packed_fa2.BACKEND)
            fa2.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            batch = model_inputs(PackedCollator(attention_backend='fa2')(rows))
            expected = dense(**model_inputs(PackedCollator()(rows)))
            actual = fa2(**batch)
            torch.testing.assert_close(actual.logits, expected.logits, rtol=3e-5, atol=3e-7)
            actual.loss.backward(); expected.loss.backward()
            for p, q in zip(fa2.parameters(), dense.parameters()):
                torch.testing.assert_close(p.grad, q.grad, rtol=3e-5, atol=2e-7)
            fa2.zero_grad(); dense.zero_grad()
            with tempfile.TemporaryDirectory() as out:
                args = TrainingArguments(output_dir=out, use_cpu=True, report_to='none',
                    remove_unused_columns=False, gradient_accumulation_steps=2,
                    per_device_train_batch_size=1, max_steps=1, save_strategy='no',
                    max_grad_norm=0., dataloader_pin_memory=False, disable_tqdm=True)
                opt = torch.optim.SGD(fa2.parameters(), lr=.03)
                trainer = PackedTrainer(model=fa2, args=args, train_dataset=rows,
                    data_collator=PackedCollator(attention_backend='fa2'),
                    optimizers=(opt, torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.)))
                trainer.train()
                ref = torch.optim.SGD(dense.parameters(), lr=.03)
                dense(**model_inputs(PackedCollator()(rows))).loss.backward(); ref.step()
                for p, q in zip(fa2.parameters(), dense.parameters()):
                    torch.testing.assert_close(p, q, rtol=3e-5, atol=2e-7)
                trainer.save_model(out)
                loaded = build_model(ModelArguments(model_name_or_path=out, attention_backend='fa2'))
                for name, value in fa2.state_dict().items():
                    torch.testing.assert_close(value, loaded.state_dict()[name], rtol=0, atol=0)

    def test_kernel_wrapper_and_version_guard(self):
        import sys
        from types import SimpleNamespace
        packed_fa2.load_kernel.cache_clear()
        calls=[]
        def fake(*args, **kwargs):
            calls.append(kwargs)
            return 'output'
        with patch.object(packed_fa2, 'version', return_value='2.8.3.post2+local'), \
             patch.dict(sys.modules, {'flash_attn':SimpleNamespace(flash_attn_varlen_func=fake)}):
            self.assertEqual(packed_fa2.load_kernel()(1, causal=True), ('output', None))
            self.assertEqual(calls, [dict(causal=True, dropout_p=0.)])
        packed_fa2.load_kernel.cache_clear()
        with patch.object(packed_fa2, 'version', return_value='2.7.0'):
            with self.assertRaises(RuntimeError): packed_fa2.load_kernel()


if __name__ == '__main__': unittest.main()
