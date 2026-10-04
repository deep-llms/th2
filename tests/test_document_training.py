"""CPU checks for the disposable full-training benchmark, not GPU management."""
import copy
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from transformers import TrainingArguments
from deep_kv.model import DeepKV
from deep_kv.packing import group_texts
from scripts.benchmark_document_training import (BenchmarkModel, BenchTrainer, Collator, context,
                                                  tokenize_with_segments)
from tests.test_deep_kv import config


def fake_varlen(q, k, v, *, cu_seqlens_q, cu_seqlens_k, **kwargs):
    outputs = []
    for a, b in zip(cu_seqlens_q[:-1], cu_seqlens_q[1:]):
        output = F.scaled_dot_product_attention(q[a:b].transpose(0,1)[None],
                    k[a:b].transpose(0,1)[None], v[a:b].transpose(0,1)[None],
                    is_causal=True, enable_gqa=True)
        outputs.append(output[0].transpose(0,1))
    return torch.cat(outputs), None


class DocumentTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.rows = [dict(input_ids=[3,4,31,8,9,10,31,31], segments=[0,0,0,1,1,1,1,2]),
                     dict(input_ids=[5,6,31,12,13,14,15,31], segments=[3,3,3,4,4,4,4,4])]

    def model(self, cls=BenchmarkModel):
        return cls.from_scratch(config(), 'A', consumer=2, deep_target=4,
                               checkpoint_layers=False, checkpoint_lm=False, lm_chunk=3)

    def test_packing_tracks_appended_boundary_not_literal_eos(self):
        class Tokenizer:
            def __call__(self, texts, **kwargs): return dict(input_ids=[[3,31,4],[5,6]])
        result = tokenize_with_segments({'text':['a','b']}, [7,8], Tokenizer(), 31)
        packed = group_texts(result, 7)
        self.assertEqual(packed['input_ids'], [[3,31,4,31,5,6,31]])
        self.assertEqual(packed['segments'], [[7,7,7,7,8,8,8]])
        ctx = context(Collator('fa4_isolated')([dict(input_ids=packed['input_ids'][0],
                      segments=packed['segments'][0])]), 'fa4_isolated')
        self.assertEqual(ctx.cu_seqlens.tolist(), [0,4,7])
        self.assertEqual(ctx.position_ids.tolist(), [[0,1,2,3,0,1,2]])
        self.assertEqual(ctx.targets().tolist(), [[True,True,True,False,True,True]])

    def test_adapter_matches_original_and_independent_varlen(self):
        import sys, types
        fake = types.ModuleType('flash_attn.cute.interface')
        fake.flash_attn_varlen_func = fake_varlen
        with patch.dict(sys.modules, {'flash_attn.cute.interface':fake}):
            for reference_mode, candidate_mode in [('sdpa_cross','fa4_cross'),
                                                   ('sdpa_cross','sdpa_causal'),
                                                   ('sdpa_isolated','fa4_isolated')]:
                reference = self.model(DeepKV)
                candidate = self.model()
                candidate.load_state_dict(reference.state_dict())
                a = context(Collator(reference_mode)(self.rows), reference_mode)
                b = context(Collator(candidate_mode)(self.rows), candidate_mode)
                x = reference(a); y = candidate(b)
                torch.testing.assert_close(x['lm_sum'], y['lm_sum'])
                (x['lm_sum']/x['lm_count']).backward(); (y['lm_sum']/y['lm_count']).backward()
                for (name,p), (_,q) in zip(reference.named_parameters(), candidate.named_parameters()):
                    torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-4, msg=name)

    def test_global_target_normalization_for_uneven_microbatches(self):
        model = self.model()
        reference = copy.deepcopy(model)
        mode = 'sdpa_isolated'
        # Different target counts per row (5 and 6) test real token weighting.
        micro = [Collator(mode)([r]) for r in self.rows]
        with tempfile.TemporaryDirectory() as path:
            args = TrainingArguments(output_dir=path, use_cpu=True, report_to=[],
                                     gradient_accumulation_steps=2)
            trainer = BenchTrainer(model=model, mode=mode, args=args)
            count = trainer._get_num_items_in_batch(micro, torch.device('cpu'))
            self.assertEqual(count.item(), 11)
            for batch in micro:
                trainer.compute_loss(model, batch, num_items_in_batch=count).backward()
        full = reference(context(Collator(mode)(self.rows), mode))
        (full['lm_sum']/full['lm_count']).backward()
        for (name,p), (_,q) in zip(model.named_parameters(), reference.named_parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-4, msg=name)

    def test_hf_trainer_accumulation_matches_full_batch_update(self):
        from datasets import Dataset
        models = [self.model(), self.model()]
        with tempfile.TemporaryDirectory() as path:
            for i, (micro, accum) in enumerate(((1,2), (2,1))):
                args = TrainingArguments(output_dir=f'{path}/{i}', use_cpu=True, report_to=[],
                    per_device_train_batch_size=micro, gradient_accumulation_steps=accum,
                    max_steps=1, save_strategy='no', remove_unused_columns=False,
                    max_grad_norm=0., disable_tqdm=True)
                trainer = BenchTrainer(model=models[i], mode='sdpa_isolated', args=args,
                    train_dataset=Dataset.from_list(self.rows), data_collator=Collator('sdpa_isolated'),
                    optimizers=(torch.optim.SGD(models[i].parameters(), lr=.01), None))
                trainer.train()
        for (name,p), (_,q) in zip(models[0].named_parameters(), models[1].named_parameters()):
            torch.testing.assert_close(p, q, atol=1e-7, rtol=1e-5, msg=name)


if __name__ == '__main__': unittest.main()
