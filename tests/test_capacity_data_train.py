import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, TrainingArguments

from capacity_allocation.data import load_text_data, preprocess_text, group_texts
from capacity_allocation.modeling import build_model, experiment_config
from train import CausalTrainer, main as train_main
from evaluate_capacity import main as eval_main


def fixture(root):
    vocabulary = {'[UNK]': 0, **{f'w{i}': i+1 for i in range(95)}, '[EOS]': 96}
    backend = Tokenizer(WordLevel(vocabulary, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]',
                                        eos_token='[EOS]', pad_token='[EOS]')
    tokenizer.save_pretrained(root/'tokenizer')
    for split in ('train', 'eval'):
        for lang in ('en', 'vi'):
            rows = [' '.join(f'w{(i+j)%95}' for j in range(19)) for i in range(24)]
            target = root/split/lang
            if split == 'train':
                Dataset.from_dict({'text': rows[:12]}).save_to_disk(str(target/'shard_0000'))
                Dataset.from_dict({'text': rows[12:]}).save_to_disk(str(target/'shard_0001'))
            else:
                Dataset.from_dict({'text': rows}).save_to_disk(str(target))
    return tokenizer


class DataAndTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_old_layout_and_explicit_language_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture(root)
            self.assertEqual(len(load_text_data(root/'train')), 24)
            self.assertEqual(len(load_text_data(root/'train', ('vi', 'en'))), 48)
            self.assertEqual(len(load_text_data(root/'eval')), 24)
            with self.assertRaises(FileNotFoundError):
                load_text_data(root/'train', ('zh',))
            with self.assertRaises(ValueError):
                load_text_data(root/'train', ('../train',))
            with self.assertRaises(ValueError):
                load_text_data(root/'train', ('en', 'en'))

    def test_packing_matches_old_concatenate_and_group(self):
        examples = {'input_ids': [[1,2,3], [4,96,6,7,8], [9]]}
        self.assertEqual(group_texts(examples, 4), {
            'input_ids': [[1,2,3,4], [96,6,7,8]], 'labels': [[1,2,3,4], [96,6,7,8]]})
        # Exactly the old behavior: no padding/EOS insertion; drop the batch tail.
        self.assertEqual(group_texts({'input_ids': [[1,2]]}, 4)['input_ids'], [])

    def test_batched_multiprocess_cache_and_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            tokenizer = fixture(root)
            raw = load_text_data(root/'train')
            for workers in (1, 2):
                data = preprocess_text(raw, tokenizer, block_size=8, num_proc=workers,
                                       batch_size=4, cache_dir=root/'cache')
                # Each 4-document batch has 76 tokens -> 9 full blocks; 6 batches.
                self.assertEqual(len(data), 54)
                expected = []
                for start in range(0, 24, 4):
                    ids = tokenizer(raw[start:start+4]['text'], add_special_tokens=False)['input_ids']
                    expected.extend(group_texts({'input_ids': ids}, 8)['input_ids'])
                self.assertEqual(data[:]['input_ids'].tolist(), expected)
                self.assertEqual(data[:]['labels'].tolist(), expected)
                before = {p.name:p.stat().st_mtime_ns for p in (root/'cache').glob('*.arrow')}
                cached = preprocess_text(raw, tokenizer, block_size=8, num_proc=workers,
                                         batch_size=4, cache_dir=root/'cache')
                after = {p.name:p.stat().st_mtime_ns for p in (root/'cache').glob('*.arrow')}
                self.assertEqual(before, after)
                self.assertEqual(cached._fingerprint, data._fingerprint)
            with self.assertRaises(ValueError):
                preprocess_text(raw, tokenizer, num_proc=0)

    def test_shifted_loss_and_accumulation(self):
        with tempfile.TemporaryDirectory() as folder:
            model = build_model(experiment_config('A128', tiny=True))
            trainer = CausalTrainer(model=model, args=TrainingArguments(
                output_dir=folder, use_cpu=True, report_to='none', gradient_accumulation_steps=2))
            x = torch.randint(0, 97, (4, 8))
            batches = [dict(input_ids=a, labels=a.clone()) for a in x.chunk(2)]
            count = trainer._get_num_items_in_batch(batches, torch.device('cpu'))
            self.assertEqual(count.item(), 28)
            loss = sum(trainer.compute_loss(model, dict(batch), num_items_in_batch=count)
                       for batch in batches)
            torch.testing.assert_close(loss, model(x, labels=x).loss)

    def test_training_save_resume_and_evaluator(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture(root)
            for arm in ('B0','A128','A256','A512','C','D'):
                output = root/arm
                argv = ['train.py', '--arm', arm, '--tiny_test',
                        '--tokenizer_name', str(root/'tokenizer'),
                        '--data_dir', str(root/'train'), '--eval_data_dir', str(root/'eval'),
                        '--output_dir', str(output), '--block_size', '8',
                        '--preprocessing_num_workers', '2', '--preprocessing_batch_size', '4',
                        '--preprocessing_cache_dir', str(root/'cache'),
                        '--use_cpu', '--report_to', 'none', '--num_train_epochs', '1',
                        '--per_device_train_batch_size', '2', '--gradient_accumulation_steps', '2',
                        '--per_device_eval_batch_size', '7', '--learning_rate', '.001',
                        '--stop-at-step', '2', '--save_steps', '2', '--logging_steps', '1',
                        '--gradient_checkpointing', '--gradient_checkpointing_kwargs', '{"use_reentrant":false}']
                with self.subTest(arm=arm), patch.object(sys, 'argv', argv):
                    train_main()
                result = json.loads((output/'result.json').read_text())
                self.assertTrue(result['success'])
                self.assertEqual(result['global_step'], 2)
                self.assertGreater(result['schedule_steps'], 2)
                self.assertEqual(result['status'], 'stopped_at_step')
                self.assertEqual(result['eval_metrics']['eval_scored_targets'], 54*7)
                self.assertTrue((output/'checkpoint-2'/'optimizer.pt').is_file())
                self.assertTrue((output/'final'/'tokenizer.json').is_file())
                with patch.object(sys, 'argv', argv), self.assertRaises(ValueError):
                    train_main()
                if arm == 'A128':
                    evaluation = root/'evaluation.json'
                    with patch.object(sys, 'argv', ['evaluate_capacity.py',
                        '--checkpoint', str(output/'final'), '--data_dir', str(root/'eval'),
                        '--block_size', '8', '--preprocessing_num_workers', '2',
                        '--preprocessing_batch_size', '4', '--preprocessing_cache_dir', str(root/'cache'),
                        '--device', 'cpu', '--precision', 'fp32', '--batch-size', '7',
                        '--output', str(evaluation)]):
                        eval_main()
                    self.assertAlmostEqual(json.loads(evaluation.read_text())['nll'],
                                           result['eval_metrics']['eval_loss'], places=6)
                    # Simulate interrupted checkpoint; automatic resume must retain schedule.
                    (output/'result.json').rename(output/'short_result.json')
                    resumed = argv.copy()
                    resumed[resumed.index('--stop-at-step')+1] = '3'
                    with patch.object(sys, 'argv', resumed):
                        train_main()
                    uninterrupted = resumed.copy()
                    uninterrupted[uninterrupted.index('--output_dir')+1] = str(root/'reference')
                    with patch.object(sys, 'argv', uninterrupted):
                        train_main()
                    a = AutoModelForCausalLM.from_pretrained(output/'final', local_files_only=True)
                    b = AutoModelForCausalLM.from_pretrained(root/'reference'/'final', local_files_only=True)
                    for key, value in a.state_dict().items():
                        torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
