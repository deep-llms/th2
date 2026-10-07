"""Offline evaluation checks; real harness, tiny models, no GPU or task downloads."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from safetensors.torch import save_file
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from deep_kv import ALL_PROXY_ARMS
from deep_kv.model import DeepKV
from deep_kv.proxy import ProxyModel, ProxySettings
from eval.models import EvaluationModel, load_checkpoint
from eval.benchmarks import resolve_tasks, task_configs, eval_benchmarks
from eval.ppl import compute_perplexity, eval_ppl
from eval.eval_parallel import build_cmd
from tests.test_proxy_memory import config
from tests.test_proxy_heads import batch
from tests.test_fa4_baseline import reference_kernel
from tests.test_proxy_training import proxy_fixture
from tests.test_train import invoke


def make_model(arm, backend='sdpa'):
    m = ProxyModel.from_scratch(config(), arm, consumer=2, deep_target=8,
        proxy_settings=ProxySettings(width=8, features=9, chunk_size=2),
        checkpoint_layers=False, checkpoint_aux=False, checkpoint_lm=False,
        attention_backend=backend, lm_chunk=3, sequence_length=8)
    with torch.no_grad():
        for head in m.heads.values():
            if hasattr(head, 'alpha'):
                head.alpha.fill_(.7)  # Exercise learned proxy use, not the zero-gate baseline.
        m.mu.normal_()
        m.mu_initialized.fill_(True)
    return m.eval()


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_verified_local_dataset_loading(self):
        import hashlib
        from eval.benchmarks import local_dataset_paths
        from lm_eval.tasks import TaskManager, get_task_dict
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); folder = root / 'hellaswag'; folder.mkdir()
            path = folder / 'validation.parquet'
            Dataset.from_list([dict(ctx_a='A person', ctx_b='walks', endings=['a','b','c','d'],
                                    label='0', activity_label='Walking')]).to_parquet(path)
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps(dict(schema_version=1, repositories=[dict(path='hellaswag',
                tasks=['hellaswag'], files=[dict(path='hellaswag/validation.parquet', bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())], load_configs=dict(hellaswag=dict(
                loader='parquet', data_files=dict(train=['hellaswag/validation.parquet'],
                validation=['hellaswag/validation.parquet']), kwargs={}))) ])))
            configs = local_dataset_paths(root, manifest)
            tasks = get_task_dict(task_configs(['hellaswag'], configs), task_manager=TaskManager())
            self.assertEqual(len(tasks['hellaswag'].eval_docs), 1)
            self.assertEqual(tasks['hellaswag'].doc_to_target(tasks['hellaswag'].eval_docs[0]), 0)
            with self.assertRaisesRegex(ValueError, 'Missing local datasets'):
                task_configs(['xnli_en'], configs)
            with path.open('ab') as stream: stream.write(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                local_dataset_paths(root, manifest)

    def test_logits_match_training_loss_all_proxy_arms_both_backends(self):
        ctx = batch()
        with patch('deep_kv.fa4.load_kernel', return_value=(reference_kernel, {'version': 'CPU-test-double'})):
            for arm in ('A',) + ALL_PROXY_ARMS:
                for backend in ('sdpa', 'fa4'):
                    with self.subTest(arm=arm, backend=backend), torch.no_grad():
                        m = make_model(arm, backend)
                        before = {k: v.clone() for k, v in m.state_dict().items()}
                        wrapper = EvaluationModel(m, 8)
                        result = wrapper(ctx.input_ids, labels=ctx.input_ids, segments=ctx.segments,
                                         position_ids=ctx.position_ids)
                        hidden = m.hidden_states(ctx)[0]
                        torch.testing.assert_close(result.logits, m.backbone.lm_head(hidden), rtol=0, atol=0)
                        training = m(ctx, compute_auxiliary_losses=False)
                        torch.testing.assert_close(result.loss, training['lm_sum'] / training['lm_count'],
                                                   rtol=1e-6, atol=1e-6)
                        changed = ctx.input_ids.clone(); changed[:, :3] = 20
                        other = wrapper(changed, segments=ctx.segments).logits
                        torch.testing.assert_close(result.logits[:, 3:], other[:, 3:], rtol=0, atol=0)
                        if m.family:
                            with m.without_proxy():
                                disabled = wrapper(ctx.input_ids, segments=ctx.segments).logits
                            self.assertGreater((disabled - result.logits).abs().max().item(), 1e-7)
                        for key, value in m.state_dict().items():
                            torch.testing.assert_close(before[key], value, rtol=0, atol=0)

    def test_variable_lengths_padding_and_literal_eos(self):
        for arm in ('P3-block', 'P7', 'P7-ems', 'P7-simple'):
            m = make_model(arm)
            wrapper = EvaluationModel(m, 8)
            # Real EOS inside the text is NOT a document boundary.
            ids = torch.tensor([[3, 31, 5]])
            alone = wrapper(ids).logits
            padded = wrapper(torch.tensor([[0, 0, 3, 31, 5, 0]]),
                             attention_mask=torch.tensor([[0, 0, 1, 1, 1, 0]])).logits
            torch.testing.assert_close(alone, padded[:, 2:5], rtol=1e-5, atol=1e-6)
            self.assertEqual(wrapper(ids[:, :1]).logits.shape, (1, 1, 32))
            with self.assertRaises(ValueError): wrapper(ids, use_cache=True)
            with self.assertRaises(ValueError): wrapper(torch.ones(1, 9, dtype=torch.long))
            with self.assertRaises(NotImplementedError): wrapper.generate(ids)

    def test_padding_and_invalid_layouts_both_backends(self):
        with patch('deep_kv.fa4.load_kernel', return_value=(reference_kernel, {'version': 'CPU-test-double'})):
            for backend in ('sdpa', 'fa4'):
                for arm in ('P3-block', 'P7', 'P7-ems', 'P7-simple'):
                    wrapper = EvaluationModel(make_model(arm, backend), 8)
                    ids = torch.tensor([[3, 31, 5]])
                    alone = wrapper(ids).logits
                    padded = torch.tensor([[9, 9, 3, 31, 5, 9, 9]])
                    mask = torch.tensor([[0, 0, 1, 1, 1, 0, 0]])
                    torch.testing.assert_close(alone, wrapper(padded, attention_mask=mask).logits[:, 2:5],
                                               rtol=1e-5, atol=1e-6)
                    # Reusing a document ID could disagree between dense and varlen attention.
                    with self.assertRaisesRegex(ValueError, 'contiguous segment'):
                        wrapper(ids, segments=torch.tensor([[0, 1, 0]]))
                    with self.assertRaisesRegex(ValueError, 'contiguous span'):
                        wrapper(ids, attention_mask=torch.tensor([[1, 0, 1]]))
                    with self.assertRaisesRegex(ValueError, 'zero or one'):
                        wrapper(ids, attention_mask=torch.tensor([[1, -1, 1]]))
                    with self.assertRaisesRegex(ValueError, 'nonnegative segment'):
                        wrapper(ids, segments=torch.zeros(1, 3))

    def test_score_serialization_preserves_numbers(self):
        import numpy as np
        from eval.eval_checkpoint import json_default
        scores = dict(doc_id=np.int64(3), acc=np.float32(.5),
                      values=np.array([1.25, 2.5]), logits=torch.tensor([.25, .75]))
        restored = json.loads(json.dumps(scores, default=json_default, allow_nan=False))
        self.assertEqual(restored, dict(doc_id=3, acc=.5, values=[1.25, 2.5], logits=[.25, .75]))
        self.assertIsInstance(restored['doc_id'], int)
        with self.assertRaises(ValueError):
            json.dumps(dict(score=float('nan')), default=json_default, allow_nan=False)

    def test_strict_checkpoint_loading_all_proxy_arms(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); ckpt = root / 'checkpoint-7'; ckpt.mkdir()
            (ckpt / 'trainer_state.json').write_text(json.dumps(dict(global_step=7)))
            for arm in ('A',) + ALL_PROXY_ARMS:
                m = make_model(arm)
                recipe = dict(model_config=config().to_dict(), model=dict(tokenizer_name='local-tokenizer'),
                    data=dict(block_size=8), training=dict(seed=42, bf16=False),
                    pilot=dict(arm=arm, proxy_screen=True, consumer=2, deep_target=8, lm_chunk=3,
                               causal_attention=False, **{'proxy_' + k: v for k, v in asdict(m.settings).items()}))
                (root / 'train_config.json').write_text(json.dumps(recipe))
                state = {k: v.clone() for k, v in m.state_dict().items()}
                save_file(state, ckpt / 'model.safetensors')
                loaded, _, metadata = load_checkpoint(ckpt)
                self.assertEqual(metadata['step'], 7)
                self.assertEqual(metadata['arm'], arm)
                for key, value in loaded.wrapped.state_dict().items():
                    torch.testing.assert_close(value, state[key], rtol=0, atol=0)
                del state['mu']
                save_file(state, ckpt / 'model.safetensors')
                with self.assertRaises(RuntimeError): load_checkpoint(ckpt)
            save_file({k: v.clone() for k, v in m.state_dict().items()}, ckpt / 'model.safetensors')
            recipe['pilot']['attention_backend'] = 'fa4'
            (root / 'train_config.json').write_text(json.dumps(recipe))
            with self.assertRaisesRegex(ValueError, 'FA4 evaluation requires CUDA BF16'):
                load_checkpoint(ckpt)
            _, _, metadata = load_checkpoint(ckpt, attention_backend='sdpa')
            self.assertEqual(metadata['trained_attention_backend'], 'fa4')
            self.assertEqual(metadata['attention_backend'], 'sdpa')
            (root / 'train_config.json').unlink()
            with self.assertRaisesRegex(ValueError, 'missing.*recipe'): load_checkpoint(ckpt)

    def test_legacy_plain_and_bf16_models(self):
        with tempfile.TemporaryDirectory() as tmp, torch.no_grad():
            root = Path(tmp)
            for arm in ('A', 'B', 'F', 'G'):
                m = DeepKV.from_scratch(config(), arm, consumer=2, deep_target=8,
                                       checkpoint_layers=False, checkpoint_lm=False, checkpoint_aux=False).eval()
                path = root / arm; path.mkdir()
                recipe = dict(model_config=config().to_dict(), model=dict(tokenizer_name='local-tokenizer'),
                    data=dict(block_size=8), training=dict(seed=42, bf16=False),
                    pilot=dict(arm=arm, proxy_screen=False, consumer=2, deep_target=8,
                               causal_attention=False, lm_chunk=128))
                (path / 'train_config.json').write_text(json.dumps(recipe))
                save_file({k: v.clone() for k, v in m.state_dict().items()}, path / 'model.safetensors')
                loaded, _, _ = load_checkpoint(path)
                ctx = batch()
                actual = loaded(ctx.input_ids, segments=ctx.segments, labels=ctx.input_ids)
                expected = m(ctx, compute_auxiliary_losses=False)
                torch.testing.assert_close(actual.loss, expected['lm_sum']/expected['lm_count'])
            native = make_model('A').backbone
            native.save_pretrained(root / 'native')
            loaded, _, metadata = load_checkpoint(root / 'native')
            self.assertEqual(metadata['kind'], 'huggingface')
            ids = torch.tensor([[3, 4, 5]])
            torch.testing.assert_close(loaded(ids).logits, native(ids).logits)
            for arm in ('A', 'P7-simple'):
                m = make_model(arm)
                wrapper = EvaluationModel(m, 8, torch.bfloat16)
                ctx = batch()
                actual = wrapper(ctx.input_ids, segments=ctx.segments, labels=ctx.input_ids)
                with torch.autocast('cpu', dtype=torch.bfloat16):
                    expected = m(ctx, compute_auxiliary_losses=False)
                torch.testing.assert_close(actual.loss, expected['lm_sum']/expected['lm_count'])
                self.assertTrue(all(p.dtype == torch.float32 for p in m.parameters()))

    def test_english_selection_and_no_environment_mutation(self):
        self.assertEqual(resolve_tasks(english_only=True),
            ['xnli_en', 'belebele_eng_Latn', 'xstorycloze_en', 'paws_en', 'hellaswag'])
        self.assertEqual(resolve_tasks(['hellaswag', 'xnli', 'hellaswag'], True), ['hellaswag', 'xnli_en'])
        self.assertEqual(resolve_tasks(['piqa', 'arc_easy', 'winogrande'], True), ['piqa', 'arc_easy', 'winogrande'])
        for tasks in (['xcopa'], ['xnli_vi'], ['typo']):
            with self.assertRaises(ValueError): resolve_tasks(tasks, True)
        self.assertIn('hellaswag_vi', resolve_tasks(['hellaswag']))
        self.assertEqual(task_configs(['xnli_en'])[0]['dataset_path'], 'facebook/xnli')
        # Resolve actual pinned-harness templates without accessing the datasets.
        from lm_eval.tasks import TaskManager
        manager = TaskManager()
        for name in resolve_tasks(english_only=True) + resolve_tasks(['piqa', 'arc_easy', 'arc_challenge', 'winogrande'], True):
            self.assertEqual(manager._get_config(name)['output_type'], 'multiple_choice')

    def test_real_trainer_checkpoint_harness_and_ppl(self):
        from lm_eval.models.huggingface import HFLM
        from lm_eval.api.instance import Instance
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = proxy_fixture(root)
            invoke(root, {**args, 'arm': 'P3-block', 'stop_after': 1})
            model, tokenizer_path, metadata = load_checkpoint(root / 'run' / 'checkpoint-1')
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
            tokenizer.pad_token = tokenizer.eos_token
            before = {k: v.clone() for k, v in model.state_dict().items()}
            lm = HFLM(pretrained=model, tokenizer=tokenizer, backend='causal', device='cpu',
                      batch_size=2, max_length=8, softmax_dtype='float32')
            requests = [Instance(request_type='loglikelihood', doc={}, arguments=(context, continuation), idx=i)
                        for i, (context, continuation) in enumerate([('3 4', ' 5'), ('6', ' 7 8'), ('', ' 9')])]
            scores = lm.loglikelihood(requests, disable_tqdm=True)
            for request, (score, _) in zip(requests, scores):
                context, continuation = request.args
                if context:
                    c, target = lm._encode_pair(context, continuation)
                else:
                    c, target = [lm.prefix_token_id], lm.tok_encode(continuation)
                ids = torch.tensor([c + target])
                logits = model(ids[:, :-1]).logits.float().log_softmax(-1)
                expected = logits[0, len(c)-1:, :].gather(-1, torch.tensor(target)[:, None]).sum()
                self.assertAlmostEqual(score, expected.item(), places=5)
            # Full simple_evaluate loop against a tiny local multiple-choice task.
            data = root / 'task.jsonl'
            data.write_text('\n'.join(json.dumps(dict(prompt='3 4', choices=['5', '6'], answer=i % 2)) for i in range(4)))
            task = dict(task='local_proxy_eval', dataset_path='json',
                        dataset_kwargs=dict(data_files={'validation': str(data)}), validation_split='validation',
                        output_type='multiple_choice', doc_to_text='{{prompt}}', doc_to_choice='choices',
                        doc_to_target='answer', metric_list=[dict(metric='acc', aggregation='mean', higher_is_better=True)])
            from lm_eval.tasks import TaskManager
            (root / 'local_task.yaml').write_text(json.dumps(task))
            manager = TaskManager(include_path=str(root))
            with patch('eval.benchmarks.task_configs', return_value=[task]), \
                 patch('lm_eval.evaluator.TaskManager', return_value=manager):
                results = eval_benchmarks(model, tokenizer, ['hellaswag'], batch_size=2, device='cpu', english_only=True)
            self.assertIn('acc,none', results['results']['local_proxy_eval'])
            self.assertEqual(len(results['samples']['local_proxy_eval']), 4)
            docs = ['3 4 5', '6 7', '8 9 10 11 12 13 14 15 16']
            Dataset.from_dict({'text': docs}).save_to_disk(root / 'ppl')
            result = eval_ppl(model, tokenizer, root / 'ppl', block_size=8, stride=4, device='cpu')['dataset']
            self.assertEqual(result['num_tokens'], sum(len(tokenizer.encode(t)) for t in docs))
            self.assertEqual(result['documents'], 3)
            with self.assertRaises(ValueError): compute_perplexity(model, torch.tensor([[1, 2]]), 8, 8, 'cpu')
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)
            # Real CLI PPL evaluation on the same trained checkpoint.
            cli = subprocess.run([sys.executable, '-m', 'eval.eval_checkpoint', '--checkpoint',
                str(root / 'run' / 'checkpoint-1'), '--device', 'cpu', '--ppl-only', '--eval-dir',
                str(root / 'ppl'), '--output-dir', str(root / 'evaluation')], capture_output=True, text=True)
            self.assertEqual(cli.returncode, 0, cli.stdout + cli.stderr)
            self.assertEqual(json.loads((root / 'evaluation' / 'eval_metadata.json').read_text())['status'], 'completed')

    def test_parallel_forwards_evaluation_options_and_failure(self):
        args = argparse.Namespace(bench_only=True, ppl_only=False, bf16=True, eval_dir=None,
            english_only=True, tokenizer_name='tokenizer', attention_backend='sdpa',
            batch_size=2, num_fewshot=0, seed=99, limit=4, block_size=None, stride=None,
            tasks=['hellaswag', 'xnli'], langs=None, dataset_root='/local/data', dataset_manifest='/local/spec.json')
        cmd = build_cmd('eval.py', 'checkpoint', args)
        for flag in ('--english-only', '--tasks', '--seed', '--limit', '--tokenizer-name', '--attention-backend', '--dataset-root', '--dataset-manifest'):
            self.assertIn(flag, cmd)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'invalid').mkdir()
            run = subprocess.run([sys.executable, '-m', 'eval.eval_parallel', '--checkpoints', str(root / 'invalid'),
                '--bench-only', '--english-only', '--num-gpus', '1', '--gpu-ids', '0',
                '--output-dir', str(root / 'out'), '--log', str(root / 'launcher.log')], capture_output=True, text=True)
            self.assertEqual(run.returncode, 1, run.stdout + run.stderr)

    def test_real_english_templates_and_cli_output(self):
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast
        from eval.eval_checkpoint import main
        # Mock only dataset retrieval. Use the real pinned task templates,
        # preprocessing, harness scoring, checkpoint loader and output writer.
        examples = {
            'facebook/xnli': dict(premise='3 4', hypothesis='5 6', label=0),
            'facebook/belebele': dict(flores_passage='3 4', question='5 6', mc_answer1='7',
                mc_answer2='8', mc_answer3='9', mc_answer4='10', correct_answer_num='1'),
            'juletxara/xstory_cloze': dict(input_sentence_1='3', input_sentence_2='4', input_sentence_3='5',
                input_sentence_4='6', sentence_quiz1='7', sentence_quiz2='8', answer_right_ending=1),
            'google-research-datasets/paws-x': dict(sentence1='3 4', sentence2='5 6', label=1),
            'Rowan/hellaswag': dict(ctx_a='3', ctx_b='4', endings=['5', '6', '7', '8'], label='0', activity_label='9'),
            'baber/piqa': dict(goal='3 4', sol1='5', sol2='6', label=0),
            'allenai/ai2_arc': dict(question='3 4', choices=dict(text=['5','6','7','8'], label=['A','B','C','D']), answerKey='B'),
            'allenai/winogrande': dict(sentence='3 _ 4', option1='5', option2='6', answer='1'),
        }
        requested = []
        def local_data(path, name=None, **kwargs):
            requested.append((path, name))
            self.assertIn(path, examples)
            rows = Dataset.from_list([examples[path]] * 2)
            return DatasetDict({split: rows for split in ('train', 'validation', 'test', 'eval')})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); checkpoint = root / 'checkpoint-7'; checkpoint.mkdir()
            m = make_model('P7-simple')
            save_file({k: v.clone() for k, v in m.state_dict().items()}, checkpoint / 'model.safetensors')
            (checkpoint / 'trainer_state.json').write_text(json.dumps(dict(global_step=7)))
            recipe = dict(model_config=config().to_dict(), model=dict(tokenizer_name=str(checkpoint)),
                data=dict(block_size=64), training=dict(seed=42, bf16=False),
                pilot=dict(arm='P7-simple', proxy_screen=True, consumer=2, deep_target=8, lm_chunk=3,
                           causal_attention=False, **{'proxy_' + k: v for k, v in asdict(m.settings).items()}))
            (root / 'train_config.json').write_text(json.dumps(recipe))
            backend = Tokenizer(models.WordLevel({**{str(i): i for i in range(31)}, '<|endoftext|>': 31}, unk_token='0'))
            backend.pre_tokenizer = pre_tokenizers.Whitespace()
            tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='0', eos_token='<|endoftext|>', model_max_length=64)
            tok.save_pretrained(checkpoint)
            output = root / 'evaluation'
            groups = ['hellaswag','xnli','belebele','xstorycloze','paws-x','piqa','arc_easy','arc_challenge','winogrande']
            argv = ['eval_checkpoint.py', '--checkpoint', str(checkpoint), '--device', 'cpu',
                    '--bench-only', '--english-only', '--output-dir', str(output), '--batch-size', '2', '--tasks', *groups]
            with patch.object(sys, 'argv', argv), patch('datasets.load_dataset', side_effect=local_data):
                main()
            metrics = json.loads((output / 'eval_benchmarks.json').read_text())
            self.assertEqual(set(metrics), set(resolve_tasks(groups, english_only=True)))
            for task in metrics:
                self.assertIsInstance(metrics[task]['acc,none'], (int, float))
            self.assertEqual(set(requested), {('facebook/xnli','en'),('facebook/belebele','eng_Latn'),
                ('juletxara/xstory_cloze','en'),('google-research-datasets/paws-x','en'),
                ('Rowan/hellaswag',None),('baber/piqa',None),('allenai/ai2_arc','ARC-Easy'),
                ('allenai/ai2_arc','ARC-Challenge'),('allenai/winogrande','winogrande_xl')})
            samples = [json.loads(row) for row in (output / 'eval_samples.jsonl').read_text().splitlines()]
            self.assertEqual(len(samples), 18)
            self.assertTrue(all(isinstance(sample['doc_id'], int) for sample in samples))
            metadata = json.loads((output / 'eval_metadata.json').read_text())
            self.assertEqual(metadata['status'], 'completed')
            self.assertEqual(metadata['arm'], 'P7-simple')
            self.assertEqual(metadata['step'], 7)
            self.assertEqual(len(metadata['checkpoint_sha256']), 64)
            # Fresh-output guard must reject a second evaluation without loading.
            with patch.object(sys, 'argv', argv), patch('eval.eval_checkpoint.load_checkpoint') as load:
                with self.assertRaises(SystemExit) as raised: main()
                self.assertEqual(raised.exception.code, 2)
                load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
