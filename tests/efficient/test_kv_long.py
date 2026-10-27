import json
import math
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from efficient_llm.diagnostic_data import choose_documents, sha256
from efficient_llm.kv_long import DOCUMENTS, STEPS, TOKENS, jobs_for, verify_probe, verify_handoff
from efficient_llm.kv_mlp_study import recovery_summary
from run_experiments import load_jobs


class LongRecoveryTests(unittest.TestCase):
    def test_handoff_rejects_unfinished_predecessor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'prior';(run/'supervised/run').mkdir(parents=True)
            burn=run/'source/resources/llm_pretrain_burn.py'
            (run/'supervised/run/complete.json').write_text('{"status":"ok"}')
            (run/'supervised/supervisor.json').write_text('{"training_status":"ok"}')
            proof=run/'supervised/burn-verified.json'
            proof.write_text('{"collective_progress_verified":true}')
            self.assertEqual(verify_handoff(root,burn,sha256)['root'],str(run))
            proof.write_text('{"collective_progress_verified":false}')
            with self.assertRaisesRegex(ValueError,'incomplete'):verify_handoff(root,burn,sha256)

    def test_batched_fast_tokenizer_preserves_selection_and_duplicates(self):
        backend = Tokenizer(WordLevel({'[UNK]':0, 'a':1, 'b':2, 'c':3, 'd':4}, unk_token='[UNK]'))
        backend.pre_tokenizer = Whitespace()
        tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]')
        texts = ['a b c d a b', 'b c d a', 'a', 'c d a b c', 'a b c d a b',
                 'a  b c d a b', 'd a b c d', 'b b c c d', 'c c c c c']
        ds = [{'text':t} for t in texts]
        for seed in range(5):
            for count in (1, 3, 5, 20):
                expected = choose_documents(ds, tok, count=count, length=4, seed=seed,
                    excluded=set(), source='test', eod_id=9)
                for batch in (2, 4, 256):
                    actual = choose_documents(ds, tok, count=count, length=4, seed=seed,
                        excluded=set(), source='test', eod_id=9, tokenizer_batch_size=batch)
                    self.assertEqual(actual, expected)

    def test_jobs_gate_both_smokes_then_matched_long_pair(self):
        config = jobs_for('/model','/probe','/config',29721,'/parent')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'jobs.json';path.write_text(json.dumps(config));load_jobs(path)
        self.assertEqual(STEPS,20752)
        self.assertEqual(DOCUMENTS,1328128)
        self.assertEqual(TOKENS,1360003072)
        jobs=config['jobs'];self.assertEqual(len(jobs),6)
        self.assertEqual(jobs[0]['gpus'],[0])
        for job in jobs[1:5]:
            self.assertEqual(job['gpus'],list(range(8)))
            self.assertIn('/parent/supervised/run/A/final',job['argv'])
            self.assertNotIn('--qkv-only',job['argv'])
        for job in jobs[1:3]:self.assertIn('--smoke',job['argv'])
        for job in jobs[3:5]:
            self.assertIn(str(STEPS),job['argv']);self.assertIn('10376',job['argv'])
            self.assertEqual(job['required_outputs'][0]['json_equals']['input_tokens'],TOKENS)
        self.assertNotIn('--sharing-layout',jobs[3]['argv'])
        self.assertIn('uniform25',jobs[4]['argv'])

    def test_long_summary_requires_same_dense_parent_recipe_and_actual_order(self):
        keys=['learning_rate','lr_scheduler_type','warmup_steps','weight_decay','max_grad_norm',
              'per_device_train_batch_size','gradient_accumulation_steps','seed','data_seed','bf16']
        def score(loss):return dict(ppl=math.exp(loss),per_document_nll=[loss]*8)
        base=dict(status='passed',arm='A',smoke=False,phase='new_recovery',trainable_scope='all',
            steps=STEPS,world_size=8,input_tokens=TOKENS,parent_steps=512,
            parent_result_sha256='parent',initial_model_files={'weights':'same'},mlp_widths=[32]*4,
            kv_sources=[0,1,2,3],sample_order_sha256_per_rank=['same']*8,probe_sha256='same',
            training_arguments=dict.fromkeys(keys,1),validation={'0':score(3.1),str(STEPS):score(3.)},
            inference_validation=score(3.))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'A').mkdir();(root/'B').mkdir()
            (root/'A/result.json').write_text(json.dumps(base))
            b=deepcopy(base);b.update(arm='B',phase='placement_conversion',sharing_layout='uniform25',
                kv_sources=[0,1,2,2],inference_validation=score(3.02))
            path=root/'B/result.json';path.write_text(json.dumps(b))
            args=SimpleNamespace(runs=root,output=root/'summary.json')
            recovery_summary(args,matched_placement=True)
            self.assertEqual(json.loads(args.output.read_text())['decision'],'evaluate_efficiency')
            for field,value in [('parent_result_sha256','different'),('initial_model_files',{}),
                    ('trainable_scope','affected_qkv'),('sample_order_sha256_per_rank',['bad']*8)]:
                changed={**b,field:value};path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):recovery_summary(args,matched_placement=True)

    def test_probe_extension_is_exact_and_disjoint(self):
        def record(i):return dict(text_sha256='text'+str(i),tokens_sha256='tokens'+str(i))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);parent=root/'parent';parent.mkdir()
            (parent/'manifest.json').write_text(json.dumps(dict(documents={'calibration':[record(0)],'evaluation':[record(9)]})))
            for name,n in [('short',2),('long',4)]:
                path=root/name;path.mkdir()
                torch.save(dict(calibration=torch.arange(1,13).reshape(4,3)[:n],evaluation=torch.tensor([[8,9,10]])),path/'tokens.pt')
                report=dict(status='passed',length=2,calibration_documents=n,evaluation_documents=1,
                    tokens_sha256=sha256(path/'tokens.pt'),seed=9042,evaluation_reused=True,
                    recovery_from_manifest_sha256=sha256(parent/'manifest.json'),
                    documents=dict(calibration=[record(i) for i in range(1,n+1)],evaluation=[record(9)]))
                (path/'manifest.json').write_text(json.dumps(report))
            result=verify_probe(root/'long',root/'short',parent,documents=4)
            self.assertEqual(result['exact_short_training_prefix_documents'],2)
            path=root/'long/manifest.json';r=json.loads(path.read_text())
            r['documents']['calibration'][-1]=record(0);path.write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError,'overlapping'):verify_probe(root/'long',root/'short',parent,documents=4)


if __name__ == '__main__':unittest.main()
