import json
from pathlib import Path
import tempfile
import unittest
from eval.downstream_suite import collect, GROUPS, COUNTS, PRIMARY


class DownstreamCollectionTests(unittest.TestCase):
    def fixture(self, root):
        checkpoints={'D':'/test/D/checkpoint-10000','Partial-25-cosine':'/test/P/checkpoint-10000'}
        for arm,checkpoint in checkpoints.items():
            (root/arm).mkdir()
            for group,tasks in GROUPS.items():
                result=dict(kv_metadata=dict(status='ok',global_step=20752,checkpoint=checkpoint,
                    arguments=dict(limit=1,num_fewshot=0,seed=42,precision='bf16',attention_backend='sdpa',max_length=1024),
                    manifest_sha256='dataset',tokenizer_sha256='tokenizer',weight_dtype='torch.bfloat16',
                    packages={'lm-eval':'0.4.13'},weights_sha256=arm),
                    results={t:{PRIMARY[t]:.5} for t in tasks},
                    samples={t:[dict(doc_id=0,doc={'x':'text'},arguments=[['Q',' A']],target='A')] for t in tasks},
                    **{'n-samples':{t:dict(original=COUNTS[t],effective=1) for t in tasks},
                       'n-shot':{t:0 for t in tasks}})
                (root/arm/(group+'.json')).write_text(json.dumps(result))
        return checkpoints

    def test_pair_and_average(self):
        with tempfile.TemporaryDirectory() as p:
            root=Path(p);cp=self.fixture(root);r=collect(root,cp,1)
            self.assertEqual(r['status'],'ok');self.assertTrue(r['diagnostic'])
            self.assertEqual(r['scores']['D']['multiple_choice_average'],.5)

    def test_changed_prompt_and_missing_population_fail(self):
        with tempfile.TemporaryDirectory() as p:
            root=Path(p);cp=self.fixture(root)
            f=root/'Partial-25-cosine/commonsense.json';r=json.loads(f.read_text())
            r['samples']['hellaswag'][0]['arguments']=[['different',' A']];f.write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError,'Unpaired'):collect(root,cp,1)
            r['samples']['hellaswag']=[];f.write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError,'population'):collect(root,cp,1)

    def test_wrong_protocol_fails(self):
        with tempfile.TemporaryDirectory() as p:
            root=Path(p);cp=self.fixture(root)
            f=root/'D/science.json';r=json.loads(f.read_text());r['kv_metadata']['arguments']['seed']=0
            f.write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError,'protocol'):collect(root,cp,1)

    def test_changed_checkpoint_fails(self):
        with tempfile.TemporaryDirectory() as p:
            root=Path(p);cp=self.fixture(root)
            with self.assertRaisesRegex(ValueError,'preflight'):
                collect(root,cp,1,{k:'wrong' for k in cp})

    def test_queue_smoke_gates_full_and_uses_eight_workers(self):
        from scripts.launch_kv_downstream import jobs_for
        from run_experiments import load_jobs
        with tempfile.TemporaryDirectory() as p:
            path=Path(p)/'jobs.json';path.write_text(json.dumps(jobs_for('/cp','/data','/manifest','/tokenizer')))
            load_jobs(path)
            jobs=json.loads(path.read_text())['jobs']
            self.assertEqual([j['name'] for j in jobs],['adapter-check','smoke','full'])
            self.assertEqual(jobs[-1]['gpus'],list(range(8)))
            self.assertIn('--limit',jobs[1]['argv']);self.assertNotIn('--limit',jobs[2]['argv'])

    def test_tokenizer_manifest_uses_byte_and_hash_records(self):
        import hashlib
        from scripts.launch_kv_downstream import verify_tokenizer
        with tempfile.TemporaryDirectory() as p:
            root=Path(p);f=root/'config.json';f.write_bytes(b'{}')
            files={'config.json':dict(bytes=2,sha256=hashlib.sha256(b'{}').hexdigest())}
            verify_tokenizer(root,files)
            f.write_bytes(b'[]')
            with self.assertRaisesRegex(ValueError,'tokenizer'):verify_tokenizer(root,files)
            f.write_bytes(b'{} ')
            with self.assertRaisesRegex(ValueError,'tokenizer'):verify_tokenizer(root,files)


if __name__=='__main__':unittest.main()
