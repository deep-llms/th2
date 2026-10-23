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
                result=dict(bdot_metadata=dict(status='ok',global_step=10000,checkpoint=checkpoint,
                    arguments=dict(limit=1,num_fewshot=0,seed=42,precision='bf16',attention_backend='fa4',max_length=1024),
                    manifest_sha256='dataset',tokenizer_sha256='tokenizer',master_dtype='torch.float32',
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
            f=root/'D/science.json';r=json.loads(f.read_text());r['bdot_metadata']['arguments']['seed']=0
            f.write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError,'protocol'):collect(root,cp,1)


if __name__=='__main__':unittest.main()
