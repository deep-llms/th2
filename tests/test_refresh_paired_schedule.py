import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from eval.paired import protocol,position
from scripts.refresh_paired_schedule import refresh


class RefreshTests(unittest.TestCase):
    def fixture(self,root):
        queue=root/'queue';run=queue/'supervised/run';run.mkdir(parents=True)
        frozen=root/'protocol.json';sample=root/'sample.json';expected=root/'expected.json'
        old=dict(version='fixture',position=dict(documents=1,length=2),rules={'fixed':True},implementation={'old.py':'old'})
        old['protocol_sha256']=protocol.digest(old);frozen.write_text(json.dumps(old))
        rows=[dict(document_id='first',input_ids=[1,2],token_sha256=protocol.digest([1,2]))]
        data=dict(rows=rows,length=2,tokenizer_sha256='tok',protocol_sha256=old['protocol_sha256'])
        data['sample_sha256']=protocol.digest(data);sample.write_text(json.dumps(data))
        expected.write_text(json.dumps(dict(before={'old.py':'old'},added=['parallel.py'])))
        (run/'run.json').write_text(json.dumps(dict(status='running',jobs=[dict(name='finetune-expanded',status='running')])))
        (run/'jobs.snapshot.json').write_text(json.dumps(dict(jobs=[dict(name='paired-plan',argv=['--protocol',str(frozen),'--position-sample',str(sample)])])))
        new={k:v for k,v in old.items() if k!='protocol_sha256'};new['implementation']={'old.py':'new','parallel.py':'added'}
        return queue,frozen,sample,expected,new

    def test_refreeze_preserves_examples_and_original_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);queue,frozen,sample,expected,new=self.fixture(root)
            old_protocol=frozen.read_bytes();old_sample=sample.read_bytes()
            with patch('eval.paired.protocol.definition',return_value=new):
                refresh(queue,expected,root/'receipt.json')
                position.read_sample(sample,protocol.read(frozen))
            self.assertEqual(frozen.with_name(frozen.name+'.before-parallel-20261013').read_bytes(),old_protocol)
            self.assertEqual(sample.with_name(sample.name+'.before-parallel-20261013').read_bytes(),old_sample)
            self.assertEqual(json.loads(old_sample)['rows'],json.loads(sample.read_text())['rows'])
            self.assertTrue(json.loads((root/'receipt.json').read_text())['rows_unchanged'])

    def test_refuses_changed_science_or_started_paired_stage(self):
        for mode in ('rules','active'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);queue,frozen,sample,expected,new=self.fixture(root)
                before=frozen.read_bytes()
                if mode=='rules':new['rules']={'fixed':False}
                else:(queue/'supervised/run/paired').mkdir()
                with patch('eval.paired.protocol.definition',return_value=new),self.assertRaises(ValueError):
                    refresh(queue,expected,root/'receipt.json')
                self.assertEqual(frozen.read_bytes(),before)
                self.assertFalse((root/'receipt.json').exists())
