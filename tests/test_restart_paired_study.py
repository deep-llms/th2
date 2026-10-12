import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from eval.paired import protocol
from scripts.restart_paired_study import prepare, PREVIOUS_RUN_SHA256, STAGES


class RestartTests(unittest.TestCase):
    def fixture(self, root):
        source=root/'old';run=source/'supervised/run';(run/'paired').mkdir(parents=True)
        (run/'finetune').mkdir()
        def write(p,v):p.write_text(json.dumps(v))
        burn=dict(collective_progress_verified=True)
        write(source/'supervised/supervisor.json',dict(burn=burn))
        write(source/'supervised/burn-verified.json',burn)
        write(run/'run.json',dict(status='failed',jobs=[dict(name='paired-evaluate')]))
        write(run/'finetune/summary.json',dict(status='passed'))
        frozen=source/'protocol.json';sample=source/'sample.json'
        old=dict(position=dict(documents=2,length=2),rules={'fixed':True},implementation={'eval/paired/run.py':PREVIOUS_RUN_SHA256})
        old['protocol_sha256']=protocol.digest(old);write(frozen,old)
        rows=[dict(document_id=str(i),input_ids=[1,i],token_sha256=protocol.digest([1,i])) for i in range(2)]
        data=dict(rows=rows,length=2,tokenizer_sha256='tok',protocol_sha256=old['protocol_sha256'])
        data['sample_sha256']=protocol.digest(data);write(sample,data)
        plan=dict(protocol=str(frozen),protocol_sha256=old['protocol_sha256'],position_sample=str(sample),sample_sha256=data['sample_sha256'])
        plan['plan_sha256']=protocol.digest(plan);write(run/'paired/plan.json',plan)
        jobs=[dict(name=s,argv=['python','--protocol',str(frozen),'--position-sample',str(sample)],required_outputs=[dict(path=s+'.json')]) for s in STAGES]
        jobs[1]['gpus']=list(range(8))
        write(run/'jobs.snapshot.json',dict(jobs=jobs))
        new=copy.deepcopy(old);del new['protocol_sha256'];new['implementation']['eval/paired/run.py']='fixed'
        return source,new

    def test_fresh_retry_preserves_old_files_and_science(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,new=self.fixture(root)
            before={p:p.read_bytes() for p in source.rglob('*.json')}
            target=root/'new'
            with patch('eval.paired.protocol.definition',return_value=new):prepare(source,target)
            for p,data in before.items():self.assertEqual(p.read_bytes(),data)
            receipt=json.loads((target/'restart-receipt.json').read_text());self.assertTrue(receipt['rows_unchanged'])
            jobs=json.loads((target/'jobs.json').read_text())['jobs']
            self.assertEqual([j['name'] for j in jobs],STAGES)
            self.assertEqual(jobs[1]['gpus'],list(range(8)))
            for job in jobs:
                self.assertIn(str(target/'paired-protocol.json'),job['argv'])
                self.assertIn(str(target/'position-sample.json'),job['argv'])
            with self.assertRaises(FileExistsError):prepare(source,target)

    def test_rejects_changes_and_missing_handoff(self):
        for mode in ('science','implementation','burn'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);source,new=self.fixture(root)
                if mode=='science':new['rules']['fixed']=False
                elif mode=='implementation':new['implementation']['other.py']='changed'
                else:(source/'supervised/burn-verified.json').write_text('{}')
                with patch('eval.paired.protocol.definition',return_value=new),self.assertRaises(ValueError):prepare(source,root/'new')
                self.assertFalse((root/'new').exists())
