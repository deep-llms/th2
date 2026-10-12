"""Offline launch-plan/data contracts; never starts GPU management."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch
from tests.test_train import fixture  # Set CPU/offline defaults before dataset imports.
from datasets import Dataset
from eval import suite
from eval.benchmarks import local_dataset_paths
from scripts.expanded_evaluation_study import make,SUPERVISED
from scripts.materialize_expanded_inputs import materialize
from tests.test_paired_eval import snapshot


class StudyTests(unittest.TestCase):
    def test_position_count_is_explicit_and_frozen(self):
        from eval.paired import protocol
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'protocol.json'
            frozen=protocol.freeze(path,882)
            self.assertEqual(protocol.read(path)['position']['documents'],882)
            self.assertEqual(frozen['confirmatory']['tasks'],protocol.Z)
            with self.assertRaises(ValueError):protocol.freeze(Path(tmp)/'invalid.json',0)
            frozen['position']['documents']=883;path.write_text(json.dumps(frozen))
            with self.assertRaises(ValueError):protocol.read(path)

    def test_explicit_new_protocols_preserve_defaults(self):
        self.assertEqual(suite.selected_protocols(),suite.PROTOCOLS)
        chosen=suite.selected_protocols(['zero','five','math','continuation'],True)
        self.assertEqual(chosen['five'],(5,['mmlu','openbookqa','social_iqa','boolq']))
        self.assertEqual(chosen['math'][0],5)
        self.assertEqual(chosen['continuation'][0],5)
        self.assertNotIn('paloma',chosen)
        with self.assertRaises(ValueError):suite.selected_protocols(['ten'],True)
        with self.assertRaises(ValueError):suite.selected_protocols(['zero','zero'])

    def test_parallel_allocations_and_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobs=[dict(output=str(Path(tmp)/str(i)),argv=['fixture',str(i)]) for i in range(8)]
            seen=[];lock=threading.Lock()
            def run(cmd,**kw):
                with lock:seen.append((cmd,kw['env']['CUDA_VISIBLE_DEVICES']))
                kw['stdout'].write('fixture\n')
            with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'2,5'}),patch('eval.suite.subprocess.run',side_effect=run):
                suite.execute(jobs,2)
            self.assertEqual(len(seen),8);self.assertTrue({g for _,g in seen}<={'2','5'})
            with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'2'}),self.assertRaises(ValueError):suite.execute([],2)
            with patch('eval.suite.subprocess.run',side_effect=subprocess.CalledProcessError(1,['fixture'])):
                with self.assertRaises(subprocess.CalledProcessError):suite.execute([dict(output=str(Path(tmp)/'bad'),argv=['fixture'])])

    def test_offline_raw_materialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            row=dict(lang='en',id='q',question=dict(stem=' Why? ',choices=dict(label=['A','B'],text=['yes','no'])),answerKey='B')
            repo=snapshot(root,'xcsqa_en',[row],['validation'])
            source=root/'raw.json';source.write_text(json.dumps(dict(schema_version=1,repositories=[repo])))
            materialize(root,source,root/'manifest.json')
            config=local_dataset_paths(root,root/'manifest.json')['xcsqa_en']
            from datasets import load_dataset
            rows=load_dataset(config['dataset_path'],**config['dataset_kwargs'])['validation']
            self.assertEqual(rows[0],dict(id='q',question='Why?',choices=['yes','no'],label=1))
            materialize(root,source,root/'reused.json')
            self.assertEqual(json.loads((root/'manifest.json').read_text()),json.loads((root/'reused.json').read_text()))
            (root/'xcsqa_en.parquet').write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError,'checksum'):materialize(root,source,root/'bad.json')

    def test_extended_position_identity_and_duplicate_guards(self):
        import hashlib
        from transformers import AutoTokenizer
        from scripts.extended_position_sample import materialize as extended,sha
        from eval.paired.run import token_hash
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root)
            tok=AutoTokenizer.from_pretrained(root/'model',local_files_only=True)
            texts=['1 2 3 4 5','6 7 8 9 10']
            Dataset.from_dict({'text':[texts[1]]}).save_to_disk(root/'new')
            arrow=next((root/'new').glob('*.arrow'))
            old_manifest=root/'old.json';training_file='subsets/qwen3_0.6b_base_en_30B/train/fixture.arrow'
            file_spec=dict(bytes=1,sha256='fixture')
            old_manifest.write_text(json.dumps(dict(files={training_file:file_spec})))
            audit=root/'overlap.json';audit.write_text(json.dumps(dict(old_manifest_sha256=sha(old_manifest),
                excluded_text_sha256=[],files=[dict(path=training_file,**file_spec)])))
            records=[]
            for source,text in zip(('old_validation','new_validation'),texts):
                records.append(dict(source=source,row=0,text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    prefix_sha256=hashlib.sha256(json.dumps(tok(text)['input_ids'][:4]).encode()).hexdigest()))
            selection=dict(rows=records,old_manifest_sha256=sha(old_manifest),tokenizer_sha256=token_hash(tok),
                overlap_audit_file='overlap.json',overlap_audit_sha256=sha(audit),sources=dict(new_validation=dict(path=str(arrow.relative_to(root)),
                    bytes=arrow.stat().st_size,sha256=sha(arrow))))
            frozen=dict(protocol_sha256='fixture',position=dict(documents=2,length=4))
            path=root/'selection.json';path.write_text(json.dumps(selection));old=Dataset.from_dict({'text':[texts[0]]})
            sample=extended(path,old,root,old_manifest,tok,root/'sample.json',frozen)
            self.assertEqual(len(sample['rows']),2)
            self.assertTrue(sample['rows'][1]['document_id'].startswith('new_validation:0:'))
            selection['rows'][1]=dict(selection['rows'][0]);path.write_text(json.dumps(selection))
            with self.assertRaisesRegex(ValueError,'duplicated'):
                extended(path,old,root,old_manifest,tok,root/'duplicate.json',frozen)
            old_manifest.write_text('changed')
            with self.assertRaisesRegex(ValueError,'training manifest differs'):
                extended(path,old,root,old_manifest,tok,root/'bad-source.json',frozen)

    def test_overlap_scan_finds_training_match_and_checks_file_integrity(self):
        import hashlib
        from scripts.prepare_position_extension import overlap,SUBSET
        from scripts.extended_position_sample import sha
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);dest=root/SUBSET/'train/shard_000000'
            Dataset.from_dict({'text':['duplicate text','different text']}).save_to_disk(dest)
            path=next(dest.glob('*.arrow'))
            (root/'dataset_manifest.json').write_text(json.dumps(dict(files={str(path.relative_to(root)):
                dict(bytes=path.stat().st_size,sha256=sha(path))})))
            result=overlap(root,[dict(text='duplicate text'),dict(text='new text')])
            self.assertEqual(result['training_documents'],2)
            self.assertEqual(result['excluded_text_sha256'],[hashlib.sha256(b'duplicate text').hexdigest()])
            path.write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError,'Training file differs'):
                overlap(root,[dict(text='new text')])

    def test_queue_can_recompute_historical_scores_without_legacy_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobs=make(Path(tmp)/'study',Path.cwd(),{'A10':'/a10','A20':'/a20','P20':'/p20'},
                '/data','/manifest','/protocol','/position',None)
            for job in jobs:
                self.assertNotIn('--legacy-f',job['argv'])
                self.assertNotIn(None,job['argv'])

    def test_queue_order_and_unchanged_finetune_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'study'
            jobs=make(root,Path.cwd(),{'A10':'/a10','A20':'/a20','P20':'/p20'},'/data','/manifest','/protocol','/position','/legacy')
            names=[j['name'] for j in jobs]
            self.assertLess(names.index('expanded-full'),names.index('finetune-expanded'))
            self.assertLess(names.index('finetune-expanded'),names.index('paired-plan'))
            self.assertLess(names.index('paired-evaluate'),names.index('analyze-Z'))
            self.assertIn('--defer-paloma',jobs[names.index('paired-plan')]['argv'])
            self.assertIn('--legacy-f',jobs[names.index('paired-plan')]['argv'])
            fit=json.loads((root/'finetune-plan/jobs.json').read_text())['jobs']
            self.assertTrue(any('glue_mnli_mismatched' in j['name'] for j in fit))
            for path in (root/'finetune-plan/configs').glob('*.json'):
                cfg=json.loads(path.read_text())
                self.assertEqual(cfg['expected_step'],20000)
                self.assertEqual(cfg['save_total_limit'],0)
                self.assertTrue(cfg['output_dir'].startswith(str(root/'supervised/run/finetune')))
                self.assertIn(cfg['seed'],[42,43,44])
                self.assertEqual(cfg['per_device_train_batch_size'],16)
            scope=json.loads((root/'study-scope.json').read_text())
            self.assertEqual(scope['supervised_tasks'],SUPERVISED)

if __name__=='__main__':unittest.main()
