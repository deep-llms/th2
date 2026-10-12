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
