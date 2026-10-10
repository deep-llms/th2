import json
from pathlib import Path
import tempfile
import unittest
from scripts.checkpoint_downstream_study import make

class CheckpointStudyTests(unittest.TestCase):
    def test_three_checkpoints_keep_separate_outputs_and_exact_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'study'
            spec={f'/checkpoints/{label}':dict(label=label,arm=arm,step=step,checkpoint_sha256=label)
                  for label,arm,step in [('A-10000','A',10000),('P6-iso-10000','P6-iso',10000),('A-10800','A',10800)]}
            items=make(root,Path.cwd(),spec,dict(pairs='/data',extended='/more-data'),dict(pairs='/manifest',extended='/more-manifest'))
            self.assertFalse((root/'supervised/run').exists())
            self.assertEqual(len(items),18)
            studies=[j for j in items if j['name'].startswith('finetune-')]
            self.assertEqual(len(studies),6)
            for job in studies:
                self.assertEqual(job['gpus'],list(range(8)))
                self.assertIn('--run-dir',job['argv'])
            for checkpoint,entry in spec.items():
                for group in ('pairs','extended'):
                    plan=root/'finetune-plans'/entry['label']/group
                    for path in (plan/'configs').glob('*.json'):
                        cfg=json.loads(path.read_text())
                        self.assertEqual(cfg['checkpoint'],checkpoint)
                        self.assertEqual(cfg['expected_step'],entry['step'])
                        self.assertEqual(cfg['save_total_limit'],0)
                        self.assertTrue(cfg['output_dir'].startswith(str(root/'supervised/run/finetune'/entry['label']/group)))
                    nested=json.loads((plan/'jobs.json').read_text())['jobs']
                    self.assertTrue(all(j['gpus']==list(range(8)) for j in nested if j['name'].startswith(('search-','confirm-'))))
            for job in items:
                if job['name'].startswith('validate-'):
                    self.assertIn('--checkpoint-spec',job['argv'])

if __name__=='__main__':unittest.main()
