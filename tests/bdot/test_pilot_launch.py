"""CPU guards for the real pilot's initialization, scientific gates and queue."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from test_model import model
from scripts.bdot_pipeline_check import initial_pair, summarize_run, compare_runs
from scripts.bdot_pilot_pipeline import gated_jobs
from run_experiments import load_jobs

ROOT = Path(__file__).resolve().parents[2]


class PilotLaunchTests(unittest.TestCase):
    def test_random_initializations_map_exactly(self):
        cfg = model(routing='fixed', curriculum='none', r=0, step_film=False).config.to_dict()
        native = copy.deepcopy(cfg)
        del native['bdot']
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, config in [('native', native), ('bdot', cfg)]:
                (root/(name+'.json')).write_text(json.dumps(config))
            result = initial_pair(root/'native.json', root/'bdot.json', 0)
            self.assertEqual(result['status'], 'passed')
            self.assertEqual(result['layers'], cfg['num_hidden_layers'])

    def test_pipeline_loss_gate_rejects_mismatch_and_missing_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entries = [dict(step=s, loss=3.+s/10000) for s in range(410,501,10)]
            entries += [dict(step=500, eval_loss=3., eval_num_tokens=1020, eval_num_sequences=1)]
            (root/'trainer_state.json').write_text(json.dumps(dict(global_step=500, log_history=entries)))
            (root/'train_results.json').write_text(json.dumps(dict(global_step=500, train_runtime=100.)))
            a = summarize_run(root)
            self.assertEqual(compare_runs(a, a)['status'], 'passed')
            b = dict(a, last_window_lm_loss=a['last_window_lm_loss']+.02)
            self.assertEqual(compare_runs(a,b)['status'], 'failed')
            with self.assertRaisesRegex(ValueError,'validation sets differ'):
                compare_runs(a, dict(a, validation_tokens=10))
            entries.pop(0)
            (root/'trainer_state.json').write_text(json.dumps(dict(global_step=500, log_history=entries)))
            with self.assertRaisesRegex(ValueError,'Incomplete'):
                summarize_run(root)

    def test_generated_grouped_queue_is_gated_with_complete_schedule(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            out=root/'prepared'
            subprocess.run([sys.executable, str(ROOT/'scripts/make_bdot_pilot.py'),
                '--tokenizer', str(root/'tokenizer'), '--data', str(root/'train'),
                '--eval-data', str(root/'validation'), '--revision', 'a'*40,
                '--accelerate-config', str(root/'accelerate.yaml'), '--output', str(out),
                '--expert-backend', 'grouped'],check=True,capture_output=True)
            jobs=gated_jobs(out,root/'accelerate.yaml')
            file=root/'jobs.json';file.write_text(json.dumps(jobs));load_jobs(file)
            self.assertEqual([j['name'] for j in jobs['jobs']],['T12','T14','D','P1-A','P1-B'])
            self.assertIn('1000',jobs['jobs'][0]['argv'])
            for arm in ['D','P1-A','P1-B']:
                cfg=json.loads((out/(arm+'.model.json')).read_text())
                args=json.loads((out/(arm+'.train.json')).read_text())
                self.assertEqual(cfg['bdot']['expert_backend'],'grouped')
                self.assertEqual(args['max_steps'],4768)
                self.assertEqual(args['warmup_steps'],96)
                self.assertNotIn('stop_after',args)
                self.assertEqual(args['per_device_train_batch_size']*args['gradient_accumulation_steps']*8,512)
                self.assertIsNone(args['save_total_limit'])

    def test_cache_preparation_accepts_full_schedule_without_cutoff(self):
        from datasets import Dataset
        from test_data_and_trainer import tokenizer_assets
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            tokenizer_assets(root/'tokenizer')
            Dataset.from_dict({'text':['a b c d '*32]*64}).save_to_disk(root/'train')
            Dataset.from_dict({'text':['a b c d '*32]*2}).save_to_disk(root/'validation')
            options=dict(tokenizer_name=str(root/'tokenizer'), data_dir=str(root/'train'),
                eval_data_dir=str(root/'validation'), output_dir=str(root/'run'),
                max_steps=1, block_size=8, report_to='none', do_eval=True,
                preprocessing_num_workers=1, document_end_token='[EOS]')
            file=root/'args.json';file.write_text(json.dumps(options))
            subprocess.run([sys.executable, str(ROOT/'scripts/warm_training_cache.py'),
                '--arguments',str(file),'--output',str(root/'ready.json')],
                check=True,capture_output=True,text=True)
            receipt=json.loads((root/'ready.json').read_text())
            self.assertEqual(receipt['status'],'passed')
            self.assertGreaterEqual(receipt['train_sequences'],512)


if __name__ == '__main__':
    unittest.main()
