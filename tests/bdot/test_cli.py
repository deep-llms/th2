import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from datasets import Dataset
from test_model import model
from test_data_and_trainer import tokenizer_assets


class CLITests(unittest.TestCase):
    def test_native_entrypoint_resume_and_standalone_eval(self):
        project=Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);tok=tokenizer_assets(p/'tokenizer')
            model().config.save_pretrained(p/'model')
            Dataset.from_dict({'id':[str(i) for i in range(32)],'text':['a b c d '*8]*32}).save_to_disk(p/'raw')
            args=dict(config_name=str(p/'model'),tokenizer_name=str(p/'tokenizer'),data_dir=str(p/'raw'),
                output_dir=str(p/'run'),data_order='document_stream',dataset_revision='fixture',
                validation_manifest=str(p/'manifest.json'),validation_tokens=40,map_batch_size=5,
                data_cache_dir=str(p/'cache'),block_size=8,eval_strategy='steps',eval_steps=1,
                eval_max_sequences=2,do_eval=True,use_cpu=True,report_to='none',
                max_steps=2,stop_after=1,save_steps=1,logging_steps=1,per_device_train_batch_size=1,
                per_device_eval_batch_size=2,gradient_accumulation_steps=2,dataloader_pin_memory=False,
                disable_tqdm=True)
            cfg=p/'args.json'
            def run(argv):
                result=subprocess.run([sys.executable,*map(str,argv)],cwd=project,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1'),
                    capture_output=True,text=True,timeout=180)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            cfg.write_text(json.dumps(args));run(['train.py',cfg])
            args['stop_after']=2;cfg.write_text(json.dumps(args));run(['train.py',cfg])
            stats=json.loads((p/'run/train_results.json').read_text())
            self.assertEqual(stats['global_step'],2)
            self.assertTrue((p/'run/final_eval.json').is_file())
            ev=dict(checkpoint=str(p/'run'),eval_data_dir=str(p/'raw'),output_file=str(p/'ppl.json'),
                    use_cpu=True,batch_size=2,data_cache_dir=str(p/'cache'))
            (p/'eval.json').write_text(json.dumps(ev));run(['eval/packed_ppl.py',p/'eval.json'])
            actual=json.loads((p/'ppl.json').read_text())
            final=json.loads((p/'run/final_eval.json').read_text())
            self.assertAlmostEqual(actual['loss'],final['eval_loss'],places=6)

if __name__=='__main__':unittest.main()
