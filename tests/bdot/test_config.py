import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import torch
from transformers import AutoConfig,Qwen3ForCausalLM
from bdot.model import BDOTForCausalLM

ROOT=Path(__file__).resolve().parents[2]

class ConfigTests(unittest.TestCase):
    def test_parameter_counts_without_allocating_real_models(self):
        counts={}
        for name in ('dense_native','D','P1-A','P1-B','R','N-A','N-B'):
            config=AutoConfig.from_pretrained(ROOT/'configs/bdot'/f'{name}.json',local_files_only=True)
            config._attn_implementation='sdpa'
            with torch.device('meta'):
                m=BDOTForCausalLM(config) if hasattr(config,'bdot') else Qwen3ForCausalLM(config)
            counts[name]=sum(p.numel() for p in m.parameters())
            self.assertLess(abs(counts[name]/344400000-1),.005)
        self.assertEqual(counts['P1-A'],counts['P1-B'])
        self.assertEqual(counts['N-A'],counts['P1-A'])

    def test_queue_generation_only(self):
        from run_experiments import load_jobs
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'plan'
            result=subprocess.run([sys.executable,str(ROOT/'scripts/make_bdot_pilot.py'),
                '--tokenizer','/assets/tokenizer','--data','/assets/data','--revision','a'*40,
                '--eval-data','/assets/validation',
                '--manifest','/assets/validation.json','--accelerate-config','/assets/accelerate.yaml',
                '--output',str(out)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            jobs=load_jobs(out/'jobs.json');self.assertEqual([j['name'] for j in jobs],['D','P1-A','P1-B'])
            config=json.loads((out/'P1-A.train.json').read_text())
            self.assertEqual(config['per_device_train_batch_size']*8*config['gradient_accumulation_steps'],512)
            self.assertEqual(config['max_steps'],4768)
            self.assertEqual(config['warmup_steps'],96)
            self.assertNotIn('stop_after',config)
            self.assertEqual(config['data_order'],'packed_shuffle')
            self.assertEqual(config['train_sampling_strategy'],'sequential')
            self.assertEqual(config['preprocessing_num_workers'],128)
            self.assertEqual(config['eval_max_sequences'],1024)
            self.assertEqual(config['eval_data_dir'],'/assets/validation')
            self.assertEqual(jobs[0]['gpus'],list(range(8)))
            self.assertIsNone(config['save_total_limit'])

if __name__=='__main__':unittest.main()
