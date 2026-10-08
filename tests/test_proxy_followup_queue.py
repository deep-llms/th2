import json
from pathlib import Path
import tempfile
import unittest
from scripts.proxy_followup_queue import GROUPS, make


class FollowupQueueTest(unittest.TestCase):
    def test_mixed_seeds_preserve_recipe_and_validate_before_next_run(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'jobs.json'
            items = make('proxy_heads.b200.json', path)
            self.assertEqual(len(items), 20)
            training = [item for item in items if 'gpus' in item]
            expected = [(seed, arm) for seed, arms in GROUPS for arm in arms]
            self.assertEqual(len(training), 9)
            for item, (seed, arm) in zip(training, expected):
                argv = item['argv']
                def arg(name): return argv[argv.index('--'+name)+1]
                self.assertEqual(argv[:2], ['env', f'PYTHONHASHSEED={seed}'])
                self.assertEqual((arg('seed'), arg('data_seed')), (str(seed), str(seed)))
                self.assertEqual(arg('arm'), arm)
                if arm == 'A': self.assertNotIn('--proxy_module_seed', argv)
                else: self.assertEqual(arg('proxy_module_seed'), str(seed+1))
                self.assertEqual(item['gpus'], list(range(8)))
                for key, value in dict(stop_after=2500,max_steps=28600,warmup_steps=1430,
                    per_device_train_batch_size=16,gradient_accumulation_steps=4,
                    block_size=2048,eval_rows=4882,monitor_rows=128,attention_backend='fa4').items():
                    self.assertEqual(arg(key), str(value))
                self.assertEqual(arg('output_dir'), f'{{run_dir}}/seed-{seed}/{arm}')
                following=items[items.index(item)+1]
                self.assertEqual(following['name'], f'seed-{seed}-validate-{arm}')
                self.assertEqual(following['required_outputs'][0]['json_equals'], {'status':'passed'})
            self.assertEqual(json.loads(path.read_text())['jobs'], items)
            outputs=[o['path'] for j in items for o in j['required_outputs']]
            self.assertEqual(len(outputs),len(set(outputs)))
            with self.assertRaises(FileExistsError):make('proxy_heads.b200.json',path)


if __name__ == '__main__':unittest.main()
