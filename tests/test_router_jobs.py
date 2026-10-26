import json
from pathlib import Path
import tempfile
import unittest
from scripts.launch_group_diagnostic import router_jobs_for
from run_experiments import load_jobs


class RouterJobTests(unittest.TestCase):
    def test_sequential_gpu_gates_and_fixed_pilot(self):
        config = router_jobs_for('/model', '/probe', '/accelerate.yaml', 29681)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'jobs.json'
            path.write_text(json.dumps(config))
            load_jobs(path)
        jobs = config['jobs']
        self.assertEqual(len(jobs), 3)
        self.assertEqual(jobs[0]['gpus'], [0])
        for job in jobs[1:]:
            self.assertEqual(job['gpus'], list(range(8)))
            self.assertTrue(job['required_outputs'])
        self.assertIn('efficient_llm.router_pilot', jobs[1]['argv'])
        self.assertEqual(jobs[1]['argv'][jobs[1]['argv'].index('--steps')+1], '512')
        self.assertIn('efficient_llm.decode_profile', jobs[2]['argv'])
