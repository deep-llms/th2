import json
from pathlib import Path
import tempfile
import unittest
from scripts.launch_group_diagnostic import router_jobs_for, context_jobs_for, deploy_jobs_for, research_jobs_for, numerical_jobs_for, window_jobs_for
from run_experiments import load_jobs


class RouterJobTests(unittest.TestCase):
    def test_window_screen_is_gated_inference_only(self):
        config=window_jobs_for('/model','/inputs','/accelerate.yaml',29721)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'jobs.json';path.write_text(json.dumps(config));load_jobs(path)
        jobs=config['jobs']
        self.assertEqual([j['gpus'] for j in jobs],[[0],list(range(8)),list(range(8))])
        self.assertIn('gate',jobs[0]['argv'])
        self.assertIn('profile',jobs[1]['argv'])
        self.assertIn('quality',jobs[2]['argv'])
        self.assertFalse(jobs[1]['required_outputs'][0]['json_equals']['native_backend_accepted'])
        self.assertTrue(all('train' not in j['argv'] for j in jobs))

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

    def test_context_study_gates_and_budget(self):
        config=context_jobs_for('/model','/root/probe','/accelerate.yaml',29681)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'jobs.json';path.write_text(json.dumps(config));load_jobs(path)
        jobs=config['jobs']
        self.assertEqual([j['gpus'] for j in jobs],[[0],list(range(8)),list(range(8))])
        self.assertIn('efficient_llm.context_check',jobs[0]['argv'])
        self.assertIn('/root/long-probe',jobs[1]['argv'])
        for option,value in [('--keep','4'),('--window','128'),('--steps','512')]:
            self.assertEqual(jobs[1]['argv'][jobs[1]['argv'].index(option)+1],value)
        self.assertIn('efficient_llm.context_profile',jobs[2]['argv'])

    def test_deployment_uses_existing_fit_and_gate_first(self):
        config=deploy_jobs_for('/model','/inputs','/accelerate.yaml',29721)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'jobs.json';path.write_text(json.dumps(config));load_jobs(path)
        jobs=config['jobs']
        self.assertEqual([j['gpus'] for j in jobs],[[0],list(range(8)),list(range(8))])
        self.assertIn('gate',jobs[0]['argv'])
        self.assertIn('profile',jobs[1]['argv'])
        self.assertIn('retrieval',jobs[2]['argv'])
        self.assertTrue(all('train' not in j['argv'] for j in jobs))

    def test_research_does_not_accept_backend_or_use_it_for_retrieval(self):
        for factory in (numerical_jobs_for,research_jobs_for):
            config=factory('/model','/inputs','/accelerate.yaml',29721)
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'jobs.json';path.write_text(json.dumps(config));load_jobs(path)
            self.assertTrue(all(j['gpus']==list(range(8)) for j in config['jobs']))
        jobs=config['jobs']
        self.assertIn('--diagnostic-only',jobs[0]['argv'])
        self.assertEqual(jobs[0]['required_outputs'][0]['json_equals']['status'],'measured')
        self.assertIn('--reference-only',jobs[1]['argv'])
