"""Real two-process CPU/Gloo checks for the production DDP update path."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from pcc.distributed import identical_parameters, wrap
from pcc.joint_config import JointSettings, load_config, settings_for
from pcc.joint_training import evaluate, load_checkpoint, make_optimizer, train, train_update
from test_joint import contexts, tiny


def worker(rank, directory, world=2):
    root = Path(directory)
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + str(root / 'rendezvous'), rank=rank, world_size=world)
    try:
        settings = JointSettings(updates=4, context=8, tokens_per_update=64, warmup=1,
                                 eval_every=2, monitor_tokens=16, dev_tokens=35, s=1, d=2)
        data, dev = contexts(32), contexts(5)
        dev.valid[-1, 3:] = False
        for arm in ('Base', 'Shallow', 'Deep'):
            reference, model = tiny(arm), tiny(arm)
            def optimizer(m):
                return torch.optim.SGD([{'params': list(m.parameters()), 'lr': .01, 'peak_lr': .01}])
            a, b, parallel = optimizer(reference), optimizer(model), wrap(model)
            for step in (1, 2):
                expected = train_update(reference, a, data, step, settings, 1, mixed_precision=False)
                actual = train_update(model, b, data, step, settings, 1, mixed_precision=False, parallel=parallel)
                assert abs(expected['nll'] - actual['nll']) < 1e-6
                assert actual['input_tokens'] == step * 64 and actual['target_tokens'] == 56
                for p, q in zip(reference.parameters(), model.parameters()):
                    torch.testing.assert_close(p, q, atol=2e-7, rtol=1e-5)
                identical_parameters(model)
            # Five rows across two ranks, with a padded final row: no repeats,
            # omissions or changed ordering. Also test fewer rows than ranks.
            for rows in (None, 1):
                expected = evaluate(model, dev, 1, mixed_precision=False, rows=rows)
                actual = evaluate(model, dev, 1, mixed_precision=False, rows=rows, distributed=True)
                np.testing.assert_array_equal(expected[0], actual[0])
                np.testing.assert_array_equal(expected[1], actual[1])
            del parallel

        identity = {'settings': asdict(settings), 'arm': 'Deep', 'distributed': {'world_size': world}}
        torch.manual_seed(100 + rank)
        train(tiny(), data, dev, root / 'full', identity, mixed_precision=False, distributed=True)
        train(tiny(), data, dev, root / 'partial', identity, mixed_precision=False, stop_after=2, distributed=True)
        dist.barrier()
        partial = torch.load(root / 'partial/latest.pt', weights_only=True)
        assert len(partial['rank_rng']) == world
        assert not torch.equal(partial['rank_rng'][0]['torch'], partial['rank_rng'][1]['torch'])
        model = tiny()
        torch.manual_seed(999)
        load_checkpoint(root / 'partial/latest.pt', model, make_optimizer(model), identity, rng_rank=rank)
        assert torch.equal(torch.get_rng_state(), partial['rank_rng'][rank]['torch'])
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, partial['model'][name], atol=0, rtol=0)
        train(tiny(), data, dev, root / 'resumed', identity, mixed_precision=False,
              resume=root / 'partial/latest.pt', distributed=True)
        dist.barrier()
        full = torch.load(root / 'full/final.pt', weights_only=True)
        resumed = torch.load(root / 'resumed/final.pt', weights_only=True)
        for name in full['model']:
            # Four-rank Gloo can change floating-point summation order when a
            # fresh reducer rebuilds buckets. Loaded weights/RNG above must be
            # exact; subsequent updates must agree numerically. Preserve the
            # established bitwise two-rank regression check.
            torch.testing.assert_close(full['model'][name], resumed['model'][name],
                atol=1e-8 if world == 4 else 0, rtol=1e-6 if world == 4 else 0)
        assert full['next_context'] == resumed['next_context'] == 32
        if rank == 0:
            (root / 'passed.json').write_text(json.dumps({'status': 'ok'}))
    finally:
        dist.destroy_process_group()


class DistributedTests(unittest.TestCase):
    def test_gradient_accumulation_evaluation_and_resume_two_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(worker, args=(directory,), nprocs=2, join=True)
            self.assertEqual(json.loads((Path(directory) / 'passed.json').read_text())['status'], 'ok')

    def test_four_process_local_gradient_accumulation_evaluation_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(worker, args=(directory, 4), nprocs=4, join=True)
            self.assertEqual(json.loads((Path(directory) / "passed.json").read_text())["status"], "ok")

    def test_long_eight_gpu_manifest_preserves_global_budget(self):
        from pcc.joint import manifest
        from run_experiments import load_jobs
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = root / 'config.json'
            cfg.write_text(json.dumps({'model_path': 'model', 'train_data': 'train', 'val_data': 'val',
                                       'experiment': 'joint-v2-ddp', 'train_documents': 400000}))
            settings = settings_for(load_config(cfg))
            self.assertEqual(settings.updates, 6144)
            self.assertEqual(settings.updates * settings.tokens_per_update, 201326592)
            manifest(cfg, list(range(8)), root / 'jobs.json', root / 'inputs')
            jobs = load_jobs(root / 'jobs.json')
            self.assertEqual(len(jobs), 7)
            for job in jobs[:6]:
                self.assertEqual(job['gpus'], list(range(8)))
                self.assertIn('--physical-gpus', job['argv'])
                self.assertEqual(job['required_outputs'][0]['json_equals']['updates'], 6144)
            with self.assertRaises(ValueError):
                manifest(cfg, 0, root / 'invalid.json')
            from pcc.joint import main
            with patch('sys.argv', ['pcc.joint', 'train', '--config', str(cfg), '--output', str(root / 'out'),
                                   '--data-dir', str(root / 'inputs'), '--arm', 'Base', '--physical-gpu', '0']):
                with self.assertRaisesRegex(ValueError, 'requires eight GPUs'):
                    main()


if __name__ == '__main__':
    unittest.main()
