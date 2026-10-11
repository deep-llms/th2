import json
from pathlib import Path
import tempfile
import unittest

import torch
from scripts.checkpoint_ppl_study import evaluate, make, report, prepare_output, ARMS, STEPS
from tests.test_proxy_training import proxy_fixture
from tests.test_train import invoke


def output_worker(rank, tmp):
    from datetime import timedelta
    import time
    import torch.distributed as dist
    root = Path(tmp)
    dist.init_process_group('gloo', init_method=f'file://{root}/rendezvous',
        rank=rank, world_size=2, timeout=timedelta(seconds=20))
    try:
        if rank == 1:
            # Force rank zero to create the directory before this worker checks it.
            deadline = time.monotonic() + 10
            while not (root/'output').exists():
                if time.monotonic() > deadline: raise TimeoutError('Main rank did not create output')
                time.sleep(.01)
        prepare_output(root/'source', root/'output', is_main_process=rank == 0)
        assert (root/'output').is_dir()
        try:
            prepare_output(root/'source', root/'output', is_main_process=rank == 0)
        except ValueError as exc:
            assert 'fresh evaluation' in str(exc)
        else:
            raise AssertionError('Existing output accepted')
    finally:
        dist.destroy_process_group()


class CheckpointPPLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_output_reservation_with_late_distributed_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            torch.multiprocessing.spawn(output_worker, args=(tmp,), nprocs=2, join=True)

    def test_saved_final_control_and_intermediate_checkpoint(self):
        for arm in ARMS:
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);cfg=proxy_fixture(root);source=root/arm
                invoke(root,{**cfg,'arm':arm,'output_dir':str(source),'stop_after':2})
                evaluate(source,root/'final',2)
                final=json.loads((root/'final/summary.json').read_text())
                self.assertLessEqual(final['final_control_gap'],1e-5)
                self.assertTrue(final['model_state_unchanged'])
                evaluate(source,root/'early',1)
                early=json.loads((root/'early/summary.json').read_text())
                self.assertIsNone(early['final_control_gap'])
                self.assertEqual(early['step'],1)
                self.assertEqual(early['metrics']['eval_target_tokens'],final['metrics']['eval_target_tokens'])
                self.assertFalse(list((root/'early').glob('checkpoint-*')))
                with self.assertRaisesRegex(ValueError,'fresh evaluation'):evaluate(source,root/'early',1)
                path=source/'train_config.json';saved=json.loads(path.read_text())
                saved['eval_fingerprint']='incorrect';path.write_text(json.dumps(saved))
                with self.assertRaisesRegex(ValueError,'fingerprint'):evaluate(source,root/'bad',1)

    def test_queue_controls_first_and_paired_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.mkdir();output=root/'output';output.mkdir()
            (source/'continuation-summary.json').write_text('{"status":"passed"}')
            for step in STEPS:
                for arm in ARMS:
                    c=source/'seed-1042'/arm/f'checkpoint-{step}';c.mkdir(parents=True)
                    (c/'trainer_state.json').write_text(json.dumps(dict(global_step=step)))
                    d=output/f'{arm}-{step}';d.mkdir()
                    (d/'summary.json').write_text(json.dumps(dict(status='passed',arm=arm,step=step,
                        model_state_unchanged=True,checkpoint_unchanged=True,eval_fingerprint='same',
                        world_size=8,metadata=dict(attention_backend='fa4'),lm_loss=3.,perplexity=20.,
                        final_control_gap=0. if step==20000 else None,
                        metrics=dict(eval_rows=4882,eval_input_tokens=9998336,eval_target_tokens=9981660))))
            jobs=make(source,output)
            self.assertEqual([j['name'] for j in jobs[:2]],['A-20000','P6-iso-20000'])
            self.assertEqual(len(jobs),13)
            self.assertTrue(all(j['gpus']==list(range(8)) for j in jobs[:-1]))
            results=report(output)
            self.assertEqual([r['step'] for r in results['results']],sorted(STEPS))
            self.assertTrue(all(r['p6_ppl_reduction_percent']==0 for r in results['results']))


if __name__=='__main__':unittest.main()
