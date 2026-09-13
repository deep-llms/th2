"""CPU/mocked queue tests: never touch live GPUs or observer processes."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import queue_seed29 as job
from ccm.cli import parser


class QueueTests(unittest.TestCase):
    def test_common_command_is_independent_seed29(self):
        cmd = job.train_command()
        a = parser().parse_args(cmd[cmd.index('train'):])
        self.assertEqual((a.phase, a.arm, a.seed), ('common', 'base', 29))
        self.assertIsNone(a.checkpoint)
        self.assertIsNone(a.table)
        self.assertEqual(a.model_config, str(job.ASSETS))
        self.assertEqual(a.output, str(job.OUT/'common'))
        self.assertEqual((a.microbatch_segments, a.loss_chunk), (8, 1024))
        self.assertTrue(a.activation_checkpointing)
        self.assertFalse(a.engineering or a.online)
        self.assertIn('--nproc_per_node=8', cmd)

    def test_wait_does_not_touch_gpu_or_stop_flag(self):
        w = job.Seed29()
        w.record = Mock()
        with patch.object(job, 'previous_status', side_effect=[dict(stage='eval_grad', time='t'),
                       dict(stage='complete', success=True, time='t')]), \
             patch.object(job, 'previous_complete', side_effect=[False, True]), \
             patch.object(job, 'observer_identity', return_value=(5, 'identity')), \
             patch.object(job.time, 'sleep'), patch.object(job, 'stop') as stop, \
             patch.object(job, 'start') as start, patch.object(job, 'inspect') as inspect:
            w.wait_previous()
            stop.assert_not_called(); start.assert_not_called(); inspect.assert_not_called()

    def test_incomplete_or_wrong_marker_is_not_completion(self):
        with tempfile.TemporaryDirectory() as d, patch.object(job, 'PREVIOUS', Path(d)):
            self.assertFalse(job.previous_complete())
            p = Path(d)/'complete.json'
            p.write_text('{')
            self.assertFalse(job.previous_complete())
            p.write_text(json.dumps(dict(success=True, stage='complete', event='last_arm_finished')))
            with self.assertRaises(RuntimeError):
                job.previous_complete()
            p.write_text(json.dumps(dict(success=True, stage='complete', event=job.COMPLETE_EVENT)))
            self.assertTrue(job.previous_complete())

    def test_three_minute_window_keeps_same_workers(self):
        w = job.Seed29(); w.record = Mock()
        clock = [0.]
        def sleep(t): clock[0] += t
        result = ([], {10: ('identity',)}, 9, ('parent',))
        with patch.object(job.time, 'monotonic', side_effect=lambda: clock[0]), \
             patch.object(job.time, 'sleep', side_effect=sleep), \
             patch.object(job, 'previous_status', return_value=dict(stage='complete', success=True)), \
             patch.object(job, 'previous_complete', return_value=True), \
             patch.object(job, 'inspect', return_value=result) as inspect, patch.object(job, 'stop') as stop:
            w.stable_burn_window()
            self.assertGreaterEqual(clock[0], 180)
            self.assertGreaterEqual(inspect.call_count, 7)
            stop.assert_not_called()

    def test_recovery_and_restart_disabled_before_takeover(self):
        w = job.Seed29(); w.record = Mock()
        with patch.object(job.Workflow, 'recover_idle') as recover, \
             patch.object(job.Workflow, 'keep_burns') as keep, patch.object(job, 'inspect'):
            w.recover_idle(); w.keep_burns(job.ALL)
            recover.assert_not_called(); keep.assert_not_called()

    def test_disarm_waits_for_exact_observer_exit(self):
        with tempfile.TemporaryDirectory() as d:
            w = job.Seed29(); w.record = Mock()
            ident = (2, '123', [b'python', b'scripts/pilot_stage2.py'])
            with patch.object(job, 'PREVIOUS', Path(d)), patch.object(job, 'previous_complete', return_value=True), \
                 patch.object(job, 'observer_identity', side_effect=[(99, ident), None]), \
                 patch.object(job, 'identity', side_effect=[ident, ProcessLookupError()]), \
                 patch.object(job.time, 'sleep') as sleep, patch.object(job, 'stop') as stop:
                w.disarm_previous()
                self.assertTrue((Path(d)/'STOP_IDLE_WATCH').exists())
                sleep.assert_called_once_with(30)
                stop.assert_not_called()

    def test_full_order(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)/'fresh'; events = []
            w = job.Seed29()
            def record(**kw):
                (out/'status.json').write_text(json.dumps(kw))
                if kw.get('success'): (out/'STOP_IDLE_WATCH').touch()
            w.record = record
            w.wait_previous = lambda: events.append('wait_full_completion')
            w.stable_burn_window = lambda: events.append('burns_180s')
            w.disarm_previous = lambda: events.append('disarm_and_exit')
            w.free_after_wait = lambda g: events.append('wait30_free')
            w.burns = lambda g: events.append('burns_restored')
            w.run = lambda name, cmd, indices=(), spare=(): events.append(name)
            with patch.object(job, 'OUT', out), patch.object(job, 'PROJECT', Path.cwd()), \
                 patch.dict(os.environ, CONDA_DEFAULT_ENV='train_env'), patch.object(job, 'preflight'), \
                 patch.object(job, 'inspect'), patch.object(job, 'snapshot', return_value=[]), \
                 patch.object(job, 'stop', side_effect=lambda g: events.append('stop_verified_workers')):
                w.execute()
            self.assertEqual(events, ['wait_full_completion', 'burns_180s', 'validate_previous', 'validate_data',
                             'disarm_and_exit', 'stop_verified_workers', 'wait30_free', 'wait30_free',
                             'train_common_seed29', 'wait30_free', 'validate_common', 'burns_restored'])
            self.assertTrue((out/'complete.json').exists())

    def test_failed_predecessor_never_reclaims(self):
        with tempfile.TemporaryDirectory() as d:
            w = job.Seed29(); w.wait_previous = Mock(side_effect=RuntimeError('prior failure'))
            with patch.object(job, 'OUT', Path(d)/'fresh'), patch.object(job, 'PROJECT', Path.cwd()), \
                 patch.dict(os.environ, CONDA_DEFAULT_ENV='train_env'), patch.object(job, 'preflight'), \
                 patch.object(job, 'stop') as stop, patch.object(job, 'start') as start:
                with self.assertRaisesRegex(RuntimeError, 'prior failure'): w.execute()
                self.assertFalse(w.reclaim_authorized)
                stop.assert_not_called(); start.assert_not_called()


if __name__ == '__main__':
    unittest.main()
