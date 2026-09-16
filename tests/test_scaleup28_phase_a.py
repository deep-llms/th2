"""CPU/mocked Phase-A queue checks: never inspect or signal real GPU processes."""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import pilot_scaleup28_phase_a as job
import pilot_scaleup28_phase_a2 as job2
import scaleup28_config as config
from ccm.cli import parser
from ccm.scaleup_jobs import make_jobs


def generated_common_jobs():
    args = parser().parse_args(['scaleup-jobs', '--queue', 'common', '--data', str(config.DATA28),
        '--vocabulary', str(config.DATA/'vocabulary.npz'), '--model-config', str(config.ASSETS),
        '--gpus', '0', '1', '2', '3', '4', '5', '6', '7', '--output', 'unused.json'])
    jobs = make_jobs(args)['jobs']
    resolved = {}
    for j in jobs:
        resolved[j['name']] = [a.replace('{python}', config.PYTHON).replace('{run_dir}', str(config.OUT))
                               for a in j['argv']]
    return resolved


class PhaseATests(unittest.TestCase):
    def test_common_commands_equal_decided_queue(self):
        # Decisions 1(c)/3a are locked in ccm.scaleup_jobs; the controller must
        # run exactly those commands, only with resolved python/run paths.
        generated = generated_common_jobs()
        self.assertEqual(generated['common-pipeline'], job.common_command('common-pipeline', 32))
        self.assertEqual(generated['common-stability'], job.common_command('common-stability', 1024))
        self.assertEqual(generated['common-base'], job.common_command('common-base'))
        self.assertEqual(generated['verify-common-base'], job.verify_common_command())
        self.assertEqual(config.SMOKES, (('common-pipeline', 32), ('common-stability', 1024)))

    def test_shallow_followup_commands(self):
        for seed in (17, 29):
            train = job.shallow_train_command(seed)
            self.assertIn('pilot12-shallow-followup', train)
            self.assertIn(str(config.COMMON_ROOT[seed]/'common/checkpoint-15259'), train)
            self.assertIn(str(config.TABLES[seed]/'shallow'), train)
            self.assertEqual(train[train.index('--save-every')+1], '1000')
            self.assertEqual(train[train.index('--seed')+1], str(seed))
            ev = job.shallow_eval_command(seed)
            self.assertIn(str(config.TABLES[seed]/'contextual'), ev)
            self.assertIn(str(config.shallow_root(seed)/'train/shallow/checkpoint-3815'), ev)
            cmp_cmd = job.shallow_compare_command(seed)
            self.assertEqual(cmp_cmd[cmp_cmd.index('--left')+1], str(config.CONTEXT_EVAL[seed]))
            self.assertEqual(cmp_cmd[cmp_cmd.index('--replicates')+1], '10000')
            self.assertEqual(cmp_cmd[cmp_cmd.index('--cluster')+1], 'doc_id')
        with self.assertRaises(RuntimeError):
            job.shallow_train_command(43)

    def test_prepare_command_and_pins(self):
        prep = job.prepare_command()
        self.assertEqual(prep[prep.index('--dataset-revision')+1], config.REVISION)
        self.assertEqual(prep[prep.index('--historical-data')+1], str(config.DATA/'corpus'))
        self.assertEqual(prep[prep.index('--output')+1], str(config.DATA28))
        self.assertNotIn('--budget', prep)
        self.assertNotIn('--engineering', prep)
        for value in [config.CORE, config.CORPUS_HASH, config.VOCAB_HASH,
                      *config.COMMON_HASH.values(), *config.TABLE_HASH.values(),
                      *config.CONTEXT_EVAL_CKPT.values()]:
            self.assertRegex(value, '^[0-9a-f]{64}$')
        from ccm.cli import code_hash
        self.assertEqual(config.CORE, code_hash())
        self.assertNotEqual(config.SESSION, config.OLD_SESSION)
        # a2 recovery uses fresh roots; the old queue's roots stay addressable
        # as verification/cleanup targets only.
        self.assertIn('20260916', str(config.OUT))
        self.assertIn('20260916', str(config.DATA28))
        self.assertIn('20260915', str(config.OUT_A1))
        self.assertIn('20260915', str(config.DATA28_PARTIAL))
        self.assertNotEqual(config.OUT, config.OUT_A1)
        self.assertNotEqual(config.DATA28, config.DATA28_PARTIAL)

    def test_previous_state_gate(self):
        done = dict(success=True, event=config.OLD_EVENT)
        status = dict(success=True, stage='complete')
        self.assertTrue(job.previous_state_ok(done, status, 60))
        self.assertFalse(job.previous_state_ok(done, status, 300))
        self.assertFalse(job.previous_state_ok(dict(done, event='other'), status, 60))
        self.assertFalse(job.previous_state_ok(done, dict(status, stage='failed'), 60))
        self.assertFalse(job.previous_state_ok(dict(success=False, event=config.OLD_EVENT), status, 60))

    def test_recover_refuses_before_reclaim_and_with_live_child(self):
        w = job.PhaseA.__new__(job.PhaseA)
        w.stage, w.burn_number, w.child, w.owns_output = 'failed', 0, None, True
        w.reclaimed, w.prep = False, None
        with patch.object(job.runtime.Workflow, 'record') as rec, \
             patch.object(job.runtime, 'snapshot') as snap:
            job.runtime.OUT = Path(tempfile.mkdtemp())
            w.recover_idle()
            snap.assert_not_called()
            self.assertEqual(rec.call_args.kwargs['event'], 'failure_preserved_previous_burn_owner')
        w.reclaimed = True
        w.child = SimpleNamespace(poll=lambda: None, pid=123)
        with patch.object(job.runtime.Workflow, 'record') as rec:
            w.recover_idle()
            self.assertEqual(rec.call_args.kwargs['event'], 'recovery_refused_live_child')

    def test_stop_own_preparation_only_touches_own_child(self):
        w = job.PhaseA.__new__(job.PhaseA)
        w.prep = None
        w.stop_own_preparation()  # no child: no-op
        signals = []
        w.prep = SimpleNamespace(poll=lambda: None, pid=77,
                                 send_signal=lambda s: signals.append(s),
                                 wait=lambda timeout: 0)
        with patch.object(job.PhaseA, 'record'):
            w.stop_own_preparation()
        self.assertEqual(len(signals), 1)

    def test_a2_foreign_allowlist(self):
        ok = job2.foreign_process_allowed
        self.assertTrue(ok([b'/usr/bin/python3', b'-B', b'-c', b'x', b'--multiprocessing-fork']))
        self.assertTrue(ok([b'/mnt/local/conda-py310/envs/deepeyes/bin/python', b'trainer.py']))
        self.assertTrue(ok([b'/usr/bin/python3', b'/tmp/llm_pretrain_burn.py']))
        self.assertFalse(ok([b'/usr/bin/bash', b'--multiprocessing-fork']))
        self.assertFalse(ok([b'/usr/bin/python3', b'some_random_service.py']))
        self.assertFalse(ok([b'']))
        self.assertFalse(ok([]))

    @unittest.skipUnless(hasattr(os, 'pidfd_open'), 'pidfd API unavailable in this test env')
    def test_a2_pinned_signal_tolerates_reparent_rejects_reuse(self):
        import os as _os
        import signal as _sig
        import subprocess as _sp
        pid = _os.getpid()
        real = job2.identity(pid)
        # Benign reparenting: only ppid differs -> stable identity holds -> signal 0 delivers.
        reparented = (999999, real[1], real[2])
        self.assertTrue(job2.pinned_signal(pid, reparented, 0))
        # PID reuse: start time differs -> refuse (returns False, never raises).
        reused = (real[0], str(int(real[1]) + 5), real[2])
        self.assertFalse(job2.pinned_signal(pid, reused, _sig.SIGTERM))
        # Cmdline change -> refuse.
        self.assertFalse(job2.pinned_signal(pid, (real[0], real[1], real[2] + [b'x']), _sig.SIGTERM))
        # Gone process -> False.
        child = _sp.Popen(['true'])
        child.wait()
        self.assertFalse(job2.pinned_signal(child.pid, real, _sig.SIGTERM))

    def test_a2_ancestor_walk_stops_at_non_allowlisted(self):
        # chain: gpu-worker(pid) -> torchrun(2) -> bash(3); bash must NOT be collected.
        table = {
            10: (2, '100', [b'/usr/bin/python3', b'--multiprocessing-fork']),
            2: (3, '90', [b'python', b'-m', b'torch.distributed.run']),
            3: (1, '80', [b'/usr/bin/bash']),
        }
        with patch.object(job2, 'identity', side_effect=lambda p: table[p]):
            anc = job2.allowlisted_ancestors(10)
        self.assertEqual(set(anc), {2})  # torchrun kept, bash (non-allowlisted) stops the walk

    def test_a2_cleanup_guards_and_success(self):
        w = job2.PhaseA2.__new__(job2.PhaseA2)
        events = []
        with tempfile.TemporaryDirectory() as tmp:
            old = Path(tmp)/'a1'
            seed29 = old/'shallow12_seed29/train/shallow'
            seed29.mkdir(parents=True)
            (seed29/'run.json').write_text(json.dumps(dict(
                study='pilot12-shallow-followup', seed=29, arm='shallow')))
            (old/'shallow29_train.log').write_text('log')
            (old/'cache-123.arrow').write_text('x')
            sub = old/'shallow12_seed17'
            sub.mkdir()
            (sub/'tmp-x').write_text('y')
            (sub/'metrics.json').write_text('{}')
            with patch.object(job2, 'OUT_A1', old), \
                 patch.object(job2, 'DATA28_PARTIAL', Path(tmp)/'absent'), \
                 patch.object(job2.PhaseA2, 'record', lambda self, **kw: events.append(kw)):
                # refuse completed section
                (old/'shallow12_seed29/complete.json').write_text('{}')
                with self.assertRaisesRegex(RuntimeError, 'completed'):
                    w.cleanup()
                (old/'shallow12_seed29/complete.json').unlink()
                # refuse wrong contract
                (seed29/'run.json').write_text(json.dumps(dict(study='pilot12', seed=17, arm='base')))
                with self.assertRaisesRegex(RuntimeError, 'contract mismatch'):
                    w.cleanup()
                (seed29/'run.json').write_text(json.dumps(dict(
                    study='pilot12-shallow-followup', seed=29, arm='shallow')))
                w.cleanup()
            self.assertFalse((old/'shallow12_seed29').exists())
            self.assertFalse((old/'shallow29_train.log').exists())
            self.assertFalse((old/'cache-123.arrow').exists())
            self.assertFalse((sub/'tmp-x').exists())
            self.assertTrue((sub/'metrics.json').exists())
            removed = [e for e in events if e.get('event') == 'removed_cache_tmp_files'][0]
            self.assertEqual(removed['count'], 2)

    def test_a2_partial_root_hard_path_guard(self):
        w = job2.PhaseA2.__new__(job2.PhaseA2)
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp)/'not-the-real-partial-root'
            fake.mkdir()
            with patch.object(job2, 'OUT_A1', Path(tmp)/'missing-old'), \
                 patch.object(job2, 'DATA28_PARTIAL', fake), \
                 patch.object(job2.PhaseA2, 'record', lambda self, **kw: None):
                with self.assertRaisesRegex(RuntimeError, 'refuse delete'):
                    w.cleanup()
            self.assertTrue(fake.exists())

    def test_a2_clear_targets_aborts_on_unrecognized_gpu_process(self):
        w = job2.PhaseA2.__new__(job2.PhaseA2)
        gpus = [{'index': 0, 'pids': [500]}]
        with patch.object(job2, 'snapshot', return_value=gpus), \
             patch.object(job2, 'identity', return_value=(1, '10', [b'/usr/bin/postgres'])):
            with self.assertRaisesRegex(RuntimeError, 'refuse to signal'):
                w.clear_targets()

    def test_a2_clear_targets_collects_workers_and_launcher(self):
        w = job2.PhaseA2.__new__(job2.PhaseA2)
        gpus = [{'index': 0, 'pids': [10]}, {'index': 1, 'pids': [11]}]
        table = {
            10: (2, '100', [b'/usr/bin/python3', b'--multiprocessing-fork']),
            11: (2, '100', [b'/usr/bin/python3', b'--multiprocessing-fork']),
            2: (3, '90', [b'python', b'-m', b'torch.distributed.run']),
            3: (1, '80', [b'/usr/bin/bash']),
        }
        with patch.object(job2, 'snapshot', return_value=gpus), \
             patch.object(job2, 'identity', side_effect=lambda p: table[p]):
            _, targets = w.clear_targets()
        self.assertEqual(set(targets), {10, 11, 2})  # workers + torchrun launcher, not bash

    def test_a2_burns_and_disarm(self):
        w = job2.PhaseA2.__new__(job2.PhaseA2)
        w.burn_number = 0
        with patch.object(job2, 'start') as started:
            w.burns([0, 1])
            args = started.call_args.args
            self.assertIn('phase_a2', args[1])
            self.assertEqual(args[3], 30801)
        with self.assertRaisesRegex(RuntimeError, 'no previous observer'):
            w.disarm()

    def test_validator_parses_all_gate_actions(self):
        # The controller's gate invocations must match the validator CLI.
        import validate_scaleup28_phase_a as gates
        p = gates.argparse.ArgumentParser()
        for action, extra in [('inputs', []), ('extension', []),
                              ('shallow-train', ['--seed', '17']), ('shallow-eval', ['--seed', '29']),
                              ('followup', ['--seed', '17']), ('smoke', ['--name', 'pipeline']),
                              ('smoke', ['--name', 'stability']), ('common', [])]:
            with patch.object(sys, 'argv', ['x', action]+extra):
                q = gates.argparse.ArgumentParser(description='t')
                q.add_argument('action', choices=('inputs', 'extension', 'shallow-train',
                                                  'shallow-eval', 'followup', 'smoke', 'common'))
                q.add_argument('--seed', type=int, choices=(17, 29))
                q.add_argument('--name', choices=('pipeline', 'stability'))
                q.parse_args([action]+extra)
        self.assertEqual(gates.SMOKE_STEPS, dict(pipeline=32, stability=1024))


if __name__ == '__main__':
    unittest.main()
