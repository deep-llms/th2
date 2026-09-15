"""CPU/mocked Phase-A queue checks: never inspect or signal real GPU processes."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import pilot_scaleup28_phase_a as job
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
        self.assertIn('20260915', str(config.OUT))
        self.assertIn('20260915', str(config.DATA28))

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
