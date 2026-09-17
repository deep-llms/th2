"""CPU/mocked Phase-B panel queue checks: never touch real GPUs or tmux."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import call, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import pilot_scaleup28_panel as panel
import scaleup28_config as config
from ccm.cli import parser
from ccm.scaleup_jobs import make_jobs


def generated_panel_jobs():
    args = parser().parse_args(['scaleup-jobs', '--queue', 'panel', '--data', str(config.DATA28),
        '--vocabulary', str(config.DATA/'vocabulary.npz'), '--checkpoint', str(config.THETA),
        '--gpus', '0', '1', '2', '3', '4', '5', '6', '7', '--output', 'unused.json'])
    return make_jobs(args)['jobs']


class PanelTests(unittest.TestCase):
    def test_generated_manifest_matches_controller_expectations(self):
        jobs = generated_panel_jobs()
        self.assertEqual(len(jobs), panel.EXPECTED_JOBS)
        self.assertEqual(jobs[0]['name'], 'verify-common-input')
        self.assertEqual(jobs[-1]['name'], 'replication-decision')
        kinds = [panel.job_kind(j) for j in jobs]
        # 2 compile jobs + 11 trainings on 8 GPUs; 12 single-GPU evaluations;
        # 9 contrasts + 1 decision as burn-covered CPU; the rest plain CPU.
        self.assertEqual(kinds.count('gpu8'), 13)
        self.assertEqual(kinds.count('gpu0'), 12)
        self.assertEqual(kinds.count('cpu_long'), 10)
        self.assertEqual(kinds.count('cpu'), 13)
        for j in jobs:
            argv = panel.substitute(j['argv'])
            self.assertNotIn('{python}', ' '.join(argv))
            self.assertNotIn('{run_dir}', ' '.join(argv))
            self.assertNotIn('--final-evaluation', argv)
            self.assertNotIn('delta', argv)
            index = argv.index('ccm')
            parser().parse_args(argv[index+1:])
        trains = [j for j in jobs if 'train' in j['argv']]
        self.assertEqual(len(trains), 11)
        for j in trains:
            argv = j['argv']
            self.assertEqual(argv[argv.index('--checkpoint')+1], str(config.THETA))
            phase = argv[argv.index('--phase')+1]
            self.assertEqual(argv[argv.index('--save-every')+1], '977' if phase == 'stage1' else '2000')
        decision = jobs[-1]
        self.assertEqual(decision['required_outputs'][0]['json_equals'], dict(automatic_launch=False))

    def test_generation_command_mirrors_test_invocation(self):
        cmd = panel.generation_command()
        self.assertEqual(cmd[cmd.index('--queue')+1], 'panel')
        self.assertEqual(cmd[cmd.index('--checkpoint')+1], str(config.THETA))
        self.assertEqual(cmd[cmd.index('--data')+1], str(config.DATA28))
        self.assertEqual(cmd[cmd.index('--gpus')+1:cmd.index('--gpus')+9],
                         [str(i) for i in range(8)])

    def test_verify_required_subset_and_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(panel, 'PANEL_OUT', Path(tmp)):
                (Path(tmp)/'ok.json').write_text(json.dumps(dict(success=True, extra=1)))
                panel.verify_required(dict(name='x', required_outputs=[
                    dict(path='ok.json', json_equals=dict(success=True))]))
                with self.assertRaisesRegex(RuntimeError, 'contract failed'):
                    panel.verify_required(dict(name='x', required_outputs=[
                        dict(path='ok.json', json_equals=dict(success=False))]))
                with self.assertRaisesRegex(RuntimeError, 'Missing/empty'):
                    panel.verify_required(dict(name='x', required_outputs=[dict(path='absent.json')]))
                with self.assertRaisesRegex(RuntimeError, 'escapes'):
                    panel.verify_required(dict(name='x', required_outputs=[dict(path='../oops.json')]))

    def test_burn_state_transitions_never_mix_sessions(self):
        w = panel.Panel.__new__(panel.Panel)
        w.burn_number, w.reclaimed, w.burn_state = 0, True, 'none'
        events = []
        with patch.object(panel, 'stop', lambda idx: events.append(('stop', tuple(idx)))), \
             patch.object(panel.Panel, 'burns', lambda self, idx: events.append(('burns', tuple(idx)))), \
             patch.object(panel.Panel, 'free_after_wait', lambda self, idx: events.append(('free', tuple(idx)))):
            w.ensure_burn_state('spare')
            w.ensure_burn_state('all')     # must stop SPARE fully before ALL
            w.ensure_burn_state('none')
            w.ensure_burn_state('all')
            w.ensure_burn_state('spare')   # must stop ALL fully before SPARE
        self.assertEqual(events, [
            ('burns', tuple(config.SPARE)),
            ('stop', tuple(config.SPARE)), ('free', tuple(config.ALL)), ('burns', tuple(config.ALL)),
            ('stop', tuple(config.ALL)), ('free', tuple(config.ALL)), ('free', tuple(config.ALL)),
            ('burns', tuple(config.ALL)),
            ('stop', tuple(config.ALL)), ('free', tuple(config.ALL)), ('burns', tuple(config.SPARE))])

    def test_panel_config_pins(self):
        for value in (config.THETA_HASH, config.THETA_META_HASH,
                      config.EXT_MANIFEST_HASH, config.MAPPING_HASH):
            self.assertRegex(value, '^[0-9a-f]{64}$')
        self.assertIn('20260917', str(config.PANEL_OUT))
        self.assertEqual(str(config.THETA),
                         str(config.OUT/'common-base/checkpoint-38147'))
        self.assertNotEqual(config.PANEL_SESSION, config.SESSION)
        self.assertTrue(config.A2_SCRIPT.endswith(b'phase_a2.py'))

    def test_disarm_targets_a2_observer(self):
        src = open(Path(__file__).resolve().parents[1]/'scripts/pilot_scaleup28_panel.py').read()
        self.assertIn("A2_EVENT", src)
        self.assertIn("OUT/'STOP_IDLE_WATCH'", src)
        self.assertIn('run_panel_job', src)


if __name__ == '__main__':
    unittest.main()
