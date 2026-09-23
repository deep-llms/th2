"""Pure-data tests only; never inspect or signal actual GPU workers."""
import copy
import unittest

from scripts.reclaim_b200_burn_20260923 import BURN_HASH, HOST, WORKERS, validate


class ReclaimGuardTests(unittest.TestCase):
    def fixture(self):
        status = [{'index': i, 'pids': [pid]} for i, pid in enumerate(WORKERS)]
        parent = {'pid': 431, 'ppid': 1, 'start_ticks': 258980356,
                  'argv': ['/usr/bin/python3', '/tmp/llm_pretrain_burn.py']}
        workers = [{'pid': pid, 'ppid': 431, 'start_ticks': 258980483,
                    'argv': ['/usr/bin/python3', '-B', '-c',
                             'from multiprocessing.spawn import spawn_main; spawn_main()',
                             '--multiprocessing-fork']} for pid in WORKERS]
        return [status, parent, workers, HOST, BURN_HASH]

    def test_exact_inspected_identity(self):
        validate(*self.fixture())

    def test_unknown_processes_reused_pids_and_changed_launcher_are_rejected(self):
        for edit in (lambda v: v[0][0]['pids'].append(999),
                     lambda v: v[2][0].update(start_ticks=258980484),
                     lambda v: v[2][0].update(ppid=999),
                     lambda v: v[1].update(pid=1),
                     lambda v: v[1].update(argv=['python', 'train.py'])):
            value = copy.deepcopy(self.fixture())
            edit(value)
            with self.assertRaises(ValueError):
                validate(*value)

    def test_wrong_host_or_script_rejected(self):
        for index in (3, 4):
            value = self.fixture()
            value[index] = 'different'
            with self.assertRaises(ValueError):
                validate(*value)
