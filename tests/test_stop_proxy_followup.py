"""Owned supervisor discovery, without signaling any real process."""
import json
from contextlib import ExitStack
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts import stop_proxy_followup as stop


class OwnedSupervisorTests(unittest.TestCase):
    def inspect(self, pane=10, owner=20, worker_parent=20, wrong_argv=False):
        command=['python','-u','-m','scripts.train_then_burn','--output',str(stop.ROOT/'supervised'),
                 '--config',str(stop.ROOT/'jobs.json'),'--host',stop.HOST]
        if wrong_argv: command[command.index('--output')+1]='/other/supervised'
        processes={10:dict(pid=10,ppid=1),20:dict(pid=20,ppid=10),30:dict(pid=30,ppid=1)}
        processes.update({p:dict(pid=p,ppid=worker_parent) for p in range(100,108)})
        gpus=[dict(index=i,pids=[100+i]) for i in range(8)]
        def read(path):
            if str(path).endswith('/DISABLED'):return dict(owner_pid=owner)
            if str(path).endswith('supervisor.json'):return dict(training_status='running')
            return dict(status='running')
        with ExitStack() as stack:
            for name,value in [('socket.gethostname',lambda:stop.HOST),
                               ('scripts.stop_proxy_followup.pane_pid',lambda:pane),
                               ('scripts.stop_proxy_followup.read',read),
                               ('scripts.stop_proxy_followup.process',lambda pid:processes[pid]),
                               ('scripts.stop_proxy_followup.snapshot',lambda _:gpus),
                               ('pathlib.Path.is_file',lambda _:True),
                               ('pathlib.Path.read_bytes',lambda _:b'\0'.join(x.encode() for x in command))]:
                stack.enter_context(patch(name,value))
            return stop.inspect_owned()

    def test_direct_and_wrapper_supervisor(self):
        self.assertEqual(self.inspect(pane=20)[0],20)
        self.assertEqual(self.inspect(pane=10)[0],20)

    def test_reject_unrelated_guard_and_pid_one(self):
        for owner in (1,30,'20',None):
            with self.assertRaises(RuntimeError):self.inspect(owner=owner)

    def test_reject_wrong_command_and_unowned_worker(self):
        with self.assertRaises(RuntimeError):self.inspect(wrong_argv=True)
        with self.assertRaises(RuntimeError):self.inspect(worker_parent=30)


if __name__=='__main__':unittest.main()
