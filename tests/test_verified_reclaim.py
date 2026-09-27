import copy
import unittest
import os
import signal
import subprocess
import sys
from unittest.mock import patch
from scripts.verified_gpu_reclaim import validate
from scripts.verified_gpu_reclaim import pidfd_open, pidfd_send_signal


class VerifiedReclaimTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == 'linux', 'Linux PID handles')
    def test_libc_pidfd_fallback_only_signals_owned_child(self):
        # Exercise the B200 conda case without GPU access or external processes.
        with patch('scripts.verified_gpu_reclaim.os', wraps=os) as mocked_os, \
                patch('scripts.verified_gpu_reclaim.signal', wraps=signal) as mocked_signal:
            del mocked_os.pidfd_open
            del mocked_signal.pidfd_send_signal
            with subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']) as child:
                fd = None
                try:
                    fd = pidfd_open(child.pid)
                    pidfd_send_signal(fd, signal.SIGTERM)
                    self.assertEqual(child.wait(timeout=5), -signal.SIGTERM)
                    with self.assertRaises(ProcessLookupError):
                        pidfd_send_signal(fd, signal.SIGTERM)
                finally:
                    if fd is not None:
                        os.close(fd)
                    if child.poll() is None:
                        child.kill()

    def fixture(self):
        status=[{'index':i,'uuid':f'gpu-{i}','pids':[100+i]} for i in range(8)]
        records={str(100+i):{'pid':100+i,'ppid':50,'start_ticks':1000+i,'command_sha256':'worker',
                 'exe':'python','command_prefix':['python'],'launcher_kind':None} for i in range(8)}
        records['50']={'pid':50,'ppid':1,'start_ticks':900,'command_sha256':'parent','exe':'python',
                       'command_prefix':['python','polite_burn.py'],'launcher_kind':'gpu_burn'}
        expected={'host':'thiennh-p6-tpbw-worker-0','gpus':copy.deepcopy(status),
                  'workers':list(range(100,108)),'stop_roots':[50],'processes':copy.deepcopy(records)}
        return expected,status,records,expected['host']

    def test_exact_inspected_ownership_only(self):
        validate(*self.fixture())
        for mutation in ('pid','parent','command','gpu','uuid','host','unknown_parent','unrelated_parent','pid1'):
            expected,status,records,host=self.fixture()
            if mutation=='pid':records['100']['start_ticks']+=1
            elif mutation=='parent':records['50']['start_ticks']+=1
            elif mutation=='command':records['100']['command_sha256']='new'
            elif mutation=='gpu':status[0]['pids'].append(999)
            elif mutation=='uuid':status[0]['uuid']='other-node-device'
            elif mutation=='host':host='other-host'
            elif mutation=='unknown_parent':
                records['50']['launcher_kind']=None;expected['processes']=copy.deepcopy(records)
            elif mutation=='unrelated_parent':
                for i in range(100,108):records[str(i)]['ppid']=1
                expected['processes']=copy.deepcopy(records)
            elif mutation=='pid1':expected['stop_roots']=[1]
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                validate(expected,status,records,host)
