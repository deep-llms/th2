import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.a2_why_pause_depth import copy_state, verify_owner


class PauseTests(unittest.TestCase):
    def test_copy_is_independent_and_detects_member_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); src = root/'src'; src.mkdir()
            files = {f'f{i}': 4 for i in range(20)}
            for name in files: (src/name).write_bytes(b'data')
            marker = dict(complete=True, state_finite=True, files=files)
            (src/'depth-checkpoint.json').write_text(json.dumps(marker))
            dst = root/'dst'; hashes = copy_state(src, dst, marker)
            self.assertEqual(len(hashes), 21)
            (src/'f0').write_bytes(b'new!')
            self.assertEqual((dst/'f0').read_bytes(), b'data')
            (src/'f1').write_bytes(b'x')
            with self.assertRaises(ValueError): copy_state(src, root/'bad', marker)

    def test_owner_checks_full_identity(self):
        live = dict(host='node', gpus=[dict(index=0,uuid='uuid',pids=[10])])
        expected = dict(host='node', gpu_mapping=[[0,'uuid',[10]]],
                        processes={'10':dict(pid=10,start_ticks=123)}, supervisor_pid=20)
        with patch('scripts.a2_why_pause_depth.inspect',return_value=live), \
             patch('scripts.a2_why_pause_depth.socket.gethostname',return_value='node'), \
             patch('scripts.a2_why_pause_depth.process',return_value=dict(pid=10,start_ticks=123)), \
             patch('scripts.a2_why_pause_depth.read',return_value=dict(owner_pid=20)):
            self.assertEqual(verify_owner(expected),live)
            expected['processes']['10']['start_ticks']=124
            with self.assertRaises(RuntimeError):verify_owner(expected)


if __name__=='__main__':unittest.main()
