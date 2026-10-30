import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from datetime import datetime,timezone
from scripts.queue_watch import deadline
ROOT=Path(__file__).resolve().parents[1]
POLICY=ROOT/'resources/a2_stage1b_time_policy_20261030.json'
SHA='e9cf0904bf92c8365b209ce940150cb33ce83c5987512242d3dcb2ef4d79a3ac'


class WatchPolicy(unittest.TestCase):
    def test_only_exact_user_override_removes_deadline(self):
        self.assertEqual(deadline(None,POLICY,SHA),float('inf'))
        self.assertEqual(deadline('2026-10-31T04:06:27+08:00'),datetime.fromisoformat('2026-10-31T04:06:27+08:00').timestamp())
        for args in [(None,None,None),('2026-10-31',None,None),(None,POLICY,'0'*64),
                     ('2026-10-31T04:06:27+08:00',POLICY,SHA)]:
            with self.assertRaises(ValueError):deadline(*args)

    def test_same_state_unbounded_watch_still_honors_completion_stop(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);state=root/'state';state.mkdir();prompt=root/'prompt';prompt.write_text('test only')
            stub=root/'codex-stub';stub.write_text('#!/usr/bin/env python3\nfrom pathlib import Path\nPath('+repr(str(state/'STOP'))+').touch()\n')
            stub.chmod(0o700)
            subprocess.run(['python3',str(ROOT/'scripts/queue_watch.py'),'--thread','test-only','--cwd',str(root),
                '--prompt',str(prompt),'--state',str(state),'--interval','1800','--first-at','2000-01-01T00:00:00+00:00',
                '--time-policy',str(POLICY),'--time-policy-sha256',SHA,'--codex',str(stub)],check=True,timeout=10)
            self.assertEqual(json.loads((state/'status.json').read_text())['status'],'stopped')
            deliveries=[json.loads(p.read_text()) for p in state.glob('20*.json')]
            self.assertEqual(len(deliveries),1);self.assertIsNone(deliveries[0]['until'])
            self.assertEqual(deliveries[0]['time_policy_sha256'],SHA)


if __name__=='__main__':unittest.main()
