"""Real spawned-process scheduling tests without touching any GPU."""
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from eval.paired.parallel import Pool, options, visible_devices


def fake_job(cmd, cache):
    _, opts=options(cmd)
    output=Path(opts['--output']);output.parent.mkdir(parents=True,exist_ok=True)
    if opts['--checkpoint']=='fail':raise RuntimeError('intentional worker failure')
    if opts['--checkpoint']=='crash':os._exit(17)
    start=time.monotonic()
    time.sleep(.15 if opts['--checkpoint']!='slow' else 30)
    output.write_text(json.dumps(dict(gpu=os.environ['CUDA_VISIBLE_DEVICES'],pid=os.getpid(),start=start,end=time.monotonic())))
    return dict(model_loaded=False)


def command(root, index, checkpoint='model'):
    return ['python','-m','eval.paired','evaluate','--checkpoint',checkpoint,'--path','production',
            '--task','piqa','--output',str(root/str(index))]


class ParallelTests(unittest.TestCase):
    def test_allocation_and_concurrent_isolation(self):
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'4,7'}):
            self.assertEqual(visible_devices(),['4','7'])
            with self.assertRaises(ValueError):visible_devices(3)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with Pool(['4','7'],root/'status',handler=fake_job) as pool:
                pool.run([command(root,i) for i in range(6)])
                self.assertEqual(len(pool.records),6)
                self.assertTrue(all(r['status']=='ok' for r in pool.records))
                pids=[p.pid for p in pool.processes]
            self.assertTrue(all(not p.is_alive() for p in pool.processes))
            rows=[json.loads((root/str(i)).read_text()) for i in range(6)]
            self.assertEqual({r['gpu'] for r in rows},{'4','7'})
            self.assertEqual({r['pid'] for r in rows},set(pids))
            self.assertTrue(any(a['pid']!=b['pid'] and max(a['start'],b['start'])<min(a['end'],b['end']) for a in rows for b in rows))
            for gpu in ('4','7'):
                work=sorted((r for r in rows if r['gpu']==gpu),key=lambda r:r['start'])
                self.assertTrue(all(a['end']<=b['start'] for a,b in zip(work,work[1:])))

    def test_failure_or_dead_worker_stops_owned_pool(self):
        for failure in ('fail','crash'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);start=time.monotonic()
                with self.assertRaises(RuntimeError):
                    with Pool(['0','1'],root/'status',handler=fake_job) as pool:
                        pool.run([command(root,0,failure),command(root,1,'slow')])
                self.assertLess(time.monotonic()-start,15)
                self.assertTrue(all(not p.is_alive() for p in pool.processes))

    def test_duplicate_output_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with Pool(['0'],root/'status',handler=fake_job) as pool:
                with self.assertRaisesRegex(ValueError,'Duplicate'):
                    pool.run([command(root,0),command(root,0)])


if __name__=='__main__':unittest.main()
