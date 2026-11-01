import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class DepthDistributedResumeTests(unittest.TestCase):
    def test_two_rank_native_resume(self):
        with tempfile.TemporaryDirectory() as d:
            proc=subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1',
                '--nproc_per_node=2','--module','tests.a2.depth_resume_distributed_fixture','--output',d],
                env={**os.environ,'CUDA_VISIBLE_DEVICES':'','ACCELERATE_TORCH_DEVICE':'cpu','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=120)
            self.assertEqual(proc.returncode,0,proc.stdout[-15000:])
            for rank in range(2):
                r=json.loads((Path(d)/f'verified-rank{rank}.json').read_text())
                self.assertTrue(r['passed']);self.assertEqual(r['world'],2);self.assertEqual(r['global_step'],16)
                self.assertTrue(r['order']['exact_order'])


if __name__=='__main__':unittest.main()
