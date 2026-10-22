"""Comparator must reject lost routing/RNG/data/optimizer state, not just exit 0."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from resume_gpu_probe import compare, write


class ResumeReportTests(unittest.TestCase):
    def fixture(self, root, mutation=None):
        states={str(i):dict(model=f'w{i}',optimizer=f'o{i}',scheduler=f's{i}',
                           completed_steps=i,routing_counter=2*i) for i in range(7)}
        batches=[dict(step=i,input_sha256=f'i{i}-{j}',labels_sha256=f'l{i}-{j}',
                      rng_sha256=f'r{i}-{j}',progress=i/6,routing_counter=2*i+j,
                      routes=f'route{i}-{j}',loss=float(i)) for i in range(6) for j in range(2)]
        updates=[dict(step=i,lr=1/i) for i in range(1,7)]
        full=dict(status='completed',world_size=1,global_step=6,max_steps=6,
                  batches=batches,updates=updates,snapshots=states)
        write(root/'full_trace/rank0.json',full)
        for name,start,end in [('resume_hard',4,6),('resume_soft',2,4)]:
            trace=dict(full,global_step=end,initial=states[str(start)],final=states[str(end)],
                       batches=copy.deepcopy(batches[2*start:2*end]),updates=updates[start:end])
            if mutation is not None and name=='resume_hard':mutation(trace)
            write(root/(name+'_trace')/'rank0.json',trace)

    def test_accepts_exact_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            self.assertEqual(compare(root,1,4,2,6),0)

    def test_rejects_corrupt_data_rng_routes_optimizer_and_loss(self):
        for field in ('input_sha256','rng_sha256','routes','loss','optimizer'):
            with self.subTest(field=field),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                def mutate(trace):
                    if field=='optimizer':trace['initial']={**trace['initial'],'optimizer':'wrong'}
                    else:trace['batches'][0][field]=100. if field=='loss' else 'wrong'
                self.fixture(root,mutate)
                self.assertEqual(compare(root,1,4,2,6),1)
                summary=json.loads((root/'summary.json').read_text())
                self.assertEqual(summary['results'][0]['status'],'failed')
                self.assertEqual(summary['results'][1]['status'],'passed')


if __name__=='__main__':
    unittest.main()
