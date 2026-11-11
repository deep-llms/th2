import json,tempfile,unittest
from pathlib import Path
from scripts.launch_dot_pool import job_config
from run_experiments import load_jobs
class QueueContracts(unittest.TestCase):
 def test_actual_native_manifest(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'jobs.json';p.write_text(json.dumps(job_config('/mnt/local/_outputs/deep-llms_th2/dot_pool_20261112/run-a01')));jobs=load_jobs(p)
   self.assertEqual(len(jobs),1);self.assertEqual(jobs[0]['gpus'],list(range(8)));self.assertEqual(jobs[0]['required_outputs'][0]['json_equals'],{'reported':True})
if __name__=='__main__':unittest.main()
