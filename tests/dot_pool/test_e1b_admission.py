"""Launcher packaging: a checkout must not depend on ignored local result files."""
import gzip,hashlib,json,shutil,tempfile,unittest
from pathlib import Path
from scripts.launch_dot_pool_e1b import pilot_reference

class AdmissionPackaging(unittest.TestCase):
 def test_reference_works_without_results_directory(self):
  project=Path(__file__).resolve().parents[2];pin=json.loads((project/'resources/dot_pool_e1b_v15_admission.json').read_text())
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);(root/'resources').mkdir()
   shutil.copyfile(project/'resources/dot_pool_v12_reference.md.gz',root/'resources/dot_pool_v12_reference.md.gz')
   self.assertFalse((root/'results').exists())
   value=pilot_reference(root,pin)
   self.assertEqual(hashlib.sha256(value).hexdigest(),pin['pilot_report_sha256'])
   self.assertIn(b'local',value)
 def test_changed_report_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);(root/'resources').mkdir();(root/'resources/dot_pool_v12_reference.md.gz').write_bytes(gzip.compress(b'changed'))
   with self.assertRaises(ValueError):pilot_reference(root,{'pilot_report_sha256':'0'*64})
 def test_missing_report_fails_without_output_mutation(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td)
   with self.assertRaises(FileNotFoundError):pilot_reference(root,{'pilot_report_sha256':'0'*64})
   self.assertEqual(list(root.iterdir()),[])

if __name__=='__main__':unittest.main()
