import importlib.util, json, tempfile, unittest
from pathlib import Path
s=importlib.util.spec_from_file_location('prune',Path(__file__).parents[1]/'scripts/a2_prune_backup_inputs.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class PruneTest(unittest.TestCase):
 def fixture(self,base):
  b=base/'backup';o=base/'original';a=b/'audit';a.mkdir(parents=True);o.mkdir()
  rel='wc69-a2-family-mixture-20261107-a02/mixture/sources/dclm/text.utf8'
  for root in (b,o):
   p=root/rel;p.parent.mkdir(parents=True);p.write_bytes(b'input')
  (b/'model.safetensors').write_bytes(b'precious weights')
  (b/'report.json').write_text('{"result": 42}')
  row=dict(relative=rel,source=str(o/rel),bytes=5,sha256=m.sha(b/rel))
  return b,o,a,row
 def test_exact_deletion_preserves_original_and_results(self):
  with tempfile.TemporaryDirectory() as t:
   b,o,a,row=self.fixture(Path(t));plan=dict(rows=m.preflight([row],b,o),inventory=m.inventory(b,a));m.write(a/'plan.json',plan)
   proof=m.execute(plan,b,o,a)
   self.assertEqual(proof['removed_bytes'],5);self.assertEqual((o/row['relative']).read_bytes(),b'input')
   self.assertTrue((b/'model.safetensors').exists());self.assertTrue((b/'report.json').exists())
 def test_hash_mismatch_blocks_preflight(self):
  with tempfile.TemporaryDirectory() as t:
   b,o,a,row=self.fixture(Path(t));(o/row['relative']).write_bytes(b'wrong')
   with self.assertRaises(AssertionError):m.preflight([row],b,o)
   self.assertTrue((b/row['relative']).exists())
 def test_changed_protected_file_blocks_all_deletion(self):
  with tempfile.TemporaryDirectory() as t:
   b,o,a,row=self.fixture(Path(t));plan=dict(rows=m.preflight([row],b,o),inventory=m.inventory(b,a));(b/'report.json').write_text('changed')
   with self.assertRaises(AssertionError):m.execute(plan,b,o,a)
   self.assertTrue((b/row['relative']).exists())
 def test_symlink_and_hardlink_rejected(self):
  import os
  with tempfile.TemporaryDirectory() as t:
   b,o,a,row=self.fixture(Path(t));p=b/row['relative'];p.unlink();p.symlink_to(o/row['relative'])
   with self.assertRaises(ValueError):m.preflight([row],b,o)
   p.unlink();os.link(o/row['relative'],p)
   with self.assertRaises(AssertionError):m.preflight([row],b,o)
 def test_scope(self):
  for x in ['../text.utf8','/tmp/text.utf8','run/results/tokens.npy','wc69-a2-family-mixture-20261107-a02/mixture/sources/../tokens.bin','wc69-a2-family-mixture-20261107-a02/mixture/sources/dclm/model.safetensors','6y3o-a2-inputs-20261028-a01/run/inputs/data/heal.bin']:
   self.assertFalse(m.allowed(x),x)
if __name__=='__main__':unittest.main()
