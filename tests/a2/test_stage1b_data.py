import json,tempfile,unittest
from pathlib import Path
from tests.a2.test_stage1_data import Stage1DataTests
from scripts.a2_stage1_data import prepare
from scripts.a2_prepare_corpora import sha
class Stage1bDataTests(Stage1DataTests):
 def test_fwe_routing_and_locked_prior_receipt(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);self.fixture(p)
   for name in ('parts','packed'):(p/name/'dclm').rename(p/name/'fineweb_edu')
   plan=json.loads((p/'plan.json').read_text());plan['corpora']['fineweb_edu']=plan['corpora'].pop('dclm');(p/'plan.json').write_text(json.dumps(plan))
   mpath=p/'packed/fineweb_edu/manifest.json';m=json.loads(mpath.read_text());m['plan_sha256']=sha(p/'plan.json');mpath.write_text(json.dumps(m))
   receipt=dict(status='verified',corpus='fineweb_edu',manifest_sha256=sha(mpath),full_output_hashes_passed=True,native_reader_passed=True,streams=m['streams']);r=p/'receipt.json';r.write_text(json.dumps(receipt))
   prepare(p/'plan.json',sha(mpath),p/'out',p/'spec.md',length=2,minimum=2,corpus='fineweb_edu',prior_verification=r,replication_spec=p/'spec.md')
   out=json.loads((p/'out/manifest.json').read_text());self.assertEqual(out['corpus'],'fineweb_edu');self.assertEqual(out['stage1b_spec_sha256'],sha(p/'spec.md'));self.assertTrue(out['T7']['passed'])
   receipt['manifest_sha256']='0'*64;r.write_text(json.dumps(receipt))
   with self.assertRaisesRegex(ValueError,'Prior full verification'):prepare(p/'plan.json',sha(mpath),p/'bad',p/'spec.md',length=2,minimum=2,corpus='fineweb_edu',prior_verification=r)
