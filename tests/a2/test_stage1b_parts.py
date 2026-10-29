import json,tempfile,unittest
from pathlib import Path
from tests.a2.test_stage1 import Stage1Tests
from scripts.a2_stage1b import execute
from efficient_llm.a2_stage1b.protocol import REPLICATION_SPEC_SHA
from efficient_llm.diagnostic_data import sha256
class Stage1bPartsTests(unittest.TestCase):
 def setup(self,root):
  helper=Stage1Tests();pre,inputs,prov=helper.setup_execution(root);prov['stage1b_spec_sha256']=REPLICATION_SPEC_SHA
  gate=json.loads((pre/'tests.json').read_text());gate['provenance']=prov
  for p in gate['parents'].values():p['provenance']=prov
  (pre/'tests.json').write_text(json.dumps(gate));v=json.loads(inputs.read_text());v.update(provenance=prov,dataset='fwe',replication_spec=str(Path('docs/A2_stage1b_fwe_spec.md').resolve()));inputs.write_text(json.dumps(v));return helper,pre,inputs,prov
 def lock(self,p):return {'path':str(p),'sha256':sha256(p)}
 def prior(self,root,part):
  interim=root/'interim.json';interim.write_text(json.dumps(dict(status='verified',report_sent=True)))
  p=root/'prior.json';p.write_text(json.dumps(dict(stage1b_spec_sha256=REPLICATION_SPEC_SHA,complete=self.lock(part/'complete.json'),runs=self.lock(part/'runs.json'),choice=self.lock(part/'lr_choice.json'),interim=self.lock(interim))));return self.lock(p)
 def test_two_parts_do_not_repeat_grid_or_main_and_edge_once(self):
  for edge,count in [(False,13),(True,15)]:
   with tempfile.TemporaryDirectory() as d:
    root=Path(d);helper,pre,inputs,prov=self.setup(root);runner=helper.mock_runner(prov,edge=edge)
    execute(root/'p1',inputs,pre,12345,0,runner=runner);first=list(helper.calls)
    self.assertEqual([r['arm'] for r in first[-2:]],['P300','P1T']);self.assertTrue((root/'p1/GRID_NOTE.md').exists());self.assertFalse(json.loads((root/'p1/complete.json').read_text())['training_complete'])
    execute(root/'p2',inputs,pre,12345,36000,runner=runner,phase='part2',prior_lock=self.prior(root,root/'p1'))
    self.assertEqual(len(helper.calls),count);self.assertEqual(len({r['run_id'] for r in helper.calls}),count);self.assertEqual(helper.calls[len(first)]['arm'],'X72')
    decision=json.loads((root/'p2/decision_stage1_fwe.json').read_text());self.assertIn('E_fwe',decision['budgets']['0.8B']);self.assertNotIn('E_dclm',decision['budgets']['0.8B'])
 def test_missing_interim_and_24h_refused(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);helper,pre,inputs,prov=self.setup(root);runner=helper.mock_runner(prov)
   with self.assertRaisesRegex(ValueError,'interim'):execute(root/'bad',inputs,pre,1,0,runner=runner,phase='part2')
   self.assertEqual(helper.calls,[])
   with self.assertRaisesRegex(ValueError,'24h'):execute(root/'late',inputs,pre,1,24*3600,runner=runner)
