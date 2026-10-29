import ast,json,tempfile,unittest
from pathlib import Path
from efficient_llm.a2_stage1b.protocol import Run,phase_runs,cross_dataset_label
from scripts.a2_stage1b_preflight import finalize_preflight
from scripts.launch_a2_stage1b_preflight import preflight_jobs
from scripts.a2_stage1 import validate_fit
from efficient_llm.a2.protocol import lr_for_update
class Stage1bPreflightTests(unittest.TestCase):
 def test_registered_order_and_exact_identity(self):
  self.assertEqual([r.arm for r in phase_runs('part1',.001)],['P300','P1T'])
  self.assertEqual([(r.arm,r.budget) for r in phase_runs('part2',.001)],[('X72','3.2B'),('P300','0.8B'),('P1T','0.8B'),('X72','0.8B'),('X72','0.2B')])
  r=Run('X72','3.2B',.001);self.assertEqual(r.record()['dataset'],'fwe');self.assertEqual(r.parent,'S72');self.assertEqual(r.teacher,'P300');self.assertTrue(r.run_id.startswith('ST1-fwe-'))
  row=dict(r.record(),provenance={'v':1},status='complete',coverage=dict(examples=r.steps*64,microbatches=r.steps*2,world_size=8,exact_order=True),lr_updates=[lr_for_update(i,r.steps,r.lr) for i in range(1,r.steps+1)],roundtrip_difference=0)
  validate_fit(row,r,{'v':1});row['dataset']='dclm'
  with self.assertRaises(ValueError):validate_fit(row,r,{'v':1})
 def test_same_numeric_functions_and_routing_only_changes(self):
  old=Path('efficient_llm/a2_stage1/model_stage.py').read_text();new=Path('efficient_llm/a2_stage1b/model_stage.py').read_text()
  for name in ('t8','evaluation'):
   fn=lambda s:next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name==name)
   self.assertEqual(ast.dump(fn(old)),ast.dump(fn(new)))
  # The training engine, native model/KD/score and T17 NEM route stay shared.
  self.assertIn("OldRun('nem','P300','S1','0.05B'",new)
  self.assertNotIn("load_parent(assets,'dclm'",new)
 def test_preflight_jobs_real_gates(self):
  jobs=preflight_jobs(Path('/inputs'),Path('/accelerate'),12345)['jobs'];self.assertEqual([x['name'] for x in jobs],['P300','P1T','S72','numerical','throughput','t17','report'])
  self.assertTrue(all(x['gpus']==list(range(8)) for x in jobs));self.assertIn('scripts.a2_stage1b_preflight',jobs[-1]['argv'])
 def fixture(self,r,step=1.,s72=True,t17=True):
  prov={'fixture':1};base={k:{'loss':3.} for k in ('final','tuning','small','secondary')}
  for n in ('P300','P1T','S72'):(r/(n+'.json')).write_text(json.dumps(dict(parent=n,provenance=prov,T1={'passed':s72 if n=='S72' else True},sanity={'passed':True},baseline=base,pre=base)))
  (r/'numerical.json').write_text(json.dumps(dict(passed=True,provenance=prov,T5_loss=0.)));(r/'throughput.json').write_text(json.dumps(dict(passed=True,provenance=prov,step_seconds=step)))
  (r/'t17').mkdir();(r/'t17/result.json').write_text(json.dumps(dict(roundtrip_difference=0.)));(r/'t17/T17.json').write_text(json.dumps(dict(passed=t17,provenance=prov)))
  p=r/'inputs.json';p.write_text(json.dumps(dict(provenance=prov,T3={'passed':True},data_manifest={'T7':{'passed':True},'T18':{}},preflight_cap_seconds=7200)));return p
 def test_budget20h_and_s72_only_failure(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);p=self.fixture(r,step=1.4,s72=False);out=finalize_preflight(r,p);self.assertFalse(out['eligible_x72']);self.assertTrue('fwe' in out['tests']['T18']['parent_loss']['P300'])
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);p=self.fixture(r,step=1.4);self.assertTrue(finalize_preflight(r,p)['report_before_launch'])
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);p=self.fixture(r,t17=False)
   with self.assertRaises(ValueError):finalize_preflight(r,p)
 def test_cross_labels(self):
  self.assertEqual(cross_dataset_label('persists'),'replicated');self.assertEqual(cross_dataset_label('reversed'),'not replicated')
