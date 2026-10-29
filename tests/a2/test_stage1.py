import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from efficient_llm.a2_stage1.protocol import Run,GRID,remaining,grid_choice,decision
from efficient_llm.a2.protocol import lr_for_update
from scripts.a2_stage1 import execute,validate_fit,finalize_preflight
from scripts.launch_a2_stage1 import preflight_jobs
from run_experiments import load_jobs

class Stage1Tests(unittest.TestCase):
 def row(self,arm,lr,val,div=False):return dict(arm=arm,lr=lr,status='complete',val_tune_post=val,diverged=div)
 def test_grid_common_tie_edge_divergence(self):
  rows=[self.row(a,lr,v) for lr,v in zip(GRID,[2.001,2.,2.03]) for a in ('P300','P1T')]
  x=grid_choice(rows);self.assertEqual(x['chosen_lr'],1e-4);self.assertIsNone(x['edge_lr'])
  rows[2]['diverged']=True;x=grid_choice(rows);self.assertEqual(x['edge_lr'],3e-5)
  with self.assertRaises(ValueError):grid_choice(rows+rows[:1])
 def test_boundaries_and_count(self):
  for value,label in [(None,'INCOMPLETE'),(.008,'persists'),(-.008,'reversed'),(.003,'grey'),(-.0029,'erased')]:self.assertEqual(decision(value,'length')['label'],label)
  for value,label in [(.01,'anneal first'),(-.01,'prune before you anneal'),(.003,'grey'),(0,'no effect')]:self.assertEqual(decision(value,'state')['label'],label)
  self.assertEqual(6*762+sum(r.steps for r in remaining(3e-4)),51108)
  self.assertEqual(Run('X72','3.2B',3e-4).parent,'S72');self.assertEqual(Run('X72','3.2B',3e-4).teacher,'P300')
 def test_job_contract(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'jobs.json';p.write_text(json.dumps(preflight_jobs('/inputs','/config',12345)));load_jobs(p)
 def setup_execution(self,root):
  prov={'spec_sha256':'fixture'};parents={}
  for p in ('P300','P1T','S72'):
   scores={k:{'loss':3.} for k in ('final','tuning','small','secondary')}
   parents[p]=dict(provenance=prov,baseline=scores,pre=scores)
  gate=dict(status='passed',provenance=prov,report_before_launch=False,eligible_x72=True,step_seconds=.001,estimated_remaining_seconds=2000,preflight_artifacts={},parents=parents)
  pre=root/'pre';pre.mkdir();(pre/'tests.json').write_text(json.dumps(gate))
  spec=root/'spec.md';spec.write_text('fixture');inputs=root/'inputs.json';inputs.write_text(json.dumps(dict(provenance=prov,spec=str(spec),data_manifest={})))
  return pre,inputs,prov
 def mock_runner(self,prov,edge=False,diverge_main=False):
  self.calls=[]
  def f(argv,**kwargs):
   out=Path(argv[argv.index('--output')+1]);r=json.loads(Path(argv[argv.index('--run')+1]).read_text());self.calls.append(r)
   run=Run(r['arm'],r['budget'],r['lr']);n=run.steps
   score={1e-4:2.0 if edge else 2.1,3e-4:2.05 if edge else 2.,1e-3:2.2,3e-5:1.9}[run.lr]
   row=dict(r,status='complete',provenance=prov,diverged=diverge_main and run.budget=='3.2B',val_tune_post=score,val_post=score,delta_post=score-3,val_secondary_post=score,delta_secondary_post=score-3,coverage=dict(examples=n*64,microbatches=n*2,world_size=8,exact_order=True),lr_updates=[lr_for_update(i,n,run.lr) for i in range(1,n+1)],roundtrip_difference=0.)
   out.mkdir();(out/'result.json').write_text(json.dumps(row));return SimpleNamespace(returncode=0)
  return f
 def test_exact13_and_fresh_budgets(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);pre,inputs,prov=self.setup_execution(root)
   execute(root/'run',inputs,pre,12345,0,runner=self.mock_runner(prov))
   self.assertEqual(len(self.calls),13);self.assertEqual([r['arm'] for r in self.calls[6:9]],['P300','P1T','X72'])
   self.assertTrue(all(r['lr']==3e-4 for r in self.calls[6:]));self.assertFalse(json.loads((root/'run/complete.json').read_text())['full_stage_complete'])
 def test_one_edge_pair_only_and_stop_non_grid(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);pre,inputs,prov=self.setup_execution(root);execute(root/'run',inputs,pre,12345,0,runner=self.mock_runner(prov,edge=True));self.assertEqual(len(self.calls),15)
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);pre,inputs,prov=self.setup_execution(root)
   with self.assertRaisesRegex(ValueError,'Non-grid divergence'):execute(root/'run',inputs,pre,12345,0,runner=self.mock_runner(prov,diverge_main=True))
   self.assertEqual(len(self.calls),7);self.assertFalse((root/'run/complete.json').exists())

 def test_preflight_refuses_failed_t17_and_excludes_only_x72(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);prov={'fixture':1};base={k:{'loss':3.} for k in ('final','tuning','small','secondary')}
   for name in ('P300','P1T','S72'):
    (r/(name+'.json')).write_text(json.dumps(dict(parent=name,provenance=prov,T1={'passed':name!='S72'},sanity={'passed':True},baseline=base,pre=base)))
   (r/'numerical.json').write_text(json.dumps(dict(passed=True,provenance=prov,T5_loss=0.)))
   (r/'throughput.json').write_text(json.dumps(dict(passed=True,provenance=prov,step_seconds=1.)))
   (r/'t17').mkdir();(r/'t17/result.json').write_text(json.dumps(dict(roundtrip_difference=0.)))
   check=dict(passed=False,provenance=prov);(r/'t17/T17.json').write_text(json.dumps(check))
   ip=r/'inputs.json';ip.write_text(json.dumps(dict(provenance=prov,T3={'passed':True},data_manifest={'T7':{'passed':True},'T18':{'sequences':800000}})))
   with self.assertRaisesRegex(ValueError,'T17'):finalize_preflight(r,ip)
   check['passed']=True;(r/'t17/T17.json').write_text(json.dumps(check))
   result=finalize_preflight(r,ip);self.assertFalse(result['eligible_x72']);self.assertEqual(result['status'],'passed')
