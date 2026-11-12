import csv,json,tempfile,unittest,tarfile
from pathlib import Path
from unittest.mock import patch
from scripts.dot_pool_report import main,diagnostic_row
from scripts.dot_pool_report_recovery import recover
from efficient_llm.dot_pool.data import sha

class ReportTests(unittest.TestCase):
 def test_distinct_run_and_diagnostic_seeds(self):
  source={'seed':0,'mean_cost':.2}
  row=diagnostic_row('E2','DOT',1,source)
  self.assertEqual(row['seed'],1);self.assertEqual(row['diagnostic_seed'],0)
  self.assertEqual(source,{'seed':0,'mean_cost':.2})
 def test_full_report_with_shuffle_seed_and_preserved_flags(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td)
   def put(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))
   for study,arms in [('E1',['D','L','SO','DOT','SR']),('E2',['D','PL','SO','DOT','SR'])]:
    for arm in arms:
     for seed in [0,1]:
      run=root/f'{study}-{arm}-s{seed}';steps=154000 if study=='E1' else 5722
      put(run/'config.json',dict(study=study,arm=arm,seed=seed,steps=steps,counts={'parameters':123},model={}))
      metrics={k:{'accuracy':.95 if k=='id_test' else .1} for k in ['id_test','ood_test','train','atoms','ood_atoms','ood_dev']} if study=='E1' else {'val':{'nll':3.4+seed*.001}}
      final=dict(metrics=metrics,flags=[],step=steps)
      put(run/'final_eval.json',final);put(run/f'eval-{steps}.json',final)
      put(run/'train_complete.json',dict(elapsed=100.,step=steps,flags=[]))
      put(run/'analysis_complete.json',dict(complete=True,flags=[]))
      (run/'metrics.jsonl').write_text(json.dumps(dict(step=steps,nll=.1))+'\n')
      if study=='E2':
       put(run/'eval-4577.json',final)
       put(run/'downstream.json',dict(metrics={'hellaswag':.3,'piqa':.6,'arc_easy':.4,'lambada_openai':.3}))
       if arm in ['SO','DOT']:put(run/'shuffle.json',dict(seed=0,mean_cost=.1))
      elif arm=='DOT':put(run/'reentry.json',dict(groups={'all':{'ci':[.1,.2]}},bootstrap_seed=0))
   put(root/'status.json',dict(flags=[{'flag':'original_report_failure'}]))
   with patch('sys.argv',['report','--root',td]):main()
   out=json.loads((root/'readout.json').read_text())
   self.assertEqual([len(out['tables'][k]) for k in ['E1','E2']],[10,10]);self.assertEqual(len(out['downstream']),10)
   readings=[r['name'] for r in out['readings'] if r['name'].startswith('E2-R4')]
   self.assertEqual(len(readings),4);self.assertEqual(len(set(readings)),4)
   self.assertIn('E2-R4-DOT-s1',readings)
   rows=json.loads((root/'shuffle_summary.json').read_text())['rows']
   self.assertTrue(all(r['diagnostic_seed']==0 for r in rows));self.assertEqual({r['seed'] for r in rows},{0,1})
   with (root/'e1_reentry.csv').open() as f:self.assertEqual(len(list(csv.DictReader(f))),2)
   self.assertTrue(any('original_report_failure' in f for f in out['flags']))
   self.assertIn('Downstream (reported only)',(root/'REPORT.md').read_text())
   self.assertEqual(json.loads((root/'report_complete.json').read_text())['coverage'],{'E1':10,'E2':10})

   put(root/'phase4.json',dict(extra_epochs=0,elapsed_seconds=53000))
   (root/'source.py').write_text('pass\n');put(root/'source-manifest.json',dict(files={'source.py':sha(root/'source.py')}))
   expected=dict(burn={},source=str(root),source_manifest=str(root/'source-manifest.json'),source_manifest_sha256=sha(root/'source-manifest.json'),data_locks={},analysis_archives={})
   for run in sorted(root.glob('E[12]-*')):
    records=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),tier=1,locations=[str(p)]) for p in run.iterdir() if p.is_file()]
    put(run/'artifact_manifest.json',dict(artifacts=records))
    archive=root/(run.name+'.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:tar.add(run/'artifact_manifest.json',arcname='artifact_manifest.json')
    expected['analysis_archives'][run.name]=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive))
   with patch('scripts.dot_pool_report_recovery.check_burn',return_value={}):
    done=recover(root,root/'recovered',expected)
    self.assertTrue(done['verified']);self.assertTrue(done['original_inputs_unchanged'])
    with self.assertRaises(AssertionError):recover(root,root/'recovered',expected)
   proof=json.loads((root/'recovered/report/recovery_verification.json').read_text())
   self.assertEqual(len(proof['report_rows_replayed']),20)
   self.assertEqual(proof['downstream_rows_replayed'],10)
   self.assertEqual(json.loads((root/'status.json').read_text())['flags'],[{'flag':'original_report_failure'}])

if __name__=='__main__':unittest.main()
