import copy, unittest
from eval.a2_stage1b_regression import compare, NAME
from eval.downstream_suite import COUNTS,PRIMARY
from scripts.launch_a2_stage1b_regression import job_config,budget
class Regression(unittest.TestCase):
 def data(self):
  s=dict(request_sha256='r',prepared_sha256='p',checkpoint_identity_sha256='i',likelihood_requests=2,batch_size=2,precision='fp32_parameters_bf16_cuda_autocast',max_length=4096,scores=[[-2.0,True],[-3.0,False]])
  r=dict(scores={NAME:{t:dict(primary_metric=PRIMARY[t],examples=COUNTS[t],prompt_sha256=t,accuracy=.5,metrics={PRIMARY[t]:.5}) for t in COUNTS}})
  return s,r
 def test_exact(self):
  s,r=self.data();v=compare(s,s,r,r);self.assertTrue(v['passed']);self.assertEqual(v['max_abs_loglikelihood_difference'],0)
 def test_ll_diagnostic(self):
  s,r=self.data();n=copy.deepcopy(s);n['scores'][1][0]-=.25
  v=compare(s,n,r,r);self.assertTrue(v['passed']);self.assertEqual(v['max_abs_loglikelihood_difference'],.25);self.assertEqual(v['max_difference_index'],1)
 def test_one_task_failure_not_average(self):
  s,r=self.data();n=copy.deepcopy(r);ts=list(COUNTS);n['scores'][NAME][ts[0]]['accuracy']+=.1;n['scores'][NAME][ts[1]]['accuracy']-=.1
  self.assertFalse(compare(s,s,r,n)['passed'])
 def test_unaligned_refused(self):
  s,r=self.data();n=copy.deepcopy(s);n['request_sha256']='changed'
  with self.assertRaises(ValueError):compare(s,n,r,r)
 def test_nonfinite_refused(self):
  s,r=self.data();n=copy.deepcopy(s);n['scores'][0][0]=float('nan')
  with self.assertRaises(ValueError):compare(s,n,r,r)
 def test_failed_gate_still_handoff(self):
  j=job_config(dict(path='/p',sha256='a'*64),1800,'part1')['jobs'][0]
  self.assertEqual(j['gpus'],list(range(8)));self.assertEqual(j['required_outputs'][-1]['json_equals'],dict(status='compared'))


class Admission(unittest.TestCase):
 def test_fail_and_changed_source_refused(self):
  import tempfile,json,hashlib
  from pathlib import Path
  from eval.a2_run import write
  from eval.a2_stage1b_regression import admit_regression
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);src=root/'source';(src/'eval').mkdir(parents=True);(src/'efficient_llm').mkdir();(src/'eval/x.py').write_text('x=1')
   sm=write(root/'source-manifest.json',dict(files={'eval/x.py':hashlib.sha256((src/'eval/x.py').read_bytes()).hexdigest()}))
   base=root/'supervised/run';base.mkdir(parents=True)
   report=write(base/'regression.json',dict(status='compared',passed=True))
   complete=write(base/'complete.json',dict(status='ok',models=1,additional_results={'regression':report}))
   terminal=write(base/'terminal.json',dict(status='ok',jobs=[dict(name='stage1b-regression',returncode=0,status='ok',artifacts=[dict(path='regression.json',sha256=report['sha256']),dict(path='complete.json',sha256=complete['sha256'])])]))
   bundle=write(root/'bundle.json',dict(source_manifest=sm,complete=complete,run_complete=terminal,
    supervisor=write(root/'supervisor.json',dict(training_status='ok',training_returncode=0)),
    burn=write(root/'burn.json',dict(collective_progress_verified=True)),training_predecessor=write(root/'prior.json',{})))
   self.assertTrue(admit_regression(bundle,src)[2]['passed'])
   (src/'eval/x.py').write_text('x=2')
   with self.assertRaises(ValueError):admit_regression(bundle,src)
   (src/'eval/x.py').write_text('x=1')
   (base/'regression.json').write_text(json.dumps(dict(status='compared',passed=False)))
   with self.assertRaises(ValueError):admit_regression(bundle,src)

if __name__=='__main__':unittest.main()
