import copy
import tempfile
import unittest
from pathlib import Path
from scripts.verify_a2_stage1b_final import verify_gate,verify_terminal
from eval.a2_request_bridge import digest
from eval.a2_run import write,lock


class FinalVerification(unittest.TestCase):
    def gate(self):
        identity={'name':'parent'};prepared=dict(requests=[dict(continuation=[1]*1000),dict(continuation=[2]*1000)])
        a=[[1]*1000,[2]*1000];b=copy.deepcopy(a);b[0][0]=3
        gate=dict(passed=True,checkpoint_identity_sha256=digest(identity),prepared_sha256=digest(prepared),
            production_batch_size=2,precision='fp32_parameters_bf16_cuda_autocast',
            batch1_token_predictions=a,batch2_token_predictions=b,selected_request_indices=[0,1],
            original_gate=dict(manual_max_nats_per_token=0,batch_max_nats_per_token=.011164188385009766,production_max_nats_per_token=.011164188385009766),
            manual_max_nats_per_token=0,batch_max_nats_per_token=.011164188385009766,
            greedy_token_mismatches=1,compared_tokens=2000,greedy_token_mismatch_rate=.0005,
            thresholds=dict(manual_nats_per_token=2e-5,batch_nats_per_token=.1,greedy_token_mismatch_rate=.001),
            token_diagnostic_ll_errors={'1':0,'2':0})
        return gate,identity,prepared

    def test_recompute_actual_token_denominator_instead_of_trusting_passed(self):
        gate,identity,prepared=self.gate();self.assertEqual(verify_gate(gate,identity,prepared)['greedy_token_mismatches'],1)
        for key,value in [('greedy_token_mismatches',0),('compared_tokens',10000),('manual_max_nats_per_token',1e-6),('selected_request_indices',[0,0])]:
            bad=copy.deepcopy(gate);bad[key]=value
            with self.assertRaises(ValueError):verify_gate(bad,identity,prepared)
        bad=copy.deepcopy(gate);bad['original_gate']['manual_max_nats_per_token']=bad['manual_max_nats_per_token']=3e-5
        with self.assertRaises(ValueError):verify_gate(bad,identity,prepared)
        bad=copy.deepcopy(gate);bad['batch2_token_predictions'][0][:3]=[3,3,3]
        bad.update(greedy_token_mismatches=3,greedy_token_mismatch_rate=.0015)
        with self.assertRaises(ValueError):verify_gate(bad,identity,prepared)

    def test_native_success_must_bind_all_four_outputs_and_source(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'source';source.mkdir();(source/'x.py').write_text('x=1\n')
            write(root/'source-manifest.json',dict(code_commit='a'*40,files={'x.py':lock(source/'x.py')['sha256']}))
            base=root/'supervised/run';base.mkdir(parents=True)
            artifacts=[]
            for name in ('complete','results','batch_decision','paired_accuracy'):
                item=write(base/'downstream'/(name+'.json'),{})
                artifacts.append(dict(path='downstream/'+name+'.json',sha256=item['sha256']))
            run=dict(status='ok',jobs=[dict(name='stage1b-downstream-final',status='ok',returncode=0,artifacts=artifacts)])
            write(base/'complete.json',run)
            write(root/'supervised/supervisor.json',dict(training_status='ok',training_returncode=0,
                finished_at='recorded',free_after_training=[dict(pids=[]) for _ in range(8)]))
            write(root/'supervised/burn-verified.json',dict(collective_progress_verified=True))
            verify_terminal(root,'a'*40)
            with self.assertRaises(ValueError):verify_terminal(root,'b'*40)
            (source/'x.py').write_text('x=2\n')
            with self.assertRaises(ValueError):verify_terminal(root,'a'*40)
            (source/'x.py').write_text('x=1\n')
            import json
            run['jobs'][0]['artifacts'].pop();(base/'complete.json').write_text(json.dumps(run))
            with self.assertRaises(ValueError):verify_terminal(root,'a'*40)


if __name__=='__main__':unittest.main()
