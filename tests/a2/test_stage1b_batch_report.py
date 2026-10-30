"""Full-sized synthetic artifact test; no GPU or invented scientific result."""
import tempfile
import unittest
from pathlib import Path
from eval.a2_run import write
from eval.a2_request_bridge import digest
from eval.a2_stage1b_batch_report import compare
from eval.downstream_suite import COUNTS,PRIMARY


class BatchReport(unittest.TestCase):
    def test_locked_full_artifacts_prediction_flips_and_accuracy_decision(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);identity=dict(dataset='fwe',parent='P300',kind='parent')
            requests=[dict(context=[1],continuation=[2],text=['q','a']) for _ in range(78259)]
            protocol=dict(tasks=list(COUNTS),limit=None,seed=42,num_fewshot=0,max_length=4096)
            prepared=dict(requests=requests,protocol=protocol,request_sha256=digest(requests),protocol_sha256=digest(protocol))
            inputs=dict(identity=write(root/'identity.json',identity),prepared=write(root/'prepared.json',prepared),scores={},replays={},predictions={})
            for batch in (1,2):
                score=dict(batch_size=batch,checkpoint_identity=identity,checkpoint_identity_sha256=digest(identity),
                    prepared_sha256=digest(prepared),request_sha256=digest(requests),likelihood_requests=78259,
                    scores=[[-1.,False]]*78259,max_length=4096,precision='fp32_parameters_bf16_cuda_autocast',packages={})
                inputs['scores'][str(batch)]=write(root/f'score{batch}.json',score)
                token=dict(batch_size=batch,checkpoint_identity_sha256=digest(identity),prepared_sha256=digest(prepared),
                    request_sha256=digest(requests),predictions=[[7]]*78259)
                inputs['predictions'][str(batch)]=write(root/f'tokens{batch}.json',token)
                samples={}
                for task,n in COUNTS.items():
                    samples[task]=[dict(doc_id=i,doc={},arguments=[['q','a']],target='c',
                        filtered_resps=[[-1.,False]] if task=='lambada_openai' else [[-1.,False],[-2.,False],[-3.,False]]) for i in range(n)]
                # Two wrong MC predictions change; aggregate accuracy is unchanged.
                if batch==2:samples['sciq'][0]['filtered_resps']=[[-2.,False],[-1.,False],[-3.,False]]
                replay=dict(results={t:{PRIMARY[t]:.5} for t in COUNTS},samples=samples,
                    a2_metadata=dict(protocol=protocol,prepared_sha256=digest(prepared),request_sha256=digest(requests),
                        scored_file_sha256=inputs['scores'][str(batch)]['sha256']),
                    **{'n-samples':{t:dict(original=n,effective=n) for t,n in COUNTS.items()},'n-shot':{t:0 for t in COUNTS}})
                inputs['replays'][str(batch)]=write(root/f'replay{batch}.json',replay)
            choices={t:(lambda doc:['a','b','c']) for t in COUNTS if t!='lambada_openai'}
            result=compare(inputs,choices)
            self.assertTrue(result['hard_batch_passed']);self.assertEqual(result['production_batch_size'],2)
            self.assertEqual(result['prediction_flips']['sciq']['flips'],1)
            self.assertEqual(result['prediction_flips']['lambada_openai']['flips'],0)
            self.assertEqual(result['likelihood']['summary']['max_abs'],0)
            self.assertEqual(result['full_compared_tokens'],78259)
            # Honest integrity failure: argmax evidence changed without updating its lock.
            Path(inputs['predictions']['2']['path']).write_text('{}')
            with self.assertRaises(ValueError):compare(inputs,choices)


if __name__=='__main__':unittest.main()
