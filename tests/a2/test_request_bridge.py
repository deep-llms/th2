import copy
import json
from pathlib import Path
import tempfile
import unittest

from lm_eval.api.instance import Instance
from eval.a2_request_bridge import make_bridge, RequestsCaptured
from eval.a2_likelihood import score_requests
from eval.a2_harness import make_harness
from test_harness import tokenizer
from test_model import tiny_model


class RequestBridgeTests(unittest.TestCase):
    def instances(self):
        return [Instance(request_type='loglikelihood',doc={},arguments=pair,idx=i)
                for i,pair in enumerate([('w1 w2',' w3 w4'),('', 'w5'),('w1 ', 'w2')])]

    def prepared(self):
        bridge=make_bridge(tokenizer())
        with self.assertRaises(RequestsCaptured):bridge.loglikelihood(self.instances(),disable_tqdm=True)
        self.assertEqual(bridge.calls,1)
        self.assertIsNotNone(bridge.captured)
        return bridge.captured

    def test_persisted_cli_capture_score_replay(self):
        import hashlib
        from unittest.mock import patch
        from eval.a2_requests import main
        from eval.a2_request_bridge import digest
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);tokdir=root/'tokenizer';tokenizer().save_pretrained(tokdir)
            hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in tokdir.iterdir()}
            spec=root/'spec.json';spec.write_text(json.dumps(dict(tokenizer_provenance={
                'gpt_neox':dict(files=hashes)})))
            data=root/'data.jsonl';data.write_text(json.dumps(dict(
                question='w1 w2',choices=['w3','w4'],answer=0))+'\n')
            manifest=root/'manifest.json';manifest.write_text('{}')
            config=dict(task='fixture',dataset_path='json',dataset_kwargs=dict(
                data_files={'validation':str(data)},cache_dir=directory),validation_split='validation',
                output_type='multiple_choice',doc_to_text='question',doc_to_target='answer',
                doc_to_choice='choices',num_fewshot=0,
                metric_list=[dict(metric='acc',aggregation='mean',higher_is_better=True)])
            prepared_path=root/'prepared.json';scored_path=root/'scored.json';result_path=root/'result.json'
            common=['--tokenizer',str(tokdir),'--input-spec',str(spec),'--manifest',str(manifest),
                    '--dataset-root',directory,'--tasks','fixture','--max-length','16']
            def invoke(args):
                with patch('sys.argv',['a2_requests']+args), patch(
                    'eval.benchmark_data.local_dataset_paths',return_value={}), patch(
                    'eval.benchmark_data.task_configs',return_value=[config]), patch(
                    'eval.downstream_suite.COUNTS',{'fixture':1}):main()
            invoke(['prepare',*common,'--output',str(prepared_path)])
            prepared=json.loads(prepared_path.read_text())
            self.assertNotIn('results',prepared)
            scored=score_requests(tiny_model(),prepared['requests'],max_length=16)
            scored.update(prepared_sha256=digest(prepared),request_sha256=prepared['request_sha256'])
            scored_path.write_text(json.dumps(scored))
            invoke(['replay',*common,'--prepared',str(prepared_path),'--scored',str(scored_path),
                    '--output',str(result_path)])
            result=json.loads(result_path.read_text())
            self.assertEqual(result['n-samples']['fixture'],dict(original=1,effective=1))
            self.assertEqual(result['a2_metadata']['prepared_sha256'],digest(prepared))
            with self.assertRaises(SystemExit):
                invoke(['prepare',*common,'--output',str(prepared_path)])

    def test_persisted_protocol_binding(self):
        from eval.a2_requests import validate_prepared, validate_scored
        from eval.a2_request_bridge import digest
        prepared=self.prepared()
        protocol=dict(max_length=16,seed=42,tasks=['fixture'])
        prepared.update(protocol=protocol,protocol_sha256=digest(protocol))
        prepared=json.loads(json.dumps(prepared))
        scored=dict(prepared_sha256=digest(prepared),request_sha256=prepared['request_sha256'],
            max_length=16,precision='fp32_parameters_bf16_cuda_autocast',likelihood_requests=3)
        validate_prepared(prepared,protocol);validate_scored(prepared,scored)
        for key,value in [('max_length',8),('likelihood_requests',2),('precision','bf16'),
                          ('prepared_sha256','0'*64)]:
            with self.assertRaises(ValueError):validate_scored(prepared,{**scored,key:value})
        with self.assertRaises(ValueError):validate_prepared(prepared,{**protocol,'seed':43})
        bad=copy.deepcopy(prepared);bad['requests'][0]['continuation'].append(3)
        with self.assertRaises(ValueError):validate_prepared(bad,protocol)

    def test_capture_native_score_and_replay(self):
        prepared=self.prepared()
        scored=score_requests(tiny_model(),prepared['requests'],max_length=16)
        scored['request_sha256']=prepared['request_sha256']
        bridge=make_bridge(tokenizer(),prepared=prepared,scored=scored)
        self.assertEqual(bridge.loglikelihood(self.instances(),disable_tqdm=True),scored['scores'])
        with self.assertRaises(ValueError):bridge.loglikelihood(self.instances(),disable_tqdm=True)

    def test_reject_changed_missing_and_nonfinite_scores(self):
        prepared=self.prepared();scored=dict(request_sha256=prepared['request_sha256'],scores=[[-1.,False]]*3)
        for invalid in (dict(scored,request_sha256='0'*64),dict(scored,scores=scored['scores'][:-1]),
                        dict(scored,scores=[[float('nan'),False]]*3),dict(scored,scores=[[1.,False]]*3)):
            with self.assertRaises(ValueError):make_bridge(tokenizer(),prepared=prepared,scored=invalid)
        bad=copy.deepcopy(prepared);bad['requests'][0]['context'].append(7)
        with self.assertRaises(ValueError):make_bridge(tokenizer(),prepared=bad,scored=scored)
        replay=make_bridge(tokenizer(),prepared=prepared,scored=scored)
        with self.assertRaises(ValueError):replay.loglikelihood(list(reversed(self.instances())),disable_tqdm=True)

    def test_canonical_task_aggregation_matches_direct_harness(self):
        from lm_eval import evaluator
        from lm_eval.api.task import ConfigurableTask
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'local.jsonl'
            path.write_text('\n'.join(json.dumps(row) for row in [
                dict(question='w1 w2',choices=['w3','w4 w5'],answer=0),
                dict(question='w6',choices=['w7 w8','w9'],answer=1)])+'\n')
            def task():
                result=ConfigurableTask(config=dict(task='a2_local_fixture',dataset_path='json',
                    dataset_kwargs=dict(data_files={'validation':str(path)},cache_dir=directory),
                    validation_split='validation',output_type='multiple_choice',
                    doc_to_text='question',doc_to_target='answer',doc_to_choice='choices',
                    num_fewshot=0,metric_list=[dict(metric='acc',aggregation='mean',higher_is_better=True),
                                              dict(metric='acc_norm',aggregation='mean',higher_is_better=True)]))
                result.set_fewshot_seed(42)
                return result
            def evaluate(model):
                return evaluator.evaluate(lm=model,task_dict={'a2_local_fixture':task()},
                    bootstrap_iters=0,log_samples=True)
            capture=make_bridge(tokenizer())
            with self.assertRaises(RequestsCaptured):evaluate(capture)
            prepared=capture.captured;model=tiny_model()
            scored=score_requests(model,prepared['requests'],max_length=16)
            scored['request_sha256']=prepared['request_sha256']
            replay=evaluate(make_bridge(tokenizer(),prepared=prepared,scored=scored))
            direct=evaluate(make_harness(model,tokenizer(),max_length=16))
            self.assertEqual(replay['results'],direct['results'])
            for left,right in zip(replay['samples']['a2_local_fixture'],direct['samples']['a2_local_fixture']):
                self.assertEqual(left['arguments'],right['arguments'])
                self.assertEqual(left['target'],right['target'])


if __name__=='__main__':unittest.main()
