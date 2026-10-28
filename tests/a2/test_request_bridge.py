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
