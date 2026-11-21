import copy
import json
import unittest

import numpy as np
import torch

import test_adaptation_train as fixture
from efficient_llm.warm_order.adaptation_reservation import reserve, bind, validate_binding
from efficient_llm.warm_order.adaptation_train import train
from efficient_llm.warm_order.data import sha


class AdaptationBinding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.AdaptationTrain.setUpClass()
        cls.root = fixture.AdaptationTrain.root
        c = fixture.AdaptationTrain.config; stem = c['data']; r = cls.root
        np.savez(r/'known.npz', known_e2_rows=np.arange(160, dtype=np.int64))
        proof = dict(token_sha256=sha(stem+'.bin'), starts_sha256=sha(stem+'.starts'),
                     output_sha256=sha(r/'known.npz'), known_union_rows=160)
        (r/'known.json').write_text(json.dumps(proof))
        plan = dict(schema=1, domain='dclm', row_space='E2_destination', data=stem,
                    expected_stream_hashes={p:sha(p) for p in (stem+'.bin',stem+'.starts')},
                    exposure_proof=str(r/'known.json'), exposure_proof_sha256=sha(r/'known.json'),
                    exposure_rows=str(r/'known.npz'), stream_rows=200, length=8,
                    steps=c['steps'], batch=c['batch'], calibration_rows=4, confirmation_rows=4,
                    selection_seed=129)
        cls.reservation=r/'reservation'; reserve(plan,cls.reservation)
        cls.digest=sha(cls.reservation/'manifest.json')
        cls.config=bind(c,cls.reservation,cls.digest,0)

    @classmethod
    def tearDownClass(cls): fixture.AdaptationTrain.tearDownClass()

    def test_bound_pair_and_exact_resume(self):
        row_traces=[]
        for branch in ('A','B'):
            c=dict(self.config,branch=branch); whole=self.root/('bound-'+branch); part=self.root/('resume-'+branch)
            validate_binding(c)
            train(c,whole); train(c,part,stop_after=2); train(c,part,resume=True)
            fixture.same(torch.load(whole/'fullstate.pt',weights_only=False),
                         torch.load(part/'fullstate.pt',weights_only=False))
            row_traces.append([json.loads(s)['rows'] for s in (whole/'trace.jsonl').read_text().splitlines()])
            recipe=json.loads((whole/'recipe.json').read_text())
            self.assertEqual(recipe['config']['row_reservation']['manifest_sha256'],self.digest)
            self.assertIn(c['calibration_indices'],recipe['inputs'])
        self.assertEqual(row_traces[0],row_traces[1])
        self.assertEqual(np.array(row_traces[0]).ravel().tolist(),np.load(self.config['indices']).tolist())

    def test_binding_and_seed_tamper_before_output(self):
        c=copy.deepcopy(self.config); c['calibration_indices']=c['confirmation_indices']
        c['expected_inputs']={p:sha(p) for p in [c['data']+'.bin',c['data']+'.starts',c['indices'],c['calibration_indices'],c['confirmation_indices']]}
        with self.assertRaisesRegex(ValueError,'binding'):train(c,self.root/'bad-binding')
        self.assertFalse((self.root/'bad-binding').exists())
        c=bind(fixture.AdaptationTrain.config,self.reservation,self.digest,1)
        with self.assertRaisesRegex(ValueError,'parent seed'):train(c,self.root/'bad-seed')
        self.assertFalse((self.root/'bad-seed').exists())
        c=copy.deepcopy(self.config); c['row_reservation']['manifest_sha256']='bad'
        with self.assertRaisesRegex(ValueError,'manifest hash'):train(c,self.root/'bad-hash')

    def test_unbound_cpu_calibration_overlap_cannot_bypass_guard(self):
        c=copy.deepcopy(fixture.AdaptationTrain.config); c['calibration_indices']=c['indices']
        with self.assertRaisesRegex(ValueError,'calibration/confirmation overlap'):
            train(c,self.root/'overlap')
        self.assertFalse((self.root/'overlap').exists())


if __name__=='__main__':unittest.main()
