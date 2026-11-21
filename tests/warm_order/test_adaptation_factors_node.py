import copy
import json
import unittest
from pathlib import Path

import numpy as np

from test_adaptation_pipeline import AdaptationPipeline
from efficient_llm.warm_order.adaptation_factors_node import prepare,run_prepared,finish
from efficient_llm.warm_order.data import sha


class FactorsNode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):AdaptationPipeline.setUpClass()
    @classmethod
    def tearDownClass(cls):AdaptationPipeline.tearDownClass()

    def test_prepared_reference_gate_replay_and_small_archive(self):
        f=AdaptationPipeline;root=f.root/'node_factors';root.mkdir()
        ix=f.root/'node_factors_eval.npy';cal=f.root/'node_factors_cal.npy'
        np.save(ix,np.array([192,193]));np.save(cal,np.array([194]))
        stem=f.config['data'];pins={p:sha(p) for p in (stem+'.bin',stem+'.starts',str(ix))}
        c=dict(schema=1,domain='dclm',target=2,path=[0,2,1,3],measurement_device='cpu',
               parent=str(f.root/'parent'),parent_hashes=f.pins,phase_files=f.phase,phase_step=4,
               data=stem,indices=str(ix),expected_inputs=pins,batch_size=1,
               calibration_indices=str(cal),calibration_sha256=sha(cal),
               flags=['exploratory','reused_confirmation','historical_overlap_unknown','fixed_parent_input_reader'])
        plan=f.root/'node_factors_plan.json';plan.write_text(json.dumps({'seeds':{'0':c,'1':copy.deepcopy(c)}}))
        prepared=prepare(plan,sha(plan),root/'prepared')
        self.assertEqual(prepared['actual_panels_loaded'],2)
        for seed in ('0','1'):
            s=root/'prepared'/f's{seed}'
            with self.assertRaisesRegex(ValueError,'config mismatch'):run_prepared(s,'wrong','cpu')
            result=run_prepared(s,prepared['seeds'][seed]['config_sha256'],'cpu')
            self.assertEqual(result,dict(cases=50,sequences=2,gate=True))
            with self.assertRaisesRegex(ValueError,'Fresh'):run_prepared(s,prepared['seeds'][seed]['config_sha256'],'cpu')
        (root/'execution').mkdir()
        for name in ('source.json','admission.json'):(root/'execution'/name).write_text('{}')
        proof=finish(root)
        self.assertFalse(proof['reference_activations_exported'])
        self.assertTrue(all(not Path(p).suffix=='.pt' for p in proof['files']))
        self.assertEqual(sha(root/'tier1.tar.gz'),proof['archive_sha256'])


if __name__=='__main__':unittest.main()
