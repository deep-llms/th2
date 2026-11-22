import copy
import json
import unittest

import numpy as np
import torch

from test_attention_shift import AttentionShift
from efficient_llm.warm_order.attention_shift_train import train, read
from efficient_llm.warm_order.attention_shift_eval import evaluate, replay, load_bias, load_means
from efficient_llm.warm_order.data import sha


class ShiftEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        AttentionShift.setUpClass();cls.f=AttentionShift.f;cls.c=AttentionShift.c
        cls.configs={b:dict(cls.c,branch=b) for b in ('C','S')};cls.runs={}
        for b,c in cls.configs.items():
            out=cls.f.root/('eval-fit-'+b);train(c,out)
            cls.runs[b]=dict(path=str(out),sha256=sha(out/'fullstate.pt'))
        root=cls.f.root/'mean-reference';root.mkdir();hashes={}
        for b in ('A','B'):
            torch.save(dict(mean=torch.linspace(-.02,.03,8,dtype=torch.float64)*(1 if b=='A' else -1)),root/(b+'-basis.pt'))
            hashes[b]=sha(root/(b+'-basis.pt'))
        (root/'calibration.json').write_text(json.dumps(dict(binding=dict(
            checkpoints=dict(parent_file_pins=cls.c['parent_hashes']),
            data_pins={p:cls.c['expected_inputs'][p] for p in (cls.c['data']+'.bin',cls.c['data']+'.starts')},
            calibration_sha256=cls.c['expected_inputs'][cls.c['calibration_indices']]),
            node_files={b+'-basis.pt':h for b,h in hashes.items()})))
        cls.reference=dict(directory=str(root),metadata_sha256=sha(root/'calibration.json'),basis_hashes=hashes)

    @classmethod
    def tearDownClass(cls):AttentionShift.tearDownClass()

    def test_streaming_ten_cells_and_portable_replay(self):
        a=self.f.root/'eval-batch1';z=self.f.root/'eval-batch3'
        first=evaluate(self.configs,self.runs,self.reference,a,batch=1)
        second=evaluate(self.configs,self.runs,self.reference,z,batch=3)
        self.assertEqual(replay(a)['cases'],10)
        self.assertEqual(first['tokens_per_case'],24)
        for k,v in first['cases'].items():
            self.assertAlmostEqual(v['nll_nat'],second['cases'][k]['nll_nat'],places=6)
            if k.endswith('/zero'):self.assertEqual(v['gain_vs_zero_nat'],0.)
        with np.load(a/'scores.npz') as raw:
            gain=(raw['S/zero']-raw['S/learned_S']).sum()/raw['token_count'].sum()
            self.assertEqual(gain,first['cases']['S/learned_S']['gain_vs_zero_nat'])
        self.assertFalse(read(a/'manifest.json')['weights_or_vectors_exported'])
        (a/'scores.npz').write_bytes((a/'scores.npz').read_bytes()+b'tamper')
        with self.assertRaisesRegex(ValueError,'hash'):replay(a)

    def test_means_fits_and_manifest_bindings(self):
        bad=copy.deepcopy(self.reference);bad['basis_hashes']['A']='bad'
        with self.assertRaisesRegex(ValueError,'hash'):load_means(bad,self.c,8)
        bad=copy.deepcopy(self.configs);bad['S']['seed']=1
        with self.assertRaisesRegex(ValueError,'unmatched'):evaluate(bad,self.runs,self.reference,self.f.root/'bad-eval')
        r=self.f.root/'eval-fit-C';recipe=r/'recipe.json';original=recipe.read_bytes()
        try:
            recipe.write_text('{}')
            with self.assertRaisesRegex(ValueError,'recipe'):
                load_bias(r,self.configs['C'],self.runs['C']['sha256'],8,'bad')
        finally:recipe.write_bytes(original)


if __name__=='__main__':unittest.main()
