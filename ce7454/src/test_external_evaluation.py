import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from predict_external import predict_external, score_predictions
from prepare_evaluation_data import prepare
from train import set_norm


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.fc=torch.nn.Linear(3,3);self.bn=torch.nn.BatchNorm1d(3)
    def forward(self,x):return self.bn(self.fc(x.mean((-2,-1))))


class ExternalTests(unittest.TestCase):
    def test_named_scoring(self):
        p={'checkpoint_sha256':'abc','filenames':['b.jpg','a.jpg'],
           'logits':[[0,2,0],[2,0,0]],'predictions':[1,0]}
        for labels in ({'a.jpg':0,'b.jpg':1},{'a.jpg':[0,1.0],'b.jpg':[1,1.0]}):
            r=score_predictions(p,{'labels':labels,'note':'provenance'})
            self.assertEqual(r['correct'],2);self.assertEqual(r['confusion'],[[1,0,0],[0,1,0],[0,0,0]])
        with self.assertRaises(ValueError):score_predictions(p,{'labels':{'a.jpg':0}})

    def test_reject_untrusted_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'evaluation_manifest.json').write_text('{}')
            with self.assertRaises(ValueError):prepare(root,root/'cache','bad')
            self.assertFalse((root/'cache').exists())

    def test_complete_both_splits_without_mutating_model(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cache=root/'cache';cache.mkdir()
            ready={'status':'passed','purpose':'evaluation_only','manifest_sha256':'abc','splits':{}}
            for split,n in [('dev',900),('test_official',1500)]:
                # Sparse zero-backed files, isolated from all real project data.
                a=np.lib.format.open_memmap(cache/(split+'_images.npy'),mode='w+',dtype=np.uint8,shape=(n,512,512,3));a.flush();del a
                meta=cache/(split+'_meta.json');meta.write_text(json.dumps({'filenames':[f'{i}.jpg' for i in range(n)]}))
                ready['splits'][split]={'meta_sha256':hashlib.sha256(meta.read_bytes()).hexdigest()}
            (cache/'ready.json').write_text(json.dumps(ready))
            set_norm('imagenet');m=Tiny().eval();before={k:v.clone() for k,v in m.state_dict().items()}
            r=predict_external(m,cache,root/'out','checkpoint')
            self.assertEqual(r['splits']['dev']['n'],900);self.assertEqual(r['splits']['test_official']['n'],1500)
            for k,v in m.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
            for split in r['splits']:
                p=json.loads((root/'out'/f'{split}_predictions.json').read_text())
                self.assertEqual(p['filenames'],[f'{i}.jpg' for i in range(len(p['filenames']))])
                self.assertEqual(len(set(p['predictions'])),1)


if __name__=='__main__':unittest.main()
