import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from efficient_llm.warm_order.adaptation_reservation import reserve, verify, select
from efficient_llm.warm_order.data import sha


class AdaptationReservation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        r = self.root; stem = str(r/'stream')
        np.arange(60*8, dtype=np.uint32).reshape(60,8).tofile(stem+'.bin')
        starts = np.zeros((60,8), dtype=np.uint8); starts[:,0] = 1
        starts[[5,10],4] = 1; starts.tofile(stem+'.starts')
        np.savez(r/'known.npz', known_e2_rows=np.array([0,3,9], dtype=np.int64))
        proof = dict(token_sha256=sha(stem+'.bin'), starts_sha256=sha(stem+'.starts'),
                     output_sha256=sha(r/'known.npz'), known_union_rows=3)
        (r/'proof.json').write_text(json.dumps(proof))
        self.plan = dict(schema=1, domain='dclm', row_space='E2_destination', data=stem,
            expected_stream_hashes={p:sha(p) for p in (stem+'.bin',stem+'.starts')},
            exposure_proof=str(r/'proof.json'), exposure_proof_sha256=sha(r/'proof.json'),
            exposure_rows=str(r/'known.npz'), stream_rows=60, length=8, steps=6, batch=2,
            calibration_rows=4, confirmation_rows=4, selection_seed=20261121)

    def tearDown(self): self.temp.cleanup()

    def test_deterministic_actual_boundary_filter_and_split_disjointness(self):
        a = reserve(self.plan, self.root/'a'); b = reserve(self.plan, self.root/'b')
        self.assertEqual(a['eligible_rows'],58)
        self.assertFalse(a['whole_document_disjointness_verified'])
        self.assertFalse(a['production_ready'])
        groups=[]
        for name in ('calibration','confirmation','train-s0','train-s1'):
            x=np.load(self.root/'a'/(name+'.npy')); groups.append(x)
            self.assertEqual(sha(self.root/'a'/(name+'.npy')),sha(self.root/'b'/(name+'.npy')))
        all_ids=np.concatenate(groups)
        self.assertEqual(len(np.unique(all_ids)),32)
        self.assertEqual(np.intersect1d(all_ids,[0,3,9,5,10]).size,0)
        self.assertEqual(verify(self.root/'a')['status'],'PASS')

    def test_hash_and_rehashed_overlap_rejected(self):
        out=self.root/'a'; reserve(self.plan,out)
        train=np.load(out/'train-s0.npy'); train[0]=np.load(out/'confirmation.npy')[0]
        np.save(out/'train-s0.npy',train)
        with self.assertRaisesRegex(ValueError,'hash'):verify(out)
        p=out/'manifest.json'; m=json.loads(p.read_text()); m['files']['train-s0.npy']=sha(out/'train-s0.npy')
        p.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError,'overlap'):verify(out)

    def test_preflight_failures_do_not_create_reservation(self):
        bad=copy.deepcopy(self.plan); bad['expected_stream_hashes'][bad['data']+'.bin']='bad'
        with self.assertRaisesRegex(ValueError,'stream hash'):reserve(bad,self.root/'bad')
        self.assertFalse((self.root/'bad').exists())
        bad=copy.deepcopy(self.plan); bad['row_space']='packed_heal'
        with self.assertRaisesRegex(ValueError,'row space'):reserve(bad,self.root/'space')
        counts=dict(calibration=4,confirmation=4,**{'train-s0':12,'train-s1':12})
        with self.assertRaisesRegex(ValueError,'Insufficient'):
            select(np.arange(5),np.array([],dtype=np.int64),counts,0)
        with self.assertRaisesRegex(ValueError,'Unique'):
            select(np.array([1,1,2]),np.array([],dtype=np.int64),counts,0)


if __name__=='__main__':unittest.main()
