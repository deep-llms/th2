import json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from efficient_llm.warm_order.path_background_study import _summary
from efficient_llm.warm_order.path_background_delivery import finish
from efficient_llm.warm_order.data import sha
from scripts.verify_path_background_export import verify

class Delivery(unittest.TestCase):
    def test_small_archive_replay_and_trusted_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)/'node';(base/'execution').mkdir(parents=True)
            for name in ('source.json','admission.json'):(base/'execution'/name).write_text('{}')
            (base/'prepared').mkdir();(base/'prepared/prepared.json').write_text('{}')
            for seed in (0,1):
                root=base/'prepared'/f's{seed}';s=root/'study';s.mkdir(parents=True)
                plan=dict(parent_hashes={'fullstate.pt':'a'*64},expected_inputs={'rows':'b'*64});(root/'plan.json').write_text(json.dumps(plan))
                labels=['A/identity','B/identity','A/project','B/project'];raw=dict(row_ids=np.array([2,7]),token_count=np.array([3,5]))
                raw.update({name:np.array([3.+i,8.+i]) for i,name in enumerate(labels)});np.savez(s/'scores.npz',**raw)
                (s/'REPORT.json').write_text(json.dumps(dict(scores=_summary(raw,labels))))
                for name in ('parity.json','complete.json'):(s/name).write_text('{}')
                torch.save(torch.eye(2),s/'basis.pt')
                manifest=dict(small_files={n:sha(s/n) for n in ('REPORT.json','scores.npz','parity.json')},labels=labels,plan_sha256=sha(root/'plan.json'),model_pins=plan['parent_hashes'],input_pins=plan['expected_inputs'],basis_sha256=sha(s/'basis.pt'))
                (s/'manifest.json').write_text(json.dumps(manifest))
            receipt=finish(base);result=verify(base/'tier1.tar.gz',receipt['archive_sha256'],Path(tmp)/'local')
            self.assertEqual(result['status'],'PASS');self.assertFalse(result['basis_and_weights_exported'])
            self.assertEqual(result['seeds']['0']['sequences'],2)
            with self.assertRaises(ValueError):verify(base/'tier1.tar.gz','bad',Path(tmp)/'bad')
            with self.assertRaises(ValueError):finish(base)

if __name__=='__main__':unittest.main()
