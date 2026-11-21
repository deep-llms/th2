import json,unittest
from pathlib import Path
import numpy as np
import torch
import test_adaptation_prepare as fixture
from efficient_llm.warm_order.path_background_study import prepare,run,replay
from efficient_llm.warm_order.data import sha

class BackgroundStudy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.AdaptationPrepare.setUpClass();cls.f=fixture.AdaptationPrepare;cls.root=cls.f.root
    @classmethod
    def tearDownClass(cls):fixture.AdaptationPrepare.tearDownClass()

    def test_real_schema_preparation_frozen_study_and_replay_tamper(self):
        root=self.root;p=root/'background-input';p.mkdir()
        np.save(p/'cal.npy',np.arange(64));np.save(p/'eval.npy',np.arange(64,128))
        base=fixture.fixture.AdaptationTrain.config['data'];files={}
        for seed in (0,1):
            c=dict(parent=self.f.dirs[str(seed)],parent_hashes=self.f.pins[str(seed)],data=base,
                   calibration_indices=str(p/'cal.npy'),confirmation_indices=str(p/'eval.npy'))
            c['expected_inputs']={n:sha(n) for n in (base+'.bin',base+'.starts',c['calibration_indices'],c['confirmation_indices'])}
            f=p/f'A-s{seed}.json';f.write_text(json.dumps(c));files[f.name]=sha(f)
        (p/'manifest.json').write_text(json.dumps(dict(files=files)))
        out=root/'background-out';receipt=prepare(p,sha(p/'manifest.json'),out)
        self.assertEqual(len(receipt),2)
        # Smaller registered synthetic study, both immutable parent schemas prepared above.
        pp=out/'s0/plan.json';plan=json.loads(pp.read_text());plan.update(anchor_rows=2,evaluation_rows=3,rank=2,control_seeds=[42]);pp.write_text(json.dumps(plan))
        before={s:{f:sha(Path(self.f.dirs[s])/f) for f in pins} for s,pins in self.f.pins.items()}
        result=run(out/'s0',sha(pp));self.assertEqual(result['status'],'PASS');self.assertEqual(result['cases'],14)
        self.assertEqual(before,{s:{f:sha(Path(self.f.dirs[s])/f) for f in pins} for s,pins in self.f.pins.items()})
        r=out/'s0/study/REPORT.json';report=json.loads(r.read_text());report['scores']['original_gap']+=1;r.write_text(json.dumps(report))
        m=out/'s0/study/manifest.json';manifest=json.loads(m.read_text());manifest['small_files']['REPORT.json']=sha(r);m.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,'replay'):replay(out/'s0/study')
        with self.assertRaisesRegex(ValueError,'Fresh'):run(out/'s0',sha(pp))
        with self.assertRaisesRegex(ValueError,'identity'):prepare(p,'bad',root/'bad-prep')

if __name__=='__main__':unittest.main()
