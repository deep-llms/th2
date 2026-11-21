import copy
import json
import unittest
from dataclasses import asdict

import numpy as np
import torch

import test_adaptation_train as fixture
from efficient_llm.warm_order.adaptation_prepare import prepare
from efficient_llm.warm_order.adaptation_reservation import validate_binding
from efficient_llm.warm_order.adaptation_train import PARENT_FILES, train
from efficient_llm.warm_order.horizon_train import train as parent_train
from efficient_llm.warm_order.data import sha


class AdaptationPrepare(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.AdaptationTrain.setUpClass(); cls.root=fixture.AdaptationTrain.root
        r=cls.root; recipe=json.loads((r/'parent/recipe.json').read_text())
        parent_train(dict(recipe['config'],seed=1),r/'parent1','cpu')
        cls.dirs={'0':str(r/'parent'),'1':str(r/'parent1')}
        cls.pins={s:{n:sha(r/name/n) for n in PARENT_FILES} for s,name in [('0','parent'),('1','parent1')]}
        registration={'parents':{f'D-s{s}':dict(file_pins=p,model=asdict(fixture.AdaptationTrain.model_config),
                                             terminal_lr=.00003) for s,p in cls.pins.items()}}
        cls.registration=r/'parents.json'; cls.registration.write_text(json.dumps(registration))
        stem=fixture.AdaptationTrain.config['data']
        np.savez(r/'excluded.npz',known_e2_rows=np.arange(160,dtype=np.int64))
        (r/'exposure.json').write_text(json.dumps(dict(token_sha256=sha(stem+'.bin'),starts_sha256=sha(stem+'.starts'),
                                                    output_sha256=sha(r/'excluded.npz'),known_union_rows=160)))
        cls.plan=dict(schema=1,domain='dclm',row_space='E2_destination',data=stem,
            expected_stream_hashes={p:sha(p) for p in (stem+'.bin',stem+'.starts')},
            exposure_proof=str(r/'exposure.json'),exposure_proof_sha256=sha(r/'exposure.json'),
            exposure_rows=str(r/'excluded.npz'),stream_rows=200,length=8,steps=4,batch=2,
            calibration_rows=4,confirmation_rows=4,selection_seed=130,
            parent_registration_sha256=sha(cls.registration))
        cls.plan_path=r/'plan.json';cls.plan_path.write_text(json.dumps(cls.plan))

    @classmethod
    def tearDownClass(cls):fixture.AdaptationTrain.tearDownClass()

    def test_actual_parent_reservation_config_handoff_without_parent_mutation(self):
        rng=torch.get_rng_state().clone();out=self.root/'prepared'
        report=prepare(self.plan_path,sha(self.plan_path),self.registration,self.dirs,[0,1,4],1,out)
        torch.testing.assert_close(rng,torch.get_rng_state(),atol=0,rtol=0)
        self.assertFalse(report['production_ready']);self.assertEqual(report['train_jobs_launched'],0)
        self.assertEqual(report['actual_parents_loaded'],2)
        for path,digest in report['files'].items():self.assertEqual(sha(out/path),digest)
        for seed in (0,1):
            configs=[json.loads((out/f'{b}-s{seed}.json').read_text()) for b in ('A','B')]
            for c in configs:validate_binding(c)
            self.assertEqual({k:v for k,v in configs[0].items() if k!='branch'},
                             {k:v for k,v in configs[1].items() if k!='branch'})
            self.assertEqual(self.pins[str(seed)],{n:sha(self.root/('parent' if seed==0 else 'parent1')/n) for n in PARENT_FILES})
        c=json.loads((out/'A-s0.json').read_text());train(c,self.root/'handoff-check',stop_after=1)
        state=torch.load(self.root/'handoff-check/fullstate.pt',weights_only=False)
        self.assertEqual(state['optimizer']['state'][0]['step'].item(),81)
        self.assertEqual(state['recipe']['config']['row_reservation']['manifest_sha256'],sha(out/'reservation/manifest.json'))
        with self.assertRaisesRegex(ValueError,'Fresh'):
            prepare(self.plan_path,sha(self.plan_path),self.registration,self.dirs,[0,1,4],1,out)

    def test_hash_schedule_and_wrong_parent_seed_fail_before_reservation(self):
        with self.assertRaisesRegex(ValueError,'plan hash'):
            prepare(self.plan_path,'bad',self.registration,self.dirs,[0,4],1,self.root/'bad')
        self.assertFalse((self.root/'bad').exists())
        with self.assertRaisesRegex(ValueError,'schedule'):
            prepare(self.plan_path,sha(self.plan_path),self.registration,self.dirs,[1,4],1,self.root/'schedule')
        reg=json.loads(self.registration.read_text());reg['parents']['D-s1']=copy.deepcopy(reg['parents']['D-s0'])
        rp=self.root/'wrong-seed-registration.json';rp.write_text(json.dumps(reg))
        plan=dict(self.plan,parent_registration_sha256=sha(rp));pp=self.root/'wrong-plan.json';pp.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError,'parent seed'):
            prepare(pp,sha(pp),rp,{'0':self.dirs['0'],'1':self.dirs['0']},[0,4],1,self.root/'seed')
        self.assertFalse((self.root/'seed').exists())


if __name__=='__main__':unittest.main()
