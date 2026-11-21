import json
import unittest
import numpy as np

from test_adaptation_pipeline import AdaptationPipeline
from efficient_llm.warm_order.adaptation_pipeline import load_snapshot_panel
from efficient_llm.warm_order.adaptation_factors_pipeline import evaluate_stream,replay,run
from efficient_llm.warm_order.data import sha


class FactorsPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        AdaptationPipeline.setUpClass()
        cls.fixture=AdaptationPipeline

    @classmethod
    def tearDownClass(cls):AdaptationPipeline.tearDownClass()

    def test_real_schema_fixture_streaming_replay_and_tamper(self):
        f=self.fixture
        models,proof=load_snapshot_panel(f.root/'parent',f.pins,f.phase)
        ix=f.root/'factors_rows.npy';np.save(ix,np.array([192,194,193]))
        stem=f.config['data'];pins={p:sha(p) for p in (stem+'.bin',stem+'.starts',str(ix))}
        outputs=[]
        for batch in (1,2):
            out=f.root/f'factors_batch{batch}'
            result=evaluate_stream(models,stem,ix,pins,batch,out,proof)
            self.assertEqual(replay(out)['cases'],50);self.assertEqual(result['tokens'],18)
            outputs.append(out)
        with np.load(outputs[0]/'scores.npz') as a,np.load(outputs[1]/'scores.npz') as b:
            for key in a.files:np.testing.assert_allclose(a[key],b[key],atol=2e-6,rtol=1e-6)
        bad=dict(pins);bad[str(ix)]='invalid'
        with self.assertRaisesRegex(ValueError,'identity'):
            evaluate_stream(models,stem,ix,bad,1,f.root/'bad_factors',proof)
        out=outputs[0];m=json.loads((out/'manifest.json').read_text());m['path']=[0,1,2,3]
        (out/'manifest.json').write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError,'registration'):replay(out)
        out=outputs[1];r=json.loads((out/'REPORT.json').read_text());r['cases']['B/attn/000']+=1
        (out/'REPORT.json').write_text(json.dumps(r))
        with self.assertRaisesRegex(ValueError,'hash'):replay(out)

    def test_bound_configuration_actual_schema_loader(self):
        f=self.fixture;ix=f.root/'factors_config_rows.npy';np.save(ix,np.array([192]))
        stem=f.config['data'];pins={p:sha(p) for p in (stem+'.bin',stem+'.starts',str(ix))}
        c=dict(schema=1,domain='dclm',target=2,path=[0,2,1,3],measurement_device='cpu',
               parent=str(f.root/'parent'),parent_hashes=f.pins,phase_files=f.phase,phase_step=4,
               data=stem,indices=str(ix),expected_inputs=pins,batch_size=1,
               flags=['exploratory','reused_confirmation','historical_overlap_unknown','fixed_parent_input_reader'])
        result=run(c,f.root/'factors_config');self.assertEqual(result['sequences'],1)
        self.assertEqual(replay(f.root/'factors_config')['cases'],50)
        configpath=f.root/'factors_config/config.json'
        changed=json.loads(configpath.read_text());changed['phase_step']=3;configpath.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError,'hash'):replay(f.root/'factors_config')
        with self.assertRaisesRegex(ValueError,'snapshot'):run(dict(c,phase_step=3),f.root/'bad_factors_step')
        with self.assertRaisesRegex(ValueError,'flags'):run(dict(c,flags=[]),f.root/'bad_factors_flags')


if __name__=='__main__':unittest.main()
