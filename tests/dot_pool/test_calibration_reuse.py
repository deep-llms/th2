import json,tempfile,unittest
from pathlib import Path
from efficient_llm.dot_pool import SPEC_SHA
from efficient_llm.dot_pool.data import sha
from scripts.dot_pool_calibration_reuse import validated_score
class CalibrationEvidence(unittest.TestCase):
 def test_hashes_and_completeness_gate_reuse(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);state=root/'final_state';state.mkdir()
   cfg=dict(study='E1',arm='D',seed=0,steps=38500,lr=.001,batch=4096,microbatch=4096,model={},data='fixture',data_ready_sha256='fixture',spec_sha256=SPEC_SHA,actual_world=4)
   def save(name,value):(root/name).write_text(json.dumps(value))
   save('config.json',cfg);save('train_complete.json',dict(step=38500,flags=[]))
   save('final_eval.json',dict(step=38500,flags=[],metrics=dict(id_dev=dict(accuracy=.8,targets=1000))))
   for name in ['state.pt',*[f'rank{i}.pt' for i in range(4)]]:(state/name).write_bytes(b'unit test checkpoint fixture')
   save('final_state/complete.json',dict(step=38500,world=4,files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in state.iterdir()}))
   score,proof=validated_score(root,cfg);self.assertEqual(score,.8);self.assertTrue(proof['checkpoint_members_verified'])
   (state/'rank3.pt').write_bytes(b'tampered')
   with self.assertRaises(ValueError):validated_score(root,cfg)
 def test_foreign_configuration_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td)
   for n in ['config.json','train_complete.json','final_eval.json']:(root/n).write_text('{}')
   with self.assertRaises(ValueError):validated_score(root,dict(study='E1'))
if __name__=='__main__':unittest.main()
