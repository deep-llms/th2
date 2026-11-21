"""Exercise orchestration bindings and export boundary; no GPUs/process control."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from efficient_llm.warm_order.adaptation_finish import small_study_files
from efficient_llm.warm_order.adaptation_run import bound_config
from efficient_llm.warm_order.data import sha

class AdaptationDelivery(unittest.TestCase):
    def test_prepared_config_binding_and_gate_forwarding(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);p=root/'A-s0.json';c=dict(branch='A',row_reservation={'seed':0});p.write_text(json.dumps(c))
            m=root/'manifest.json';m.write_text(json.dumps({'files':{'A-s0.json':sha(p)}}));h=sha(m)
            with patch('efficient_llm.warm_order.adaptation_run.validate_binding') as binding, patch('efficient_llm.warm_order.adaptation_run.check_runtime') as runtime:
                result=bound_config(root,h,'A-s0','/node/gate.json','a'*64)
                binding.assert_called_once_with(result)
                self.assertEqual({k:v for k,v in result.items() if k!='gpu_admission'},c)
                runtime.assert_called_once_with(result,'cuda',None,False)
                self.assertEqual(result['gpu_admission'],{'path':'/node/gate.json','sha256':'a'*64})
                p.write_text('{}')
                with self.assertRaisesRegex(ValueError,'config changed'):bound_config(root,h,'A-s0','/node/gate.json','a'*64)
            with self.assertRaisesRegex(ValueError,'Registered'):bound_config(root,h,'unregistered','/node/gate.json','a'*64)

    def test_small_export_whitelist_excludes_weights_and_raw_moments(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'manifest.json').write_text(json.dumps({'snapshots':['step_000000','step_002000']}))
            (root/'fullstate.pt').write_bytes(b'weights must stay on node')
            (root/'moments_000.npz').write_bytes(b'raw moments must stay on node')
            names={str(p.relative_to(root)) for p in small_study_files(root)}
            self.assertEqual(len(names),33)
            self.assertIn('step_002000/loss/scores.npz',names)
            self.assertIn('step_000000/geometry/remaining/row_ids.npy',names)
            self.assertFalse(any(n.endswith('.pt') or 'moments_' in n for n in names))

if __name__=='__main__':unittest.main()
