import copy,json,tempfile,unittest
from pathlib import Path
from efficient_llm.warm_order.data import sha
from efficient_llm.warm_order.horizon_train import write_json
from efficient_llm.warm_order.adaptation_train import source_hashes
from efficient_llm.warm_order.adaptation_runtime import training_policy
from scripts.layer_adaptation_requalify_queue import validate_record

class ReusedGate(unittest.TestCase):
    def test_reused_identity_state_and_trace_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);c=dict(branch='A',row_reservation=dict(seed=0,manifest_sha256='rows'),parent_hashes={'parent':'pin'})
            admission=dict(prepared_sha256='prepared');r=dict(branch='A',seed=0,parent_hashes=c['parent_hashes'],source=source_hashes(),
                training_policy=training_policy('cuda'),prepared_sha256='prepared',reservation_sha256='rows',step=40,
                exact_resume=True,finite=True,parent_step0_exact=True,native_component_parity_pass=True)
            for sub in ('whole','split'):
                (root/sub).mkdir();(root/sub/'fullstate.pt').write_bytes(b'fixture state')
                (root/sub/'trace.jsonl').write_text(''.join(json.dumps({'step':i})+'\n' for i in range(1,41)))
                r[sub+'_fullstate_sha256']=sha(root/sub/'fullstate.pt')
            (root/'numerical.json').write_text('{}');r['numerical_sha256']=sha(root/'numerical.json');write_json(root/'complete.json',r)
            h=sha(root/'complete.json');self.assertEqual(validate_record(root,c,admission,h),r)
            (root/'split/fullstate.pt').write_bytes(b'altered')
            with self.assertRaisesRegex(ValueError,'checkpoint'):validate_record(root,c,admission,h)
            (root/'split/fullstate.pt').write_bytes(b'fixture state');(root/'split/trace.jsonl').write_text('{}\n')
            with self.assertRaisesRegex(ValueError,'traces'):validate_record(root,c,admission,h)
            (root/'complete.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'receipt'):validate_record(root,c,admission,h)

if __name__=='__main__':unittest.main()
