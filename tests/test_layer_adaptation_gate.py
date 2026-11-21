import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from scripts.layer_adaptation_gate_queue import aggregate,NAMES
from efficient_llm.warm_order.data import sha

class GateAggregation(unittest.TestCase):
    def test_all_branches_and_actual_file_hashes_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);prep=base/'prepared';prep.mkdir()
            for name in NAMES:
                branch,seed=name.split('-s');c=dict(branch=branch,parent_hashes={'fullstate.pt':'parent'+seed},row_reservation={'seed':int(seed)})
                (prep/(name+'.json')).write_text(json.dumps(c));run=base/'runs'/name;run.mkdir(parents=True)
                for sub in ('whole','split'):
                    (run/sub).mkdir();(run/sub/'fullstate.pt').write_bytes(b'fixture')
                (run/'numerical.json').write_text('{}')
                r=dict(branch=branch,seed=int(seed),parent_hashes=c['parent_hashes'],source={'s':'h'},prepared_sha256='prepared',step=40,
                       exact_resume=True,finite=True,parent_step0_exact=True,native_component_parity_pass=True,
                       whole_fullstate_sha256=sha(run/'whole/fullstate.pt'),split_fullstate_sha256=sha(run/'split/fullstate.pt'),
                       numerical_sha256=sha(run/'numerical.json'),torch='fixture',precision='bf16',tf32=False,reservation_sha256='rows')
                (run/'complete.json').write_text(json.dumps(r))
            admission=dict(prepared=str(prep/'manifest.json'),prepared_sha256='prepared')
            with patch('scripts.layer_adaptation_gate_queue.source_hashes',return_value={'s':'h'}):
                r=aggregate(base,admission);self.assertEqual(r['production_updates_executed'],0);self.assertEqual(len(r['branches']),4)
                (base/'runs/B-s1/whole/fullstate.pt').write_bytes(b'tampered')
                with self.assertRaisesRegex(ValueError,'checkpoint'):aggregate(base,admission)

if __name__=='__main__':unittest.main()
