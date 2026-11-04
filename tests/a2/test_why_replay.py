import json
from pathlib import Path
import tempfile
import unittest
from efficient_llm.a2_why.common import write
from efficient_llm.a2_why.protocol import descriptor
from efficient_llm.a2.protocol import lr_for_update
from scripts.a2_why_replay import replay

class ReplayTests(unittest.TestCase):
    def test_independent_replay_rejects_corrupt_sample_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);item=descriptor('YT1','dclm','P300');name=item['run_id'];steps=item['steps']
            r=root/'heals'/name
            write(r/'result.json',dict(run_id=name,registered_run=item,block='S1',lr_updates=[lr_for_update(i,steps,.001) for i in range(1,steps+1)],delta_post=.25,roundtrip_difference=0))
            for label,value in (('parent',2.),('pre-S1',3.),('post-S1',2.25)):
                write(root/'perseq'/name/(label+'.json'),dict(loss=value,per_sequence_nll=[value]*2461,tokens=2461*4095))
            order=[[list(range(batch*32+rank*4,batch*32+rank*4+4)) for batch in range(steps*2)] for rank in range(8)]
            write(r/'order.json',dict(rank_microbatches=order))
            self.assertEqual(replay(root)['status'],'PASS')
            order[0][0][0]=999;write(r/'order.json',dict(rank_microbatches=order))
            failed=replay(root);self.assertEqual(failed['status'],'FLAGGED');self.assertEqual([x['test'] for x in failed['checks'] if not x['passed']],[name+':order'])

if __name__=='__main__':unittest.main()
