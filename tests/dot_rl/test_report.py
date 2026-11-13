import dataclasses,json,tempfile,unittest
from pathlib import Path
import numpy as np
from efficient_llm.dot_pool.rl import write,ARMS
from scripts.dot_rl_report import readings,report

class Report(unittest.TestCase):
    def test_threshold_noise(self):
        v={'L':[.3,.4],'P-rand':[.01,.02],'J-RL':[.1,.3],'WS-RL':[.5,.6],'FR-adaptive':[.1,.11],'FR-fixed':[.01,.02]}
        r=readings(v)
        self.assertEqual(next(x for x in r if x['reading']=='RL-R2')['label'],'warm-up helps')
        self.assertTrue(next(x for x in r if x['reading']=='RL-R1' and x['arm']=='J-RL')['within_seed_noise'])
    def test_complete_synthetic_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run';root.mkdir();write(root.parent/'data/manifest.json',dict(counts={}))
            runs=[]
            for arm in ARMS+['FR-adaptive','FR-fixed']:
                for seed in [0,1]:
                    run=root/f'{arm}-s{seed}';run.mkdir();cfg=dict(arm=arm,seed=seed,output=str(run));runs.append(cfg);write(run/'config.json',cfg)
                    fr=arm.startswith('FR');write(run/'complete.json',dict(epochs=2000,updates=3000,selected_update=100,T12=dict(passed=True)))
                    if fr:(run/'metrics.jsonl').write_text('')
                    for tag in (['selected','last'] if fr else ['final']):
                        modes=['loop'] if arm=='L' else ['random','loop'] if arm=='P-rand' else ['greedy','sampled']
                        values={};arrays={}
                        for mode in modes:
                            values[mode]={}
                            for split in ['id_test','ood_test','rl_dev','train','ood_rl_atoms','ood_test_atoms']:
                                correct=np.array([[True],[False]]);arrays[mode+'_'+split+'_correct']=correct
                                values[mode][split]=dict(accuracy=.5,oracle=.5,draws=1,items=2,reward=-1.)
                        write(run/(tag+'_eval.json'),dict(modes=values));np.savez(run/(tag+'_paths.npz'),**arrays)
            write(root/'run_index.json',dict(runs=runs,flags=[]));write(root/'budget.json',dict(epochs=2000,switch_epoch=500));write(root/'tests_summary.json',dict(passed=True))
            self.assertTrue(report(root));self.assertTrue(json.loads((root/'report_complete.json').read_text())['reported'])
            self.assertEqual(len(json.loads((root/'report/readings.json').read_text())['values']),6)
            self.assertEqual(len((root/'report/summary.csv').read_text().splitlines()),31)
