import unittest
from efficient_llm.a2.protocol import *

class ProtocolTests(unittest.TestCase):
    def test_matrix(self):
        runs=matrix()
        self.assertEqual(len(runs),127)
        self.assertEqual(sum(r.part=='0A' for r in runs),58)
        self.assertEqual(sum(r.part=='0B' for r in runs),69)
        self.assertEqual(len({r.run_id for r in runs}),127)
        self.assertTrue(all(r.teacher=='P300' and r.parent=='S72' for r in runs if r.arm=='X72'))
        self.assertEqual(sum(r.budget=='0.03B' for r in runs),45)
        self.assertFalse(any(r.arm=='X72' and r.budget=='0.05B' for r in runs))

    def test_schedule(self):
        self.assertEqual([steps_for(x) for x in ('0.03B','0.05B','0.2B','0.8B')],[114,190,762,3051])
        n=762; w=warmup_steps(n)
        self.assertEqual(w,16)
        self.assertEqual(lr_for_update(1,n,1),1/16)
        self.assertEqual(lr_for_update(w,n,1),1)
        self.assertEqual(lr_for_update(n,n,1),0)
        self.assertEqual(lr_for_update(n-1,n,1),1/746)
        self.assertEqual(lr_for_update(381,n,1),381/746)

    def test_s2(self):
        scores=[float(i) for i in range(12)]
        for i,v in zip((8,9,10),(-3,-2,-1)): scores[i]=v
        result=choose_s2({p:scores for p in PARENTS},['P300','P1T'])
        self.assertEqual(result['raw'],[8,9,10]); self.assertEqual(result['groups'],[1,8,9])
        self.assertEqual(choose_s2({p:[1.]*12 for p in PARENTS},PARENTS)['groups'],[1,2,3])
        self.assertEqual(layers_for((8,9,10)),list(range(16,22)))

    def test_lr_selection(self):
        rows=[dict(lr=lr,val_tune_post=v,status='complete',diverged=False) for lr,v in zip(LRS,(2.001,2.,1.9999))]
        self.assertEqual(choose_lr(rows),1e-4)
        rows[0]['val_tune_post']=2.0009
        self.assertEqual(choose_lr(rows),3e-5)
        for r in rows:r['diverged']=True
        self.assertIsNone(choose_lr(rows))
        with self.assertRaises(ValueError):choose_lr(rows[:2])

    def decide(self,main=.008,pre=.06,short=None):
        m={d:[main,main] for d in DATASETS} if isinstance(main,(int,float)) else main
        p={d:[pre,pre] for d in DATASETS} if isinstance(pre,(int,float)) else pre
        return length_decision(m,p,[.01]*12 if short is None else short)

    def test_decision_boundaries(self):
        self.assertEqual(self.decide()['label'],'GO')
        self.assertEqual(self.decide(short=[.01]*10+[-.01]*2)['label'],'GO')
        self.assertEqual(self.decide(short=[.01]*9+[-.01]*3)['label'],'GREY')
        self.assertEqual(self.decide(.002,.05)['label'],'REFRAME')
        self.assertEqual(self.decide(.003,.05)['label'],'GREY')
        self.assertEqual(self.decide(.002,.02)['label'],'STOP')
        self.assertEqual(self.decide(.002,.04)['label'],'GREY')
        pre={'dclm':[.2,.2],'fwe':[-.01,-.01],'nem':[.01,.01]}
        self.assertEqual(self.decide(.002,pre)['label'],'STOP')
        mixed={'dclm':[.003,.003],'fwe':[-.003,-.003],'nem':[.001,.001]}
        out=self.decide(mixed)
        self.assertTrue(out['clauses']['REFRAME'] and out['clauses']['STOP'])
        self.assertEqual(out['label'],'REFRAME')
        missing={d:[.01,.01] for d in DATASETS}; missing['nem'][0]=None
        self.assertEqual(self.decide(missing)['label'],'INCOMPLETE')
        self.assertEqual(self.decide(-.008,short=[-.01]*12)['label'],'GO')

    def test_lr_state(self):
        self.assertEqual(lr_state_decision([-.01]*5+[None])['label'],'prune before you anneal')
        self.assertEqual(lr_state_decision([.01]*5+[None])['label'],'anneal first')
        self.assertEqual(lr_state_decision([.02]*4+[None,None])['label'],'INCOMPLETE')
        self.assertEqual(lr_state_decision([.1]*3+[-.1]*3)['label'],'no consistent effect meeting the threshold')
        self.assertTrue(lr_state_decision([0.]*6,[3e-4])['grid_edge'])
