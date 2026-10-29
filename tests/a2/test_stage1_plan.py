import copy
import unittest
from efficient_llm.a2_stage1.protocol import ARMS, Run, SPEC_SHA
from efficient_llm.a2.protocol import lr_for_update
from eval.a2_stage1_plan import select_stage1_screen

class Stage1SelectionTest(unittest.TestCase):
    def fixture(self):
        prov=dict(spec_sha256=SPEC_SHA,source='fixed')
        rows=[]
        for arm in ARMS:
            run=Run(arm,'3.2B',.001)
            rows.append(dict(run.record(),provenance=prov,status='complete',diverged=False,
                coverage=dict(examples=run.steps*64,microbatches=run.steps*2,world_size=8,exact_order=True),
                lr_updates=[lr_for_update(i,run.steps,run.lr) for i in range(1,run.steps+1)],roundtrip_difference=0.))
        return rows,prov

    def test_exact_six_models_and_cross_teacher(self):
        rows,prov=self.fixture();selected=select_stage1_screen(rows,.001,prov)
        self.assertEqual(set(selected),{'P300-parent','P1T-parent','S72-parent','P300-student','P1T-student','X72-student'})
        self.assertEqual(selected['X72-student']['parent'],'S72')
        self.assertEqual(selected['X72-student']['teacher'],'P300')

    def test_missing_duplicate_and_wrong_lr_rejected(self):
        rows,prov=self.fixture()
        for bad,lr in [(rows[:-1],.001),(rows+[rows[0]],.001),(rows,.0003)]:
            with self.assertRaises(ValueError):select_stage1_screen(bad,lr,prov)

    def test_tampered_source_teacher_schedule_and_divergence_rejected(self):
        rows,prov=self.fixture()
        for field in ('provenance','teacher','lr_updates','diverged'):
            bad=copy.deepcopy(rows)
            if field=='provenance':bad[-1][field]['source']='other'
            elif field=='teacher':bad[-1][field]='S72'
            elif field=='lr_updates':bad[-1][field][10]*=.5
            else:bad[-1][field]=True
            with self.assertRaises(ValueError):select_stage1_screen(bad,.001,prov)

if __name__=='__main__':unittest.main()
