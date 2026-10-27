import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from efficient_llm.a2.protocol import matrix,DATASETS,ARMS,PARENTS
from efficient_llm.a2.results import report,selected_lrs,estimate_workload
from efficient_llm.a2.queue import execute_phase

class ResultTests(unittest.TestCase):
    def rows(self):
        rows=[]
        for run in matrix({(d,a):1e-4 for d in DATASETS for a in ARMS}):
            delta={'P300':.1,'P1T':.112,'S72':.1,'S188':.11,'X72':.08}[run.arm]
            rows.append(dict(run.record(),status='complete',diverged=False,rerun_of=None,
                delta_pre=.2,delta_post=delta,val_post=2+delta,
                val_tune_post={3e-5:2.3,1e-4:2.1,3e-4:2.2}[run.lr]))
        return rows

    def test_reports_retry_and_no_final_selection(self):
        rows=self.rows(); chosen=selected_lrs(rows)
        self.assertTrue(all(lr==1e-4 for lr in chosen.values()))
        first=next(r for r in rows if r['budget']=='0.03B');first['val_post']=-100
        self.assertEqual(chosen,selected_lrs(rows))
        result=report(rows,{})
        self.assertEqual(result['phase0A']['label'],'GO')
        self.assertEqual(result['phase0B_length']['label'],'GO')
        self.assertEqual(result['phase0B_lr']['label'],'prune before you anneal')
        row=next(r for r in rows if (r['dataset'],r['arm'],r['set'],r['budget'])==('nem','P1T','S1','0.2B'))
        row['diverged']=True
        self.assertEqual(report(rows,{})['phase0A']['label'],'INCOMPLETE')
        retry=dict(row,diverged=False,run_id=row['run_id']+'-rerun',lr=row['lr']/2,rerun_of=row['run_id'])
        rows.append(retry)
        self.assertEqual(report(rows,{})['phase0A']['label'],'GO')
        retry['lr']=row['lr']
        with self.assertRaises(ValueError):report(rows,{})

    def test_whole_estimate(self):
        cost=dict(step_seconds=.5,evaluation_seconds=600,measurement_seconds=120,
                  load_save_seconds=300,retry_allowance_seconds=40)
        result=estimate_workload({'0A':cost,'0B':cost})
        self.assertEqual(result['0A']['runs'],58)
        expected=(sum(r.steps for r in matrix() if r.part=='0A')*.5+1060)/3600
        self.assertEqual(result['0A']['node_hours'],expected)

    def test_secondary_scores_same_effective_runs_without_selecting(self):
        rows=self.rows()
        parents={f'{d}/{p}':dict(val=2.,secondary={'loss':3.}) for d in DATASETS for p in PARENTS}
        for row in rows:
            # Opposite secondary ranking must not change primary selection/labels.
            delta=.1 if row['arm']=='P300' else .08
            row.update(val_secondary_post=3+delta,delta_secondary_pre=.3,delta_secondary_post=delta)
        result=report(rows,parents)
        self.assertEqual(result['phase0A']['label'],'GO')
        self.assertAlmostEqual(result['secondary']['phase0A']['E'],-.02)
        self.assertNotIn('label',result['secondary']['phase0A'])
        self.assertTrue(all(v==1e-4 for v in result['chosen_lrs'].values()))
        self.assertEqual(result['secondary']['parent_length_gap']['nem'],0.)
        for row in rows:
            if row['dataset']=='nem': row.pop('val_secondary_post');row.pop('delta_secondary_post')
        self.assertIsNone(report(rows,parents)['secondary']['phase0A']['E'])

    def test_queue_grid_retry_and_freshness(self):
        pre={f'{d}/{p}':{'parent':{'final':{'loss':2.}},
                        'sets':{s:{'final':{'loss':2.1},'tuning':{'loss':2.1}} for s in ('S1','S2','S3','S4')}}
             for d in DATASETS for p in PARENTS}
        seen=[]
        def child(argv,**kwargs):
            record=json.loads(Path(argv[argv.index('--run')+1]).read_text())
            folder=Path(argv[argv.index('--output')+1]);folder.mkdir()
            fail=(record['dataset']=='dclm' and record['arm']=='P300' and record['set']=='S1'
                  and record['budget']=='0.05B' and not record['suffix'])
            result=dict(record,status='complete',diverged=fail,val_tune_post={3e-5:2.3,1e-4:2.,3e-4:2.2,5e-5:2.}[record['lr']])
            (folder/'result.json').write_text(json.dumps(result));seen.append(result)
            return subprocess.CompletedProcess(argv,0)
        with tempfile.TemporaryDirectory() as d,patch('efficient_llm.a2.queue.subprocess.run',side_effect=child):
            root=Path(d)/'phase'
            summary=execute_phase(phase='0A',root=root,launcher=['python','-m','efficient_llm.a2'],common=[],
                pre=pre,estimated={b:10 for b in ('0.03B','0.05B','0.2B','0.8B')},spent_seconds=10)
            self.assertEqual(summary['status'],'complete');self.assertEqual(len(seen),59)
            self.assertEqual(sum(bool(r['rerun_of']) for r in seen),1)
            self.assertTrue(all(r['lr']==1e-4 for r in seen if r['budget']!='0.03B' and not r['rerun_of']))
            with self.assertRaises(FileExistsError):execute_phase(phase='0A',root=root,launcher=['python'],common=[],pre=pre,estimated={},spent_seconds=0)

    def test_queue_rejects_wrong_receipt_and_silent_failure(self):
        pre={'dclm/P300':dict(parent={'final':{'loss':2.}},sets={'S1':{'final':{'loss':2.1}}},
                              provenance={'source':'expected'})}
        for corruption in ('teacher','provenance','status'):
            def child(argv,**kwargs):
                record=json.loads(Path(argv[argv.index('--run')+1]).read_text())
                folder=Path(argv[argv.index('--output')+1]);folder.mkdir()
                result=dict(record,status='complete',diverged=False,provenance={'source':'expected'})
                result[corruption]={'teacher':'S72','provenance':{},'status':'failed'}[corruption]
                (folder/'result.json').write_text(json.dumps(result))
                return subprocess.CompletedProcess(argv,0)
            with tempfile.TemporaryDirectory() as d,patch('efficient_llm.a2.queue.subprocess.run',side_effect=child):
                with self.assertRaises((ValueError,RuntimeError)):
                    execute_phase(phase='0A',root=Path(d)/'phase',launcher=['python'],common=[],pre=pre,
                                  estimated={'0.03B':10},spent_seconds=0)

    def test_branch_identity_failure_does_not_invalidate_sibling(self):
        from scripts.a2_asset_manifest import branch_identity
        main={'hidden_size':2048,'_name_or_path':'/job/samples-1000B/hf/iter_0240000'}
        hashes={p:p for p in PARENTS}
        wrong=dict(main,_name_or_path='/job/samples-300B/hf/iter_0072000')
        good=dict(main,_name_or_path='/job/samples-1000B/hf/iter_0188000')
        self.assertFalse(branch_identity(wrong,main,parent='S72',hashes=hashes,annealed72000_hash='different')['passed'])
        self.assertTrue(branch_identity(good,main,parent='S188',hashes=hashes,annealed72000_hash='different')['passed'])
