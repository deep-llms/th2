import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from efficient_llm.a2.protocol import DATASETS,PARENTS,LRS,Run,lr_for_update,matrix,FIXED_SETS
from scripts.a2_remaining import (SCIENCE_KEYS,read_locked,load_history,remaining_runs,
    branch_sequence,freeze_sets,validate_completed,verify_current_core)

ROOT=Path(__file__).resolve().parents[2]


def provenance():
    return dict({k:'fixed' for k in SCIENCE_KEYS},code_commit='a'*40,code_files={'model':'fixed'})


def row_for(run):
    return dict(run.record(),provenance=provenance(),status='complete',diverged=False,rerun_of=None,
        coverage=dict(examples=run.steps*64,microbatches=run.steps*8,world_size=8,exact_order=True),
        roundtrip_difference=0,lr_updates=[lr_for_update(i,run.steps,run.lr) for i in range(1,run.steps+1)],
        val_teacher=2.,val_post=2.5,delta_post=.5,val_tune_post=4.-run.lr*1000)


def completed_s1():
    chosen={(d,p):3e-4 for d in DATASETS for p in ('P300','P1T')}
    return [row_for(r) for r in matrix(chosen) if r.part=='0A' and r.set=='S1'
            and r.budget!='0.8B' and not r.suffix]


def survey_fixture():
    surveys={d+'/'+p:dict(dataset=d,parent=p,provenance=provenance(),T1_precision_passed=True,
        parent_sanity={'passed':True},group_bi=list(range(12)),diagnostic_flags={'bad_loss':True})
        for d in DATASETS for p in PARENTS}
    identity={'tests':{d:dict(T2=True,T3={p:{'passed':True} for p in ('S72','S188')}) for d in DATASETS}}
    assets={key:{'code_files':{'modeling_opensci.py':'same','configuration_opensci.py':'same'}} for key in surveys}
    return surveys,identity,assets


class RemainingTests(unittest.TestCase):
    def test_exact_28_pending_and_no_grid_or_s1_repeat(self):
        rows=completed_s1();self.assertEqual(len(rows),30)
        plan=remaining_runs(rows);pending=plan['pending'];self.assertEqual(len(pending),28)
        self.assertEqual(len(set(x['run_id'] for x in pending)),28)
        self.assertFalse(set(plan['reused_ids']) & {r['run_id'] for r in pending})
        self.assertEqual(sum(x['budget']=='0.03B' for x in pending),0)
        self.assertEqual(sum(x['budget']=='0.05B' and not x['suffix'] for x in pending),18)
        self.assertEqual(sum(x['budget']=='0.2B' for x in pending),6)
        self.assertEqual(sum(x['budget']=='0.8B' for x in pending),2)
        self.assertEqual(sum(bool(x['suffix']) for x in pending),2)

    def test_incomplete_duplicate_wrong_lr_history_refused(self):
        rows=completed_s1()
        with self.assertRaises(ValueError):remaining_runs(rows[1:])
        with self.assertRaises(ValueError):remaining_runs(rows+[rows[0]])
        wrong=copy.deepcopy(rows)
        i=next(i for i,r in enumerate(wrong) if r['budget']=='0.05B')
        r=wrong[i];wrong[i]=row_for(Run(r['dataset'],r['arm'],'S1','0.05B',1e-4))
        with self.assertRaises(ValueError):remaining_runs(wrong)

    def test_schedule_order_parent_provenance_and_failure_checked(self):
        row=completed_s1()[0];pre=dict(provenance=provenance(),parent_sanity={'passed':True},parent={'final':{'loss':2.}})
        validate_completed(row,provenance(),pre)
        # Original baseline may predate the coordinator-only commit; preserve it.
        old=copy.deepcopy(pre);old['provenance']['code_commit']='b'*40
        validate_completed(row,provenance(),old)
        old['provenance']['data_sha256']='wrong'
        with self.assertRaises(ValueError):validate_completed(row,provenance(),old)
        for key,value in [('status','failed'),('roundtrip_difference',.01),('diverged',True),
            ('coverage',dict(row['coverage'],world_size=1)),('lr_updates',[0]*row['steps']),
            ('provenance',{}),('val_teacher',2.1),('val_post',float('nan'))]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                validate_completed(dict(row,**{key:value}),provenance(),pre)

    def test_sha_terminal_and_current_science_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            def pin(name,obj):
                p=root/name;p.write_text(json.dumps(obj));return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            row=completed_s1()[0]
            source=dict(provenance=provenance(),results=pin('rows',[row]),
                complete=pin('complete',dict(status='complete',runs=1,scientific_complete=True)),
                supervisor=pin('supervisor',dict(training_status='ok',training_returncode=0)),
                run_complete=pin('run',dict(status='ok')),burn=pin('burn',dict(collective_progress_verified=True)),
                pre={'P300':pin('pre',dict(provenance=provenance(),parent_sanity={'passed':True},parent={'final':{'loss':2.}}))})
            lock={'schema':1,'sources':[source]}
            self.assertEqual(len(load_history(lock,provenance())['rows']),1)
            with self.assertRaises(ValueError):load_history(dict(lock,sources=[source,source]),provenance())
            with self.assertRaises(ValueError):load_history(lock,dict(provenance(),data_sha256='wrong'))
            bad=copy.deepcopy(lock);bad['sources'][0]['burn']=pin('badburn',dict(collective_progress_verified=False))
            with self.assertRaises(ValueError):load_history(bad,provenance())
            (root/'rows').write_text('[]')
            with self.assertRaisesRegex(ValueError,'changed'):load_history(lock,provenance())

    def test_all_six_branch_surveys_required_before_selection(self):
        surveys,identity,assets=survey_fixture()
        annealed={k:v for k,v in surveys.items() if v['parent'] in ('P300','P1T')}
        todo=branch_sequence(annealed,identity);self.assertEqual(len(todo),6)
        self.assertEqual({(s['dataset'],s['parent']) for s in todo},
                         {(d,p) for d in DATASETS for p in ('S72','S188')})
        with self.assertRaisesRegex(ValueError,'Unmeasured'):freeze_sets(annealed,identity,provenance(),assets)
        result=freeze_sets(surveys,identity,provenance(),assets)
        self.assertEqual(result['selection']['dclm']['usable'],list(PARENTS))
        # Diagnostic flags and held-out loss never exclude a valid parent.
        self.assertTrue(result['eligibility']['dclm/S72']['T1'])
        surveys['dclm/S72']['T1_precision_passed']=False
        result=freeze_sets(surveys,identity,provenance(),assets)
        self.assertNotIn('S72',result['selection']['dclm']['usable'])
        identity['tests']['fwe']['T3']['S188']['passed']=False;del surveys['fwe/S188']
        result=freeze_sets(surveys,identity,provenance(),assets)
        self.assertNotIn('S188',result['selection']['fwe']['usable'])
        surveys['nem/P300']['T1_precision_passed']=False
        with self.assertRaises(ValueError):freeze_sets(surveys,identity,provenance(),assets)

    def test_supervised_prerequisite_manifest_has_no_healing_or_repeated_survey(self):
        from scripts.a2_remaining import prerequisite_jobs
        from run_experiments import load_jobs
        surveys,identity,assets=survey_fixture()
        annealed={k:v for k,v in surveys.items() if v['parent'] in ('P300','P1T')}
        sequence=branch_sequence(annealed,identity)
        jobs=prerequisite_jobs(source='/source',inputs='/inputs',port=29599,
            history='/history',reference='/reference',identity='/identity',sequence=sequence)
        self.assertEqual(len(jobs),8)
        self.assertEqual([j['name'] for j in jobs[-2:]],['freeze-sets','shared-numerical'])
        self.assertNotIn('gpus',jobs[-2])
        self.assertTrue(all(j['gpus']==list(range(8)) for j in jobs if j['name']!='freeze-sets'))
        self.assertTrue(all('heal' not in j['argv'] and 'throughput' not in j['argv'] for j in jobs))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'jobs.json';p.write_text(json.dumps({'jobs':jobs}));load_jobs(p)
        with self.assertRaises(ValueError):
            prerequisite_jobs(source='/source',inputs='/inputs',port=1,history='/h',reference='/r',
                identity='/i',sequence=sequence+[sequence[0]])

    def test_cpu_cli_plan_then_frozen_sets(self):
        from unittest.mock import patch
        from scripts import a2_remaining as module
        surveys,identity,assets=survey_fixture()
        annealed={k:v for k,v in surveys.items() if v['parent'] in ('P300','P1T')}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);branch=root/'surveys';branch.mkdir()
            for key,value in surveys.items():
                if key not in annealed:(branch/(key.replace('/','-')+'.json')).write_text(json.dumps(value))
            for name,value in [('history',{}),('reference',provenance()),('identity',identity),('assets',assets)]:
                (root/(name+'.json')).write_text(json.dumps(value))
            args=['remaining','--history',str(root/'history.json'),'--reference',str(root/'reference.json'),
                  '--identity',str(root/'identity.json'),'--output',str(root/'plan.json')]
            def imported(*unused):return dict(rows=completed_s1(),surveys=copy.deepcopy(annealed))
            with patch('sys.argv',args),patch.object(module,'load_history',side_effect=imported):module.main()
            self.assertEqual(len(json.loads((root/'plan.json').read_text())['pending']),28)
            args[args.index('--output')+1]=str(root/'sets.json')
            args+=['--survey-dir',str(branch),'--assets',str(root/'assets.json')]
            with patch('sys.argv',args),patch.object(module,'load_history',side_effect=imported):module.main()
            frozen=json.loads((root/'sets.json').read_text())
            self.assertEqual(frozen['status'],'frozen')
            self.assertEqual(set(frozen['sets']),set(DATASETS))
            self.assertEqual(len(frozen['survey_evidence']),12)

    def test_existing_numerical_code_lock_passes(self):
        verify_current_core(ROOT)

    def test_prerequisite_launch_gate_inputs_and_budget(self):
        from unittest.mock import patch
        from scripts.a2_remaining import prepare_prerequisites
        surveys,identity,assets=survey_fixture()
        annealed={k:v for k,v in surveys.items() if v['parent'] in ('P300','P1T')}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'secondary_map.json').write_text('{}')
            (root/'history.json').write_text('{}')
            with patch('efficient_llm.a2.__main__.provenance',return_value=provenance()), \
                 patch('scripts.a2_remaining.load_history',return_value=dict(rows=completed_s1(),surveys=annealed)):
                result=prepare_prerequisites(ROOT,root,root/'history.json','a'*40,15000)
                self.assertEqual(len(result['sequence']),6)
                self.assertEqual(len(result['plan']['pending']),28)
                self.assertEqual(result['budget']['worst_seconds'],9720)
                required={item['path'] for item in result['asset_plan']['downloads']}
                self.assertTrue(any('/dclm/P300/' in p for p in required))
                self.assertTrue(all(any('/'+ds+'/'+p+'/' in name for name in required)
                    for ds in DATASETS for p in ('S72','S188')))
                # Branches deliberately use pinned main-line model Python; only
                # P1T *weights* are unnecessary for this prerequisite stage.
                self.assertFalse(any('/P1T/model.safetensors' in name for name in required))
                for spent in (-1,float('nan'),60000):
                    with self.subTest(spent=spent),self.assertRaises(ValueError):
                        prepare_prerequisites(ROOT,root,root/'history.json','a'*40,spent)



if __name__=='__main__':unittest.main()
