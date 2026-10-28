import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from efficient_llm.a2.protocol import shared_check_sets,layers_for
from scripts.a2_remaining import freeze_sets,remaining_runs
from scripts.a2_heal_stage import reviewed_prerequisites,assemble_gate,prepare_heals,heal_jobs
from tests.a2.test_remaining import survey_fixture,provenance,completed_s1,ROOT


class HealStageTests(unittest.TestCase):
    def fixture(self,root):
        surveys,identity,assets=survey_fixture();current=provenance()
        current['code_commit']='b'*40
        selection=freeze_sets(surveys,identity,current,assets)
        def pin(name,value):
            p=root/name;p.write_text(json.dumps(value));return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        history=pin('history',{})
        identity_sha=hashlib.sha256(json.dumps(identity).encode()).hexdigest()
        frozen=dict(selection,status='frozen',survey_evidence=surveys,history_sha256=history['sha256'],identity_sha256=identity_sha)
        sets=pin('sets',frozen)
        check=dict(passed=True,scope='shared_mechanism',representative='dclm/P300',provenance=provenance(),sets_sha256=sets['sha256'],
            removal_patterns=[layers_for(g) for g in shared_check_sets(selection['sets'],assets).values()]+[[0],[11],[23]])
        lock=dict(schema=1,identity_sha256=identity_sha,sets=sets,check=pin('check',check),
            complete=pin('complete',{'status':'ok'}),supervisor=pin('supervisor',{'training_status':'ok','training_returncode':0}),
            burn=pin('burn',{'collective_progress_verified':True}))
        gate=dict(status='passed',tests={k:{'passed':True} for k in ('T4','T5','T6','T7','T8','T10_formula','T11','T12','T13')})
        return current,identity,assets,lock,history,gate

    def test_pinned_full_check_reuse_retains_original_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            current,identity,assets,lock,history,oldgate=self.fixture(Path(d))
            result=reviewed_prerequisites(lock,current,assets,identity,history['sha256'])
            check=result['numerical']
            self.assertEqual(check['provenance'],current)
            self.assertEqual(check['historical_receipt']['provenance'],provenance())
            gate=assemble_gate(oldgate,result,current,identity,lock['identity_sha256'],remaining_runs(completed_s1()))
            self.assertEqual(len(gate['parents']),6);self.assertEqual(gate['launch_scope'],'all')
            bad=dict(current,code_files={'model':'changed'})
            with self.assertRaises(ValueError):reviewed_prerequisites(lock,bad,assets,identity,history['sha256'])
            with self.assertRaises(ValueError):reviewed_prerequisites(lock,current,assets,identity,'wrong')
            with self.assertRaises(ValueError):assemble_gate(oldgate,result,current,identity,'wrong',{})
            Path(lock['check']['path']).write_text('{}')
            with self.assertRaises(ValueError):reviewed_prerequisites(lock,current,assets,identity,history['sha256'])

    def test_gate_requires_all_annealed_parents_and_recipe_tests(self):
        with tempfile.TemporaryDirectory() as d:
            current,identity,assets,lock,history,oldgate=self.fixture(Path(d))
            reviewed=reviewed_prerequisites(lock,current,assets,identity,history['sha256'])
            reviewed['records']['sets']['eligibility']['nem/P1T']['T1']=False
            with self.assertRaises(ValueError):assemble_gate(oldgate,reviewed,current,identity,lock['identity_sha256'],{})
            reviewed['records']['sets']['eligibility']['nem/P1T']['T1']=True
            oldgate['tests']['T8']['passed']=False
            with self.assertRaises(ValueError):assemble_gate(oldgate,reviewed,current,identity,lock['identity_sha256'],{})

    def test_pending_stage_manifest_all_gpus_and_completion_contract(self):
        from run_experiments import load_jobs
        jobs=heal_jobs('/i','/h','/p','/old','/grid',29590,18000,46200)
        self.assertEqual(len(jobs),1);self.assertEqual(jobs[0]['gpus'],list(range(8)))
        self.assertNotIn('efficient_llm.a2.queue',jobs[0]['argv'])
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'jobs.json';p.write_text(json.dumps({'jobs':jobs}));load_jobs(p)

    def test_whole_pending_budget_and_original_t8_reuse(self):
        from scripts import a2_heal_stage as m
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);current,identity,assets,lock,history,oldgate=self.fixture(root)
            (root/'assets.json').write_text(json.dumps(assets));(root/'prereq.json').write_text(json.dumps(lock))
            reviewed=reviewed_prerequisites(lock,current,assets,identity,history['sha256'])
            real_read=m.read
            def reader(path):
                if str(path).endswith('grid/gate.json'):return oldgate
                return real_read(path)
            with patch.object(m,'current_provenance',return_value=current), \
                 patch.object(m,'load_history',return_value={'rows':completed_s1()}), \
                 patch.object(m,'reviewed_prerequisites',return_value=reviewed), \
                 patch.object(m,'assemble_gate',return_value={}), \
                 patch.object(m,'verify_reuse',return_value={'step_seconds':1.2}) as reuse, \
                 patch.object(m,'verify_grid',return_value={'status':'passed'}),patch.object(m,'read',side_effect=reader):
                state=prepare_heals(ROOT,root,root/'history',root/'prereq.json','/prior','/grid','b'*40,18000)
                self.assertEqual(len(state['plan']['pending']),28);reuse.assert_called_once()
                self.assertLess(state['budget']['worst_seconds']+18000,64800)
                with self.assertRaises(ValueError):prepare_heals(ROOT,root,root/'history',root/'prereq.json','/prior','/grid','b'*40,40000)

    def test_cli_scores_six_baselines_before_exact_executor(self):
        from types import SimpleNamespace
        from scripts import a2_heal_stage as m
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);current,identity,assets,lock,history,oldgate=self.fixture(root)
            reviewed=reviewed_prerequisites(lock,current,assets,identity,history['sha256'])
            state=dict(budget={},plan=remaining_runs(completed_s1()),gate={},prerequisites_lock=lock,
                reviewed=reviewed,reference=current,history={'rows':completed_s1()},estimates={})
            (root/'assets.json').write_text(json.dumps(assets));calls=[]
            args=['stage','--inputs',str(root),'--history','/h','--prerequisites','/p','--prior','/old',
                '--grid','/g','--port','29590','--spent-seconds','18000','--output',str(root/'out')]
            def child(argv,**kw):
                self.assertIn('pre',argv);self.assertNotIn('heal',argv)
                ds=argv[argv.index('--dataset')+1];parent=argv[argv.index('--parent')+1];calls.append((ds,parent))
                base=dict(provenance=current,sets_sha256=reviewed['sets_sha256'],parent_sanity={'passed':True},
                    parent={'final':{'loss':2.}},sets=reviewed['records']['sets']['sets'][ds])
                Path(argv[argv.index('--output')+1]).write_text(json.dumps(base));return SimpleNamespace(returncode=0)
            with patch('sys.argv',args),patch.object(m,'prepare_heals',return_value=state), \
                 patch.object(m,'survey_diagnostics',return_value={}),patch.object(m.subprocess,'run',side_effect=child), \
                 patch.object(m,'execute_remaining') as execute:
                m.main()
            self.assertEqual(calls,[(ds,p) for ds in ('dclm','fwe','nem') for p in ('P300','P1T')])
            self.assertEqual(len(execute.call_args.kwargs['pre']),6)
            self.assertEqual(execute.call_args.kwargs['history'],state['history'])

    def test_diagnostics_never_exclude_a_parent(self):
        from scripts.a2_heal_stage import survey_diagnostics
        evidence={d+'/'+p:{'final':{'loss':3.}} for d in ('dclm','fwe','nem') for p in ('P300','S72','S188')}
        before=copy.deepcopy(evidence);result=survey_diagnostics({'survey_evidence':evidence})
        self.assertTrue(all(r['annealing_gain_below_002'] and r['stable_length_nonpositive'] for r in result.values()))
        self.assertTrue(all(r['changes_eligibility'] is False for r in result.values()))
        self.assertEqual(evidence,before)


if __name__=='__main__':unittest.main()
