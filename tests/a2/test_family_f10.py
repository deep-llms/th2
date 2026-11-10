import unittest
import numpy as np
from eval.a2_family_f10 import holm, test_statistic
from eval.a2_family_f10_tasks import answer_metrics, likelihood_metrics


class F10Tests(unittest.TestCase):
    def test_holm_unsorted_and_monotonic(self):
        np.testing.assert_allclose(holm([.04, .001, .02, .5]), [.08, .004, .06, .5])
        np.testing.assert_allclose(holm([.03, .031, .032]), [.09, .09, .09])

    def test_degenerate_and_nonfinite(self):
        self.assertEqual(test_statistic(0., np.zeros(10))['p_normal'], 1.)
        self.assertEqual(test_statistic(1., np.ones(10))['p_normal'], 0.)
        with self.assertRaises(ValueError): holm([float('nan')])

    def test_sign_symmetric_p_values(self):
        x=np.array([.2,.4,.5,.7,.9])
        a=test_statistic(.5,x); b=test_statistic(-.5,-x)
        self.assertEqual(a['p_normal'],b['p_normal'])
        self.assertEqual(a['p_bootstrap_two_sided'],b['p_bootstrap_two_sided'])
        self.assertAlmostEqual(a['bootstrap_SE_pp'],np.std(x,ddof=1))

    def test_alias_normalization_and_f1(self):
        self.assertEqual(answer_metrics('The New_York!', ['NYC','New York']),dict(exact_match=1.,token_f1=1.))
        self.assertEqual(answer_metrics('red dog',['a red cat'])['token_f1'],.5)
        self.assertEqual(answer_metrics('',['Paris'])['token_f1'],0.)

    def test_conf_is_not_choice_probability(self):
        rows=[dict(doc_id='0',requests=[0,1],character_lengths=[2,10],gold=1)]
        r=likelihood_metrics(rows,[[-1.,True],[-2.,False]],'mmlu')[0]
        self.assertEqual(r['conf'],-.2)
        self.assertEqual(r['correctness'],1.)
        self.assertEqual(r['acc'],0.)
        self.assertGreater(r['p_correct'],.5)

    def test_joint_cohort_does_not_require_all_eight(self):
        from eval.a2_family_f10_report import cohort_bootstrap
        values=np.tile([0,0,0,-2,0,0,0,-4],(6,1)).astype(float)
        correct=np.ones_like(values);correct[:3,4:]=0;correct[3:,:4]=0
        p,b,n=cohort_bootstrap(values,correct,np.random.default_rng(0),resamples=100)
        self.assertEqual(n,[3,3]);self.assertEqual(p,3.)
        np.testing.assert_allclose(b,3.)

    def test_joint_cohort_sign_and_fixed_selection(self):
        from eval.a2_family_f10_report import cohort_bootstrap
        values=np.array([[0,0,0,-2],[100,0,100,0],[0,0,0,-4.]])
        correct=np.ones_like(values);correct[1,0]=0
        p,b,n=cohort_bootstrap(values,correct,np.random.default_rng(0),resamples=100)
        self.assertEqual(n,[2]);self.assertEqual(p,3.)
        self.assertTrue(((b>=2)&(b<=4)).all())

    def test_native_manifest_and_registered_order(self):
        import tempfile,json
        from pathlib import Path
        from scripts.launch_a2_family_f10 import job_manifest
        from run_experiments import load_jobs
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'jobs.json';p.write_text(json.dumps(job_manifest('/x/config.json')))
            jobs=load_jobs(p);self.assertEqual(len(jobs),1)
        from scripts.a2_family_f10_run import Queue
        q=Queue.__new__(Queue);q.cfg={'oneshot_entries':[{'model_id':'x'}],'groups':[[{'model_id':'p','arm':'P300','dataset':'dclm'}],[],[],[],[]]}
        # Actual queue code has no trainer entrypoint or optimizer operation.
        import inspect
        code=inspect.getsource(Queue.run)
        positions=[code.index(x) for x in ["self.progress('FT14')","self.progress('FT15')","self.progress('F10b_nine_task_oneshots')","self.progress('F10c_group_'","self.progress('analysis')","self.progress('backup')"]]
        self.assertEqual(positions,sorted(positions))

    def test_new_raw_metric_replay_rejects_tampering(self):
        from eval.a2_family_f10_report import replay_new
        from eval.a2_request_bridge import digest
        rows=[dict(doc_id='0',requests=[0,1],character_lengths=[2,10],gold=1)]
        prep={'tasks':{'mmlu':rows},'request_sha256':'requests'}
        scores=dict(scores=[[-1,True],[-2,False]],prepared_sha256=digest(prep),request_sha256='requests')
        vals={'tasks':{'mmlu':likelihood_metrics(rows,scores['scores'],'mmlu')}}
        self.assertTrue(replay_new(prep,scores,vals))
        vals['tasks']['mmlu'][0]['conf']+=.01
        with self.assertRaises(ValueError):replay_new(prep,scores,vals)

    def test_joint_bootstrap_matches_old_f9(self):
        from eval.a2_family_f10_report import sample
        from eval.a2_family_f9_v110 import sampler
        from eval.downstream_suite import COUNTS
        from unittest.mock import patch
        small={t:5 for t in COUNTS}
        records={m:{t:dict(identity=t,values=[i+j for i in range(5)]) for t in COUNTS} for j,m in enumerate(['a','b'])}
        with patch('eval.a2_family_f9_v110.COUNTS',small):
            old=sampler(records)(['a','b'])[1]
        new=sample(records)
        for t in old:
            np.testing.assert_array_equal(old[t][0],new[t][1]);np.testing.assert_array_equal(old[t][1],new[t][2])

    def test_model_selection_exact_registered_counts(self):
        import json
        from eval.a2_family_f9_v110 import select
        from eval.a2_family_f10 import select_models
        from pathlib import Path
        entries=sum(select(json.loads(Path('resources/a2_family_f9_plan_20261110.json').read_text())['models'])[0],[])
        ones,groups=select_models(entries,json.loads(Path('resources/a2_family_f10_plan_20261111.json').read_text()))
        self.assertEqual([len(g) for g in groups],[8,8,8,6,12])
        self.assertEqual(len(ones),11)
        self.assertEqual(len({e['model_id'] for e in sum(groups,[])}),42)
        for e in ones:
            self.assertEqual(e['removed_layers'],list(range(24,33)) if e['family']=='smollm3' else list(range(16,22)))
        self.assertFalse(any(e['arm']=='X72' for e in sum(groups,[])))

    def test_backup_is_verified_and_stays_on_node(self):
        import tempfile,json
        from pathlib import Path
        from scripts.a2_family_f10_backup import backup
        from eval.a2_family_f10 import write
        from scripts.a2_quant_io import lock
        from efficient_llm.diagnostic_data import sha256
        with tempfile.TemporaryDirectory() as d:
            outer=Path(d);root=outer/'result';root.mkdir();src=outer/'source';src.mkdir();(src/'code.py').write_text('# fixture\n')
            cfg={'source_manifest':write(outer/'source-manifest.json',dict(files={'code.py':sha256(src/'code.py')}))}
            for k in ('prepared','input_manifest','f10a','preparation_ready','cpu_proof'):
                cfg[k]=write(outer/(k+'.json'),{})
            ex=write(outer/'examples.json',{})
            cfg['FT16']=write(outer/'FT16.json',dict(fewshot_examples=ex));cfg['f10a_root']=str(outer/'a');(outer/'a').mkdir()
            for n in ('tests.json','complete.json'):write(outer/'a'/n,{})
            (outer/'a/tests.csv').write_text('fixture\n')
            (outer/'family_f9_scientific.tar.gz').write_bytes(b'old-fixture')
            cfg['f9_manifest']=write(outer/'f9_verified.json',dict(archive_sha256=sha256(outer/'family_f9_scientific.tar.gz')))
            write(outer/'config.json',cfg);write(root/'scores.json',dict(values=[1,2,3]))
            proof=backup(root,cfg);v=json.loads(Path(proof['path']).read_text())
            self.assertTrue(v['verified']);self.assertTrue(v['raw_stays_on_node'])
            self.assertFalse(list(root.rglob('*.part*')))
            self.assertIn('scores.json',v['files'])

    def test_launcher_serializes_offline_pane_and_freezes_proof(self):
        import tempfile,json,socket,subprocess,shlex
        from pathlib import Path
        from unittest.mock import patch
        from scripts.launch_a2_family_f10 import launch
        from eval.a2_family_f10 import SPEC_SHA,write
        from efficient_llm.diagnostic_data import sha256
        from scripts.a2_quant_io import lock
        with tempfile.TemporaryDirectory(dir='temp',prefix='f10_launcher_fixture_') as d:
            base=Path(d).resolve();prep=base/'prep';prep.mkdir();output=base/'output'
            proof=base/'proof.json';write(proof,dict(launch_ready=True,files={'scripts/launch_a2_family_f10.py':sha256('scripts/launch_a2_family_f10.py')}))
            cfg={'host':socket.gethostname(),'benchmarks':{},'fresh_model_hashes':[str(i) for i in range(37)],'f9_root':'/fixture/old'}
            for k in ('prepared','input_manifest','f9_manifest','f10a'):cfg[k]=write(base/(k+'.json'),{})
            cfg['f9_config']=write(base/'f9_config.json',dict(locked_inputs=[],opensci_cx_root='/fixture/data'))
            cfg['FT16']=write(base/'FT16.json',dict(counts={'mmlu':14042,'triviaqa':17944},recorded_before_scoring=True,input_hashes={}))
            c=write(prep/'config.json',cfg);write(prep/'ready.json',dict(cpu_ready=True,new_models=42,oneshots=11,model_hashes=37,config=c))
            def fake_run(args,**kwargs):return subprocess.CompletedProcess(args,1 if 'has-session' in args else 0)
            with patch('scripts.launch_a2_family_f10.predecessor',return_value={'host':cfg['host'],'gpus':[]}),patch('efficient_llm.a2.data.PackedRows'),patch('scripts.launch_a2_family_f10.subprocess.run',side_effect=fake_run) as run:
                launch(str(prep),str(output),'fixture_session','b'*40,str(proof))
            saved=json.loads((output/'config.json').read_text())
            self.assertIn(str(output/'source'),saved['cpu_proof']['path'])
            calls=[c for c in run.call_args_list if c.args[0][:2]==['tmux','new-session']]
            shell=calls[0].args[0][-1];tokens=shlex.split(shell)
            self.assertEqual(tokens[:2],['exec','env'])
            self.assertIn('HF_HUB_OFFLINE=1',tokens);self.assertIn('PYTHONPATH='+str(output/'source'),tokens)
            self.assertEqual(saved['spec_sha256'],SPEC_SHA)

    def test_readiness_wait_does_not_relaunch_or_accept_dead_preparer(self):
        import tempfile,subprocess
        from pathlib import Path
        from unittest.mock import patch
        from scripts.launch_a2_family_f10 import wait_for_preparation
        with tempfile.TemporaryDirectory() as d:
            with patch('scripts.launch_a2_family_f10.subprocess.run',return_value=subprocess.CompletedProcess([],0,stdout='1\n')):
                with self.assertRaises(RuntimeError):wait_for_preparation(d,'dead')
            (Path(d)/'ready.json').write_text('{}')
            with patch('scripts.launch_a2_family_f10.subprocess.run') as run:
                wait_for_preparation(d,'already-ready');run.assert_not_called()
