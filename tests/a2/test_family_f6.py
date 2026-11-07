import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from efficient_llm.a2_family_f6.protocol import descriptor,initial_runs,select_extension,reading,joint_reading
from efficient_llm.a2_family_f6.readout import adapted,normalized,exposure,analyze
from efficient_llm.a2_family_f6.common import read,write
from scripts.a2_family_v14_gate import sha,verify_completion
from scripts.a2_family_report import pair_scores


class Registration(unittest.TestCase):
    def test_exact_matrix_and_fresh_recipe(self):
        runs=initial_runs()
        self.assertEqual(len(runs),5)
        self.assertEqual(runs[0]['run_id'],'ST1-dclm-S72-S1-0.2B-lr1e-3')
        self.assertEqual(sum(r['steps'] for r in runs),762+4*3051)
        for r in runs:
            self.assertEqual(r['removed'],list(range(16,22)))
            self.assertEqual((r['lr'],r['seed'],r['stream_seed']),(.001,0,2030))
        with self.assertRaises(ValueError):descriptor('fwe','S72','0.2B')
        with self.assertRaises(ValueError):descriptor('nem','S72','0.8B')

    def test_one_dataset_relative_change_not_absolute(self):
        d=select_extension(dict(dclm=.01,fwe=.02),dict(dclm=.004,fwe=.01))
        self.assertEqual(d['dataset'],'dclm')
        self.assertTrue(all(r['trigger'] for r in d['datasets'].values()))
        self.assertIsNone(select_extension(dict(dclm=.01,fwe=.02),dict(dclm=.012,fwe=.023))['dataset'])
        self.assertEqual(select_extension(dict(dclm=.01,fwe=.02),dict(dclm=None,fwe=.01))['dataset'],'fwe')
        self.assertIsNone(select_extension({}, {})['dataset'])

    def test_joint_readings(self):
        self.assertEqual(reading(.01,.006)['label'],'falls')
        self.assertEqual(reading(.01,.011)['label'],'flat')
        self.assertEqual(reading(.01,.014)['label'],'rises')
        self.assertEqual(joint_reading({d:dict(label='falls') for d in ('dclm','fwe')},dict(label='changing',direction='down')),'stable-state effect')
        self.assertEqual(joint_reading({d:dict(label='flat') for d in ('dclm','fwe')},dict(label='changing',direction='down')),'family difference')


class Replay(unittest.TestCase):
    def test_combined_report_retains_smol_qualification_and_relative_damage(self):
        from scripts.a2_family_f6_run import build_report
        from scripts.a2_quant_io import lock
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'f6';old=Path(d)/'old';root.mkdir();old.mkdir()
            old_value=dict(overlap={'fwe':{'primary':'full'}},lr_selection={'lr':.0001},
                rows=[dict(arm='k6',data='fwe',budget='0.8B',lr=.0001,stream_seed=2030,delta_post=.16)],
                pairs=[dict(quantity='E',data='fwe',seed=2030,budget='0.8B',lr=.0001,full={'E':.013})],
                readings={'FE2':dict(label='changing',direction='down',unverified=True)})
            write(old/'decision_family.json',old_value)
            write(root/'audit.json',dict(exclude_final_sequence_ids=[875,963,1320]))
            cfg=dict(results_root=str(root),previous_results_root=str(old),overlap_audit=lock(root/'audit.json'),
                     historical={ds:dict(E_full=.01,E_reduced=.01) for ds in ('dclm','fwe')},ft9_reference=3.0669867735128715)
            result=build_report(cfg,[],None)
            self.assertAlmostEqual(result['Smol_E_over_short_delta'][0]['E_over_short_delta'],.013/.16)
            self.assertTrue(result['joint_reading_unverified'])
            self.assertFalse(result['FT9_pass'])
            self.assertEqual(read(root/'decision_family.json')['previous_F0_F5']['readings'],old_value['readings'])

    def test_original_arrays_ft9_and_overlap_replay(self):
        a=json.loads(gzip.decompress(Path('resources/a2_family_f6_historical_20261108.json.gz').read_bytes()))['runs']
        self.assertEqual(a['ST1-dclm-S72-S1-0.2B-lr1e-3']['scores']['final']['loss'],3.0669867735128715)
        for ds,expected in [('dclm',.0111319744049041),('fwe',.014780744431005743)]:
            short,long=[adapted(a[f'ST1-{ds}-{arm}-S1-0.2B-lr1e-3']) for arm in ('S72','S188')]
            full=pair_scores(short,long,repeats=0)
            reduced=pair_scores(short,long,[875,963,1320],repeats=0)
            self.assertAlmostEqual(full['E'],expected,places=12)
            self.assertEqual(reduced['dropped_prediction_tokens'],12285)
            if ds=='fwe':self.assertLess(abs(reduced['E']-full['E']),.001)

    def test_wrong_estimator_or_corrupt_scalar_refused(self):
        good=dict(per_sequence_nll=[1.,2.],loss=1.5,tokens=8190,sequences=2)
        normalized(good)
        with self.assertRaises(ValueError):normalized(dict(good,tokens=8000))
        with self.assertRaises(ValueError):normalized(dict(good,loss=2.))

    def test_actual_order_exposure_and_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);records=[]
            for rank in range(8):
                p=root/f'sample-order-rank{rank}.jsonl'
                p.write_text(json.dumps(dict(sample_ids=list(range(rank*8,(rank+1)*8))))+'\n')
                records.append(dict(path=str(p),sha256=sha(p)))
            r=dict(raw_order_files=records,registered_run=dict(steps=1))
            audit=dict(prefix_coverage={'2030':{'hits':[dict(token_start=64*4096-10,token_stop=64*4096+10)]}})
            out=exposure(r,audit);self.assertEqual(out['status'],'seen');self.assertEqual(out['documents'][0]['actual_consumed_tokens'],10)
            audit['prefix_coverage']['2030']['hits'][0]=dict(token_start=64*4096,token_stop=64*4096+10)
            self.assertEqual(exposure(r,audit)['status'],'not seen')
            p.write_text('changed')
            with self.assertRaises(ValueError):exposure(r,audit)

    def test_backup_preserves_raw_traces_excludes_weights(self):
        from scripts.a2_family_f6_backup import backup
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'REPORT.md').write_text('report');write(root/'decision_family.json',{})
            p=root/'heals/r/final';p.mkdir(parents=True);(p/'model.safetensors').write_bytes(b'weights')
            for i in range(8):(p.parent/f'sample-order-rank{i}.jsonl').write_text('{}\n')
            result=backup(root)
            self.assertEqual(result['raw_rank_traces'],8)
            files=read(root/'backup/manifest.json')['files']
            self.assertFalse(any('safetensors' in f for f in files))
            for item in result['files']:self.assertEqual(sha(item['path']),item['sha256'])


class Numerics(unittest.TestCase):
    def test_traced_trainer_identical_to_a2_s(self):
        import torch
        from transformers import default_data_collator,set_seed
        from test_model import tiny_model
        from test_train import Rows
        from efficient_llm.a2.model import KDModel,prune
        from efficient_llm.a2.train import make_arguments
        from efficient_llm.a2_breadth.heal import ReportTrainer as Old,Trace
        from efficient_llm.a2_family_f6.heal import ReportTrainer as New
        teacher=tiny_model().requires_grad_(False);base=copy.deepcopy(teacher).requires_grad_(True);prune(base,[1,2]);states=[];orders=[];lrs=[]
        with tempfile.TemporaryDirectory() as d:
            for cls in (Old,New):
                set_seed(0);model=copy.deepcopy(base);trace=Trace()
                kwargs=dict(trace_path=Path(d)/'trace.jsonl') if cls is New else {}
                trainer=cls(model=KDModel(model,3),teacher=teacher,
                            args=make_arguments(Path(d)/cls.__module__,12,.001,4,1,cuda=False,global_sequences=8),
                            train_dataset=Rows(96),data_collator=default_data_collator,callbacks=[trace],**kwargs)
                trainer.train();states.append(copy.deepcopy(model.state_dict()));orders.append(trainer.seen);lrs.append(trace.lrs)
                if cls is New:
                    trainer.trace_file.close()
                    saved=[json.loads(x)['sample_ids'] for x in (Path(d)/'trace.jsonl').read_text().splitlines()]
                    self.assertEqual(saved,trainer.seen)
            self.assertEqual(lrs[0],lrs[1]);self.assertEqual(orders[0],orders[1])
            for k in states[0]:self.assertTrue(torch.equal(states[0][k],states[1][k]),k)
            self.assertTrue(all(p.grad is None for p in teacher.parameters()))


class Orchestration(unittest.TestCase):
    def test_preparation_publishes_frozen_ready_only_after_input_verification(self):
        from scripts import a2_family_f6_prepare as prep
        from scripts.a2_quant_io import lock
        from scripts.a2_family_v14_gate import SPEC_SHA,verify_ready
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'src';transition=root/'transition';previous=root/'previous';out=root/'out'
            (source/'resources').mkdir(parents=True);transition.mkdir()
            (transition/'gate.py').write_text('gate')
            bridge=previous/'source/scripts/a2_family_depth_bridge.py';bridge.parent.mkdir(parents=True);bridge.write_text('patched')
            write(transition/'installed.json',dict(installed=True,gate_sha256=sha(transition/'gate.py'),patched_sha256=sha(bridge)))
            write(source/'resources/a2_family_f6_cpu_20261108.json',dict(status='PASS',launch_ready=True,files={}))
            assets={ds+'/'+arm:dict(weights_dir=str(root),files={}) for ds in ('dclm','fwe') for arm in ('S72','S188')}
            write(root/'assets.json',assets);write(root/'overlap.json',dict(exclude_final_sequence_ids=[875,963,1320]))
            history=Path('resources/a2_family_f6_historical_20261108.json.gz')
            (source/'history.gz').write_bytes(history.read_bytes())
            (source/'docs').mkdir();(source/'docs/A2_family_spec_v1.4.md').write_bytes(Path('docs/A2_family_spec_v1.4.md').read_bytes())
            plan=dict(host='fixture',family_root=str(previous),transition_root=str(transition),spec_sha256=SPEC_SHA,
                      assets=lock(root/'assets.json'),overlap_audit=lock(root/'overlap.json'),historical_file=dict(path='history.gz',sha256=sha(history)))
            verified=[]
            def verify(cfg):
                self.assertFalse((transition/'ready.json').exists());verified.append(True);return []
            def freeze(src,dest,commit,extra_files):
                frozen=dest/'source';(frozen/'scripts').mkdir(parents=True);(frozen/'docs').mkdir()
                (frozen/'scripts/a2_family_f6_run.py').write_text('fixture')
                (frozen/'docs/A2_family_spec_v1.4.md').write_bytes((src/'docs/A2_family_spec_v1.4.md').read_bytes())
                write(dest/'source-manifest.json',dict(files={str(p.relative_to(frozen)):sha(p) for p in frozen.rglob('*') if p.is_file()}))
                return frozen
            with patch.object(prep.socket,'gethostname',return_value='fixture'),patch.object(prep,'verify_inputs',side_effect=verify),patch.object(prep,'freeze_source',side_effect=freeze):
                prep.prepare(plan,source,out,'0'*40)
                with self.assertRaises(ValueError):prep.prepare(plan,source,out,'0'*40)
            self.assertEqual(verified,[True])
            _,cfg=verify_ready(read(transition/'ready.json'))
            self.assertEqual(cfg['ft9_reference'],3.0669867735128715)
            self.assertEqual(len(cfg['runs']),9)

    def test_bad_actual_model_hash_is_rejected_before_data_or_ready(self):
        from scripts.a2_family_f6_prepare import verify_inputs
        from scripts.a2_quant_io import lock
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);model=root/'model';model.mkdir();(model/'weights').write_bytes(b'wrong')
            assets={ds+'/'+arm:dict(weights_dir=str(model),code_dir=str(model),files={'weights':'0'*64},code_files={})
                    for ds in ('dclm','fwe') for arm in ('S72','S188')}
            write(root/'assets.json',assets)
            with patch('efficient_llm.a2.protocol.parent_ref',return_value={}):
                with self.assertRaisesRegex(ValueError,'input bytes differ'):
                    verify_inputs(dict(assets=lock(root/'assets.json')))
            self.assertFalse((root/'ready.json').exists())

    def exercise(self,trigger,fail=False):
        from scripts import a2_family_f6_run as runner
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'f6';root.mkdir();old=Path(d)/'old';transition=Path(d)/'transition'
            write(old/'backup/verified.json',dict(verified=True,files=[]));write(transition/'ready.json',{})
            cfg=dict(results_root=str(root),previous_results_root=str(old),transition_root=str(transition),overlap_audit={})
            calls=[]
            def launch(cfg,config,item,events):
                calls.append(item['run_id']);events.append(dict(name=item['run_id'],returncode=1 if fail and len(calls)==1 else 0))
            def report(cfg,events,extension,final=False):
                if final:
                    (root/'REPORT.md').write_text('complete report, including failures')
                    write(root/'decision_family.json',dict(events=events))
                return dict(E02=dict(dclm=.01,fwe=.02),overlap=dict(primary_FWE='full'),
                     pairs=[dict(dataset=ds,budget='0.8B',full={'E':v}) for ds,v in [('dclm',.004 if trigger else .011),('fwe',.01 if trigger else .021)]])
            with patch.object(runner,'owner'),patch.object(runner,'verify_ready'),patch('scripts.a2_family_f6_prepare.verify_inputs'),patch.object(runner,'launch',side_effect=launch),patch.object(runner,'build_report',side_effect=report):
                runner.execute(cfg,root/'config.json',root/'receipt.json')
            value=verify_completion(root/'receipt.json')
            self.assertEqual(value['extension_dataset'],'dclm' if trigger else None)
            self.assertEqual(len(calls),7 if trigger else 5)
            self.assertEqual(calls[0],'ST1-dclm-S72-S1-0.2B-lr1e-3')
            self.assertEqual(len(set(calls)),len(calls))
            if fail:self.assertEqual(value['events'][0]['returncode'],1)
    def test_complete_sequence_no_extension(self):self.exercise(False)
    def test_larger_relative_dataset_only(self):self.exercise(True)
    def test_ft9_failure_preserved_continues_registered_runs(self):self.exercise(True,True)
