import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scripts.a2_family_report import arrays,pair_scores,exposure
from efficient_llm.a2_family.common import sha

class FamilyReadout(unittest.TestCase):
    def score(self,values,counts=(1,9,2)):
        return dict(per_sequence_nll=values,per_sequence_tokens=list(counts),loss=float(np.dot(values,counts)/sum(counts)))
    def row(self,parent,pre,post):
        return dict(baseline={'final':self.score(parent)},pre={'final':self.score(pre)},scores={'final':self.score(post)})
    def test_weighted_common_deletion_and_paired_intervals(self):
        a=self.row([3,2,4],[4,3,5],[3.5,2.2,4.1]);b=self.row([2,1,3],[4,3,5],[2.7,1.4,3.3])
        full=pair_scores(a,b,repeats=1000);reduced=pair_scores(a,b,[1],repeats=1000)
        self.assertAlmostEqual(full['E'],.2);self.assertAlmostEqual(full['G'],1);self.assertAlmostEqual(full['E_over_G'],.2)
        np.testing.assert_allclose(full['bootstrap']['E_ci95'],[.2,.2],atol=1e-14)
        self.assertEqual(reduced['dropped_prediction_tokens'],9);self.assertEqual(reduced['kept_prediction_tokens'],3)
        self.assertAlmostEqual(reduced['E'],.2)
        bad=copy.deepcopy(b);bad['baseline']['final']['per_sequence_tokens'][0]=2
        with self.assertRaises(ValueError):pair_scores(a,bad)
        with self.assertRaises(ValueError):pair_scores(a,b,[0,1,2])
    def test_actual_trace_exposure_at_token_boundary_and_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'rank0.jsonl';p.write_text(json.dumps(dict(sample_ids=list(range(64))))+'\n')
            r=dict(registered_run=dict(data='fwe',stream_seed=2030,steps=1),raw_order_files=[dict(path=str(p),sha256=sha(p))])
            config=dict(overlap={'fwe':dict(consumed_overlaps={'2030':[dict(stream_start=64*4096,stream_stop=64*4096+1)]})})
            self.assertEqual(exposure(config,r)['status'],'not seen')
            config['overlap']['fwe']['consumed_overlaps']['2030'][0]['stream_start']-=1
            self.assertEqual(exposure(config,r)['status'],'seen')
            p.write_text(json.dumps(dict(sample_ids=list(range(64))+[0]))+'\n');r['raw_order_files'][0]['sha256']=sha(p)
            with self.assertRaises(ValueError):exposure(config,r)
            p.write_text('{}\n')
            with self.assertRaises(ValueError):exposure(config,r)

class FamilyQueue(unittest.TestCase):
    def test_registered_graph_extensions_and_automatic_backup(self):
        from unittest.mock import patch
        from scripts import a2_family_run as run
        from efficient_llm.a2_family.common import write
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            def launch(cfg,module,name,arguments,events):
                event=dict(name=name,module=module,returncode=0,seconds=1);events.append(event)
                if module.endswith('.checks'):write(root/'parents'/name[3:]/'result.json',dict(mask_documents=True))
                if module.endswith('.heal'):
                    item=json.loads(Path(arguments[-1]).read_text());long=item['arm']!='k6'
                    delta={'0.2B':.02,'0.8B':.04,'3.2B':.045}[item['budget']] if long else 0.
                    if item['arm']=='k64':delta+=.02
                    if item['data']=='mix' and long:delta+=.01
                    write(root/'heals'/name/'result.json',dict(delta_post=delta,scores={'tuning':{'loss':item['lr']}}))
                return event
            def readout(*args,**kwargs):
                pairs=[]
                for data in ('fwe','mix'):
                    for budget,value in [('0.2B',.02),('0.8B',.04),('3.2B',.045)]:
                        for seed in (2030,2031):
                            for quantity in ('E','D64'):
                                v=value+(.01 if data=='mix' else 0)
                                pairs.append(dict(data=data,budget=budget,lr=2e-4/16,seed=seed,quantity=quantity,full={'E':v},reduced={'E':v}))
                return dict(coverage_complete=True,pairs=pairs,overlap={d:{'primary':'full'} for d in ('fwe','mix')})
            with patch.object(run,'launch',side_effect=launch),patch('scripts.a2_family_report.report',side_effect=readout),patch('scripts.a2_family_backup.backup',return_value={'verified':True}) as backup:
                run.execute(dict(results_root=str(root),peak_lr=2e-4),'unused',root/'receipt.json')
            receipt=json.loads((root/'receipt.json').read_text());jobs=receipt['events'];heals=[e for e in jobs if e['module'].endswith('.heal')]
            self.assertEqual(len(heals),21);self.assertEqual(len(jobs),26);self.assertEqual(len({e['name'] for e in jobs}),26)
            self.assertTrue(all(json.loads((root/'extensions.json').read_text())[k] for k in ('F3','F4','F5')))
            backup.assert_called_once();self.assertTrue(receipt['readout_complete'])
    def test_material_overlap_controls_trigger_and_missing_audit_never_invents_value(self):
        from scripts.a2_family_run import primary_contrast
        p=dict(quantity='E',data='mix',budget='0.2B',lr=5e-5,seed=2030,full={'E':.02},reduced={'E':.04})
        d=dict(pairs=[p],overlap={'mix':{'primary':'reduced'}})
        self.assertEqual(primary_contrast(d,'0.2B',5e-5,data='mix'),.04)
        d['overlap']={};self.assertIsNone(primary_contrast(d,'0.2B',5e-5,data='mix'))
    def test_backup_keeps_parent_final_arrays_and_raw_order_excludes_weights(self):
        from scripts.a2_family_backup import backup
        from efficient_llm.a2_family.common import read
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ['REPORT.md','decision_family.json','parents/k6/perseq/final/parent.json','heals/run/sample-order-rank0.jsonl','heals/run/final/config.json','heals/run/trainer/state.json']:
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
            backup(root);v=read(root/'backup/manifest.json')
            self.assertEqual(len(v['files']),4);self.assertIn('parents/k6/perseq/final/parent.json',v['files'])
            self.assertEqual(read(root/'backup/verified.json')['raw_rank_traces'],1)

class FamilyNativeManifest(unittest.TestCase):
    def test_depth_bridge_validates_native_selection_then_uses_frozen_environment(self):
        from unittest.mock import patch
        from scripts import a2_family_depth_bridge as bridge
        from efficient_llm.a2_family.common import write,read
        from efficient_llm.a2_depth.heal_long import run_identity
        from efficient_llm.a2_depth.resume import checkpoint_files
        from efficient_llm.a2_depth.common import SPEC_SHA as DEPTH_SHA
        from efficient_llm.a2.protocol import lr_for_update
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);depth=root/'depth';execution=depth/'continuations/attempt001/source'
            src=depth/'source';src.mkdir(parents=True);execution.mkdir(parents=True)
            for folder in (src,execution):(folder/'fixture.py').write_text('# immutable source\n')
            original=write(depth/'source-manifest.json',dict(files={'fixture.py':sha(src/'fixture.py')}))
            patched=write(execution.parent/'execution-source-manifest.json',dict(files={'fixture.py':sha(execution/'fixture.py')}))
            plan=dict(spec_sha256=DEPTH_SHA,results_root=str(depth/'results'),assets={},heal_streams={'dclm':{}},source_manifest=original)
            config=depth/'config.json';config.write_text(json.dumps(plan))
            configlock=dict(path=str(config),sha256=sha(config))
            launch=write(depth/'launch.json',dict(config=configlock,source_manifest=original,environment={'HF_HUB_OFFLINE':'1','A2_CODE_COMMIT':'original'}))
            identity=run_identity(plan,'P300');state=Path(plan['results_root'])/'heals/ST1-dclm-P300-S1-12.8B-lr1e-3/trainer/checkpoint-1';state.mkdir(parents=True)
            for name in checkpoint_files(8):(state/name).write_bytes(b'fixture')
            (state/'trainer_state.json').write_text(json.dumps({'global_step':1}))
            for rank in range(8):
                audit=dict(identity=identity,global_step=1,world_size=8,rank=rank,nonfinite_microbatches=0,lrs=[lr_for_update(1,48828,.001)],seen=[[0,1,2,3],[4,5,6,7]])
                (state/f'depth-audit-rank{rank}.json').write_text(json.dumps(audit))
            marker=dict(spec_sha256=DEPTH_SHA,identity=identity,global_step=1,world_size=8,complete=True,state_finite=True,files={n:(state/n).stat().st_size for n in checkpoint_files(8)})
            (state/'depth-checkpoint.json').write_text(json.dumps(marker))
            pin=root/'pause/checkpoint-1';pin.parent.mkdir();shutil.copytree(state,pin)
            saved=write(root/'pause/saved.json',dict(checkpoint=str(pin),original=str(state),global_step=1,hashes={p.name:sha(p) for p in state.iterdir()}))
            pause=write(root/'pause/terminal.json',dict(state='paused',handoff_verified=True,arm='P300',saved_state=saved))
            results=root/'family';backup=write(results/'backup/archive.json',{'scientific':'fixture'})
            write(results/'backup/verified.json',dict(verified=True,files=[backup]))
            cfg=dict(depth_pause=pause,depth_launch=launch,depth_root=str(depth),depth_execution_source=str(execution),depth_execution_manifest=patched,results_root=str(results))
            self.assertEqual(bridge.validate(cfg)[2]['step'],1)
            cfgpath=root/'family.json';write(cfgpath,cfg)
            with patch('sys.argv',['bridge','--config',str(cfgpath),'--receipt',str(root/'receipt.json')]),patch.object(bridge,'require_free') as free,patch.object(bridge.subprocess,'run') as run:
                run.return_value.returncode=0;bridge.main()
                free.assert_called_once_with(list(range(8)))
                self.assertEqual(run.call_args.kwargs['cwd'],execution)
                self.assertEqual(run.call_args.kwargs['env']['PYTHONPATH'],str(execution))
                self.assertEqual(run.call_args.kwargs['env']['A2_CODE_COMMIT'],'original')
                self.assertIn(str(config),run.call_args.args[0])
            (pin/'optimizer.pt').write_bytes(b'changed')
            with self.assertRaises(ValueError):bridge.validate(cfg)
    def test_plan_requires_complete_inputs_and_binds_original_depth(self):
        from scripts.a2_family_plan import inputs
        from efficient_llm.a2_family.common import write,read
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);common=root/'common';mix=root/'mix';depth=root/'depth';resources=root/'resources'
            models=write(resources/'a2_family_downloads_20261107.json',dict(checkpoints=[]))
            mixplan=write(resources/'a2_family_mixture_plan_20261107.json',{})
            write(resources/'a2_family_lr_state_20261107.json',dict(peak_lr=2e-4))
            write(resources/'a2_family_historical_20261107.json',dict(datasets={}))
            mr=write(common/'models_ready.json',dict(status='verified',manifest_sha256=models['sha256']))
            cx=write(common/'cx/complete.json',dict(status='complete'))
            fwe=write(common/'fwe/complete.json',dict(status='complete',streams={s:dict(path='fixture',sha256='fixture') for s in ('2030','2031')}))
            write(common/'complete.json',dict(status='complete',failures={},ready=dict(models=mr,cx=cx,fwe=fwe),model_manifest_sha256=models['sha256']))
            ms=write(mix/'stream/manifest.json',dict(status='complete',checks=[dict(test='FT8',passed=False)]))
            write(mix/'complete.json',dict(status='complete',plan_sha256=mixplan['sha256'],stream=ms,flags=['finite pool']))
            dc=write(depth/'config.json',dict(results_root=str(depth/'results')))
            sm=write(depth/'source-manifest.json',dict(files={}))
            write(depth/'launch.json',dict(config=dc,source_manifest=sm))
            write(depth/'continuations.json',dict(attempts=[dict(root=str(depth/'continuations/attempt001'),config=dc,source_manifest=sm,execution_source_manifest=sm)]))
            opensci=write(root/'opensci/manifest.json',{})
            cfg=inputs(common,mix,depth,resources,opensci)
            self.assertEqual(cfg['depth_config'],dc);self.assertEqual(cfg['cx_complete'],cx)
            self.assertEqual(len(cfg['input_flags']),2);self.assertTrue(all(f['data']=='mix' for f in cfg['input_flags']))
            self.assertNotIn('depth_pause',cfg)
            write(mix/'complete.json',dict(status='incomplete'))
            with self.assertRaises(ValueError):inputs(common,mix,depth,resources,opensci)
            write(common/'complete.json',dict(status='incomplete',failures={'cx':'bad'}))
            with self.assertRaises(ValueError):inputs(common,mix,depth,resources,opensci)
    def test_native_order_has_no_separate_owner_or_fixed_deadline(self):
        from scripts.launch_a2_family import jobs
        from run_experiments import load_jobs
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'jobs.json';p.write_text(json.dumps(jobs('/frozen/config.json')))
            parsed=load_jobs(p);self.assertEqual(len(parsed),2)
            manifest=json.loads(p.read_text())['jobs']
            self.assertEqual(manifest[0]['name'],'a2-family');self.assertEqual(manifest[1]['name'],'a2-depth-resume-after-family')
            self.assertTrue(all(x['timeout_seconds'] is None and x['gpus']==list(range(8)) for x in manifest))
            self.assertEqual(manifest[1]['argv'][3],'scripts.a2_family_depth_bridge')
    def test_family_pause_copy_keeps_optimizer_rng_bytes_independent(self):
        from scripts.a2_family_pause_depth import copy_state
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'state';source.mkdir();files={f'member{i}':4 for i in range(20)}
            for name in files:(source/name).write_bytes(b'abcd')
            marker=dict(complete=True,state_finite=True,files=files);(source/'depth-checkpoint.json').write_text(json.dumps(marker))
            result=copy_state(source,root/'pinned',marker)
            self.assertEqual(len(result),21);(source/'member0').write_bytes(b'new!')
            self.assertEqual((root/'pinned/member0').read_bytes(),b'abcd')
            marker['state_finite']=False
            with self.assertRaises(ValueError):copy_state(source,root/'bad',marker)

class FamilyReportIntegration(FamilyReadout):
    def test_report_reads_actual_arrays_and_never_calls_partial_coverage_complete(self):
        from efficient_llm.a2_family.common import write,read
        from scripts.a2_family_report import report
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);events=[]
            for arm,base,post in [('k6',[3,2,4],[3.5,2.2,4.1]),('k20',[2,1,3],[2.7,1.4,3.3])]:
                name='run-'+arm;folder=root/'heals'/name;folder.mkdir(parents=True);trace=folder/'sample-order-rank0.jsonl';trace.write_text(json.dumps(dict(sample_ids=list(range(64))))+'\n')
                r=self.row(base,[4,3,5],post);r.update(run_id=name,registered_run=dict(arm=arm,data='fwe',budget='0.2B',lr=5e-5,steps=1,tokens=64*4096,stream_seed=2030),raw_order_files=[dict(path=str(trace),sha256=sha(trace))],flags=[])
                write(folder/'result.json',r);events.append(dict(name=name,module='efficient_llm.a2_family.heal',returncode=0))
            write(root/'lr_selection.json',dict(lr=5e-5))
            cfg=dict(results_root=str(root),peak_lr=2e-4,overlap={'fwe':dict(final_indices=[1],consumed_overlaps={'2030':[]})})
            result=report(cfg,events,final=True)
            self.assertFalse(result['coverage_complete']);self.assertEqual(len(result['missing']),5)
            self.assertTrue(result['overlap']['fwe']['immaterial']);self.assertAlmostEqual(result['pairs'][0]['full']['E'],.2)
            self.assertEqual(result['pairs'][0]['reduced']['dropped_prediction_tokens'],9)
            self.assertTrue((root/'REPORT.md').is_file());self.assertEqual(len(read(root/'decision_family.json')['rows']),2)

class FamilySafetyAndAudit(unittest.TestCase):
    def test_pause_rejects_other_arm_and_changed_frozen_source(self):
        from unittest.mock import patch
        from scripts import a2_family_pause_depth as p
        from efficient_llm.a2_family.common import write
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'depth';ex=root/'continuations/attempt001'
            for folder in (root/'source',ex/'source'):
                folder.mkdir(parents=True);(folder/'code.py').write_text('x=1\n')
            original=write(root/'source-manifest.json',dict(files={'code.py':sha(root/'source/code.py')}))
            execution=write(ex/'execution-source-manifest.json',dict(files={'code.py':sha(ex/'source/code.py')}))
            config=write(root/'config.json',{})
            write(root/'launch.json',dict(config=config,source_manifest=original));write(root/'continuations.json',dict(attempts=[dict(root=str(ex),execution_source_manifest=execution)]))
            identity=dict(pid=20,ppid=2,start_ticks=100)
            req=dict(depth_root=str(root),execution_root=str(ex),arm='P1T',config=config,source_manifest=original,execution_source_manifest=execution,supervisor_pid=20,gpu_mapping=[[0,'gpu',[10]]],processes={'20':identity,'10':dict(ppid=20)})
            cmd=lambda pid:['python','-m','scripts.train_then_burn','--output',str(ex/'supervised')] if pid==20 else ['python','--arm','P1T']
            with patch.object(p,'process',return_value=identity),patch.object(p,'process_argv',side_effect=cmd):
                p.validate_request(req);req['arm']='P300'
                with self.assertRaisesRegex(ValueError,'Wrong active arm'):p.validate_request(req)
                req['arm']='P1T';(ex/'source/code.py').write_text('changed')
                with self.assertRaisesRegex(ValueError,'Frozen source changed'):p.validate_request(req)
    def test_overlap_audit_whole_pool_and_both_exact_streams(self):
        from unittest.mock import patch
        from scripts import a2_family_overlap as audit
        from efficient_llm.a2_family.common import write
        from scripts.a2_stage1_data import INDEX
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);h='ab'*31+'00';heldout=[dict(set='final',index=0)]
            for split in ('tuning','final'):
                d=write(root/'cx'/split/'documents.json',dict(documents=[dict(text_sha256=h,tokens=5,predicted_tokens=4)]))
                write(root/'cx'/split/'manifest.json',dict(documents_sha256=d['sha256']))
            pool=np.zeros(2,dtype=INDEX);pool['hash']=[bytes.fromhex(h),bytes.fromhex('cd'*32)];pool.tofile(root/'pool.bin')
            streams={}
            for name in ('2030','2031','mix'):
                folder=root/name;folder.mkdir();trace=folder/'consumed_documents.jsonl';trace.write_text(json.dumps(dict(text_sha256=h,stream_start=1,stream_stop=6))+'\n')
                m=write(folder/'manifest.json',dict(streams={'heal':dict(exposure_sha256=sha(trace))}));streams[name]=dict(path=str(folder),sha256=m['sha256'])
            source=write(root/'source.json',dict(pool_overlap=[dict(text_sha256=h,heldout=heldout)]))
            ready=write(root/'ready.json',dict(status='complete',sources={'source':source}))
            cfg=dict(cx_root=str(root/'cx'),fwe_pool_documents=str(root/'pool.bin'),streams={s:streams[s] for s in ('2030','2031')},mixture_complete=ready,mixture_stream=streams['mix'])
            with patch.object(audit,'POOL_SHA',sha(root/'pool.bin')):result=audit.audit(cfg,root/'audit.json')
            self.assertEqual(result['fwe']['final_indices'],[0]);self.assertEqual(len(result['fwe']['pool_overlaps']),1)
            self.assertEqual(result['fwe']['pool_overlaps'][0]['text_sha256'],h)
            self.assertEqual(len(result['mix']['consumed_overlaps']['2030']),1)
