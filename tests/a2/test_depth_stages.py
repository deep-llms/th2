import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from efficient_llm.a2_depth.common import write
from efficient_llm.a2_depth.stream import verify


class DepthStageTests(unittest.TestCase):
    def test_plan_does_not_inherit_old_output_or_predecessor(self):
        from scripts.a2_depth_partb_plan import assemble
        from efficient_llm.a2_depth.common import SPEC_SHA
        base={k:{} for k in ('assets','data_root','data_manifest','small_indices','reference_nll','heal_streams','phase0_sets','layer_sets','deviations')}
        base.update(results_root='old',code_commit='old',predecessor='Breadth')
        refs=dict(datasets={ds:{'0.2B':dict(E=.01)} for ds in ('dclm','fwe')})
        handoff={k:'actual-evidence-fixture' for k in ('root','burn_script','supervisor','burn','complete','source_manifest','prior_status','known_burn_identities')}
        note=dict(posted_at='fixture',spec_sha256=SPEC_SHA)
        plan=assemble(base,refs,{},handoff,note,{})
        self.assertNotIn('results_root',plan);self.assertNotIn('code_commit',plan)
        self.assertEqual(plan['predecessor'],handoff)
        self.assertEqual(plan['short_S1_reference']['dclm'],refs['datasets']['dclm']['0.2B'])
        with self.assertRaisesRegex(ValueError,'Actual terminal'):assemble(base,refs,{}, {},note,{})

    def test_partB_launcher_scientific_failure_still_submits_only_registered_stage(self):
        import scripts.launch_a2_depth_partb as launcher
        from efficient_llm.a2_depth.common import SPEC_SHA
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'code';output=Path(d)/'partb'
            (source/'resources').mkdir(parents=True);(source/'docs').mkdir()
            (source/'docs/A2_depth_spec.md').write_text('fixture')
            (source/'resources/accelerate_b200.yaml').write_text('fixture')
            write(source/'resources/a2_depth_partb_cpu_tests_20261101.json',dict(status='PASS',files={}))
            report=Path(d)/'parta-report.md';report.write_text('# A2 Depth — Part A\n\nWindow table fixture\nRemaining: historical footer\n')
            refs={ds:dict(E=0,E_relative=0,A_init=0,A_init_relative=0) for ds in ('dclm','fwe')}
            cfg=dict(spec_sha256=SPEC_SHA,assets='assets',data_manifest='manifest',data_root=d,small_indices=[],
                reference_nll={},heal_streams={},phase0_sets='sets',layer_sets={},stage1_references='refs',short_S1_reference=refs,
                predecessor=dict(complete='complete',burn_script='fixture-owned-burn'),
                part_a_report=dict(path=str(report),sha256=hashlib.sha256(report.read_bytes()).hexdigest()),
                part_a_note=dict(posted_at='fixture',spec_sha256=SPEC_SHA))
            write(source/'resources/a2_depth_partb_plan_20261101.json',cfg)
            objects=dict(manifest=dict(small_indices=[]),assets={},sets={},refs=dict(datasets={ds:{'0.2B':r} for ds,r in refs.items()}),complete=dict(coverage_complete=False,readout_complete=False))
            calls=[]
            def run(cmd,**kw):
                calls.append(cmd)
                return SimpleNamespace(returncode=1 if cmd[:2]==['tmux','has-session'] else 0,stdout='',stderr='')
            def freeze(src,out,commit,**kw):
                write(out/'source-manifest.json',dict(code_commit=commit,files={}))
                return source
            with patch.object(launcher,'__file__',str(source/'scripts/launcher.py')),patch('sys.argv',['launcher','--output',str(output),'--session','fixture','--code-commit','fixture']),patch.object(launcher.socket,'gethostname',return_value='thiennh-p6-6y3o-worker-0'),patch.object(launcher,'predecessor',return_value=dict(inspection={})),patch.object(launcher,'freeze_source',side_effect=freeze),patch.object(launcher,'read_locked',side_effect=lambda key:objects[key]),patch.object(launcher.subprocess,'run',side_effect=run),patch('efficient_llm.a2.data.PackedRows',side_effect=ValueError('fixture scientific failure')),patch('accelerate.commands.config.config_args.default_config_file',str(Path(d)/'cache.yaml')),patch('accelerate.commands.config.config_args.load_config_from_file',return_value=SimpleNamespace(num_processes=8,mixed_precision='bf16',use_cpu=False)):
                launcher.main()
            jobs=json.loads((output/'jobs.json').read_text())['jobs']
            self.assertEqual(len(jobs),1);self.assertIsNone(jobs[0]['timeout_seconds'])
            self.assertIn('scripts.a2_depth_partb_run',jobs[0]['argv'])
            self.assertEqual(len([c for c in calls if c[:2]==['tmux','new-session']]),1)
            result=output/'results/a2_depth_20261031'
            tests=json.loads((result/'preparation_checks.json').read_text())['checks']
            self.assertTrue(any(t['test']=='packed_data:final' and not t['passed'] for t in tests))
            from scripts.a2_depth_partb_report import finalize
            frozen=json.loads((result/'config/plan.json').read_text());finalize(frozen)
            self.assertIn('Window table fixture',(result/'REPORT.md').read_text())
            self.assertNotIn('Remaining: historical footer',(result/'REPORT.md').read_text())

    def test_DT6_uses_historical_identity_not_same_path_comparison(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);stream=root/'stream';stream.mkdir()
            (stream/'heal.bin').write_bytes(np.arange(40,dtype='<u4').tobytes())
            (stream/'heal.starts').write_bytes(bytes([1,0,0,0])*10)
            digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            manifest=dict(status='complete',streams=dict(heal=dict(rows=10,length=4,
                token_sha256=digest(stream/'heal.bin'),starts_sha256=digest(stream/'heal.starts'))))
            write(stream/'manifest.json',manifest)
            historical=dict(data_manifest=manifest,provenance=dict(data_sha256=digest(stream/'manifest.json')))
            good=verify(stream,historical,root/'good',prefix_rows=3,selected_rows=8)
            self.assertTrue(good['passed']);self.assertEqual(good['selected_sequences'],8)
            content=bytearray((stream/'heal.bin').read_bytes());content[-1]^=1
            (stream/'heal.bin').write_bytes(content)
            bad=verify(stream,historical,root/'bad',prefix_rows=3,selected_rows=8)
            self.assertFalse(bad['passed'])
            self.assertEqual(bad['reader_prefix_sha256'],good['reader_prefix_sha256'])
            self.assertTrue(any(c['test']=='DT6_historical_full_tokens' for c in bad['flags']))

    def test_partB_requires_physical_handoff_but_not_scientific_pass(self):
        import scripts.launch_a2_depth_partb as launcher
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'parta';root.mkdir();output=root.parent/'partb';script=root/'source/resources/llm_pretrain_burn.py'
            gpu=dict(index=0,uuid='fixture',pids=[21]);live=dict(gpus=[gpu])
            records={'21':dict(pid=21,ppid=20,start_ticks=11),'20':dict(pid=20,ppid=1,start_ticks=10)}
            objects=dict(supervisor=dict(training_status='failed',finished_at='fixture'),
                burn=dict(collective_progress_verified=True,gpus=[gpu]),source_manifest=dict(files={}))
            cfg=dict(predecessor=dict(root=str(root),burn_script=str(script),supervisor='supervisor',burn='burn',
                source_manifest='source_manifest',prior_status='failed',known_burn_identities=records))
            def digest(path):return launcher.BURN_HASH if Path(path)==script else launcher.GUARD_HASH
            with patch.object(launcher,'read_locked',side_effect=lambda key:objects[key]),patch.object(launcher,'inspect',return_value=live),patch.object(launcher,'process',side_effect=lambda pid:records[str(pid)]),patch.object(launcher,'argv',return_value=str(script)),patch.object(launcher,'digest',side_effect=digest),patch.object(launcher.Path,'exists',return_value=False):
                self.assertEqual(launcher.predecessor(cfg,output)['identities'],records)
                wrong=copy.deepcopy(live);wrong['gpus'][0]['pids']=[99]
                with patch.object(launcher,'inspect',return_value=wrong):
                    with self.assertRaisesRegex(ValueError,'Different live GPU'):launcher.predecessor(cfg,output)
                objects['supervisor']['finished_at']=None
                with self.assertRaisesRegex(ValueError,'not terminal'):launcher.predecessor(cfg,output)


if __name__=='__main__':unittest.main()
