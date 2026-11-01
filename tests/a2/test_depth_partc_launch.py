import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from efficient_llm.a2_depth.common import read,write,SPEC_SHA


class DepthCLaunchTests(unittest.TestCase):
    def test_plan_requires_actual_note_and_handoff_and_preserves_registered_inputs(self):
        from scripts.a2_depth_partc_plan import assemble
        base=read('resources/a2_depth_partb_plan_20261101.json')
        refs=read('resources/a2_depth_stage1_references_20261101.json')
        history=read('resources/a2_depth_stream_reference_20261101.json')
        downstream=read('resources/a2_quant_plan_20261031.json')
        handoff={k:'fixture' for k in ('root','burn_script','supervisor','burn','complete','source_manifest','prior_status','known_burn_identities')}
        note=dict(posted_at='fixture',spec_sha256=SPEC_SHA)
        cfg=assemble(base,refs,history,downstream,handoff,note,{})
        self.assertNotIn('results_root',cfg);self.assertNotIn('code_commit',cfg)
        self.assertEqual(cfg['stage1_dclm_reference'],refs['datasets']['dclm'])
        self.assertEqual(cfg['prepared'],downstream['prepared'])
        self.assertEqual(cfg['prior_downstream'],downstream['prior_downstream']['dclm'])
        with self.assertRaises(ValueError):assemble(base,refs,history,downstream,{},note,{})
        with self.assertRaises(ValueError):assemble(base,refs,history,downstream,handoff,{}, {})

    def test_launcher_continues_scientific_flags_and_freezes_registered_native_job(self):
        import scripts.launch_a2_depth_partc as launcher
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'source';out=Path(d)/'partc'
            (source/'resources').mkdir(parents=True);(source/'docs').mkdir()
            (source/'docs/A2_depth_spec.md').write_text('fixture')
            (source/'resources/accelerate_b200.yaml').write_text('fixture')
            write(source/'resources/a2_depth_partc_integration_cpu_20261101.json',dict(status='PASS',files={}))
            report=Path(d)/'AB.md';report.write_text('Actual A+B report fixture')
            cfg=dict(spec_sha256=SPEC_SHA,data_root=d,assets='assets',data_manifest='data',prepared='prepared',
                reference_nll={},part_ab_note=dict(spec_sha256=SPEC_SHA,posted_at='fixture'),
                part_ab_report=dict(path=str(report),sha256=hashlib.sha256(report.read_bytes()).hexdigest()),
                predecessor=dict(complete='done',burn_script='owned'),stream_history={})
            plan=source/'resources/plan.json';write(plan,cfg);calls=[]
            def process(args,**kw):
                calls.append(args);return SimpleNamespace(returncode=1 if args[:2]==['tmux','has-session'] else 0,stdout='',stderr='')
            def freeze(src,output,commit,**kw):
                write(output/'source-manifest.json',dict(code_commit=commit,files={}))
                return source
            with patch.object(launcher,'__file__',str(source/'scripts/launcher.py')),patch('sys.argv',['launcher','--plan',str(plan),'--output',str(out),'--session','fixture','--code-commit','fixture']),patch.object(launcher.socket,'gethostname',return_value='thiennh-p6-6y3o-worker-0'),patch.object(launcher,'predecessor',return_value=dict(inspection={})),patch.object(launcher,'freeze_source',side_effect=freeze),patch.object(launcher,'read_locked',side_effect=lambda item:{}),patch.object(launcher.subprocess,'run',side_effect=process),patch('efficient_llm.a2.data.PackedRows',side_effect=ValueError('fixture flag')),patch('accelerate.commands.config.config_args.default_config_file',str(Path(d)/'cache.yaml')),patch('accelerate.commands.config.config_args.load_config_from_file',return_value=SimpleNamespace(num_processes=8,mixed_precision='bf16',use_cpu=False)):
                launcher.main()
            jobs=read(out/'jobs.json')['jobs'];self.assertEqual(len(jobs),1)
            self.assertIsNone(jobs[0]['timeout_seconds']);self.assertIn('scripts.a2_depth_partc_run',jobs[0]['argv'])
            self.assertEqual(sum(c[:2]==['tmux','new-session'] for c in calls),1)
            root=out/'results/a2_depth_20261031';frozen=read(root/'config/plan.json')
            self.assertEqual(frozen['source_manifest']['path'],str(out/'source-manifest.json'))
            self.assertEqual(Path(frozen['part_ab_report_local']).read_text(),report.read_text())
            self.assertTrue(any(not c['passed'] for c in read(root/'preparation_checks.json')['checks']))


if __name__=='__main__':unittest.main()
