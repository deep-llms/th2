import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from efficient_llm.a2_family.common import write,read,sha,SPEC_SHA
from scripts import a2_family_admit as admit


class Admission(unittest.TestCase):
    def exercise(self,fail=None):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'admission';family=root/'family';events=[]
            for name in ('common','mixture'):write(root/name/'complete.json',dict(status='complete'))
            cfg=dict(input_identity='fixture')
            def verify(c,paused):
                self.assertFalse(paused);events.append('verify')
                if fail=='data':raise ValueError('Changed actual data bytes')
            def audit(c,p):events.append('audit');write(p,dict(status='complete',datasets={}))
            def request(c):
                events.append('request');self.assertEqual(read(c['overlap_preflight']['path'])['status'],'complete')
                return dict(arm='P300')
            def run(cmd,**kw):
                from types import SimpleNamespace
                if cmd[0]=='tmux':return SimpleNamespace(returncode=1)
                module=cmd[3];events.append(module)
                if module.endswith('pause_depth'):
                    if fail=='pause':raise RuntimeError('Changed ownership; no signal')
                    write(out/'pause/terminal.json',dict(state='paused',handoff_verified=True,global_step=44000))
                else:
                    self.assertTrue((out/'pause/terminal.json').is_file())
                    config=write(family/'plan.json',{});jobs=write(family/'jobs.json',{})
                    write(family/'launch.json',dict(config=config,jobs=jobs,automatic_Depth_resume=True))
                return SimpleNamespace(returncode=0)
            with patch.object(admit,'verify_proof'),patch.object(admit,'inputs',return_value=cfg),patch.object(admit,'verify_inputs',side_effect=verify),patch.object(admit,'audit',side_effect=audit),patch.object(admit,'pause_request',side_effect=request),patch.object(admit,'finalize',side_effect=lambda c,p:c),patch.object(admit.subprocess,'run',side_effect=run):
                args=[root/'common',root/'mixture',root/'depth',root,out,family,'fixture','0'*40]
                if fail:
                    with self.assertRaises((ValueError,RuntimeError)):admit.execute(*args)
                    self.assertTrue((out/'failure.json').is_file());self.assertFalse((out/'complete.json').exists())
                else:
                    admit.execute(*args)
                    self.assertTrue(read(out/'complete.json')['automatic_Depth_resume'])
                    self.assertFalse(read(out/'complete.json')['training_start_verified'])
                    with self.assertRaises(FileExistsError):admit.execute(*args)
            return events

    def test_verified_data_audit_then_pause_then_one_native_queue(self):
        self.assertEqual(self.exercise(),['verify','audit','request','scripts.a2_family_pause_depth','scripts.launch_a2_family'])

    def test_changed_data_never_pauses_depth(self):
        self.assertEqual(self.exercise('data'),['verify'])

    def test_changed_ownership_never_launches_family(self):
        self.assertEqual(self.exercise('pause'),['verify','audit','request','scripts.a2_family_pause_depth'])

    def test_proof_and_overlap_require_exact_bytes(self):
        from scripts.launch_a2_family import verify_proof,verified_overlap
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);code=root/'code.py';code.write_text('immutable')
            p=root/'resources/a2_family_integration_cpu_20261107.json'
            write(p,dict(status='PASS',launch_ready=False,files={'code.py':sha(code)}))
            with self.assertRaises(ValueError):verify_proof(root)
            v=read(p);v['launch_ready']=True;write(p,v);verify_proof(root)
            code.write_text('changed')
            with self.assertRaises(ValueError):verify_proof(root)
            trace=write(root/'trace.json',{});audit=write(root/'audit.json',dict(status='complete',datasets={'mix':dict(trace_locks={'2030':trace})}))
            verified_overlap(dict(overlap_preflight=audit))
            (root/'trace.json').write_text('changed')
            with self.assertRaises(ValueError):verified_overlap(dict(overlap_preflight=audit))
