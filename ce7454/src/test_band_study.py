import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest

import torch

from band_study import command, choose_epochs
from extension_gate import wait_released
from attach_band_study import identity, original_running, supervise, stop_owned
from models import build_model, count_trainable
from research_band_stem import haar_bands


class BandTests(unittest.TestCase):
    def test_haar_and_phase_parameters(self):
        x = torch.randn(2,3,8,8, dtype=torch.float64, requires_grad=True)
        y = haar_bands(x)
        torch.testing.assert_close(y.square().sum(), x.square().sum())
        self.assertTrue(torch.autograd.gradcheck(haar_bands, (x,)))
        torch.manual_seed(0); a = build_model('band_stem')
        torch.manual_seed(0); b = build_model('phase_stem')
        self.assertEqual(count_trainable(a), 11291651)
        self.assertTrue(all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()))

    def test_models_gradient_reload_full512(self):
        for name in ('band_stem','phase_stem','band_stem_ibn'):
            torch.manual_seed(0)
            m = build_model(name)
            self.assertLessEqual(sum(p.numel() for p in m.parameters()), 12300000)
            opt = torch.optim.Adam(m.parameters(), lr=.0003)
            x = torch.randn(2,3,64,64)
            for _ in range(2):
                opt.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(m(x), torch.tensor([0,2]))
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
                opt.step()
            m.eval()
            with torch.no_grad():
                full = torch.randn(1,3,512,512)
                before = m(full)
                buf = io.BytesIO();torch.save(m.state_dict(),buf);buf.seek(0)
                copy = build_model(name).eval();copy.load_state_dict(torch.load(buf,weights_only=True))
                torch.testing.assert_close(before,copy(full),rtol=0,atol=0)
                self.assertTrue(torch.isfinite(before).all())

    def test_recipes_budget(self):
        for recipe in ('strong','moderate'):
            cmd = command('band_stem',recipe,Path('/cache'),Path('/out/run'),200)
            self.assertNotIn('--init-from', cmd)
            self.assertEqual(cmd[cmd.index('--crop')+1], '512')
            self.assertEqual(cmd[cmd.index('--epochs')+1], '200')
        r={'arm': {'mean_end_to_end_step_seconds_after_warmup':.1,'full512_eval_batch32_seconds':.1}}
        s=choose_epochs(r,30000)
        self.assertLess(s['estimated_seconds_with_margin'],30000)
        with self.assertRaises(RuntimeError):choose_epochs(r,100)

    def test_odd_evaluation_shapes(self):
        # Explicit even extension is an independent expected input. Covers
        # odd width, odd height, and both from random social resizing.
        for name in ('band_stem','phase_stem','band_stem_ibn'):
            m=build_model(name).eval()
            with torch.no_grad():
                for h,w in ((201,200),(200,201),(201,203)):
                    x=torch.randn(1,3,h,w)
                    expected=torch.nn.functional.pad(x,(0,w%2,0,h%2),mode='replicate')
                    result=m(x)
                    self.assertTrue(torch.isfinite(result).all())
                    torch.testing.assert_close(result,m(expected),rtol=0,atol=0)

    def test_release_wait_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'release.json'; done=threading.Event()
            t=threading.Thread(target=lambda:(wait_released(p),done.set()))
            t.start();time.sleep(.1);self.assertFalse(done.is_set())
            tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps({'gpus_free':True,'gpus':[4,5,6,7]}));tmp.replace(p)
            t.join(3);self.assertTrue(done.is_set())
            p.write_text('{"gpus_free":false}')
            with self.assertRaises(RuntimeError):wait_released(p)

    def test_owner_failure_cleans_only_child(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'supervised/run').mkdir(parents=True)
            status=root/'supervised/run/run.json';status.write_text('{"status":"running"}')
            owner=identity(os.getpid());self.assertTrue(original_running(root,owner))
            child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)
            other=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)
            try:
                status.write_text('{"status":"failed"}')
                with self.assertRaises(RuntimeError):supervise(child,root,owner,time.time()+10)
                stop_owned(child);self.assertIsNotNone(child.poll());self.assertIsNone(other.poll())
            finally:
                stop_owned(child);stop_owned(other)

    def test_gate_preserves_evaluator_arguments_and_exit(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); original=root/'original.py'; release=root/'released.json'
            called=root/'called.json'
            original.write_text('import sys,json\nfrom pathlib import Path\n'
                +f'Path({str(called)!r}).write_text(json.dumps(sys.argv[1:]))\nraise SystemExit(7)\n')
            cfg=root/'config.json';cfg.write_text(json.dumps({'original_evaluator':str(original),
                                                            'release_receipt':str(release)}))
            wrapper=root/'wrapper.py'
            wrapper.write_text('import sys\nsys.path.insert(0,'+repr(str(Path(__file__).resolve().parent))
                +')\nfrom extension_gate import main\nraise SystemExit(main('+repr(str(cfg))+'))\n')
            child=subprocess.Popen([sys.executable,str(wrapper),'--checkpoint','same.pt'],start_new_session=True)
            try:
                until=time.time()+5
                while not called.exists() and time.time()<until:time.sleep(.02)
                self.assertEqual(json.loads(called.read_text()),['--checkpoint','same.pt'])
                self.assertIsNone(child.poll())
                release.write_text(json.dumps({'gpus_free':True,'gpus':[4,5,6,7]}))
                self.assertEqual(child.wait(timeout=5),7)
            finally:stop_owned(child)


if __name__ == '__main__':
    unittest.main()
