import json
import random
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from efficient_llm.warm_order.data import sha
from efficient_llm.warm_order.model import Config
from efficient_llm.warm_order.soft_route import Model as SoftModel, ARMS
from efficient_llm.warm_order.soft_exploration import Model as ExposureModel
from efficient_llm.warm_order.exposure_recovery import load_parent, optimizer_for, restore_rng


class RecoveryTests(unittest.TestCase):
    def fixture(self, root, arm, exposure, seed=0):
        torch.set_num_threads(1)
        torch.manual_seed(13+seed)
        np.random.seed(13+seed)
        random.seed(13+seed)
        cfg = Config(d=8, heads=1, kv_heads=1, head_dim=8, ff=16, vocab=19, length=8, prefix=2)
        c = dict(model=asdict(cfg), arm=arm, seed=seed, steps=3,
                 batch=2, lr=.0001, router_lr=.001, canonical_probability=.999)
        if exposure:
            c['exploration_schedule'] = 'floor0.4-ramp500-hold500-decay500-zero500'
        model = (ExposureModel if exposure else SoftModel)(cfg, ARMS[arm], .999)
        optimizer = optimizer_for(model, c['lr'], c['router_lr'])
        for g in optimizer.param_groups:
            g['lr'] = g['peak'] * .1
        x = torch.randint(19, (2, 8))
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            if exposure:
                model.set_training_update(step)
            model.nll(x).mean().backward()
            optimizer.step()
        recipe = dict(config=c, source={}, inputs={})
        state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), recipe=recipe,
                     step=3, torch_rng=torch.get_rng_state(), cuda_rng=[],
                     numpy_rng=np.random.get_state(), python_rng=random.getstate(),
                     block_tokens=3*2*8*(6 if arm=='C' else 18))
        torch.save(state, root/'fullstate.pt')
        torch.save(dict(config=asdict(cfg), arm=arm, seed=seed, canonical_probability=.999,
                        model=model.state_dict()), root/'weights.pt')
        (root/'recipe.json').write_text(json.dumps(recipe))
        (root/'complete.json').write_text(json.dumps(dict(step=3, arm=arm, seed=seed,
            fullstate_sha256=sha(root/'fullstate.pt'), weights_sha256=sha(root/'weights.pt'))))
        hashes = {f: sha(root/f) for f in ('weights.pt', 'fullstate.pt', 'recipe.json', 'complete.json')}
        if exposure:
            model.disable_exploration()
        return model, optimizer, hashes, x

    def test_all_five_schemas_preserve_forward_moments_next_update_rng(self):
        for arm, exposure in (('C', False), ('G', False), ('S', False), ('G', True), ('S', True)):
            with self.subTest(arm=arm, exposure=exposure), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                a, ao, hashes, x = self.fixture(root, arm, exposure)
                name = arm + ('F' if exposure else '') + '-s0'
                b, bo, state = load_parent(root, hashes, name, expected_steps=3)
                self.assertEqual(ao.state_dict()['param_groups'], bo.state_dict()['param_groups'])
                for hard in (False, True):
                    torch.testing.assert_close(a.nll(x, hard), b.nll(x, hard), rtol=0, atol=0)
                restore_rng(state)
                sample = (torch.rand(5), np.random.rand(5), random.random())
                restore_rng(state)
                torch.testing.assert_close(sample[0], torch.rand(5), rtol=0, atol=0)
                np.testing.assert_array_equal(sample[1], np.random.rand(5))
                self.assertEqual(sample[2], random.random())
                for model, opt in ((a, ao), (b, bo)):
                    opt.zero_grad(set_to_none=True)
                    model.nll(x).mean().backward()
                    opt.step()
                for k, v in a.state_dict().items():
                    torch.testing.assert_close(v, b.state_dict()[k], rtol=0, atol=0)
                for k, state_a in ao.state_dict()['state'].items():
                    for key, value in state_a.items():
                        torch.testing.assert_close(value, bo.state_dict()['state'][k][key], rtol=0, atol=0)
                if exposure:
                    self.assertFalse(b.use_exploration)

    def test_refuse_wrong_identity_and_changed_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, hashes, _ = self.fixture(root, 'S', True)
            for name, steps in (('SF-s1', 3), ('S-s0', 3), ('SF-s0', 4)):
                with self.assertRaises(ValueError):
                    load_parent(root, hashes, name, expected_steps=steps)
            with (root/'weights.pt').open('ab') as f:
                f.write(b'tamper')
            with self.assertRaises(ValueError):
                load_parent(root, hashes, 'SF-s0', expected_steps=3)

    def test_refuse_rehashed_optimizer_reset_lr(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, hashes, _ = self.fixture(root, 'G', False)
            state = torch.load(root/'fullstate.pt', weights_only=False)
            state['optimizer']['param_groups'][0]['lr'] *= 10
            torch.save(state, root/'fullstate.pt')
            hashes['fullstate.pt'] = sha(root/'fullstate.pt')
            done = json.loads((root/'complete.json').read_text())
            done['fullstate_sha256'] = hashes['fullstate.pt']
            (root/'complete.json').write_text(json.dumps(done))
            hashes['complete.json'] = sha(root/'complete.json')
            with self.assertRaises(ValueError):
                load_parent(root, hashes, 'G-s0', expected_steps=3)

    def test_refuse_rehashed_missing_adam_moments(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, hashes, _ = self.fixture(root, 'S', False)
            state = torch.load(root/'fullstate.pt', weights_only=False)
            state['optimizer']['state'] = {}
            torch.save(state, root/'fullstate.pt')
            hashes['fullstate.pt'] = sha(root/'fullstate.pt')
            done = json.loads((root/'complete.json').read_text())
            done['fullstate_sha256'] = hashes['fullstate.pt']
            (root/'complete.json').write_text(json.dumps(done))
            hashes['complete.json'] = sha(root/'complete.json')
            with self.assertRaises(ValueError):
                load_parent(root, hashes, 'S-s0', expected_steps=3)

    def test_real_phase_trainer_exact_resume_all_schemas(self):
        from efficient_llm.warm_order.exposure_recovery_train import train
        for arm, exposure in (('C', False), ('G', False), ('S', False), ('G', True), ('S', True)):
            with self.subTest(arm=arm, exposure=exposure), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);parent=root/'parent';parent.mkdir()
                _, _, hashes, _=self.fixture(parent,arm,exposure)
                stem=str(root/'rows')
                np.random.default_rng(123).integers(0,19,(12,8),dtype=np.uint32).tofile(stem+'.bin')
                starts=np.zeros((12,8),dtype='u1');starts[:,0]=1;starts.tofile(stem+'.starts')
                np.save(root/'ids.npy',np.arange(12))
                c=dict(name=arm+('F' if exposure else '')+'-s0',parent=str(parent),parent_hashes=hashes,
                    parent_steps=3,steps=6,batch=2,threads=1,log_every=1,save_every=2,
                    schedule='inherit-terminal-lr-constant-no-floor',data=stem,indices=str(root/'ids.npy'))
                c['expected_inputs']={p:sha(p) for p in (stem+'.bin',stem+'.starts',c['indices'])}
                a,ao=train(c,root/'whole','cpu')
                train(c,root/'resumed','cpu',stop_after=2)
                b,bo=train(c,root/'resumed','cpu',resume=True)
                for k,v in a.state_dict().items():
                    torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)
                self.assertEqual(ao.state_dict()['param_groups'],bo.state_dict()['param_groups'])
                for k,va in ao.state_dict()['state'].items():
                    for key,value in va.items():
                        torch.testing.assert_close(value,bo.state_dict()['state'][k][key],rtol=0,atol=0)
                    self.assertEqual(int(va['step']),9)
                sa=torch.load(root/'whole/fullstate.pt',weights_only=False)
                sb=torch.load(root/'resumed/fullstate.pt',weights_only=False)
                torch.testing.assert_close(sa['torch_rng'],sb['torch_rng'],rtol=0,atol=0)
                self.assertEqual(sa['counts'],sb['counts'])
                self.assertEqual(sa['block_tokens'],6*2*8*(6 if arm=='C' else 18))
                self.assertTrue(all(x['exploration_epsilon']==0 for x in map(json.loads,(root/'whole/trace.jsonl').read_text().splitlines())))
                with self.assertRaises(ValueError):train(c,root/'resumed','cpu',resume=True)


if __name__ == '__main__':
    unittest.main()
