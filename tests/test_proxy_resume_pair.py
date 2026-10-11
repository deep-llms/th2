import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from deep_kv.__main__ import jobs
from scripts.proxy_resume_pair import ARMS, make, report
from scripts.stage_deep_kv_resume import digest


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root):
    source = root/'source'
    original = jobs(Path(__file__).resolve().parents[1]/'proxy_heads.b200.json',
                    stop_after=10000, arms=ARMS, seeds=[1042])
    for job in original['jobs']:
        if 'gpus' in job:
            argv = job['argv']; argv[argv.index('--save_total_limit')+1] = '2'
    for arm in ARMS:
        write(source/f'seed-1042/{arm}/train_config.json', dict(
            pilot=dict(arm=arm), world_size=8, train_fingerprint='fixed-training',
            eval_fingerprint='fixed-validation', training=dict(
                seed=1042, data_seed=1042, max_steps=28600, warmup_steps=1430,
                save_steps=250, save_total_limit=2, ignore_data_skip=False)))
        write(source/f'seed-1042/{arm}/result.json', dict(arm=arm,
            global_step=10000, schedule_steps=28600,
            training_cost=dict(optimizer_steps=10000, runtime_seconds=21000),
            evaluation=dict(eval_lm_loss=3.)))
        write(source/f'seed-1042-validate-{arm}.json', dict(status='passed'))
    write(source/'run.json', dict(status='ok'))
    write(source/'jobs.snapshot.json', original)
    return source, original


class ResumePairTests(unittest.TestCase):
    def test_both_commands_change_only_cutoff_retention_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source, original = fixture(root)
            before = {str(p): p.read_bytes() for p in source.rglob('*') if p.is_file()}
            plan = make(source, root/'new')
            self.assertEqual(plan['stop_after'], 20000)
            self.assertFalse(plan['cooldown'])
            generated = json.loads((root/'new/jobs.json').read_text())['jobs']
            self.assertEqual([j['name'] for j in generated], [
                'stage-original-checkpoints', 'seed-1042-resume-A', 'validate-A',
                'seed-1042-resume-P6-iso', 'validate-P6-iso', 'validate-continuations'])
            for arm, job in zip(ARMS, (generated[1], generated[3])):
                old = next(j for j in original['jobs'] if j['name'] == f'seed-1042-arm-{arm}')
                self.assertEqual(job['gpus'], list(range(8)))
                self.assertEqual(job['argv'][-2:], ['--resume_from_checkpoint',
                    f'{{run_dir}}/seed-1042/{arm}/checkpoint-10000'])
                restored = job['argv'][:-2].copy()
                for flag in ('--stop_after', '--save_total_limit'):
                    restored[restored.index(flag)+1] = old['argv'][old['argv'].index(flag)+1]
                self.assertEqual(restored, old['argv'])
            self.assertEqual(before, {str(p): p.read_bytes() for p in source.rglob('*') if p.is_file()})

    def test_rejects_cooldown_or_wrong_seed_source(self):
        for change in ('cooldown', 'seed'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); source, _ = fixture(root)
                path = source/'seed-1042/P6-iso/train_config.json'
                cfg = json.loads(path.read_text())
                if change == 'cooldown': cfg['pilot']['cooldown_start'] = 10000
                else: cfg['training']['data_seed'] = 42
                write(path, cfg)
                with self.assertRaises(AssertionError): make(source, root/'new')
                self.assertFalse((root/'new').exists())

    def test_final_retention_source_integrity_and_cumulative_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source, _ = fixture(root); output = root/'finished'
            receipt = dict(status='ok', source_step=10000, arms={})
            for arm in ARMS:
                old = source/f'seed-1042/{arm}'; current = output/f'seed-1042/{arm}'
                cfg = json.loads((old/'train_config.json').read_text())
                cfg['training']['save_total_limit'] = 0
                write(current/'train_config.json', cfg)
                result = json.loads((old/'result.json').read_text())
                result['global_step'] = 20000
                result['training_cost']['runtime_seconds'] = 22000
                write(current/'result.json', result)
                write(current/'resume-transition-test.json', dict(changes={
                    'save_total_limit': dict(before=2, after=0)}))
                write(output/f'validate-{arm}.json', dict(status='passed'))
                for step in range(10000, 20001, 250): (current/f'checkpoint-{step}').mkdir()
                for folder in (old, current):
                    checkpoint = folder/'checkpoint-10000'; checkpoint.mkdir(exist_ok=True)
                    (checkpoint/'model.safetensors').write_bytes(b'original-weights')
                file = old/'checkpoint-10000/model.safetensors'
                receipt['arms'][arm] = dict(source=str(old), files={
                    'model.safetensors': dict(bytes=file.stat().st_size, sha256=digest(file))})
            write(output/'seed-1042/resume_inputs.json', receipt)
            # Matched-arm validation is separately covered by the existing report tests.
            with patch('deep_kv.report.report') as compare:
                value = report(source, output)
                compare.assert_called_once()
            self.assertTrue(all(r['cumulative_runtime_seconds'] == 43000 for r in value['results']))
            self.assertTrue(all(len(r['retained_checkpoints']) == 41 for r in value['results']))
            (output/'continuation-summary.json').unlink()
            (source/'seed-1042/P6-iso/checkpoint-10000/model.safetensors').write_bytes(b'corrupt')
            with patch('deep_kv.report.report'), self.assertRaises(AssertionError): report(source, output)


if __name__ == '__main__':
    unittest.main()
