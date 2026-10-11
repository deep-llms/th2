"""Synchronization options must leave optimization and logged values exact."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load_file

from deep_kv.proxy_training import ProxyTrainer
from tests.test_proxy_memory import config as model_config, model, batch
from tests.test_proxy_training import proxy_fixture
from tests.test_train import invoke
from scripts.check_optimized_resume import state_hash


class SyncTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_diagnostic_skip_preserves_updates_and_rejects_bad_moments(self):
        a, b = model('P6-iso'), model('P6-iso')
        shape = a.mu.shape
        torch.manual_seed(24)
        sums, squares, counts = torch.randn(shape), torch.rand(shape), torch.ones(shape[0])*8
        expected = a.update_statistics(sums, squares, counts)
        self.assertEqual(b.update_statistics(sums, squares, counts, diagnostics=False), {})
        self.assertTrue(expected)
        self.assertEqual(state_hash(a.state_dict()), state_hash(b.state_dict()))
        for invalid in (torch.zeros_like(counts), counts*float('nan')):
            with self.assertRaises(ValueError): b.update_statistics(sums, squares, invalid, diagnostics=False)

    def test_real_training_and_resume_exact(self):
        for arm in ('A', 'P6-iso'):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config = proxy_fixture(root);model_config().save_pretrained(root/'model')
                config.update(arm=arm, proxy_groups=2, logging_steps=2, logging_first_step=True,
                              eval_steps=3, save_total_limit=0, checkpoint_layers=False,
                              checkpoint_aux=False, checkpoint_lm=False)
                for fast in (False, True):
                    output = root/str(fast)
                    args = {**config, 'output_dir':str(output), 'proxy_sync_optimization':fast}
                    invoke(root,{**args, 'stop_after':2})
                    invoke(root,args)
                for name in ('model.safetensors','optimizer.pt','scheduler.pt','rng_state.pth'):
                    def load(fast):
                        p=root/str(fast)/'checkpoint-3'/name
                        return load_file(p) if name.endswith('safetensors') else torch.load(p,weights_only=False)
                    self.assertEqual(state_hash(load(False)),state_hash(load(True)),arm+'/'+name)
                def logs(fast):
                    rows=json.loads((root/str(fast)/'trainer_state.json').read_text())['log_history']
                    return [{k:v for k,v in row.items() if k.startswith(('step_', 'target_', 'proxy_'))}
                            for row in rows if 'loss' in row]
                self.assertEqual(logs(False),logs(True))

    def test_nonfinite_objective_still_fails(self):
        from transformers import TrainingArguments
        with tempfile.TemporaryDirectory() as tmp:
            m=model('P6-iso')
            trainer=ProxyTrainer(model=m,args=TrainingArguments(output_dir=tmp,use_cpu=True,report_to=[]))
            trainer.sync_optimization=True;trainer.args.logging_nan_inf_filter=False
            trainer.is_in_train=True
            with torch.no_grad(): next(m.backbone.parameters()).fill_(float('nan'))
            c=batch()
            with self.assertRaisesRegex(ValueError,'Nonfinite proxy'):
                trainer.compute_loss(m,dict(input_ids=c.input_ids,attention_mask=c.valid,
                    position_ids=c.position_ids,segments=c.segments,labels=c.labels))

    def test_timing_validator_uses_all_ranks_and_refuses_mismatched_data(self):
        from scripts.p6_sync_trial import summary
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for arm in ('A','P6'):
                for mode in ('reference','optimized'):
                    for repeat in ('','-repeat'):
                        folder=root/f'{arm}-{mode}{repeat}'/'seed-1042'/('A' if arm=='A' else 'P6-iso')
                        folder.mkdir(parents=True)
                        (folder/'train_config.json').write_text(json.dumps(dict(train_fingerprint='same')))
                        (folder/'result.json').write_text(json.dumps(dict(global_step=100,evaluation=dict(eval_lm_loss=9.))))
                        for rank in range(8):
                            (folder/f'sync-audit-rank{rank}.json').write_text(json.dumps(dict(batches=[str(rank)]*8)))
                            (folder/f'step-profile-rank{rank}.json').write_text(json.dumps(dict(peak_gib=100,
                                steps=[dict(step=i,interval_seconds=2+rank/10) for i in range(1,100)])))
            summary(root)
            self.assertAlmostEqual(json.loads((root/'summary.json').read_text())['mean_seconds']['A-reference'],2.7)
            (root/'summary.json').unlink()
            (root/'A-optimized/seed-1042/A/sync-audit-rank7.json').write_text(json.dumps(dict(batches=['wrong']*8)))
            with self.assertRaisesRegex(ValueError,'Data mismatch'):summary(root)


if __name__=='__main__':unittest.main()
