import copy
from pathlib import Path
import tempfile
import unittest

from followup_study import ARMS, command, parent_path, validate_fit


class FollowupTests(unittest.TestCase):
    def test_queue_uses_only_existing_training_parents_and_fixed_split(self):
        self.assertEqual([a['gpu'] for a in ARMS], list(range(5)))
        self.assertEqual(len({a['name'] for a in ARMS}), 5)
        self.assertEqual({a['seed'] for a in ARMS}, {1})
        for arm in ARMS:
            cmd = command(arm, Path('/cache'), Path('/fresh')/arm['name'])
            get = lambda flag: cmd[cmd.index(flag)+1]
            self.assertEqual(get('--init-from'), str(parent_path(arm)))
            self.assertEqual(get('--epochs'), '40')
            self.assertEqual(get('--split-seed'), '42')
            self.assertEqual(get('--val-frac'), '0.1')
            self.assertEqual(get('--crop'), '512')
            self.assertEqual(get('--batch-size'), '16')
            self.assertEqual(get('--views'), '2')
            self.assertEqual(get('--sched'), 'cosine')
            self.assertEqual(int(get('--data-seed')), 42+arm['seed'])
            self.assertNotIn('--kd-teacher', cmd)
            self.assertNotIn('--resume', cmd)
            if arm['model'] in ('grip_nodown_ibna', 'grip_nodown_ibnb'):
                self.assertIn('grip_nodown_base', str(parent_path(arm)))
                self.assertEqual(float(get('--lr')), 1e-4)
            else:
                self.assertIn(arm['model']+'_full512', str(parent_path(arm)))
                self.assertEqual(float(get('--lr')), 3e-5)

    def test_smoke_cannot_become_scientific_parent(self):
        arm = ARMS[0]
        smoke = command(arm, Path('/cache'), Path('/fresh/entrypoint'), 1)
        actual = command(arm, Path('/cache'), Path('/fresh/actual'), 40)
        self.assertEqual(smoke[smoke.index('--init-from')+1],
                         actual[actual.index('--init-from')+1])

    def test_real_checkpoint_seed_identity_and_finite_epoch_log(self):
        import torch
        from models import build_model, count_trainable
        torch.set_num_threads(1)
        arm = ARMS[0]
        model = build_model(arm['model'])
        args = dict(model=arm['model'], epochs=1, seed=arm['seed'], data_seed=arm['data_seed'],
                    split_seed=42, val_frac=.1, train_split='train', views=2, jsd_lambda=0.,
                    clip_grad=0., norm='imagenet', init_from=str(parent_path(arm)), kd_teacher=None,
                    crop=512, rrc_size=512, batch_size=16, lr=arm['lr'], optim='adam',
                    sched='cosine', warmup_epochs=0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            torch.save(dict(model=model.state_dict(), args=args, epoch=1,
                            val_acc=.5, n_params_training=count_trainable(model)), root/'last.pt')
            header = 'epoch,lr,train_loss,train_acc,val_loss,val_acc,sec\n'
            (root/'log.csv').write_text(header+'1,0,1,.5,1,.5,123.4\n')
            self.assertEqual(validate_fit(arm, root, 1)['last_epoch_seconds'], 123.4)
            wrong = copy.deepcopy(arm); wrong['seed'] = 99
            with self.assertRaisesRegex(ValueError, 'identity'):
                validate_fit(wrong, root, 1)
            (root/'log.csv').write_text(header+'1,0,nan,.5,1,.5,123.4\n')
            with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                validate_fit(arm, root, 1)


if __name__ == '__main__':
    unittest.main()
