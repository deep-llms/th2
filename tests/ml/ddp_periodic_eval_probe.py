"""Two CPU ranks: periodic evaluation parity, unpadded shards, and native resume."""
import json
import os
from pathlib import Path
import sys

os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1')
import numpy as np
from test_periodic_eval import make_trainer, assert_training_equal, ROWS, torch, stopping
from resume_recipe import require_resume_training

root = Path(sys.argv[1])
helper = stopping.StoppingTests()
plain, plain_lr = make_trainer(root / 'plain', 8, False)
plain.train()
torch.distributed.barrier()
full, full_lr = make_trainer(root / 'full', 8)
full.train()
assert_training_equal(helper, plain, full, plain_lr, full_lr)
entries = [x for x in full.state.log_history if 'eval_loss' in x]
helper.assertEqual([x['step'] for x in entries], [0, 2, 4, 6, 8])
helper.assertTrue(all(x['eval_num_tokens'] == 11 and x['eval_num_sequences'] == 3 for x in entries))
# Rank 1 has no sequences, but must still join the final reductions.
metrics = full.evaluate(ROWS[:1])
helper.assertEqual(metrics['eval_num_tokens'], 5)
helper.assertEqual(metrics['eval_num_sequences'], 1)
torch.distributed.barrier()
first, first_lr = make_trainer(root / 'split', 2)
first.train()
torch.distributed.barrier()
checkpoint = root / 'split/checkpoint-2'
resumed, resumed_lr = make_trainer(root / 'split', 8)
require_resume_training(checkpoint, resumed.training_recipe, resumed.args)
resumed.train(resume_from_checkpoint=str(checkpoint))
helper.assertEqual(first.seen + resumed.seen, full.seen)
helper.assertEqual(first_lr.values + resumed_lr.values, full_lr.values)
helper.assert_nested_equal(full.model.state_dict(), resumed.model.state_dict())
helper.assert_nested_equal(full.optimizer.state_dict(), resumed.optimizer.state_dict())
helper.assert_nested_equal(full.lr_scheduler.state_dict(), resumed.lr_scheduler.state_dict())
torch.distributed.barrier()
rank = torch.distributed.get_rank()
states = [torch.load(root / name / f'checkpoint-8/rng_state_{rank}.pth', weights_only=False)
          for name in ('plain', 'full', 'split')]
for state in states[1:]:
    helper.assert_nested_equal(states[0]['cpu'], state['cpu'])
    helper.assertEqual(states[0]['python'], state['python'])
    np.testing.assert_equal(states[0]['numpy'], state['numpy'])
(root / f'rank{rank}.json').write_text(json.dumps(dict(status='passed', rank=rank,
    step=8, train_batches_weights_optimizer_scheduler_rng='exact_match',
    eval_steps=[x['step'] for x in entries],
    evaluated_sequences=3, evaluated_tokens=11, empty_rank_check='passed')) + '\n')
torch.distributed.barrier()
torch.distributed.destroy_process_group()
