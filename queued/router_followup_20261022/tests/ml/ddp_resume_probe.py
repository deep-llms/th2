"""Two-process CPU resume parity, including recipe checks and per-rank RNG.

Run: torchrun --standalone --nproc_per_node=2 tests/ml/ddp_resume_probe.py OUTPUT
"""
import json
import os
from pathlib import Path
import sys

os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1')
from test_stopping import StoppingTests, torch, require_resume_training
from train import check_resume_steps

root = Path(sys.argv[1])
helper = StoppingTests()
full, full_lr = helper.trainer(root/'full', 8)
full.train()
torch.distributed.barrier()
first, first_lr = helper.trainer(root/'split', 2)
first.train()
torch.distributed.barrier()
checkpoint = root/'split/checkpoint-2'
resumed, resumed_lr = helper.trainer(root/'split', 8)
check_resume_steps(checkpoint, resumed.args)
require_resume_training(checkpoint, resumed.training_recipe, resumed.args)
resumed.train(resume_from_checkpoint=str(checkpoint))
torch.distributed.barrier()
helper.assertEqual(first.seen + resumed.seen, full.seen)
helper.assertEqual(first_lr.values + resumed_lr.values, full_lr.values)
helper.assert_nested_equal(full.model.state_dict(), resumed.model.state_dict())
helper.assert_nested_equal(full.optimizer.state_dict(), resumed.optimizer.state_dict())
helper.assert_nested_equal(full.lr_scheduler.state_dict(), resumed.lr_scheduler.state_dict())
rank = torch.distributed.get_rank()
a = torch.load(root/f'full/checkpoint-8/rng_state_{rank}.pth', weights_only=False)
b = torch.load(root/f'split/checkpoint-8/rng_state_{rank}.pth', weights_only=False)
helper.assert_nested_equal(a['cpu'], b['cpu'])
(root/f'resume_rank{rank}.json').write_text(json.dumps({'rank':rank,'status':'passed',
    'world_size':torch.distributed.get_world_size(), 'step':resumed.state.global_step,
    'batches_lr_model_optimizer_scheduler_rng':'exact_match'})+'\n')
torch.distributed.barrier()
torch.distributed.destroy_process_group()
