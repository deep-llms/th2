import os,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'tests/a2'))
import torch,numpy as np
from test_family_f11 import native
from efficient_llm.a2_family_f12_width import importance,calibration_indices,select_neurons
from eval.a2_family_f12 import write
class Rows:
 def __getitem__(self,i):return {'input_ids':torch.tensor([1,2,3+i%10,4,5])}
torch.distributed.init_process_group('gloo');rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
model=native();indices=calibration_indices(40,8);a,c=importance(model,Rows(),indices,rank,world);b,d=importance(model,Rows(),indices)
np.testing.assert_allclose(a,b,rtol=0,atol=1e-15);assert c==d==[40]*5;assert select_neurons(a,32)==select_neurons(b,32)
if rank==0:write('temp/f12_inputs/distributed_proof.json',dict(passed=True,ranks=world,exact_masks=True,max_abs=float(np.max(abs(a-b))),tokens_per_layer=c))
torch.distributed.destroy_process_group()
