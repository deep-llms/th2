import random,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from efficient_llm.warm_order.horizon_probe import probe
from efficient_llm.warm_order.data import Rows
from efficient_llm.warm_order.model import Config,Model,ORDERS
class Tests(unittest.TestCase):
 def test_probe_preserves_state_rng_gradients_and_matches_reference(self):
  torch.set_num_threads(1);torch.manual_seed(4);np.random.seed(4);random.seed(4)
  c=Config(d=8,heads=1,kv_heads=1,head_dim=8,ff=16,vocab=19,length=8,prefix=2);m=Model(c).train()
  with tempfile.TemporaryDirectory() as d:
   stem=str(Path(d)/'data');x=torch.randint(19,(4,8));x.numpy().astype('uint32').tofile(stem+'.bin');st=np.zeros((4,8),dtype='u1');st[:,0]=1;st.tofile(stem+'.starts');rows=Rows(stem,8)
   m.nll(x,ORDERS[0]).mean().backward();grads={k:p.grad.clone() for k,p in m.named_parameters()};weights={k:v.clone() for k,v in m.state_dict().items()}
   tr=torch.get_rng_state().clone();nr=np.random.get_state();pr=random.getstate()
   result=probe(m,rows,np.arange(4),2,'cpu')
   self.assertTrue(m.training);torch.testing.assert_close(torch.get_rng_state(),tr,rtol=0,atol=0);self.assertEqual(random.getstate(),pr)
   after=np.random.get_state();np.testing.assert_array_equal(nr[1],after[1]);self.assertEqual(nr[2:],after[2:])
   for k,p in m.named_parameters():torch.testing.assert_close(p.grad,grads[k],rtol=0,atol=0)
   for k,v in m.state_dict().items():torch.testing.assert_close(v,weights[k],rtol=0,atol=0)
   with torch.no_grad():
    full=[float(m.nll(x,p).double().mean()) for p in ORDERS];short=[float(m.nll(x,p[:2]).double().mean()) for p in ORDERS]
   np.testing.assert_array_equal(full,result['full_order_means']);np.testing.assert_array_equal(short,result['active_order_means']);self.assertEqual(result['block_tokens'],(48+32)*4*8)
   # Failure must also restore mode and all RNGs.
   class Broken:
    def batch(self,*args):
     random.random();np.random.random();torch.rand(3);raise RuntimeError('injected')
   with self.assertRaises(RuntimeError):probe(m,Broken(),[0],1,'cpu')
   self.assertTrue(m.training);torch.testing.assert_close(torch.get_rng_state(),tr,rtol=0,atol=0);self.assertEqual(random.getstate(),pr)
