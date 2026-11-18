import unittest,tempfile,json
from pathlib import Path
import numpy as np
import torch
from efficient_llm.warm_order.budget_head import fit,threshold,GainRouter
class BudgetTests(unittest.TestCase):
 def test_head_resume_calibration_and_detachment(self):
  torch.set_num_threads(1);r=np.random.default_rng(18);x=r.normal(size=(4,7,16)).astype('f4');c=r.normal(size=(4,7,2)).astype('f4')
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);a=fit(x,c,x[:2],c[:2],0,p/'full','cpu',epochs=3,batch=8);fit(x,c,x[:2],c[:2],0,p/'part','cpu',epochs=3,batch=8,stop_after=1);b=fit(x,c,x[:2],c[:2],0,p/'part','cpu',epochs=3,batch=8,resume=True)
   for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)
   self.assertEqual(json.loads((p/'full/complete.json').read_text())['calibration'],json.loads((p/'part/complete.json').read_text())['calibration'])
   h=torch.randn(2,3,16,requires_grad=True);a(h).sum().backward();self.assertIsNone(h.grad)
 def test_constant_ties_never_use_future_quota(self):
  for q in (.25,.5,.75):
   x=np.ones(100);t=threshold(x,q);self.assertFalse((x>t).any())
   x=np.arange(100.);t=threshold(x,q);self.assertLessEqual((x>t).mean(),q)
if __name__=='__main__':unittest.main()
