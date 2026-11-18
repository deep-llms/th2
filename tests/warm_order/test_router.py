import unittest
import numpy as np
import torch
from efficient_llm.warm_order.evaluate import fit_router,summaries,paired_ci
class RouterChecks(unittest.TestCase):
    def test_learns_context_and_constant_is_not_oracle(self):
        torch.set_num_threads(1);rng=np.random.default_rng(6)
        x=rng.normal(size=(1024,16)).astype('f');y=np.ones((1024,8),dtype='f')*3
        correct=(x[:,0]>0).astype('i');y[np.arange(len(x)),correct]=2
        r,selection=fit_router(x[:768],y[:768],x[768:896],y[768:896],0,'cpu',epochs=30)
        with torch.no_grad():choices=r(torch.tensor(x[896:])).argmax(-1).numpy()
        s=summaries(y[896:],choices,int(y[:768].mean(0).argmin()))
        self.assertLess(s['learned'].mean(),s['constant'].mean()-.2)
        self.assertGreaterEqual(s['learned'].mean(),s['oracle'].mean())
        z=paired_ci(s['learned']-s['constant'],np.arange(128)//2,reps=200)
        self.assertLess(z['hi'],0)
        self.assertEqual(selection['dev_nll'],min(v['dev_selected_nll'] for v in selection['trace']))
if __name__=='__main__':unittest.main()
