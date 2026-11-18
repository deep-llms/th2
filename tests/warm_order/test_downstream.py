import unittest
import torch
from torch.nn import functional as F
from efficient_llm.warm_order.model import Model,Config
from efficient_llm.warm_order.local_policy import LocalRouter,LOCAL_ORDERS
from efficient_llm.warm_order.downstream import score,prepare_rows

class ScorerTests(unittest.TestCase):
 def setUp(self):
  torch.set_num_threads(1);torch.manual_seed(7)
  self.m=Model(Config(d=16,heads=2,kv_heads=1,head_dim=8,ff=24,vocab=31,prefix=4,length=16)).eval()
  self.router=LocalRouter(16).eval()
  with torch.no_grad():self.router.net[-1].bias.copy_(torch.tensor([0.,1.,0.,0.]))
  self.req=[{'context':[1,2,3,4,5], 'continuation':[6,7]}, {'context':[2], 'continuation':[3,4,5]}, {'context':[1,2,3,4,5], 'continuation':[8]}, {'context':list(range(1,15)), 'continuation':[2,3,4,5]}]
 def test_reference_and_singleton_padding(self):
  got,details=score(self.m,self.router,self.req,LOCAL_ORDERS,batch_size=4)
  one,_=score(self.m,self.router,self.req,LOCAL_ORDERS,batch_size=1)
  for i,row in enumerate(prepare_rows(self.req,16)):
   x=torch.tensor([row['tokens']]);h=self.m.hidden(x,LOCAL_ORDERS[1]);a=len(row['context'])-1
   z=F.linear(h[0,a:],self.m.embed.weight).float();target=torch.tensor(row['target'])
   expected=float(z.log_softmax(-1).gather(1,target[:,None]).sum())
   self.assertAlmostEqual(got[i][0],expected,places=5);self.assertAlmostEqual(got[i][0],one[i][0],places=5)
   self.assertEqual(got[i][1],bool(torch.equal(z.argmax(-1),target)))
  self.assertEqual(details['choices'],[1]*4)
 def test_target_content_cannot_change_route(self):
  seen=[];original=self.m.features
  def captured(x):
   seen.extend(x.tolist());return original(x)
  self.m.features=captured
  _,a=score(self.m,self.router,self.req,LOCAL_ORDERS)
  first=list(seen);seen.clear()
  changed=[dict(r,continuation=[30]*len(r['continuation'])) for r in self.req]
  _,b=score(self.m,self.router,changed,LOCAL_ORDERS)
  self.assertEqual(a['choices'],b['choices']);self.assertEqual(a['context_routes'],b['context_routes']);self.assertEqual(first,seen)
  self.assertNotIn(30,[t for row in seen for t in row])
 def test_common_context_truncation(self):
  req=[dict(context=list(range(1,15)),continuation=[1]*n) for n in [1,7]]
  rows=prepare_rows(req,16);self.assertEqual(rows[0]['context'],rows[1]['context']);self.assertEqual(len(rows[0]['context']),10)
  self.assertEqual(len(rows[1]['tokens']),16)
 def test_all_answer_tokens_short_context(self):
  req=[dict(context=[1],continuation=[2,3,4])]
  got,_=score(self.m,None,req,LOCAL_ORDERS)
  x=torch.tensor([[1,2,3]]);z=F.linear(self.m.hidden(x,LOCAL_ORDERS[0]),self.m.embed.weight).float()
  expected=-F.cross_entropy(z.reshape(-1,31),torch.tensor([2,3,4]),reduction='sum')
  self.assertAlmostEqual(got[0][0],float(expected),places=5)
 def test_mixed_routes_grouping(self):
  class Varying(torch.nn.Module):
   def forward(self,x):
    ix=(x[:,0]>0).long();return torch.nn.functional.one_hot(ix,4).float()
  self.m.features=lambda x:torch.stack([torch.tensor(float(row[0]%2)*2-1) for row in x])[:,None].expand(-1,16)
  got,d=score(self.m,Varying(),self.req,LOCAL_ORDERS,batch_size=4)
  self.assertEqual(set(d['choices']),{0,1})
  for i,r in enumerate(prepare_rows(self.req,16)):
   z=F.linear(self.m.hidden(torch.tensor([r['tokens']]),LOCAL_ORDERS[d['choices'][i]])[0,len(r['context'])-1:],self.m.embed.weight).float()
   want=-F.cross_entropy(z,torch.tensor(r['target']),reduction='sum')
   self.assertAlmostEqual(got[i][0],float(want.detach()),places=5)
 def test_invalid_and_empty_context(self):
  for target in [[],[1]*17]:
   with self.assertRaises(ValueError):prepare_rows([dict(context=[1],continuation=target)],16)
  self.assertEqual(prepare_rows([dict(context=[],continuation=[2])],16)[0]['context'],[0])
if __name__=='__main__':unittest.main()
