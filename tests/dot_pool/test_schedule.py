import unittest
from efficient_llm.dot_pool.train import lr_at
from scripts.dot_pool_queue import E2_STEPS

class LearningRateContract(unittest.TestCase):
 def test_registered_e2_schedules(self):
  for total in [953,*E2_STEPS.values()]:
   base=.002;start=int(total*.8)
   values=[lr_at(i,total,base,'E2') for i in range(1,total+1)]
   self.assertAlmostEqual(values[0],base/200)
   self.assertAlmostEqual(values[199],base)
   self.assertTrue(all(v==base for v in values[200:start]))
   self.assertTrue(all(0<=v<=base for v in values))
   self.assertTrue(all(a>=b for a,b in zip(values[start-1:],values[start:])))
   self.assertAlmostEqual(values[start],base*(total-start-1)/(total-start))
   self.assertEqual(values[-1],0.)
 def test_e1_remains_constant_after_warmup(self):
  self.assertEqual(lr_at(1000,154000,.001,'E1'),.001)
  self.assertEqual(lr_at(154000,154000,.001,'E1'),.001)
if __name__=='__main__':unittest.main()
