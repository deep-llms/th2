import unittest
import numpy as np
from efficient_llm.warm_order.exposure_recovery_data import select_rows


class ExtensionTests(unittest.TestCase):
    def test_skip_document_straddling_prefix_and_internal_boundaries(self):
        ids=np.arange(20*8,dtype=np.uint32).reshape(20,8)
        starts=np.zeros((20,8),dtype=np.uint8);starts[0,0]=1
        starts[6,3]=1;starts[9,0]=1;starts[12,7]=1
        chosen,proof=select_rows(ids,starts,4*8,20*8,8,length=8)
        self.assertTrue((chosen>=7).all())
        self.assertFalse(np.isin(chosen,[6,12]).any())
        self.assertEqual(proof['first_new_document_token'],6*8+3)
        np.testing.assert_array_equal(chosen,select_rows(ids,starts,32,160,8,8)[0])
        with self.assertRaises(ValueError):select_rows(ids,starts,32,160,20,8)
        with self.assertRaises(ValueError):select_rows(ids,np.zeros_like(starts),32,160,1,8)

if __name__=='__main__':unittest.main()
