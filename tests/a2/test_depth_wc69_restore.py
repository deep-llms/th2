import hashlib
from pathlib import Path
import tempfile
import unittest
import copy
from unittest.mock import patch
from scripts.a2_depth_wc69_restore import restore_files


class RestoreTests(unittest.TestCase):
    def test_exact_copy_and_existing_reuse(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);src=root/'source';src.write_bytes(b'registered bytes')
            item=dict(source=str(src),destination=str(root/'dest'),sha256=hashlib.sha256(src.read_bytes()).hexdigest())
            first=restore_files([item]);second=restore_files([item])
            self.assertEqual(first[0]['action'],'copied and verified')
            self.assertEqual(second[0]['action'],'reuse exact existing bytes')
            self.assertEqual(src.read_bytes(),(root/'dest').read_bytes())

    def test_mismatch_stops_before_any_copy_and_preserves_existing(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);src=root/'source';src.write_bytes(b'original')
            good=dict(source=str(src),destination=str(root/'new'),sha256=hashlib.sha256(src.read_bytes()).hexdigest())
            old=root/'existing';old.write_bytes(b'partial')
            bad=dict(good,destination=str(old))
            with self.assertRaises(ValueError):restore_files([good,bad])
            self.assertFalse((root/'new').exists());self.assertEqual(old.read_bytes(),b'partial')
            with self.assertRaises(ValueError):restore_files([good,dict(good,destination=str(root/'other'),sha256='wrong')])
            self.assertFalse((root/'new').exists())

    def test_new_plan_preserves_history_and_recipe_and_rejects_changed_restored_asset(self):
        from scripts import a2_depth_wc69_plan as plan
        from efficient_llm.a2_depth.common import read,write,SPEC_SHA
        from scripts.a2_quant_io import lock
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);asset=root/'asset';asset.write_bytes(b'original')
            restored=write(root/'restore.json',dict(status='verified',spec_sha256=SPEC_SHA,
                files=[dict(destination=str(asset),sha256=hashlib.sha256(asset.read_bytes()).hexdigest())]))
            reconstructed=write(root/'data.json',dict(stream=str(root/'new-stream')))
            base=read('resources/a2_depth_partc_plan_20261101.json');original=copy.deepcopy(base)
            with patch.object(plan,'rebuilt_manifest',return_value='new-manifest'):
                cfg=plan.assemble(base,reconstructed,restored,{'proof':'fixture'},{'host':plan.HOST},{'503':'fixture'})
                self.assertEqual(base,original)
                self.assertEqual(cfg['arm_order'],['X72','P300','P1T'])
                self.assertEqual(cfg['heal_streams']['dclm'],dict(path=str(root/'new-stream'),sha256='new-manifest'))
                for key in ('stream_history','stage1_dclm_reference','reference_nll','assets','prepared','final_sequences','resamples','seed','part_ab_note','predecessor'):
                    self.assertEqual(cfg[key],base[key])
                asset.write_bytes(b'changed')
                with self.assertRaises(ValueError):plan.assemble(base,reconstructed,restored,{}, {'host':plan.HOST},{})
