import copy
import json
import tempfile
import unittest
from pathlib import Path

from efficient_llm.a2.protocol import lr_for_update, parent_ref
from efficient_llm.a2_stage1.protocol import Run, SPEC_SHA
from eval.a2_checkpoint import load_verified, verify_identity
from eval.a2_stage1_checkpoint import checkpoint_identity
import test_eval_checkpoint as existing


class Stage1CheckpointTest(unittest.TestCase):
    def fixture(self, root, arm='P300'):
        old, cp, row, lock = existing.CheckpointIdentityTests().fixture(root)
        run = Run(arm, '3.2B', .001)
        assets = json.loads(Path(old['assets_lock']['path']).read_text())
        if arm == 'X72':
            entry = assets.pop('dclm/P300')
            entry.update(parent_ref('dclm', 'S72'))
            assets['dclm/S72'] = entry
        assets_lock = lock(root/'assets.json', assets)
        dest = root/run.run_id
        cp.parent.rename(dest)
        cp = dest/'final'
        ref = dict(spec_sha256=SPEC_SHA, assets_sha256=assets_lock['sha256'])
        row.update(run.record(), provenance=ref,
            coverage=dict(examples=run.steps*64, microbatches=run.steps*2,
                          world_size=8, exact_order=True),
            lr_updates=[lr_for_update(i,run.steps,run.lr) for i in range(1,run.steps+1)])
        return dict(assets_lock=assets_lock, checkpoint=str(cp),
            result_lock=lock(dest/'result.json',row),
            reference_lock=lock(root/'reference.json',ref)), row, lock

    def test_native_main_exports_reload_with_original_stage1_receipts(self):
        for arm in ('P300', 'X72'):
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as d:
                args, _, _ = self.fixture(Path(d), arm)
                identity = checkpoint_identity(**args)
                self.assertEqual(verify_identity(identity), identity)
                model = load_verified(identity)
                self.assertEqual(len(model.model.layers),18)
                self.assertEqual(model.config.a2_kept_layers,list(range(16))+[22,23])
                self.assertEqual(identity['training_receipts']['teacher'],'P300')
                self.assertEqual(identity['parent'],'S72' if arm=='X72' else 'P300')
                bad=copy.deepcopy(identity);bad['training_receipts']['teacher']='P1T'
                with self.assertRaises(ValueError):verify_identity(bad)
                weight=Path(args['checkpoint'])/'model.safetensors'
                weight.write_bytes(weight.read_bytes()+b'changed')
                with self.assertRaises(ValueError):verify_identity(identity)

    def test_wrong_recipe_provenance_or_layer_map_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            args,row,lock=self.fixture(Path(d))
            for field,value in [('teacher','P1T'),('diverged',True),('status','incomplete')]:
                bad=dict(row);bad[field]=value
                altered=dict(args,result_lock=lock(Path(args['result_lock']['path']),bad))
                with self.assertRaises(ValueError):checkpoint_identity(**altered)
            args['result_lock']=lock(Path(args['result_lock']['path']),row)
            other=lock(Path(d)/'other.json',dict(row['provenance'],spec_sha256='wrong'))
            with self.assertRaises(ValueError):checkpoint_identity(**dict(args,reference_lock=other))
            path=Path(args['checkpoint'])/'config.json'
            config=json.loads(path.read_text());config['a2_kept_layers']=list(range(18))
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):checkpoint_identity(**args)


if __name__=='__main__':unittest.main()
