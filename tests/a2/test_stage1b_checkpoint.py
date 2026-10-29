import copy,json,tempfile,unittest
from pathlib import Path
from tests.a2.test_stage1_checkpoint import Stage1CheckpointTest
from efficient_llm.a2.protocol import parent_ref
from efficient_llm.a2_stage1b.protocol import Run,REPLICATION_SPEC_SHA
from eval.a2_stage1_checkpoint import checkpoint_identity
from eval.a2_checkpoint import verify_identity,load_verified

class Stage1bCheckpointTests(unittest.TestCase):
 def test_fwe_main_identity_native_reload_and_spec_binding(self):
  for arm in ('P300','X72'):
   with self.subTest(arm=arm),tempfile.TemporaryDirectory() as d:
    root=Path(d);args,row,lock=Stage1CheckpointTest().fixture(root,arm)
    assets=json.loads(Path(args['assets_lock']['path']).read_text());parent='S72' if arm=='X72' else 'P300'
    entry=assets.pop('dclm/'+parent);entry.update(parent_ref('fwe',parent));assets['fwe/'+parent]=entry
    args['assets_lock']=lock(root/'assets.json',assets)
    run=Run(arm,'3.2B',.001);old=Path(args['checkpoint']).parent;dest=root/run.run_id;old.rename(dest)
    ref=dict(row['provenance'],assets_sha256=args['assets_lock']['sha256'],dataset='fwe',stage1b_spec_sha256=REPLICATION_SPEC_SHA)
    row.update(run.record(),provenance=ref)
    args.update(checkpoint=str(dest/'final'),result_lock=lock(dest/'result.json',row),reference_lock=lock(root/'reference.json',ref))
    identity=checkpoint_identity(**args);self.assertEqual(identity['dataset'],'fwe');self.assertEqual(verify_identity(identity),identity)
    self.assertEqual(len(load_verified(identity).model.layers),18)
    for change in ({'dataset':'dclm'},{'stage1b_spec_sha256':'wrong'}):
     bad=dict(ref,**change);altered=dict(args,reference_lock=lock(root/'bad.json',bad))
     with self.assertRaises(ValueError):checkpoint_identity(**altered)
    bad=copy.deepcopy(identity);bad['dataset']='dclm'
    with self.assertRaises(ValueError):verify_identity(bad)
