import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

from efficient_llm.a2.model import prune, save_student, reload_student
from efficient_llm.a2.protocol import Run, FIXED_SETS, layers_for, lr_for_update, parent_ref
from efficient_llm.diagnostic_data import sha256
from eval.a2_checkpoint import checkpoint_identity, verify_identity, load_verified
from test_model import tiny_model


class CheckpointIdentityTests(unittest.TestCase):
    def fixture(self, root):
        code=Path(os.environ['A2_OPENSCI_CODE'])
        source={name:sha256(code/name) for name in ('modeling_opensci.py','configuration_opensci.py')}
        model=tiny_model(num_hidden_layers=24)
        parent=root/'parent';save_student(model,parent)
        def lock(path,value):
            path.write_text(json.dumps(value));return dict(path=str(path),sha256=sha256(path))
        assets=lock(root/'assets.json',{'dclm/P300':dict(parent_ref('dclm','P300'),
            weights_dir=str(parent),code_dir=str(code),code_files=source,
            files={name:sha256(parent/name) for name in ('model.safetensors','config.json')})})
        run=Run('dclm','P300','S1','0.05B',3e-4)
        prune(model,layers_for(FIXED_SETS['S1']));cp=root/run.run_id/'final';save_student(model,cp)
        ref=dict(assets_sha256=assets['sha256'])
        row=dict(run.record(),status='complete',diverged=False,rerun_of=None,provenance=ref,
            coverage=dict(examples=run.steps*64,microbatches=run.steps*8,world_size=8,exact_order=True),
            roundtrip_difference=0,lr_updates=[lr_for_update(i,run.steps,run.lr) for i in range(1,run.steps+1)],
            val_teacher=2.,val_post=2.1,val_tune_post=2.1,delta_post=.1)
        args=dict(assets_lock=assets,dataset='dclm',parent='P300',checkpoint=str(cp),
            result_lock=lock(cp.parent/'result.json',row),reference_lock=lock(root/'reference.json',ref),
            sets_lock=lock(root/'sets.json',{'sets':{'dclm':{'S1':FIXED_SETS['S1']}}}))
        return args,cp,row,lock

    def test_actual_native_export_identity_and_reload(self):
        with tempfile.TemporaryDirectory() as d:
            args,cp,_,_=self.fixture(Path(d))
            identity=checkpoint_identity(**args);self.assertEqual(verify_identity(identity),identity)
            model=load_verified(identity)
            self.assertEqual(model.config.a2_kept_layers,identity['kept_layers'])
            self.assertEqual(len(model.model.layers),18)
            original=checkpoint_identity(assets_lock=args['assets_lock'],dataset='dclm',parent='P300')
            self.assertEqual(verify_identity(original),original)
            bad=copy.deepcopy(identity);bad['kept_layers']=[0]
            with self.assertRaises(ValueError):verify_identity(bad)
            weight=cp/'model.safetensors';weight.write_bytes(weight.read_bytes()+b'changed')
            with self.assertRaises(ValueError):verify_identity(identity)

    def test_native_scoring_binds_requests_and_checkpoint(self):
        from eval.a2_score import score_artifact
        from eval.a2_request_bridge import digest
        from eval.a2_likelihood import score_requests
        with tempfile.TemporaryDirectory() as d:
            args,_,_,_=self.fixture(Path(d));identity=checkpoint_identity(**args)
            requests=[dict(context=[2,3],continuation=[4,5]),dict(context=[0],continuation=[6])]
            protocol=dict(seed=42,num_fewshot=0,max_length=16)
            prepared=dict(schema=1,requests=requests,request_sha256=digest(requests),
                          protocol=protocol,protocol_sha256=digest(protocol))
            scored=score_artifact(identity,prepared,batch_size=2)
            reference=score_requests(load_verified(identity),requests,batch_size=1,max_length=16)
            for a,b in zip(scored['scores'],reference['scores']):
                self.assertAlmostEqual(a[0],b[0],places=5);self.assertEqual(a[1],b[1])
            self.assertEqual(scored['prepared_sha256'],digest(prepared))
            self.assertEqual(scored['checkpoint_identity_sha256'],digest(identity))
            bad=copy.deepcopy(prepared);bad['requests'][0]['context'].append(3)
            with self.assertRaises(ValueError):score_artifact(identity,bad)

    def test_reject_receipt_layer_and_code_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            args,cp,row,lock=self.fixture(Path(d))
            failed=dict(row,diverged=True)
            modified=dict(args,result_lock=lock(cp.parent/'result.json',failed))
            with self.assertRaises(ValueError):checkpoint_identity(**modified)
            args['result_lock']=lock(cp.parent/'result.json',row)
            config=json.loads((cp/'config.json').read_text());config['a2_kept_layers']=list(range(18))
            (cp/'config.json').write_text(json.dumps(config))
            with self.assertRaises(ValueError):checkpoint_identity(**args)
            config['a2_kept_layers']=[i for i in range(24) if i not in layers_for(FIXED_SETS['S1'])]
            (cp/'config.json').write_text(json.dumps(config))
            (cp/'modeling_opensci.py').write_text('# changed')
            with self.assertRaises(ValueError):checkpoint_identity(**args)


if __name__=='__main__':unittest.main()
