import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from eval.a2_plan import prepare_identities
from eval.a2_run import write,lock
from eval.a2_checkpoint import verify_identity
from test_eval_checkpoint import CheckpointIdentityTests


class EvalPlanTests(unittest.TestCase):
    def fixture(self,root):
        args,cp,row,_=CheckpointIdentityTests().fixture(root)
        stage=dict(reference=args['reference_lock'],results=write(root/'results.json',[row]),
            plan=write(root/'plan.json',{'pending':[row]}),
            complete=write(root/'complete.json',dict(status='complete',scientific_complete=True)),
            supervisor=write(root/'supervisor.json',dict(training_status='ok',training_returncode=0)),
            run_complete=write(root/'run.json',dict(status='ok')),
            burn=write(root/'burn.json',dict(collective_progress_verified=True)))
        history=write(root/'history.json',dict(sources=[]))
        stage_lock=write(root/'stage.json',stage)
        selected={'fixture':dict(kind='healed',dataset='dclm',parent='P300',run_id=row['run_id'])}
        return args,cp,row,stage,stage_lock,history,selected

    def test_actual_native_checkpoint_and_original_provenance_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);args,cp,row,stage,sl,hl,selected=self.fixture(root)
            with patch('eval.a2_plan.load_history',return_value={'rows':[]}), \
                 patch('eval.a2_plan.remaining_runs',return_value={'pending':[row]}), \
                 patch('eval.a2_plan.select_initial_screen',return_value=selected):
                result=prepare_identities(sl,hl,args['assets_lock'],args['sets_lock'],root/'out')
            identity=json.loads(Path(result['identities']['fixture']['path']).read_text())
            self.assertEqual(verify_identity(identity),identity)
            self.assertEqual(json.loads(Path(identity['training_receipts']['reference']['path']).read_text()),row['provenance'])
            self.assertEqual(json.loads((cp.parent/'result.json').read_text()),row)

    def test_reject_incomplete_handoff_before_identity_creation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);args,cp,row,stage,sl,hl,selected=self.fixture(root)
            Path(stage['burn']['path']).write_text(json.dumps(dict(collective_progress_verified=False)))
            stage['burn']=lock(stage['burn']['path']);Path(sl['path']).write_text(json.dumps(stage));sl=lock(sl['path'])
            with self.assertRaises(ValueError):prepare_identities(sl,hl,args['assets_lock'],args['sets_lock'],root/'out')
            self.assertFalse((root/'out').exists())

    def test_reject_changed_export_adjacent_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);args,cp,row,stage,sl,hl,selected=self.fixture(root)
            (cp.parent/'result.json').write_text(json.dumps(dict(row,val_post=99)))
            with patch('eval.a2_plan.load_history',return_value={'rows':[]}), \
                 patch('eval.a2_plan.remaining_runs',return_value={'pending':[row]}), \
                 patch('eval.a2_plan.select_initial_screen',return_value=selected):
                with self.assertRaises(ValueError):prepare_identities(sl,hl,args['assets_lock'],args['sets_lock'],root/'out')
            self.assertFalse((root/'out/identities.json').exists())


if __name__=='__main__':unittest.main()
