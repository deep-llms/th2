import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from efficient_llm.a2.protocol import lr_for_update
from efficient_llm.a2_stage1b.protocol import ARMS, GRID, Run, SPEC_SHA, REPLICATION_SPEC_SHA, grid_choice, phase_runs
from eval.a2_run import write, lock
from eval.a2_stage1b_plan import prepare_identities, validate_terminal_bundle


class Stage1bEvalPlanTest(unittest.TestCase):
    def fixture(self, root, phase="part1"):
        assets = write(root/'assets.json', {})
        reference = dict(spec_sha256=SPEC_SHA,stage1b_spec_sha256=REPLICATION_SPEC_SHA,dataset='fwe',assets_sha256=assets['sha256'])
        lrs = (*GRID,.003)
        rows = []
        for run in [Run(a,'0.2B',lr) for lr in lrs for a in ARMS[:2]] + phase_runs('part1',.001) + (phase_runs('part2',.001) if phase=='part2' else []):
            row = dict(run.record(),provenance=reference,status='complete',diverged=False,
                coverage=dict(examples=run.steps*64,microbatches=run.steps*2,world_size=8,exact_order=True),
                lr_updates=[lr_for_update(i,run.steps,run.lr) for i in range(1,run.steps+1)],
                roundtrip_difference=0.,val_tune_post={.0001:3.,.0003:2.9,.001:2.8,.003:2.85}[run.lr])
            rows.append(row)
            write(root/run.run_id/'result.json',row)
        grid = rows[:8]
        choice = grid_choice(grid,lrs)
        choice['own_best'] = {a:grid_choice([r for r in grid if r['arm']==a],lrs,(a,))['chosen_lr'] for a in ARMS[:2]}
        stage = dict(phase=phase,reference=write(root/'reference.json',reference),
            results=write(root/'runs.json',rows),choice=write(root/'lr_choice.json',choice),
            tests=write(root/'tests.json',dict(status='passed',eligible_x72=True,provenance=reference)),
            decision=write(root/'decision.json',dict(chosen_lr=.001,stage1_spec_sha256=SPEC_SHA,stage1b_spec_sha256=REPLICATION_SPEC_SHA,phase=phase)),
            supervisor=write(root/'supervisor.json',dict(training_status='ok',training_returncode=0)),
            burn=write(root/'burn.json',dict(collective_progress_verified=True)))
        stage['complete']=write(root/'complete.json',dict(status='complete',phase=phase,training_complete=phase=='part2',
            runs=len(rows),runs_sha256=stage['results']['sha256'],choice_sha256=stage['choice']['sha256'],decision_sha256=stage['decision']['sha256']))
        stage['run_complete']=write(root/'run_complete.json',dict(status='ok',jobs=[dict(
            name='stage1b-'+phase,status='ok',returncode=0,artifacts=[dict(path='complete.json',sha256=stage['complete']['sha256'])])]))
        return stage,write(root/'stage.json',stage),assets,rows

    def relock(self, stage, sl, field, value):
        path=Path(stage[field]['path']);path.write_text(json.dumps(value));stage[field]=lock(path)
        Path(sl['path']).write_text(json.dumps(stage));return lock(sl['path'])

    def test_complete_part1_edge_grid_and_four_identities(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);stage,sl,assets,rows=self.fixture(root)
            _,_,_,selection=validate_terminal_bundle(sl,assets)
            self.assertEqual(len(selection),4)
            with patch('eval.a2_checkpoint.checkpoint_identity',return_value={'kind':'parent'}) as parent, \
                 patch('eval.a2_stage1_checkpoint.checkpoint_identity',return_value={'kind':'healed'}) as healed:
                result=prepare_identities(sl,assets,root/'out')
            self.assertEqual(parent.call_count,2);self.assertEqual(healed.call_count,2)
            self.assertEqual(set(result['identities']),set(selection))
            for call in healed.call_args_list:
                self.assertEqual(call.kwargs['reference_lock'],stage['reference'])

    def test_bad_handoff_grid_rule_or_missing_fit_prevents_output(self):
        for case in ('burn','choice','missing_edge','missing_mid','duplicate','diverged_mid','teacher'):
            with self.subTest(case=case),tempfile.TemporaryDirectory() as d:
                root=Path(d);stage,sl,assets,rows=self.fixture(root)
                if case=='burn':sl=self.relock(stage,sl,'burn',dict(collective_progress_verified=False))
                elif case=='choice':
                    value=json.loads(Path(stage['choice']['path']).read_text());value['chosen_lr']=.0003
                    sl=self.relock(stage,sl,'choice',value)
                else:
                    rows=copy.deepcopy(rows)
                    if case=='missing_edge':rows.pop(7)
                    elif case=='missing_mid':rows.pop(-2)
                    elif case=='duplicate':rows.append(rows[0])
                    elif case=='diverged_mid':rows[-2]['diverged']=True
                    elif case=='teacher':rows[-2]['teacher']='S72'
                    sl=self.relock(stage,sl,'results',rows)
                with self.assertRaises(ValueError):prepare_identities(sl,assets,root/'out')
                self.assertFalse((root/'out').exists())

    def test_changed_terminal_artifact_and_adjacent_receipt_rejected(self):
        for case in ('terminal','adjacent','unrelated_artifact'):
            with self.subTest(case=case),tempfile.TemporaryDirectory() as d:
                root=Path(d);stage,sl,assets,rows=self.fixture(root)
                if case == 'unrelated_artifact':
                    terminal=json.loads(Path(stage['run_complete']['path']).read_text())
                    terminal['jobs'][0]['artifacts']=[dict(path='assets.json',sha256=assets['sha256'])]
                    sl=self.relock(stage,sl,'run_complete',terminal)
                else:
                    path=root/'complete.json' if case=='terminal' else root/Run('P300','3.2B',.001).run_id/'result.json'
                    path.write_text('{}')
                with self.assertRaises(ValueError):prepare_identities(sl,assets,root/'out')
                self.assertFalse((root/'out').exists())

    def test_part2_selects_only_s72_and_x72_after_all_fits(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);stage,sl,assets,rows=self.fixture(root,'part2')
            _,_,_,selected=validate_terminal_bundle(sl,assets)
            self.assertEqual(set(selected),{'S72-parent','X72-student'})
            self.assertEqual(selected['X72-student']['teacher'],'P300')
            with patch('eval.a2_checkpoint.checkpoint_identity',return_value={'kind':'parent'}) as parent, \
                 patch('eval.a2_stage1_checkpoint.checkpoint_identity',return_value={'kind':'healed'}) as healed:
                result=prepare_identities(sl,assets,root/'out')
            self.assertEqual(parent.call_count,1);self.assertEqual(healed.call_count,1)
            self.assertEqual(result['phase'],'part2')


if __name__=='__main__':unittest.main()
