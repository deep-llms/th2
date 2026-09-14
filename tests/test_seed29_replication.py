"""CPU/mocked queue checks: never inspect or signal real GPU processes."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import pilot_seed29_replication as job
import seed29_replication_config as config
from ccm.cli import parser
from ccm.runtime import delta_policy_matches
from ccm.statistics import delta_decision
from validate_stage1 import fresh_reader_hash


def decision(include=True):
    return dict(include_delta=include, replication_policy='per_seed',decision_seed=29,
        source_checkpoint_hash=config.COMMON_HASH, overall_safeguard=True,
        results={k:dict(upper95=(-.001 if k.startswith('hit') else .002),
                       replicates=10000,bootstrap_seed=20260913)
                 for k in ('hit_vs_contextual','hit_vs_shuffled','miss_vs_contextual')})


class ReplicationTests(unittest.TestCase):
    def test_gate_binding_and_boundaries(self):
        d=decision()
        self.assertTrue(delta_policy_matches(d,29,config.COMMON_HASH))
        self.assertFalse(delta_policy_matches(d,17,config.COMMON_HASH))
        self.assertFalse(delta_policy_matches(d,29,'other-writer'))
        for name,value in [('hit_vs_contextual',0.),('hit_vs_shuffled',0.),('miss_vs_contextual',.002001)]:
            bad=copy.deepcopy(d);bad['results'][name]['upper95']=value
            self.assertFalse(delta_policy_matches(bad,29,config.COMMON_HASH))
        d['overall_safeguard']=False
        self.assertFalse(delta_policy_matches(d,29,config.COMMON_HASH))
        self.assertTrue(delta_policy_matches(dict(replication_policy='seed17_then_all',decision_seed=17),29,'x'))
        self.assertFalse(delta_policy_matches(dict(replication_policy='unknown'),29,'x'))

    def test_actual_decision_policy_include_exclude_and_legacy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);paths={}
            for arm in ('delta','contextual','shuffled'):
                path=root/arm;path.mkdir();paths[arm]=str(path)
                (path/'metrics.json').write_text(json.dumps(dict(arm=arm,phase='stage1',seed=29,
                    step=977,total_steps=977,source_checkpoint_hash=config.COMMON_HASH,
                    corpus_hash='corpus',vocabulary_hash='vocab',metrics=dict(overall=dict(nll=2.)))))
            args=SimpleNamespace(**paths,cluster='doc_id',replication_policy='per_seed',output=str(root/'yes.json'))
            with patch('ccm.statistics.paired_arrays',return_value='paired'), \
                 patch('ccm.statistics.bootstrap',side_effect=[dict(upper95=-.1),dict(upper95=-.1),dict(upper95=.002)]):
                r=delta_decision(args)
            self.assertTrue(r['include_delta']);self.assertEqual(r['decision_seed'],29)
            args.output=str(root/'no.json')
            with patch('ccm.statistics.paired_arrays',return_value='paired'), \
                 patch('ccm.statistics.bootstrap',side_effect=[dict(upper95=0.),dict(upper95=-.1),dict(upper95=.002)]):
                self.assertFalse(delta_decision(args)['include_delta'])
            args.replication_policy='seed17_then_all'
            with self.assertRaises(ValueError):
                delta_decision(args)

    def test_commands_and_paired_reader(self):
        self.assertEqual(fresh_reader_hash(29),fresh_reader_hash(29))
        self.assertNotEqual(fresh_reader_hash(29),fresh_reader_hash(17))
        for include in (False,True):
            with patch.object(job.DECISION.__class__,'read_text',return_value=json.dumps(decision(include))):
                for phase,arms in [('stage1',job.STAGE1_ARMS),('stage2',config.stage2_arms(decision(include)))]:
                    for arm in arms:
                        cmd=job.train_command(phase,arm);self.assertIn('--nproc_per_node=8',cmd)
                        a=parser().parse_args(cmd[cmd.index('train'):])
                        self.assertEqual((a.seed,a.phase,a.arm),(29,phase,arm))
                        self.assertEqual(a.checkpoint,str(config.COMMON/'common/checkpoint-15259'))
                        self.assertEqual((a.microbatch_segments,a.loss_chunk),(8,1024))
                        self.assertFalse(a.engineering or a.online)
                        self.assertEqual(a.delta_decision,str(job.DECISION) if phase=='stage2' and arm=='delta' else None)
                        e=job.eval_command(phase,arm);a=parser().parse_args(e[e.index('evaluate'):])
                        self.assertEqual(a.role,'dev');self.assertFalse(a.final_evaluation)
                if not include:
                    with self.assertRaises(RuntimeError):job.train_command('stage2','delta')
        c=job.compile_command();a=parser().parse_args(c[c.index('compile'):])
        self.assertEqual((a.microbatch_segments,a.isolated_batch),(4,256))

    def test_full_sequence_both_delta_branches(self):
        for include in (False,True):
            with tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp)/'fresh';gate=out/'stage1/delta_decision.json';events=[]
                w=job.Replication()
                def record(**fields):
                    (out/'status.json').write_text(json.dumps(fields))
                    if fields.get('event')=='seed29_replication_verified_and_burns_active':
                        (out/'STOP_IDLE_WATCH').touch()
                def run(name,argv,indices=(),spare=()):
                    events.append(('run',name,list(indices)))
                    if name=='stage1_delta_decision':gate.write_text(json.dumps(decision(include)))
                w.record=record;w.run=run
                w.validate=lambda action,phase=None,arm=None,spare=():events.append(('validate',action,phase,arm))
                w.disarm=lambda:events.append(('disarm',))
                w.burns=lambda g:events.append(('burn',g))
                w.free_after_wait=lambda g:events.append(('free',g))
                with patch.object(job,'OUT',out),patch.object(job,'DECISION',gate), \
                     patch.object(job,'PROJECT',Path.cwd()),patch.dict(os.environ,CONDA_DEFAULT_ENV='train_env'), \
                     patch.object(job,'preflight'),patch.object(job,'inspect'),patch.object(job,'snapshot',return_value=[]), \
                     patch.object(job,'stop',side_effect=lambda g:events.append(('stop',g))):
                    w.execute()
                names=[e[1] for e in events if e[0]=='run']
                self.assertEqual(names[0],'compile_tables')
                self.assertEqual([n for n in names if '_train_' in n],
                    ['stage1_train_'+a for a in job.STAGE1_ARMS]+
                    ['stage2_train_'+a for a in config.stage2_arms(decision(include))])
                self.assertLess(names.index('stage1_delta_decision'),names.index('stage2_train_base'))
                self.assertTrue((out/'stage1/complete.json').is_file())
                self.assertTrue((out/'stage2/complete.json').is_file())
                self.assertTrue((out/'complete.json').is_file())
                self.assertEqual(events[:4],[('validate','inputs',None,None),('disarm',),('stop',job.ALL),('free',job.ALL)])
                self.assertEqual([e for e in events if e[0]=='burn'][-1],('burn',job.ALL))

    def test_failure_stops_queue_and_preserves_prior_owner(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'fresh';w=job.Replication()
            w.validate=Mock(side_effect=ValueError('wrong input'));w.disarm=Mock();w.record=Mock()
            with patch.object(job,'OUT',out),patch.object(job,'PROJECT',Path.cwd()), \
                 patch.dict(os.environ,CONDA_DEFAULT_ENV='train_env'),patch.object(job,'preflight'), \
                 patch.object(job,'stop') as stop:
                with self.assertRaises(ValueError):w.execute()
                w.recover_idle();stop.assert_not_called();w.disarm.assert_not_called()
            self.assertFalse((out/'complete.json').exists())
        w=job.Replication();w.run=Mock(side_effect=RuntimeError('failed train'));w.burns=Mock();w.validate=Mock()
        with self.assertRaisesRegex(RuntimeError,'failed train'):w.phase('stage1',job.STAGE1_ARMS)
        self.assertEqual(w.run.call_count,1);w.burns.assert_not_called();w.validate.assert_not_called()

    def test_a3_observer_disarm_is_cooperative(self):
        from datetime import datetime,timezone
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            (p/'complete.json').write_text(json.dumps(dict(success=True,event='a3_diagnostics_verified_and_burns_active')))
            (p/'status.json').write_text(json.dumps(dict(success=True,stage='complete',time=datetime.now(timezone.utc).isoformat())))
            w=job.Replication();w.record=Mock();ident=(123,'start',[b'scripts/pilot_a3_diagnostics.py'])
            with patch.object(job,'PREVIOUS',p),patch.object(job,'observer',side_effect=[(456,ident),None]), \
                 patch.object(job,'identity',side_effect=ProcessLookupError()),patch.object(job,'inspect'), \
                 patch.object(job,'stop') as stop:
                w.disarm();stop.assert_not_called()
            self.assertTrue(w.reclaimed);self.assertTrue((p/'STOP_IDLE_WATCH').exists())


if __name__=='__main__':
    unittest.main()
