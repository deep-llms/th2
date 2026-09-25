"""Local migration keeps budgets and produces verified independent copies."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pcc.joint_config import load_config, settings_for
from pcc.distill import settings
from scripts.local_restart import Backup


class LocalRestartTests(unittest.TestCase):
    def test_local_plan_keeps_budget_and_b200_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            path=root/'config.json'
            path.write_text(json.dumps({'experiment':'joint-local-v3','model_path':'model',
                                        'train_data':'train','val_data':'dev'}))
            config=load_config(path)
            joint=settings_for(config);student=settings(config)
            self.assertEqual(joint.version,'joint-local-v3')
            self.assertEqual(student.version,'distill-local-v2')
            for schedule in (joint,student):
                self.assertEqual(schedule.updates*schedule.tokens_per_update,201326592)
                self.assertEqual(schedule.tokens_per_update//schedule.context//4//config['microbatch'],4)
            from pcc.distributed_launch import execute
            from argparse import Namespace
            with self.assertRaisesRegex(ValueError,'four local GPUs'):
                execute(Namespace(physical_gpus=list(range(8))),config)
            with self.assertRaisesRegex(ValueError,'eight GPUs'):
                execute(Namespace(physical_gpus=list(range(4))),{**config,'experiment':'joint-v2-ddp'})

    def test_backup_atomic_replacement_and_verification_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'latest.pt'
            source.write_bytes(b'old checkpoint')
            backup=Backup(root/'backup')
            first=backup.copy(source,'teacher/latest.pt')
            self.assertEqual(first['sha256'],hashlib.sha256(b'old checkpoint').hexdigest())
            self.assertEqual(backup.copy(source,'teacher/latest.pt'),first)
            replacement=root/'replacement';replacement.write_bytes(b'new checkpoint')
            os.replace(replacement,source)
            second=backup.copy(source,'teacher/latest.pt')
            self.assertNotEqual(first['source_identity'],second['source_identity'])
            self.assertEqual((root/'backup/teacher/latest.pt').read_bytes(),b'new checkpoint')
            replacement.write_bytes(b'third checkpoint');os.replace(replacement,source)
            with patch('scripts.local_restart.hashlib.file_digest') as digest:
                digest.return_value.hexdigest.return_value='corrupted'
                with self.assertRaisesRegex(ValueError,'verification failed'):
                    backup.copy(source,'teacher/latest.pt')
            self.assertEqual((root/'backup/teacher/latest.pt').read_bytes(),b'new checkpoint')
            self.assertEqual(json.loads((root/'backup/manifest.json').read_text())['files']['teacher/latest.pt'],second)
            with self.assertRaisesRegex(ValueError,'confined'):
                backup.copy(source,'../escape.pt')

class LocalParentTests(unittest.TestCase):
    def test_standalone_teacher_loads_real_checkpoint_and_rejects_changed_receipt(self):
        from argparse import Namespace
        from dataclasses import asdict
        import torch
        from pcc.distill import file_hash, read_teacher
        from pcc.joint_training import train, rng_state
        from pcc.joint_config import SEEDS
        from test_joint import tiny, contexts, settings as tiny_settings
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'runs/seed-0-Deep'
            config={'experiment':'joint-local-v3','model_path':'unused','microbatch':1}
            fingerprints={'train':'fixed','dev':'fixed'}
            schedule=tiny_settings()
            ids={'settings':asdict(schedule),'arm':'Deep','purpose':'scientific_training',
                 'config':config,'inputs':fingerprints,'adapter_seed':SEEDS[0][0],
                 'data_order_seed':SEEDS[0][1]}
            model=tiny()
            train(model,contexts(),contexts(4),path,ids,mixed_precision=False)
            ids['distributed']={'world_size':4,'physical_gpus':list(range(4))}
            saved=torch.load(path/'final.pt',weights_only=True)
            saved['identity']=ids;saved['rank_rng']=[rng_state() for _ in range(4)]
            torch.save(saved,path/'final.pt')
            (path/'identity.json').write_text(json.dumps(ids))
            complete=json.loads((path/'complete.json').read_text());complete['world_size']=4
            (path/'complete.json').write_text(json.dumps(complete))
            queue={'status':'ok','jobs':[{'status':'ok','name':'seed-0-Deep',
                   'artifacts':[{'sha256':file_hash(path/'complete.json')}]}]}
            (root/'runs/complete.json').write_text(json.dumps(queue))
            args=Namespace(joint_root=root,seed_index=0,mode='audit')
            target=tiny()
            with patch('pcc.distill.settings_for',return_value=schedule), \
                 patch('torch.distributed.get_rank',return_value=0), \
                 patch('torch.distributed.broadcast_object_list'), \
                 patch('transformers.AutoConfig.from_pretrained'), \
                 patch('transformers.AutoModelForCausalLM.from_config'), \
                 patch('pcc.protocol.validate_config'), \
                 patch('pcc.joint_model.JointQwen',return_value=target), \
                 patch.object(target,'to',return_value=target), \
                 patch.dict(os.environ,{'LOCAL_RANK':'0'}):
                actual,parent,parent_dir=read_teacher(args,config,fingerprints)
                self.assertEqual(parent_dir,path)
                self.assertEqual(parent['sha256'],file_hash(path/'final.pt'))
                self.assertTrue(all(not p.requires_grad for p in actual.parameters()))
                for name,tensor in actual.state_dict().items():
                    torch.testing.assert_close(tensor,saved['model'][name],atol=0,rtol=0)
                complete['nll']+=.1;(path/'complete.json').write_text(json.dumps(complete))
                with self.assertRaisesRegex(ValueError,'Parent receipt changed'):
                    read_teacher(args,config,fingerprints)

class LocalQueueTests(unittest.TestCase):
    def test_scientific_gates_and_parent_backup_before_audit(self):
        import sys
        from scripts.local_restart import main
        from pcc.joint_training import code_identity
        for feedback, student_pass in ((False,False),(True,False),(True,True)):
            with self.subTest(feedback=feedback,student_pass=student_pass), tempfile.TemporaryDirectory() as directory:
                base=Path(directory);root=base/'run';root.mkdir()
                for name in ('source','inputs','resources'):(root/name).mkdir()
                (root/'resources/qwen3_joint_assets.json').write_text('{"files":[]}')
                (root/'config.json').write_text(json.dumps({'experiment':'joint-local-v3',
                    'model_path':'model','train_data':'train','val_data':'dev'}))
                (root/'cpu-ready.json').write_text(json.dumps({'status':'ok','code':code_identity()}))
                for name in ('source.json','PLAN.md','inputs/complete.json','inputs/train.npz','inputs/dev.npz'):
                    (root/name).write_text('fixture')
                backup=base/'backup';calls=[]
                def fake_run(argv,**kwargs):
                    calls.append(argv)
                    def arg(key):return argv[argv.index(key)+1]
                    if 'run_experiments.py' in argv:
                        path=Path(arg('--run-dir'));path.mkdir()
                        jobs=json.loads(Path(arg('--config')).read_text())
                        teacher=path/jobs['jobs'][0]['name'];teacher.mkdir()
                        (teacher/'final.pt').write_text('completed teacher')
                    elif 'capacity' in argv:
                        path=Path(arg('--output'));path.mkdir()
                        (path/'capacity.json').write_text(json.dumps({'status':'ok','world_size':4,
                            'resume_next_update_verified':True,'initial_native_equivalence':True,
                            'checkpoint_roundtrip':True,'parameters_changed':{'early':True,'late':True,'branch':True},
                            'teacher_unchanged':True,'student_changed':True,'student_noop_exact':True}))
                    elif 'audit' in argv:
                        seed=arg('--seed-index')
                        self.assertTrue((backup/f'teacher-seed-{seed}/runs/seed-{seed}-Deep/final.pt').exists())
                        path=Path(arg('--output'));path.mkdir()
                        (path/'complete.json').write_text(json.dumps({'status':'ok','ready_for_students':feedback}))
                    elif 'train' in argv:
                        path=Path(arg('--output'));path.mkdir(parents=True)
                        (path/'final.pt').write_text('student')
                    elif 'report' in argv:
                        path=Path(arg('--output'));path.mkdir()
                        (path/'complete.json').write_text(json.dumps({'status':'ok','seeds':[
                            {'passed':student_pass} for _ in range(int(arg('--seed-count')))]}))
                    else:raise AssertionError(argv)
                old=Path.cwd()
                try:
                    os.chdir(root)
                    with patch.object(sys,'argv',['local_restart','--root',str(root),'--backup-root',str(backup)]), \
                         patch('scripts.local_restart.require_separate_filesystem'), \
                         patch('scripts.local_restart.subprocess.run',side_effect=fake_run):
                        main()
                finally:os.chdir(old)
                value=json.loads((root/'complete.json').read_text())
                self.assertEqual(value['status'],'ok')
                teacher_calls=[a for a in calls if 'run_experiments.py' in a]
                self.assertEqual(len(teacher_calls),2 if feedback and student_pass else 1)
                student_calls=[a for a in calls if 'pcc.distill' in a and 'train' in a]
                self.assertEqual(len(student_calls),0 if not feedback else (4 if student_pass else 2))
                self.assertTrue((backup/'complete.json').exists())


if __name__=='__main__':unittest.main()
