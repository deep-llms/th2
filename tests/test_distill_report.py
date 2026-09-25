"""Report acceptance/rejection using real tiny adapter checkpoints and controlled losses."""
from argparse import Namespace
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from pcc.distill import file_hash
from pcc.distill_report import report_results
from pcc.distill_training import train
from test_distill import student, identity
from test_joint import contexts, settings


class DistillationReportTests(unittest.TestCase):
    def test_paired_report_and_changed_counts_rejected(self):
        self.exercise_report(False)

    def test_local_single_seed_report_and_changed_counts_rejected(self):
        self.exercise_report(True)

    def exercise_report(self, local):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);dev=contexts(4);data=contexts()
            counts=np.full(4,7,dtype=np.int64)
            config={'experiment':'joint-local-v3' if local else 'joint-v2-ddp','microbatch':1}
            world=4 if local else 8
            seed_count=1 if local else 2
            fingerprints={'train':'fixed','dev':'fixed'}
            for seed in range(seed_count):
                audit=root/f'audit-seed-{seed}';audit.mkdir()
                audit_value={'status':'ok','ready_for_students':True,'inputs':fingerprints,
                             'parent':{'seed':seed},'calibration':{'sigma_delta':.2},
                             'nll':{'FeedbackOff':2.9,'Deep':2.8}}
                (audit/'complete.json').write_text(json.dumps(audit_value))
                np.savez(audit/'eval.npz',target_counts=counts,sequence_indices=np.arange(4),
                         off_loss_sums=counts*2.9,deep_loss_sums=counts*2.8)
                base=root/'joint/runs'/f'seed-{seed}-Base';base.mkdir(parents=True)
                np.savez(base/'eval.npz',target_counts=counts,sequence_indices=np.arange(4),loss_sums=counts*3.)
                for arm,nll in [('LM',2.85),('PCC',2.82 if seed==0 else 2.9)]:
                    model=student(arm);path=root/'runs'/f'seed-{seed}-{arm}'
                    ids=identity(model,settings())
                    # Produce real student/optimizer histories on one CPU first.
                    train(model,data,dev,path,ids,mixed_precision=False)
                    state=torch.load(path/'final.pt',weights_only=True)
                    final=json.loads((path/'complete.json').read_text())
                    # Synthetic topology/provenance for this pure CPU report fixture.
                    ids.update(purpose='correction_distillation',seed_index=seed,inputs=fingerprints,
                               config=config,parent=audit_value['parent'],audit_sha256=file_hash(audit/'complete.json'),
                               sigma_delta=.2,world_size=world,physical_gpus=list(range(world)),code='fixed',
                               adapter_seed=seed,data_order_seed=seed,mixed_precision=False,torch=str(torch.__version__))
                    state['identity']=ids;state['rank_rng']*=world
                    torch.save(state,path/'final.pt')
                    (path/'identity.json').write_text(json.dumps(ids))
                    final.update(world_size=world,nll=nll)
                    (path/'complete.json').write_text(json.dumps(final))
                    np.savez(path/'eval.npz',loss_sums=counts*nll,target_counts=counts,sequence_indices=np.arange(4))
            args=Namespace(output=root/'report',audit_root=root,joint_root=root/'joint',
                           runs_dir=root/'runs',data_dir=root/'inputs',seed_count=seed_count)
            with patch('pcc.joint.load_inputs',return_value=(data,dev,fingerprints)), \
                 patch('pcc.distill_report.settings',return_value=settings()):
                result=report_results(args,config)
                self.assertEqual(result['decision'],'recommend_second_seed' if local else 'stop_no_consistent_student_recovery')
                self.assertTrue(result['seeds'][0]['passed'])
                if not local:self.assertFalse(result['seeds'][1]['passed'])
                if local:self.assertNotIn('OriginalBase',result['seeds'][0]['nll'])
                self.assertFalse(result['test_unlocked'])
                self.assertEqual(result['automatic_followup_training'],local)
                self.assertTrue((args.output/'validation-curves.png').is_file())
                path=root/f'runs/seed-{seed_count-1}-PCC/eval.npz'
                np.savez(path,loss_sums=counts*2.9,target_counts=counts+1,sequence_indices=np.arange(4))
                args.output=root/'bad-report'
                with self.assertRaisesRegex(ValueError,'eval alignment'):
                    report_results(args,config)
                self.assertTrue((args.output/'failure.json').exists())


if __name__=='__main__':unittest.main()
