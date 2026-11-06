import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from datetime import timedelta
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import save_file
from efficient_llm.a2.measure import score
from efficient_llm.a2_depth.heal_long import publish_final,publish_final_distributed


class Body(torch.nn.Module):
    def __init__(self):
        super().__init__();self.emb=torch.nn.Embedding(8,4)
    def forward(self,ids,**kwargs):return SimpleNamespace(last_hidden_state=self.emb(ids))


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();self.model=Body();self.lm_head=torch.nn.Linear(4,8)


class Rows:
    def __init__(self,bad=False):self.bad=bad
    def __len__(self):return 4
    def __getitem__(self,i):
        if self.bad and i==0:raise ValueError('injected row failure')
        return {'input_ids':torch.tensor([i,1,2,3])}


def distributed_probe(rank,rendezvous,folder):
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2,timeout=timedelta(seconds=30))
    try:
        torch.manual_seed(0);model=Model();failed=False
        try:score(model,Rows(bad=rank==0),amp=False)
        except RuntimeError as error:failed='Distributed scoring failed' in str(error)
        assert failed
        dist.barrier()
        healthy=score(model,Rows(),amp=False)
        serial=score(model,Rows(),amp=False,distributed=False)
        assert healthy==serial
        with patch('efficient_llm.a2_depth.heal_long.publish_final',side_effect=OSError('injected disk failure')):
            failed=False
            try:publish_final_distributed(model,Path(folder),rank,torch.device('cpu'))
            except RuntimeError as error:failed='Final export failed' in str(error)
            assert failed
        dist.barrier()
        Path(folder,f'rank{rank}.json').write_text(json.dumps(dict(score_failure=True,healthy_exact=True,export_failure=True,next_barrier=True)))
    finally:dist.destroy_process_group()


def save_fixture(model,dest):
    dest.mkdir();(dest/'config.json').write_text('{}')
    save_file({'weight':torch.tensor([2.])},str(dest/'model.safetensors'))


class ResumeHandoffTests(unittest.TestCase):
    def test_two_ranks_fail_together_then_successful_scoring(self):
        with tempfile.TemporaryDirectory() as d:
            mp.spawn(distributed_probe,args=(str(Path(d)/'rdz'),d),nprocs=2,join=True)
            for rank in range(2):self.assertTrue(all(json.loads(Path(d,f'rank{rank}.json').read_text()).values()))

    def test_partial_export_and_prior_final_survive_retry(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'final').mkdir();(root/'final/prior').write_text('preserve')
            def partial(model,dest):
                dest.mkdir();(dest/'partial').write_text('partial');raise OSError('interrupted')
            with patch('efficient_llm.a2_depth.heal_long.save_student',side_effect=partial):
                with self.assertRaises(OSError):publish_final(None,root)
            self.assertEqual((root/'final/prior').read_text(),'preserve')
            self.assertEqual(len(list(root.glob('final-pending-*/partial'))),1)
            with patch('efficient_llm.a2_depth.heal_long.save_student',side_effect=save_fixture):
                publish_final(None,root)
                publish_final(None,root)  # interruption after publish, before arm seal
            self.assertEqual(len(list(root.glob('final-preserved-*'))),2)
            self.assertTrue((root/'final/model.safetensors').is_file())

    def test_interruption_between_renames_keeps_previous_export(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'final').mkdir();(root/'final/prior').write_text('preserve')
            original=Path.rename
            def rename(p,dest):
                if p.name.startswith('final-pending-'):raise OSError('interrupted publication')
                return original(p,dest)
            with patch('efficient_llm.a2_depth.heal_long.save_student',side_effect=save_fixture),patch.object(Path,'rename',rename):
                with self.assertRaises(OSError):publish_final(None,root)
            self.assertEqual(len(list(root.glob('final-preserved-*/prior'))),1)
            with patch('efficient_llm.a2_depth.heal_long.save_student',side_effect=save_fixture):publish_final(None,root)
            self.assertEqual(len(list(root.glob('final-preserved-*/prior'))),1)

    def test_patch_copy_keeps_original_and_rejects_unbound_bytes(self):
        from scripts.a2_depth_continue import prepare_recovery_source,PATCH_FILES
        from scripts.a2_quant_io import lock
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'current';frozen=root/'original';source.mkdir();frozen.mkdir()
            sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            files={};changes={}
            for rel in sorted(PATCH_FILES|{'untouched.txt'}):
                p=frozen/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('original '+rel);files[rel]=sha(p)
                if rel in PATCH_FILES:
                    q=source/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_text('patched '+rel)
                    changes[rel]=dict(original_sha256=sha(p),patched_sha256=sha(q))
            manifest=root/'manifest.json';manifest.write_text(json.dumps(dict(files=files)))
            implementation={}
            for rel in ('scripts/a2_depth_continue.py','scripts/launch_a2_depth_partc.py'):
                q=source/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_text('launcher '+rel);implementation[rel]=sha(q)
            proof=root/'proof.json';proof.write_text(json.dumps(dict(original_source_manifest=lock(manifest),status='PASS',patches=changes,implementation=implementation,checks=[dict(passed=True)])))
            attempt=root/'attempt';attempt.mkdir();copied,receipt=prepare_recovery_source(source,frozen,attempt,proof,lock(manifest))
            self.assertTrue(all(sha(frozen/rel)==value for rel,value in files.items()))
            self.assertEqual((copied/'untouched.txt').read_bytes(),(frozen/'untouched.txt').read_bytes())
            for rel in PATCH_FILES:self.assertEqual((copied/rel).read_bytes(),(source/rel).read_bytes())
            rel=next(iter(PATCH_FILES));(source/rel).write_text('tampered')
            with self.assertRaises(ValueError):prepare_recovery_source(source,frozen,root/'bad',proof,lock(manifest))


if __name__=='__main__':unittest.main()
