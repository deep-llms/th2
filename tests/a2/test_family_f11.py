import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from eval.a2_family_f11 import *

def native():
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    cls=get_class_from_dynamic_module('modeling_opensci.OpensciForCausalLM','temp/a2_model_code',local_files_only=True)
    cfg=cls.config_class(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=5,
        num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=4096,
        use_cache=False,attention_dropout=0.,tie_word_embeddings=True)
    cfg.qk_layernorm=False;cfg._attn_implementation='sdpa';torch.manual_seed(42)
    return cls(cfg).eval()

class F11Tests(unittest.TestCase):
    def test_lm_shift_and_chunk_gradient_match_native_ce(self):
        from efficient_llm.a2_family_f11_loss import LMModel
        from efficient_llm.a2.model import prune
        for removed in ([],[1,2]):
            m=native()
            if removed:prune(m,removed)
            ids=torch.tensor([[1,2,3,4,5],[4,3,2,1,7]])
            loss=LMModel(m,2)(ids)['loss'];loss.backward()
            gradients=[p.grad.clone() for p in m.parameters()];m.zero_grad()
            logits=m(input_ids=ids,use_cache=False).logits[:,:-1].float()
            ref=torch.nn.functional.cross_entropy(logits.flatten(0,1),ids[:,1:].reshape(-1));ref.backward()
            torch.testing.assert_close(loss,ref,rtol=1e-6,atol=1e-6)
            for g,p in zip(gradients,m.parameters()):torch.testing.assert_close(g,p.grad,rtol=2e-5,atol=2e-6)

    def test_temperature_one_equals_original_native_score(self):
        from eval.a2_family_f11_calibrate import evaluate
        from efficient_llm.a2.measure import score
        from efficient_llm.a2.model import prune
        class Rows:
            def __len__(self):return 3
            def __getitem__(self,i):return {'input_ids':torch.tensor([1,2,3+i,4,5])}
        for removed in ([],[1,2]):
            m=native()
            if removed:prune(m,removed)
            a=evaluate(m,Rows(),chunk=2);b=score(m,Rows(),chunk=2,distributed=False)
            self.assertEqual(a['decoder_forwards'],3)
            np.testing.assert_allclose(a['per_sequence_nll']['100'],b['per_sequence_nll'],rtol=0,atol=1e-7)
            self.assertAlmostEqual(a['nll']['100'],b['loss'],places=7)
            self.assertTrue(all(0<=v<=np.log(32)+1e-6 for v in a['per_sequence_entropy']))

    def test_grid_extension_cannot_influence_nonedge_choice(self):
        d={str(t):(t-100)**2 for t in GUARD_GRID};d['60']=-1000
        s=choose_temperature(d);self.assertEqual(s['T_star'],1.);self.assertIsNone(s['extension'])
        d={str(t):(t-64)**2 for t in GUARD_GRID};s=choose_temperature(d)
        self.assertEqual(s['extension'],'lower');self.assertEqual(s['T_star'],.64)
        d={str(t):(t-140)**2 for t in GUARD_GRID};self.assertTrue(choose_temperature(d)['still_at_edge'])

    def test_bootstrap_matches_existing_document_stream(self):
        matrix=np.arange(20,dtype=float).reshape(5,4);c=np.array([2,3,8,2,1]);p,b=document_bootstrap(matrix,c,100)
        rng=np.random.default_rng(0);expected=[]
        for _ in range(100):
            ix=rng.integers(0,5,5);expected.append((matrix[ix]*c[ix,None]).sum(0)/c[ix].sum())
        np.testing.assert_allclose(b,expected);np.testing.assert_allclose(p,(matrix*c[:,None]).sum(0)/c.sum())

    def test_labels_and_zero_denominator(self):
        self.assertEqual(lm_reading(.008),'persists');self.assertEqual(lm_reading(-.008),'reversed')
        self.assertEqual(lm_reading(.0029),'erased');self.assertEqual(lm_reading(.003),'weak')
        self.assertEqual(calibration_reading(.02,.01),(.5,'partly calibration'))
        self.assertEqual(calibration_reading(.02,.015),(.75,'not calibration'))
        self.assertEqual(calibration_reading(0,0),(None,None))

    def test_trainer_no_teacher_full_schedule_order_and_roundtrip(self):
        from efficient_llm.a2_family_f11_loss import LMTrainer,LMModel
        from efficient_llm.a2.train import make_arguments,audit_order
        from efficient_llm.a2_family_f6.heal import Trace
        from efficient_llm.a2.model import prune,save_student,reload_student,round_to_saved_precision
        from efficient_llm.a2.protocol import lr_for_update
        from transformers import default_data_collator
        class Rows:
            def __len__(self):return 80
            def __getitem__(self,i):return dict(input_ids=torch.tensor([1,2,3,4,5]),sample_id=i)
        with tempfile.TemporaryDirectory() as d:
            m=native();prune(m,[1,2]);trace=Trace()
            args=make_arguments(Path(d)/'trainer',20,.001,2,1,cuda=False,global_sequences=4)
            tr=LMTrainer(model=LMModel(m,2),args=args,train_dataset=Rows(),data_collator=default_data_collator,callbacks=[trace],trace_path=Path(d)/'trace.jsonl')
            self.assertFalse(hasattr(tr,'teacher'));tr.train();tr.trace_file.close()
            self.assertEqual(tr.state.global_step,20)
            np.testing.assert_allclose(trace.lrs,[lr_for_update(i,20,.001) for i in range(1,21)],rtol=1e-12,atol=0)
            audit_order([tr.seen],list(range(80)),1,2)
            round_to_saved_precision(m);save_student(m,Path(d)/'final');r=reload_student(Path(d)/'final')
            ids=Rows()[0]['input_ids'][None]
            with torch.no_grad():torch.testing.assert_close(m(input_ids=ids).logits,r(input_ids=ids).logits,rtol=0,atol=0)

    def test_predecessor_requires_report_backup_and_native_handoff(self):
        from scripts.launch_a2_family_f11 import predecessor_ready
        from efficient_llm.diagnostic_data import sha256
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);r=p/'result';r.mkdir();cfg={'f10_root':d,'f10':{'output':str(r)}}
            self.assertFalse(predecessor_ready(cfg))
            write(p/'supervised/supervisor.json',dict(finished_at='done'))
            self.assertFalse(predecessor_ready(cfg))
            write(p/'supervised/burn-verified.json',dict(collective_progress_verified=True))
            write(r/'complete.json',dict(report_complete=True,nine_models=11,new_models=42,generation_models=34))
            write(r/'report_complete.json',dict(complete=True));(r/'REPORT_FULL_v1.11.md').write_text('old report')
            (r/'backup').mkdir();(r/'backup/family_f10_scientific.tar.gz').write_bytes(b'fixture')
            write(r/'backup/verified.json',dict(verified=True,files={'REPORT_FULL_v1.11.md':dict(sha256=sha256(r/'REPORT_FULL_v1.11.md'))},archive_sha256=sha256(r/'backup/family_f10_scientific.tar.gz')))
            self.assertTrue(predecessor_ready(cfg))
            (r/'REPORT_FULL_v1.11.md').write_text('tampered')
            with self.assertRaises(ValueError):predecessor_ready(cfg)

    def test_jobs_and_explicit_pane_environment(self):
        from scripts.launch_a2_family_f11 import job_manifest,environment_shell
        from run_experiments import load_jobs
        import shlex
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'jobs.json';p.write_text(json.dumps(job_manifest('/config.json')));self.assertEqual(len(load_jobs(p)),1)
        a=shlex.split(environment_shell(['python','-m','run'],'/source','a'*40,'/log'))
        self.assertEqual(a[:2],['exec','env']);self.assertIn('HF_HUB_OFFLINE=1',a);self.assertIn('PYTHONPATH=/source',a)

    def test_actual_install_and_launch_serialization_with_mock_hardware(self):
        import socket,subprocess
        from scripts.launch_a2_family_f11 import install,launch
        from scripts.a2_quant_io import lock
        from efficient_llm.diagnostic_data import sha256
        with tempfile.TemporaryDirectory(dir='temp',prefix='f11-launch-fixture-') as d:
            base=Path(d).resolve();q=base/'queue';out=base/'gpu'
            proof=base/'proof.json';write(proof,dict(launch_ready=True,files={'scripts/launch_a2_family_f11.py':sha256('scripts/launch_a2_family_f11.py')}))
            cfg=dict(host=socket.gethostname(),queue_root=str(q),output_root=str(out),queue_session='fixture_queue',session='fixture_gpu',f10_root=str(base/'f10'),f10={'output':str(base/'f10/result')})
            plan=base/'plan.json';write(plan,cfg)
            def fake(args,**kwargs):return subprocess.CompletedProcess(args,1 if 'has-session' in args else 0)
            with patch('scripts.launch_a2_family_f11.subprocess.run',side_effect=fake):install(plan,'a'*40,proof)
            saved=read(q/'plan.json');self.assertTrue(Path(saved['cpu_proof']['path']).is_file())
            saved['f10_config_lock']=write(base/'f10/config.json',{})
            saved['preparation_proof']=write(q/'actual_inputs.json',dict(files={}))
            write(base/'f10/result/FT15.json',dict(batch_size=2));(base/'f10/result/REPORT_FULL_v1.11.md').write_text('F10')
            with patch('scripts.launch_a2_family_f11.subprocess.run',side_effect=fake),patch('scripts.launch_a2_family_f11.predecessor',return_value={'gpus':[]}):
                launch(saved,q/'source','a'*40)
            new=read(out/'config.json');self.assertEqual(new['generation_batch'],2)
            self.assertTrue((out/'jobs.json').is_file());self.assertTrue((out/'source/scripts/a2_family_f11_queue.py').is_file())

if __name__=='__main__':unittest.main()
