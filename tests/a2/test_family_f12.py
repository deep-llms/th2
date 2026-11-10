import copy,json,socket,subprocess,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from test_family_f11 import native
from efficient_llm.a2_family_f12_width import slice_width,importance,select_neurons,calibration_indices
from eval.a2_family_f12 import read,write,descriptors,budget_reading

class F12Tests(unittest.TestCase):
    def test_full_width_is_exact_and_masks_ties_are_stable(self):
        m=native();x=torch.tensor([[1,2,3,4]]);before=m(x).logits.detach();weights=copy.deepcopy(m.state_dict())
        slice_width(m,[list(range(64)) for _ in range(5)])
        torch.testing.assert_close(before,m(x).logits,rtol=0,atol=0)
        for k,v in weights.items():torch.testing.assert_close(v,m.state_dict()[k],rtol=0,atol=0)
        self.assertEqual(select_neurons([[2,2,1,3]],3),[[0,1,3]])
        with self.assertRaises(ValueError):select_neurons([[float('nan')]],1)

    def test_physical_slice_matches_zeroed_neurons_logits_and_gradients(self):
        full=native();small=copy.deepcopy(full);masks=[list(range(i,64,2)) for i in [0,1,0,1,0]]
        for l,keep in zip(full.model.layers,masks):
            drop=sorted(set(range(64))-set(keep))
            with torch.no_grad():l.mlp.down_proj.weight[:,drop]=0
        slice_width(small,masks);x=torch.tensor([[1,2,3,4],[3,2,4,5]])
        a=small(x).logits;b=full(x).logits
        torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
        a.square().sum().backward();b.square().sum().backward()
        for l,r,keep in zip(small.model.layers,full.model.layers,masks):
            for n in ['gate_proj','up_proj','down_proj']:
                g=getattr(r.mlp,n).weight.grad;ref=g[:,keep] if n=='down_proj' else g[keep,:]
                torch.testing.assert_close(getattr(l.mlp,n).weight.grad,ref,rtol=3e-5,atol=3e-6)
        self.assertEqual(len(small.model.layers),5);self.assertEqual(small.config.intermediate_size,32)

    def test_importance_matches_direct_activation_sum_and_sample_seed(self):
        class Rows:
            def __getitem__(self,i):return {'input_ids':torch.tensor([1,2,3+i%10,4,5])}
        m=native();rows=Rows();ix=calibration_indices(40,8);self.assertEqual(ix,calibration_indices(40,8));self.assertEqual(len(set(ix)),8)
        means,c=importance(m,rows,ix);observed=[[] for _ in m.model.layers];hooks=[]
        for i,l in enumerate(m.model.layers):hooks.append(l.mlp.down_proj.register_forward_pre_hook(lambda mod,args,i=i:observed[i].append(args[0].detach().double().abs())))
        with torch.no_grad():
            for j in ix:m.model(rows[j]['input_ids'][None],use_cache=False)
        for h in hooks:h.remove()
        expected=np.stack([torch.cat(v,dim=1).mean((0,1)).numpy() for v in observed]);np.testing.assert_allclose(means,expected,rtol=0,atol=1e-15)
        self.assertEqual(c,[40]*5)

    def test_native_width_reload_and_kd_trainer_recipe(self):
        from efficient_llm.a2.model import KDModel,round_to_saved_precision,save_student,reload_student
        from efficient_llm.a2_family_f12_heal import ReportTrainer,Trace
        from efficient_llm.a2.train import make_arguments,audit_order
        from efficient_llm.a2.protocol import lr_for_update
        from transformers import default_data_collator
        class Rows:
            def __len__(self):return 80
            def __getitem__(self,i):return dict(input_ids=torch.tensor([1,2,3,4,5]),sample_id=i)
        with tempfile.TemporaryDirectory() as d:
            m=native();t=copy.deepcopy(m);slice_width(m,[list(range(32)) for _ in m.model.layers]);trace=Trace()
            tr=ReportTrainer(model=KDModel(m,2),teacher=t,args=make_arguments(Path(d)/'trainer',20,.001,2,1,cuda=False,global_sequences=4),train_dataset=Rows(),data_collator=default_data_collator,callbacks=[trace],trace_path=Path(d)/'trace.jsonl')
            tr.train();tr.trace_file.close();audit_order([tr.seen],list(range(80)),1,2)
            np.testing.assert_allclose(trace.lrs,[lr_for_update(i,20,.001) for i in range(1,21)],rtol=1e-12,atol=0)
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in t.parameters()))
            round_to_saved_precision(m);save_student(m,Path(d)/'final');r=reload_student(Path(d)/'final');x=Rows()[0]['input_ids'][None]
            m.eval();r.eval()
            with torch.no_grad():torch.testing.assert_close(m(x).logits,r(x).logits,rtol=0,atol=0)
            self.assertEqual(r.config.intermediate_size,32);self.assertEqual(len(r.model.layers),5)

    def test_predecessor_gate_and_native_install(self):
        from scripts.launch_a2_family_f12 import predecessor_ready,install,launch
        from efficient_llm.diagnostic_data import sha256
        from scripts.a2_quant_io import lock
        with tempfile.TemporaryDirectory(dir='temp',prefix='f12-launch-fixture-') as d:
            b=Path(d).resolve();f11=b/'f11';r=f11/'results/a2_family_20261107/f11';r.mkdir(parents=True)
            cfg=dict(host=socket.gethostname(),queue_root=str(b/'queue'),output_root=str(b/'gpu'),queue_session='fixture_cpu',session='fixture_gpu',f11_root=str(f11),f10_root=str(b/'f10'),f10={'output':str(b/'f10/result')})
            self.assertFalse(predecessor_ready(cfg));write(f11/'supervised/supervisor.json',dict(finished_at='done'));write(f11/'supervised/burn-verified.json',dict(collective_progress_verified=True))
            write(r/'complete.json',dict(report_complete=True,execution_complete=True));write(r/'report_complete.json',dict(complete=True,calibration_replayed=42,LM_replayed=True,downstream_replayed=True))
            (r/'REPORT_FULL_v1.13.md').write_text('F11');(r/'backup').mkdir();(r/'backup/family_f11_scientific.tar.gz').write_bytes(b'fixture')
            write(r/'backup/verified.json',dict(verified=True,files={'REPORT_FULL_v1.13.md':dict(sha256=sha256(r/'REPORT_FULL_v1.13.md'))},archive_sha256=sha256(r/'backup/family_f11_scientific.tar.gz')))
            self.assertTrue(predecessor_ready(cfg))
            proof=b/'proof.json';files={n:sha256(n) for n in ['scripts/launch_a2_family_f12.py','resources/a2_family_f12_inventory_20261111.json','resources/a2_family_f12_reference_13_20261111.json']};write(proof,dict(launch_ready=True,files=files))
            for key,n in [('inventory','inventory'),('reference_13_lock','reference_13')]:cfg[key]=lock(Path(f'resources/a2_family_f12_{n}_20261111.json').resolve())
            plan=b/'plan.json';write(plan,cfg)
            def fake(args,**kw):return subprocess.CompletedProcess(args,1 if 'has-session' in args else 0)
            with patch('scripts.launch_a2_family_f12.subprocess.run',side_effect=fake):install(plan,'a'*40,proof)
            saved=read(b/'queue/plan.json');saved['f10_config_lock']=write(b/'f10/config.json',{});saved['preparation_proof']=write(b/'queue/actual_inputs.json',dict(files={}))
            write(b/'f10/result/FT15.json',dict(batch_size=2))
            with patch('scripts.launch_a2_family_f12.subprocess.run',side_effect=fake),patch('scripts.launch_a2_family_f12.predecessor',return_value={'gpus':[]}):launch(saved,b/'queue/source','a'*40)
            new=read(b/'gpu/config.json');self.assertEqual(new['results_root'],str(b/'gpu/results/a2_family_20261107/f12'));self.assertEqual(new['previous_report']['path'],str(r/'REPORT_FULL_v1.13.md'))
            from run_experiments import load_jobs
            self.assertEqual(len(load_jobs(b/'gpu/jobs.json')),1)
            (r/'REPORT_FULL_v1.13.md').write_text('tamper')
            with self.assertRaises(ValueError):predecessor_ready(cfg)

    def test_production_width_parameter_count_on_meta_device(self):
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        cls=get_class_from_dynamic_module('modeling_opensci.OpensciForCausalLM','temp/a2_model_code',local_files_only=True)
        ref=read('resources/a2_family_f12_reference_13_20261111.json');cfg=cls.config_class.from_dict(ref['config'])
        cfg.intermediate_size=8192;cfg._attn_implementation='sdpa'
        with torch.device('meta'):m=cls(cfg)
        count=slice_width(m,[list(range(5440)) for _ in range(24)])
        self.assertEqual(count,ref['nonembedding_parameters']);self.assertEqual(count,1205424128)

    def test_fixed_scope_and_budget_labels(self):
        self.assertEqual([r['steps'] for r in descriptors()],[762,762,3051,3051,12207,12207]);self.assertEqual(budget_reading(.02,.023),'flat');self.assertEqual(budget_reading(.02,.025),'changing upward')

if __name__=='__main__':unittest.main()
