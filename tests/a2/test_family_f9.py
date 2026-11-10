import unittest
from unittest.mock import patch
import numpy as np
from eval import a2_family_f9 as f

class F9Statistics(unittest.TestCase):
    def test_complete_readout_pair_pool_trend_and_gain(self):
        from eval import a2_family_f9_report as r
        counts={'a':4,'boolq':4,'lambada_openai':4}
        entries=[];replays={}
        def add(name,ds,arm,role,values,nll,parent=None,budget=None):
            entries.append(dict(model_id=name,family='opensci',dataset=ds,arm=arm,role=role,
                cut=6,variant='drop',budget=budget,lr=.001,seed=2030,parent_id=parent,final_nll=nll))
            replays[name]=dict(items={t:dict(values=values,identity='same') for t in counts},
                results={'lambada_openai':{'perplexity,none':10.}})
        for ds in ('dclm','fwe'):
            for arm,v,nll in [('S72',[1,0,0,0],3.2),('P300',[1,1,0,0],3.),('S188',[1,1,0,0],3.1),('P1T',[1,1,1,1],2.8)]:
                add(ds+arm,ds,arm,'parent',v,nll)
            for budget in (.2,.8,3.2):
                add(ds+'s'+str(budget),ds,'P300','student',[1,0,0,0],3.2,ds+'P300',budget)
                add(ds+'l'+str(budget),ds,'P1T','student',[1,1,0,0],3.1,ds+'P1T',budget)
        with patch.object(r,'items',lambda v:v['items']),patch.object(f,'COUNTS',counts):
            out=r.analyze(entries,replays)
        self.assertEqual(len(out['lead']),6)
        self.assertEqual(len(out['pooled']),3)
        self.assertEqual(len(out['budget_trends']),2)
        self.assertAlmostEqual(out['lead'][0]['E_nats'],.1)
        self.assertAlmostEqual(out['lead'][0]['E_down']['lambada_openai']['estimate'],25)
        self.assertAlmostEqual(out['pooled'][0]['E_down']['lambada_openai']['estimate'],25)
        self.assertAlmostEqual(out['anneal_gains']['dclm']['delta_g_NLL'],.1)
        self.assertAlmostEqual(out['budget_trends'][0]['slope_pp_per_log_token_budget']['lambada_openai']['estimate'],0)
        text=r.render(out)
        self.assertIn('Budget trends',text)
        self.assertIn('Parent minus student accuracy',text)

    def test_joint_pair_and_pooled_cancellation(self):
        counts={'t':4,'lambada_openai':4}
        def rec(values):return {t:dict(values=values,identity='same') for t in counts}
        records={k:rec(v) for k,v in dict(sp=[1,0,1,0],lp=[1,1,1,1],
            ss=[0,0,0,0],ls=[1,1,1,1]).items()}
        with patch.object(f,'COUNTS',counts):
            result=f.pair_report(records,'sp','lp','ss','ls')
            self.assertAlmostEqual(result['MC8']['E_down']['estimate'],-50)
            self.assertEqual(result['MC8']['accuracy_kept_share'],None) # parent gap CI includes zero
            records['duplicate']=rec([1,0,1,0])
            names,draws=f.joint_draws(records,resamples=50)
            np.testing.assert_array_equal(draws['t'][1][:,0],draws['t'][1][:,4])
            records['duplicate']['t']['identity']='changed'
            with self.assertRaises(ValueError):f.joint_draws(records,resamples=50)

    def test_batch_uses_accuracy_not_request_ll_threshold(self):
        counts={'a':1000,'lambada_openai':1000}
        def result(a,l):
            return dict(samples={t:[dict(doc_id=i,doc={},target=0,arguments=[],acc=int(i<n))
                        for i in range(1000)] for t,n in [('a',a),('lambada_openai',l)]},
                        results={'a':{'acc,none':a/1000},'lambada_openai':{'acc,none':l/1000}})
        with patch.object(f,'COUNTS',counts),patch.object(f,'PRIMARY',dict.fromkeys(counts,'acc,none')):
            self.assertEqual(f.batch_decision(result(500,500),result(500,500))['batch_size'],2)
            self.assertEqual(f.batch_decision(result(500,500),result(500,502))['batch_size'],1)

class SmolScoring(unittest.TestCase):
    def test_own_eos_bridge_capture_and_replay(self):
        from tests.a2.test_harness import tokenizer
        from lm_eval.api.instance import Instance
        from eval.a2_request_bridge import make_bridge,RequestsCaptured
        tok=tokenizer();tok.eos_token='w7'
        request=Instance(request_type='loglikelihood',doc={},arguments=('', 'w5'),idx=0)
        bridge=make_bridge(tok,tokenizer_family='smollm3')
        with self.assertRaises(RequestsCaptured):bridge.loglikelihood([request],disable_tqdm=True)
        self.assertEqual(bridge.captured['requests'][0]['context'],[tok.eos_token_id])
        prepared=bridge.captured;scored=dict(request_sha256=prepared['request_sha256'],scores=[[-2.,False]])
        replay=make_bridge(tok,tokenizer_family='smollm3',prepared=prepared,scored=scored)
        self.assertEqual(replay.loglikelihood([request],disable_tqdm=True),[(-2.,False)])
        with self.assertRaises(ValueError):make_bridge(tok)

    def test_native_manual_likelihood_and_padding(self):
        import torch
        from transformers import SmolLM3Config,SmolLM3ForCausalLM
        from eval.a2_family_f9_score import smol_score
        torch.manual_seed(7)
        cfg=SmolLM3Config(vocab_size=32,pad_token_id=0,bos_token_id=1,eos_token_id=2,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
            num_attention_heads=4,num_key_value_heads=2,head_dim=8,no_rope_layers=[1,0],
            max_position_embeddings=32,tie_word_embeddings=True,attention_dropout=0.,use_cache=False)
        cfg._attn_implementation='sdpa';model=SmolLM3ForCausalLM(cfg).float().eval()
        requests=[dict(context=[1,2,3],continuation=[4,5]),dict(context=[2],continuation=[6])]
        scored=smol_score(model,requests,2,max_length=16)
        for r,(ll,greedy) in zip(requests,scored['scores']):
            ids=torch.tensor([r['context']+r['continuation']]);n=len(r['continuation'])
            with torch.no_grad():logits=model(ids[:,:-1]).logits[0,-n:].float()
            target=ids[0,-n:]
            self.assertAlmostEqual(ll,float(logits.log_softmax(-1).gather(1,target[:,None]).sum()),places=5)
            self.assertEqual(greedy,bool(torch.equal(logits.argmax(-1),target)))

class F9Admission(unittest.TestCase):
    def test_prepared_recovery_preserves_source_and_rejects_duplicate(self):
        import tempfile
        from pathlib import Path
        from scripts.activate_a2_family_f9_prepared import validate_prepared
        from eval.a2_family_f9 import write,read,SPEC_SHA
        from scripts.a2_quant_io import lock
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);result=root/'results/a2_family_20261107/downstream'
            source=root/'source';source.mkdir();(source/'fixture.py').write_text('original source')
            from efficient_llm.diagnostic_data import sha256
            manifest=write(root/'source-manifest.json',dict(code_commit='',files={'fixture.py':sha256(source/'fixture.py')}))
            att=write(result/'SOURCE_COMMIT_ATTESTATION.json',dict(verified=True,actual_source_commit='a'*40,
                checks={'fixture.py':True},source_manifest_sha256=manifest['sha256']))
            inv=write(result/'inventory.json',dict(models=[{}]*189,hash_failures=[]))
            cfg=write(result/'preparation/ready_config.json',dict(downstream_root=str(result),source_manifest=manifest,
                locked_inputs=[],inventory=inv,benchmarks={},code_commit=''))
            ready=write(result/'preparation/ready.json',dict(cpu_ready=True,models=189,flags=[],config=cfg))
            failed=write(root/'jobs.json',dict(jobs=[{'fixture':True}]))
            out=validate_prepared(root,ready['sha256'],att['sha256'],'a'*40)
            self.assertEqual(out['code_commit'],'a'*40)
            self.assertEqual(read(cfg['path'])['code_commit'],'')
            self.assertEqual(sha256(failed['path']),failed['sha256'])
            (source/'fixture.py').write_text('modified')
            with self.assertRaisesRegex(ValueError,'Frozen source changed'):
                validate_prepared(root,ready['sha256'],att['sha256'],'a'*40)
            (root/'supervised').mkdir()
            with self.assertRaisesRegex(ValueError,'refuse duplicate'):
                validate_prepared(root,ready['sha256'],att['sha256'],'a'*40)

    def test_launcher_rejects_missing_commit_before_any_io(self):
        from scripts.launch_a2_family_f9 import launch
        for value in ('','HEAD','2a74421'):
            with self.assertRaisesRegex(ValueError,'Explicit verified source commit'):
                launch('/does-not-exist','/must-not-create','fixture',value)

    def test_queue_order_and_scientific_failure_continuation(self):
        import tempfile,threading,time,json
        from pathlib import Path
        from scripts.a2_family_f9_run import Queue
        from eval.a2_family_f9 import write,read,SPEC_SHA
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'downstream';root.mkdir();events=[];eventlock=threading.Lock()
            class FixtureQueue(Queue):
                def verify_source(self):pass
                def score(self,e,g,b,label,load_only=False,spot=False):
                    with eventlock:events.append((label,e['model_id']))
                    if load_only:return dict(loaded=True)
                    if e['model_id']=='late-failure':raise ValueError('Fixture scoring failure')
                    folder=root/label/e['model_id']
                    return dict(scores=write(folder/'scores.json',{'scores':[[-1.,True]]}),
                        replay=write(folder/'replay.json',{'fixture':True}),batch=b)
                def report(self,name):
                    events.append(('report',name));(root/(name+'.md')).write_text(name)
            queue=FixtureQueue.__new__(FixtureQueue);queue.root=root;queue.start=time.monotonic()
            queue.entries=[dict(model_id=name,batch_family=family,phase=phase,part=name,spec_sha256=SPEC_SHA)
                for name,family,phase in [('p','opensci',1),('ref','opensci13',1),('smol','smollm3',1),
                    ('headline','opensci',2),('late-failure','smollm3',3)]]
            previous=root/'original.md';previous.write_text('original results')
            from scripts.a2_quant_io import lock
            queue.cfg=dict(previous_report=lock(previous));queue.flags=[];queue.outputs={};queue.mutex=threading.Lock();queue.batches={}
            with patch('scripts.a2_family_f9_run.batch_decision',return_value=dict(passed=True,batch_size=2)):
                queue.run()
            first_score=next(i for i,e in enumerate(events) if e[0]!='FT12_loads')
            self.assertEqual(first_score,5)
            self.assertLess(events.index(('report','INTERIM')),events.index(('scores','late-failure')))
            self.assertLess(events.index(('scores','headline')),events.index(('report','INTERIM')))
            done=read(root/'complete.json');self.assertTrue(done['execution_complete']);self.assertFalse(done['coverage_complete'])
            self.assertEqual(done['models'],4)
            self.assertTrue(any('late-failure' in x['test'] for x in done['flags']))
            self.assertEqual(previous.read_text(),'original results')

    def test_native_job_manifest_requires_only_evaluation_and_backup(self):
        import json,tempfile
        from pathlib import Path
        from scripts.launch_a2_family_f9 import jobs,save_jobs
        from run_experiments import load_jobs
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'jobs.json';save_jobs(p,'/cfg.json',Path('/result/downstream'))
            value=load_jobs(p)
            self.assertEqual(len(value),2)
            self.assertNotIn('train',json.dumps(value))
            self.assertEqual(set(json.loads(p.read_text())),{'jobs'})
            with self.assertRaises(ValueError):save_jobs(p,'/cfg.json',Path('/result/downstream'))

    def test_inventory_hash_rejects_changed_weight_and_missing_metadata(self):
        import tempfile,json,hashlib
        from pathlib import Path
        from scripts.a2_family_f9_prepare import hash_entry
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'config.json').write_text('{}');(root/'model.safetensors').write_bytes(b'fixture')
            expected=hashlib.sha256(b'fixture').hexdigest()
            entry=dict(path=directory,family='smollm3',expected_files={'model.safetensors':expected},
                model_id='fixture',run_id='fixture',parent_id='k6',dataset='fwe',arm='k6',cut=9,
                variant='drop',budget=.2,lr=5e-5,seed=2030,trained_node='fixture',part='F2',phase=2,batch_family='smollm3')
            result=hash_entry(entry);self.assertEqual(result['file_hashes'][str(root/'model.safetensors')],expected)
            (root/'model.safetensors').write_bytes(b'changed')
            with self.assertRaises(ValueError):hash_entry(entry)

    def test_science_archive_reuse_is_self_contained(self):
        import tempfile,json,tarfile
        from pathlib import Path
        from scripts.a2_family_f9_backup import backup
        from scripts.a2_quant_io import lock
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);root=base/'downstream';root.mkdir()
            original=base/'original.json';original.write_text('{}')
            (root/'scored_index.json').write_text(json.dumps(dict(models={'m':dict(scores=lock(original),replay=lock(original))})))
            result=backup(root);proof=json.loads(Path(result['path']).read_text());self.assertTrue(proof['verified'])
            with tarfile.open(root/'backup/family_f9_scientific.tar.gz') as archive:
                self.assertIn('reused/m/scores.json',archive.getnames())
                self.assertIn('reused/m/replay.json',archive.getnames())
            self.assertEqual(original.read_text(),'{}')

if __name__=='__main__':unittest.main()
