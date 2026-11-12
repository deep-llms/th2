import contextlib,dataclasses,io,json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from efficient_llm.dot_pool import SPEC_SHA
from efficient_llm.dot_pool.model import Config
from efficient_llm.dot_pool.train import run
from scripts.dot_pool_e1b_analysis import same_relation
from scripts.dot_pool_e1b_checks import counts
from scripts.dot_pool_e1b_queue import choose_budget

class E1b(unittest.TestCase):
    def test_counts_actual_architecture(self):
        for arm in ['D','L','SO','DOT']:counts(arm)
    def test_budget_slower_seed_and_minimum(self):
        def r(seconds):return dict(step=15400,start_step=0,training_seconds=seconds)
        self.assertEqual(choose_budget([r(2000),r(2200)])['epochs'],3500)
        v=choose_budget([r(8000),r(10000)]);self.assertEqual(v['epochs'],2000);self.assertTrue(v['minimum_overruns_11h'])
        with self.assertRaises(ValueError):choose_budget([r(2000)])
    def test_same_relation_excludes_same_bridge_and_no_partner(self):
        a=np.array([[[i,i+1]]*4 for i in [0,0,0,3,8]])
        b=np.array([[[i,i+1]]*4 for i in [0,0,3,0,8]])
        rel=np.array([1,1,1,1,2]);bridge=np.array([5,5,6,7,8]);correct=np.ones(5,bool)
        x=same_relation(a,b,rel,bridge,correct,draws=100)
        self.assertEqual(x['partner_counts'],[2,2,3,3,0]);self.assertEqual(x['groups']['all']['excluded_no_partner'],1)
        for i in range(4):
            partners=[j for j in range(5) if j!=i and rel[j]==rel[i] and bridge[j]!=bridge[i]]
            aa=set(a[i].flat);expected=np.mean([len(aa&set(b[j].flat))/len(aa|set(b[j].flat)) for j in partners])
            self.assertEqual(x['chance_same'][i],expected)
    def test_relation_only_router_has_zero_excess(self):
        rel=np.array([1,1,2,2]);a=np.array([[[r,r+1]]*4 for r in rel]);x=same_relation(a,a,rel,np.arange(4),np.array([1,0,1,0],bool),draws=50)
        self.assertEqual(x['groups']['all']['excess_same'],0);self.assertEqual(x['groups']['all']['label'],'not bridge-specific')
    def test_real_training_split_resume_same_future_losses(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);data=root/'data';data.mkdir();arr=np.array([[1,2,3,4],[2,3,4,5],[4,5,6,7],[5,6,7,8]]*2)
            for name in ['train','id_dev','ood_dev','id_test','ood_test','atoms','ood_atoms']:np.save(data/(name+'.npy'),arr)
            mc=Config(d=16,heads=2,kv_heads=1,head_dim=8,ff=24,expert_ff=12,middle=8,pool=8,vocab=2240,candidates=4,arm='DOT',label_rate=1,stem=0,coda=0)
            base=dict(study='E1',arm='DOT',seed=0,steps=20,batch=4,microbatch=4,lr=.001,eval_batch=4,eval_every=4,checkpoint_epochs=2,model=dataclasses.asdict(mc),data=str(data),spec_sha256=SPEC_SHA)
            with contextlib.redirect_stdout(io.StringIO()):
                run(dict(base,output=str(root/'full')))
                run(dict(base,steps=10,output=str(root/'first')))
                run(dict(base,output=str(root/'second'),resume=str(root/'first/final_state')))
            a=torch.load(root/'full/final_state/state.pt',weights_only=False);b=torch.load(root/'second/final_state/state.pt',weights_only=False)
            for k in a['model']:torch.testing.assert_close(a['model'][k],b['model'][k],atol=0,rtol=0)
            self.assertEqual(json.loads((root/'full/final_eval.json').read_text())['metrics'],json.loads((root/'second/final_eval.json').read_text())['metrics'])
            self.assertEqual(len(list((root/'full').glob('checkpoint-*'))),2)
            self.assertEqual(json.loads((root/'second/train_complete.json').read_text())['start_step'],10)

    def test_complete_report_and_replay_from_saved_fixture(self):
        from scripts.dot_pool_e1b_report import report
        from scripts.dot_pool_e1b_analysis import analyze
        from scripts.dot_pool_evaluate import e1_analysis
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);root=base/'run';root.mkdir();data=base/'data';data.mkdir();pilot=root/'pilot_same_relation';pilot.mkdir()
            q=np.array([[1,2000,2001,3],[2,2000,2001,4],[3,2002,2003,5],[4,2002,2003,6]])
            b=np.array([[6,2001,3,2200],[7,2001,4,2200],[8,2003,5,2200],[9,2003,6,2200]])
            np.save(data/'ood_test.npy',q);np.save(data/'ood_bridge.npy',b)
            def put(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))
            runs=[]
            for arm in ['D','L','SO','DOT']:
                for seed in [0,1]:
                    name=f'E1b-{arm}-s{seed}';path=root/name;warm=root/f'first200-{arm}-s{seed}'
                    c=dict(name=name,arm=arm,seed=seed,first200_output=str(warm),output=str(path),data=str(data),model=dict(stem=0,coda=0,middle=8),spec_sha256=SPEC_SHA,counts=counts(arm))
                    runs.append(c);put(path/'config.json',c);put(path/'train_complete.json',dict(step=154000,elapsed=10));put(warm/'train_complete.json',dict(step=15400,elapsed=1))
                    metrics={k:dict(accuracy=.5) for k in ['id_test','ood_test','id_dev','ood_dev','train','ood_atoms']}
                    put(path/'final_eval.json',dict(metrics=metrics));put(path/'artifact_manifest.json',dict(artifacts=[]))
                    for epoch in [1500,2000]:put(path/f'eval-{epoch*77}.json',dict(step=epoch*77,metrics=metrics))
                    put(warm/'eval-15400.json',dict(step=15400,metrics=metrics))
                    if arm in ['SO','DOT']:
                        routes={k:np.tile(np.array([[[0,1]]*8]),(4,1,1)) for k in ['ood_test_ids','ood_bridge_ids']}
                        routes.update({k+'_correct':np.array([True,False,True,False]) for k in ['id_test','ood_test','id_dev','ood_dev']})
                        np.savez(path/'routes_sample.npz',**routes);e1_analysis(path);v=analyze(path/'routes_sample.npz',data,path/'reentry_same_relation.json');put(pilot/f'E1-{arm}-s{seed}.json',v)
            for seed in [0,1]:put(base/f'run-a03/E1-SR-s{seed}/final_eval.json',dict(metrics={'ood_test':dict(accuracy=0)}))
            put(root/'run_index.json',dict(runs=runs,flags=[]));put(root/'budget.json',dict(epochs=2000));put(root/'tests_summary.json',dict(passed=True))
            report(root)
            v=json.loads((root/'report_complete.json').read_text());self.assertTrue(v['all_eight_complete']);self.assertTrue(v['ready_for_small_export'])
            x=json.loads((root/'report/readings.json').read_text());self.assertEqual(x['readings']['R2']['label'],'no clear difference')
            with self.assertRaises(FileExistsError):report(root)

if __name__=='__main__':unittest.main()
