import copy
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from eval.a2_request_bridge import digest
from eval.a2_suite import scoring_jobs, execute_scoring, collect, collect_files
from eval.downstream_suite import COUNTS,PRIMARY
from efficient_llm.diagnostic_data import sha256


def write_lock(path,value):
    path.write_text(json.dumps(value));return dict(path=str(path),sha256=sha256(path))


class EvalSuiteTests(unittest.TestCase):
    def test_fixed_screen_requires_all_matched_checkpoints(self):
        from eval.a2_suite import select_initial_screen
        from efficient_llm.a2.protocol import Run
        rows=[]
        for ds in ('dclm','fwe','nem'):
            for parent in ('P300','P1T'):
                for budget in (('0.2B','0.8B') if ds=='nem' else ('0.2B',)):
                    rows.append(dict(Run(ds,parent,'S1',budget,3e-4).record(),status='complete',diverged=False))
        selected=select_initial_screen(rows)
        self.assertEqual(len(selected),14)
        self.assertEqual(sum(r['kind']=='parent' for r in selected.values()),6)
        with self.assertRaises(ValueError):select_initial_screen(rows[:-1])
        with self.assertRaises(ValueError):select_initial_screen(rows+[rows[0]])
        changed=copy.deepcopy(rows);changed[0]['set']='S2'
        with self.assertRaises(ValueError):select_initial_screen(changed)

    def fixture(self):
        tasks=list(COUNTS);requests=[dict(context=[0],continuation=[2]) for _ in tasks]
        protocol=dict(tasks=tasks,limit=1,seed=42,num_fewshot=0,max_length=4096)
        prepared=dict(schema=1,requests=requests,request_sha256=digest(requests),
                      protocol=protocol,protocol_sha256=digest(protocol))
        identities={name:dict(name=name) for name in ('parent','healed')}
        scores={};replays={}
        for name,identity in identities.items():
            scores[name]=dict(checkpoint_identity=identity,checkpoint_identity_sha256=digest(identity),
                prepared_sha256=digest(prepared),request_sha256=prepared['request_sha256'],
                likelihood_requests=9,scores=[[-1.,False]]*9,max_length=4096,
                precision='fp32_parameters_bf16_cuda_autocast',packages={'torch':'fixture'})
            replays[name]=dict(a2_metadata=dict(protocol=protocol,prepared_sha256=digest(prepared),
                request_sha256=prepared['request_sha256']),
                results={t:{PRIMARY[t]:.5} for t in tasks},
                samples={t:[dict(doc_id=0,doc={'q':'fixture'},arguments=['q','a'],target=0)] for t in tasks},
                **{'n-samples':{t:dict(original=COUNTS[t],effective=1) for t in tasks},
                   'n-shot':{t:0 for t in tasks}})
        return identities,prepared,scores,replays

    def test_collection_and_exact_score_receipt(self):
        identities,prepared,scores,replays=self.fixture()
        result=collect(identities,prepared,scores,replays)
        self.assertEqual(result['scores']['healed']['multiple_choice_average'],.5)
        self.assertTrue(result['diagnostic'])
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);pl=write_lock(root/'requests.json',prepared)
            sl={k:write_lock(root/(k+'-scores.json'),v) for k,v in scores.items()}
            for k,v in replays.items():v['a2_metadata']['scored_file_sha256']=sl[k]['sha256']
            rl={k:write_lock(root/(k+'-replay.json'),v) for k,v in replays.items()}
            self.assertEqual(collect_files(identities,pl,sl,rl),result)
            replays['parent']['a2_metadata']['scored_file_sha256']='0'*64
            rl['parent']=write_lock(root/'parent-replay.json',replays['parent'])
            with self.assertRaises(ValueError):collect_files(identities,pl,sl,rl)

    def test_reject_unpaired_or_incomplete_scores(self):
        identities,prepared,scores,replays=self.fixture()
        for key,value in [('max_length',1024),('precision','bf16'),('scores',[]),
                          ('checkpoint_identity',{'name':'wrong'})]:
            changed=copy.deepcopy(scores);changed['healed'][key]=value
            with self.assertRaises(ValueError):collect(identities,prepared,changed,replays)
        changed=copy.deepcopy(replays);changed['healed']['samples']['boolq'][0]['arguments']=['different']
        with self.assertRaises(ValueError):collect(identities,prepared,scores,changed)
        changed=copy.deepcopy(replays);changed['healed']['samples']['boolq']=[]
        with self.assertRaises(ValueError):collect(identities,prepared,scores,changed)

    def test_eight_lanes_serialize_ninth_checkpoint_without_gpu_work(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);identities={f'm{i}':write_lock(root/f'm{i}.json',{'i':i}) for i in range(9)}
            prepared=write_lock(root/'requests.json',{})
            jobs=scoring_jobs(identities,prepared,root/'outputs','/explicit/native/python',list(range(8)))
            self.assertEqual([j['gpu'] for j in jobs],list(range(8))+[0])
            active=set();order=[];lock=threading.Lock()
            def fake(argv,**kw):
                gpu=kw['env']['CUDA_VISIBLE_DEVICES']
                with lock:
                    self.assertNotIn(gpu,active);active.add(gpu)
                self.assertNotIn('WORLD_SIZE',kw['env'])
                target=Path(argv[argv.index('--output')+1]);time.sleep(.005);target.write_text('{}')
                with lock:order.append(target.parent.name);active.remove(gpu)
            with patch('subprocess.run',side_effect=fake):execute_scoring(jobs,timeout_seconds=10)
            self.assertLess(order.index('m0'),order.index('m8'))
            with self.assertRaises(ValueError):scoring_jobs(identities,prepared,root/'outputs','/explicit/native/python',list(range(8)))


if __name__=='__main__':unittest.main()
