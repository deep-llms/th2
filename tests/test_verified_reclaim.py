import copy
import unittest
from scripts.verified_gpu_reclaim import validate


class VerifiedReclaimTests(unittest.TestCase):
    def fixture(self):
        status=[{'index':i,'uuid':f'gpu-{i}','pids':[100+i]} for i in range(8)]
        records={str(100+i):{'pid':100+i,'ppid':50,'start_ticks':1000+i,'command_sha256':'worker',
                 'exe':'python','command_prefix':['python'],'launcher_kind':None} for i in range(8)}
        records['50']={'pid':50,'ppid':1,'start_ticks':900,'command_sha256':'parent','exe':'python',
                       'command_prefix':['python','polite_burn.py'],'launcher_kind':'gpu_burn'}
        expected={'host':'thiennh-p6-tpbw-worker-0','gpus':copy.deepcopy(status),
                  'workers':list(range(100,108)),'stop_roots':[50],'processes':copy.deepcopy(records)}
        return expected,status,records,expected['host']

    def test_exact_inspected_ownership_only(self):
        validate(*self.fixture())
        for mutation in ('pid','parent','command','gpu','uuid','host','unknown_parent','unrelated_parent','pid1'):
            expected,status,records,host=self.fixture()
            if mutation=='pid':records['100']['start_ticks']+=1
            elif mutation=='parent':records['50']['start_ticks']+=1
            elif mutation=='command':records['100']['command_sha256']='new'
            elif mutation=='gpu':status[0]['pids'].append(999)
            elif mutation=='uuid':status[0]['uuid']='other-node-device'
            elif mutation=='host':host='other-host'
            elif mutation=='unknown_parent':
                records['50']['launcher_kind']=None;expected['processes']=copy.deepcopy(records)
            elif mutation=='unrelated_parent':
                for i in range(100,108):records[str(i)]['ppid']=1
                expected['processes']=copy.deepcopy(records)
            elif mutation=='pid1':expected['stop_roots']=[1]
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                validate(expected,status,records,host)
