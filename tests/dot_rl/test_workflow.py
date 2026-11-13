import dataclasses,json,tempfile,unittest
from pathlib import Path
import numpy as np
from efficient_llm.dot_pool.rl import write,sha,SPEC_SHA
from efficient_llm.dot_pool.rl_run import main_run
from tests.dot_rl.test_core import tiny,rows

class Workflow(unittest.TestCase):
    def test_main_real_cpu_entrypoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'data';data.mkdir()
            for name in ['train','train_compositions','atoms','id_dev','id_test','rl_train','rl_dev','ood_dev','ood_test','ood_rl_atoms','ood_test_atoms']:np.save(data/(name+'.npy'),rows())
            write(data/'manifest.json',dict(files={p.name:dict(sha256=sha(p)) for p in data.iterdir()}))
            write(root/'budget.json',dict(epochs=1,switch_epoch=0))
            for arm in ['L','P-rand','J-RL','WS-RL']:
                cfg=dict(arm=arm,seed=0,model=dataclasses.asdict(tiny()),batch=8,lr=.001,data=str(data),data_sha256=sha(data/'manifest.json'),output=str(root/arm),budget_path=str(root/'budget.json'),development_only=True,spec_sha256=SPEC_SHA)
                main_run(cfg)
                self.assertTrue(json.loads((root/arm/'complete.json').read_text())['complete'])
                self.assertEqual(json.loads((root/arm/'train_complete.json').read_text())['step'],1)
                self.assertTrue((root/arm/'final_paths.npz').exists())
