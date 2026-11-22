import json
from pathlib import Path
import unittest

import torch

import test_embedding_anchor_prepare as fixtures
from test_embedding_anchor_train import equal_tree
from efficient_llm.warm_order.data import sha
from efficient_llm.warm_order.embedding_anchor_prepare import prepare
from efficient_llm.warm_order.embedding_anchor_device import load_prepared,check_references,DeviceTrainer
from efficient_llm.warm_order.embedding_anchor_runtime import OLD_GUARD,NEW_GUARD


class PreparedDeviceTests(unittest.TestCase):
    setUp=fixtures.AnchorPreparationTests.setUp
    tearDown=fixtures.AnchorPreparationTests.tearDown

    def historical_fixture(self):
        base=self.root/'run';base.mkdir();prepare(self.path,sha(self.path),base/'prepared',fixture_only=True)
        root=base/'prepared/s0';meta=json.loads((root/'config.json').read_text())
        for name in meta['source']:
            origin=Path('efficient_llm')/name;dest=base/'source/efficient_llm'/name
            dest.parent.mkdir(parents=True,exist_ok=True)
            text=origin.read_text()
            if name=='warm_order/embedding_anchor.py':text=text.replace(NEW_GUARD,OLD_GUARD)
            dest.write_text(text);meta['source'][name]=sha(dest)
        state=torch.load(root/'initial-fullstate.pt',weights_only=False)
        state['recipe']['source']=dict(meta['source'])
        torch.save(state,root/'initial-fullstate.pt');meta['initial_sha256']=sha(root/'initial-fullstate.pt')
        (root/'config.json').write_text(json.dumps(meta))
        manifest=json.loads((base/'prepared/prepared.json').read_text())
        manifest['seeds']['0']['config_sha256']=sha(root/'config.json')
        (base/'prepared/prepared.json').write_text(json.dumps(manifest))
        return root,sha(root/'config.json')

    def test_immutable_old_cpu_bridge_and_resume(self):
        root,h=self.historical_fixture();before=sha(root/'initial-fullstate.pt')
        t,meta=load_prepared(root,h)
        assert all(v==0 for v in check_references(t,meta,root)['max_abs'].values())
        t.advance(4);p=self.root/'device.pt';cp=t.save(p)
        resumed=DeviceTrainer.restore(p,cp,t.model.c,t.data,prepared=dict(path=str(root),sha256=h))
        equal_tree(t.state(),resumed.state())
        assert sha(root/'initial-fullstate.pt')==before
        (root/'cpu-reference.pt').write_bytes((root/'cpu-reference.pt').read_bytes()+b'tamper')
        with self.assertRaisesRegex(ValueError,'hash'):load_prepared(root,h)

    def test_complete_bounded_qualification_pipeline(self):
        from efficient_llm.warm_order.embedding_anchor_qualify import qualify_seed
        root,h=self.historical_fixture();base=root.parent
        report=qualify_seed(base,sha(base/'prepared.json'),0,self.root/'qualification',device='cpu',fixture_steps=2)
        assert report['lineage_matches_single_segment'] and report['actual_updates_executed']==22
        assert not report['production_fit_admitted'] and report['production_updates_executed']==0
        assert len(report['segments'])==5
