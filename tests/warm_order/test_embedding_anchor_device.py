import copy
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

import torch

import test_embedding_anchor_eval as fixtures
from test_embedding_anchor_train import equal_tree
from efficient_llm.warm_order.embedding_anchor_runtime import verify_forward_bridge, anchor_runtime, anchor_device, OLD_GUARD, NEW_GUARD
from efficient_llm.warm_order.embedding_anchor_device import DeviceTrainer
from efficient_llm.warm_order.embedding_anchor_train import CPUStreamTrainer
from efficient_llm.warm_order.data import sha


class DeviceTests(unittest.TestCase):
    setUp = fixtures.AnchorDataEvaluationTests.setUp
    tearDown = fixtures.AnchorDataEvaluationTests.tearDown
    repin = fixtures.AnchorDataEvaluationTests.repin
    load = fixtures.AnchorDataEvaluationTests.load
    def test_device_guard_bridge(self):
        text=Path('efficient_llm/warm_order/embedding_anchor.py').read_text()
        old=self.root/'old.py';old.write_text(text.replace(NEW_GUARD,OLD_GUARD))
        verify_forward_bridge(old)
        old.write_text(old.read_text().replace('alpha * h','alpha * h + 1'))
        with self.assertRaisesRegex(ValueError,'computation'):verify_forward_bridge(old)
        with self.assertRaises(ValueError):
            with anchor_runtime('bad'):pass
        assert anchor_device()=='cpu'
        with patch('torch.cuda.device_count',return_value=0):
            with self.assertRaises(ValueError):
                with anchor_runtime('cuda'):pass

    def test_same_engine_warmup_and_resume(self):
        data=self.load()
        cpu=CPUStreamTrainer(self.c,data);cpu.advance(8)
        device=DeviceTrainer(self.c,data);device.advance(8)
        equal_tree(cpu.model.state_dict(),device.model.state_dict())
        equal_tree(cpu.optimizer.state_dict(),device.optimizer.state_dict())
        equal_tree(cpu.trace,device.trace)
        split=DeviceTrainer(self.c,data);split.advance(3)
        p=self.root/'resume.pt';h=split.save(p)
        resumed=DeviceTrainer.restore(p,h,self.c,data);resumed.advance(8)
        equal_tree(device.state(),resumed.state())
        with self.assertRaises(ValueError):DeviceTrainer(self.c,data,device='cuda')

    def test_device_phase_fork_and_stack_resume(self):
        data=self.load();warm=DeviceTrainer(self.c,data);warm.advance(768)
        p=self.root/'warm.pt';h=warm.save(p)
        for arm in ('A','C'):
            t=DeviceTrainer.restore(p,h,self.c,data,fork_arm=arm);t.advance(1152)
            start=self.root/(arm+'-start.pt');start_hash=t.save(start)
            t.advance(1156)
            part=DeviceTrainer.restore(start,start_hash,self.c,data);part.advance(1154)
            short=self.root/(arm+'-short.pt');sh=part.save(short)
            resumed=DeviceTrainer.restore(short,sh,self.c,data);resumed.advance(1156)
            equal_tree(t.state(),resumed.state())
            with self.assertRaises(ValueError):DeviceTrainer.restore(short,sh,self.c,data,fork_arm='C')

    def test_all_segment_resume_and_non_lineage(self):
        from efficient_llm.warm_order.embedding_anchor_segments import SEGMENTS,run
        data=self.load()
        for name in SEGMENTS:
            whole=run(DeviceTrainer(self.c,data),name,self.root/(name+'-whole.pt'),4)
            short=run(DeviceTrainer(self.c,data),name,self.root/(name+'-part.pt'),2)
            resumed=run(DeviceTrainer(self.c,data),name,self.root/(name+'-resumed.pt'),4,resume=short)
            a=torch.load(whole['path'],weights_only=False);b=torch.load(resumed['path'],weights_only=False)
            equal_tree(a,b)
            assert a['record']['kind']=='qualification_segment_not_lineage'
            assert 'step' not in a['state'] and 'arm' not in a['state']
            with self.assertRaises(ValueError):run(DeviceTrainer(self.c,data),name,self.root/'bad.pt',41)

    def test_device_evaluation_replay(self):
        from efficient_llm.warm_order.embedding_anchor_device_eval import evaluate
        from efficient_llm.warm_order.embedding_anchor_eval import replay
        data=self.load();t=DeviceTrainer(self.c,data);t.advance(768)
        path=self.root/'eval-warm.pt';h=t.save(path);warm=dict(path=str(path),sha256=h)
        runs={}
        for arm in ('A','C'):
            a=DeviceTrainer.restore(path,h,self.c,data,fork_arm=arm);a.advance(1152)
            p=self.root/('eval-'+arm+'.pt');runs[arm]=dict(path=str(p),sha256=a.save(p))
        report=evaluate(runs,warm,self.c,data,self.root/'eval',step=1152,ids=data.measurement_ids[:2])
        assert len(report['nll'])==72 and replay(self.root/'eval')['checkpoint_binding_verified']
        wrong=dict(A=runs['C'],C=runs['A'])
        with self.assertRaises(ValueError):evaluate(wrong,warm,self.c,data,self.root/'wrong',step=1152,ids=data.measurement_ids[:1])
