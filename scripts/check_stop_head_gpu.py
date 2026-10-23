"""Check all fresh warm starts and real-width FP32 STOP-head training on CUDA."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from datasets import Dataset
from bdot.model import BDOTForCausalLM
from eval.stop_study import fit_head,score
from packed_fa4 import register_backend


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1);backend=register_backend();results={}
    for name,cp in json.loads(a.checkpoints.read_text()).items():
        # The full runs are not created yet; this smoke reads their immutable init.
        run=Path(cp).parent
        init=run.parent.parent/(name+'-init')
        m=BDOTForCausalLM.from_pretrained(init,attn_implementation=backend).cuda().eval()
        assert int(m.completed_steps)==0
        assert all(not x.requires_grad for x in m.stop_controller.parameters())
        assert m.options.phase(0.,True)==(m.options.r,False,0.)
        rows=[dict(input_ids=list(range(100,164)),labels=list(range(100,164)),
                   attention_mask=[1]*64,segments=[0]*32+[1]*32) for _ in range(4)]
        ds=Dataset.from_list(rows)
        result=fit_head(m,ds,'fa4',size=2,samples_per_row=8,epochs=2)
        assert all(p.grad is None for n,p in m.named_parameters() if not n.startswith('stop_controller.'))
        assert not any(p.requires_grad for p in m.parameters())
        result['mixed_score']=score(m,ds,'fa4',dict(policy='learned',threshold=.5),size=2)
        results[name]=result
        del m
        torch.cuda.empty_cache()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(dict(status='ok',branches=results),f,indent=2);f.write('\n')


if __name__=='__main__':main()
