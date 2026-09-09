"""Check actual harness forward precision on tiny arms; no datasets/downloads.

This is a regression gate, NOT a benchmark. Both precision cases must pass;
also require FP32 softmax and unchanged FP32 master-weight precision.
Run on B200 only after authorization and the normal free-GPU checks.
"""
import argparse
from pathlib import Path
import sys
from unittest.mock import patch

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.runtime import offline
offline()
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from capacity_allocation.data import write_json
from capacity_allocation.modeling import ARMS, build_model, experiment_config
from eval.benchmarks import evaluate


def check(device='cpu', arms=ARMS):
    torch.set_num_threads(2)
    vocabulary={'[UNK]':0,'hello':1,**{f'w{i}':i for i in range(2,96)},'[EOS]':96}
    backend=Tokenizer(models.WordLevel(vocabulary,unk_token='[UNK]'))
    backend.pre_tokenizer=pre_tokenizers.Whitespace()
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=backend,unk_token='[UNK]',eos_token='[EOS]',pad_token='[EOS]')
    reports=[]
    class ProbeComplete(Exception): pass
    for arm in arms:
        model=build_model(experiment_config(arm,tiny=True)).to(device).eval()
        for precision in ('fp32','bf16'):
            observed=[]
            softmax_dtypes=[]
            hook=model.model.layers[0].self_attn.q_proj.register_forward_hook(
                lambda module, inputs, output: observed.append(str(output.dtype)))
            def forward_only(model, **kwargs):
                # Execute the real HFLM _model_call constructed by our wrapper.
                # Stop before scoring fabricated tasks; no synthetic accuracy.
                logits=model._model_call(torch.tensor([[1,1,1]],device=device))
                scores=torch.nn.functional.log_softmax(logits,dim=-1,dtype=model.softmax_dtype)
                softmax_dtypes.append(str(scores.dtype))
                raise ProbeComplete
            try:
                with patch('lm_eval.simple_evaluate',side_effect=forward_only):
                    evaluate(model,tokenizer,{'precision_probe':object()},device=device,precision=precision,batch_size=1)
            except ProbeComplete:
                pass
            finally:
                hook.remove()
            expected='torch.bfloat16' if precision=='bf16' else 'torch.float32'
            master_dtypes=sorted({str(p.dtype) for p in model.parameters()})
            reports.append(dict(arm=arm,precision=precision,expected=expected,observed=observed,
                softmax_dtypes=softmax_dtypes,master_dtypes=master_dtypes,
                passed=bool(observed) and set(observed)=={expected}
                    and softmax_dtypes==['torch.float32'] and master_dtypes==['torch.float32']))
    return dict(success=all(r['passed'] for r in reports),device=device,smoke_only=True,checks=reports)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--arms',nargs='+',choices=ARMS,default=list(ARMS))
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    path=Path(args.output)
    if path.exists(): parser.error('Fresh output required')
    if args.device=='cuda' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        parser.error('CUDA BF16 unavailable')
    report=check(args.device,args.arms)
    path.parent.mkdir(parents=True,exist_ok=True)
    write_json(path,report)
    print('EVAL_PRECISION_PASS' if report['success'] else 'EVAL_PRECISION_FAIL',path,flush=True)
    if not report['success']: raise SystemExit(1)


if __name__=='__main__': main()
