"""Read-only full-checkpoint task numerics against FP32 math SDPA.

No optimizer or acceptance-threshold changes. Preserve every comparison,
including failures, before deciding whether a production gate should change.
"""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from eval.finetune import PairClassifier, PairCollator, encode_pairs
from eval.models import load_checkpoint
from scripts.finetune_study import write


def difference(value, reference):
    value, reference = value.detach().double(), reference.detach().double()
    delta = value - reference
    return dict(relative_l2=float(delta.norm()/reference.norm().clamp_min(1e-30)),
                max_absolute=float(delta.abs().max()), rms=float(delta.square().mean().sqrt()),
                reference_norm=float(reference.norm()))


def diagnose(checkpoint, output, task, data_root=None, manifest=None):
    torch.manual_seed(42)
    adapter, tokenizer_path, source = load_checkpoint(checkpoint, 'cuda', attention_backend='fa4')
    model = PairClassifier(adapter.wrapped, 1 if task == 'stsb' else 2, 42).to('cuda')
    # Reproduce the original failing synthetic fixture exactly.
    ids = torch.randint(10, 10000, (2, 64)).tolist()
    batches = {'synthetic': [dict(input_ids=ids[0][:37], labels=.5 if task == 'stsb' else 0),
                             dict(input_ids=ids[1], labels=3.5 if task == 'stsb' else 1)]}
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    if data_root:
        from datasets import load_dataset
        from eval.benchmarks import local_dataset_paths
        mapping = local_dataset_paths(data_root, manifest)[task]
        # Training data only: no final-holdout metric enters this diagnostic.
        raw = load_dataset(mapping['dataset_path'], **mapping['dataset_kwargs'])['train']
        rows = raw.select([0, 1]).to_dict()
        rows['label'] = [5*x for x in rows['score']] if task == 'stsb' else rows['label']
        fields = ('sentence1', 'sentence2') if task == 'stsb' else ('passage', 'question')
        encoded = encode_pairs(rows, tokenizer, fields, 512, task_name=task)
        batches['real_training'] = [dict(input_ids=x, labels=y) for x, y in zip(encoded['input_ids'], encoded['labels'])]
    model.eval()
    pooled = []
    hook = model.score.register_forward_pre_hook(lambda module, args: pooled.append(args[0].detach().cpu()))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report = dict(status='diagnosed', task=task, source=source, batches={})
    for name, rows in batches.items():
        inputs = {k:v.cuda() for k,v in PairCollator(tokenizer.pad_token_id or 0, regression=task=='stsb')(rows).items()}
        runs = {}
        for backend, precision in [('fa4', 'bf16'), ('sdpa', 'bf16'), ('math', 'fp32')]:
            model.zero_grad(set_to_none=True)
            model.wrapped.attention_backend = 'fa4' if backend == 'fa4' else 'sdpa'
            context = sdpa_kernel(SDPBackend.MATH) if backend == 'math' else nullcontext()
            pooled.clear()
            with context, torch.autocast('cuda', dtype=torch.bfloat16, enabled=precision=='bf16'):
                out = model(**inputs)
            out['loss'].backward()
            gradients = {k:p.grad.detach().cpu().clone() for k,p in model.named_parameters()
                         if p.requires_grad and p.grad is not None}
            logits = out['logits'].detach().cpu()
            features = pooled[0].float()
            head_fp32 = F.linear(features, model.score.weight.detach().cpu(), model.score.bias.detach().cpu())
            runs[backend] = dict(logits=logits, loss=float(out['loss'].detach()), features=features,
                                 head_fp32=head_fp32, gradients=gradients)
        comparisons = {}
        for a, b in [('fa4','sdpa'),('fa4','math'),('sdpa','math')]:
            left,right=runs[a],runs[b]
            assert left['gradients'].keys()==right['gradients'].keys()
            numer=denom=0.
            for k,g in left['gradients'].items():
                ref=right['gradients'][k].double();delta=g.double()-ref
                numer+=float(delta.square().sum());denom+=float(ref.square().sum())
            comparisons[a+'-vs-'+b]=dict(logits=difference(left['logits'],right['logits']),
                hidden=difference(left['features'],right['features']),
                fp32_head_logits=difference(left['head_fp32'],right['head_fp32']),
                loss_absolute=abs(left['loss']-right['loss']), gradient_relative_l2=(numer/max(denom,1e-30))**.5,
                **({'probabilities':difference(left['logits'].softmax(-1),right['logits'].softmax(-1))} if task!='stsb' else {}))
        report['batches'][name]=dict(lengths=[len(r['input_ids']) for r in rows], comparisons=comparisons,
            outputs={key:dict(logits=r['logits'].tolist(),loss=r['loss'],fp32_head_logits=r['head_fp32'].tolist()) for key,r in runs.items()})
        del runs
    hook.remove()
    write(output,report)
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--task',choices=('stsb','boolq'),required=True)
    p.add_argument('--data-root');p.add_argument('--manifest')
    args=p.parse_args()
    diagnose(args.checkpoint,args.output,args.task,args.data_root,args.manifest)
