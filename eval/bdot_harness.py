"""Offline lm-eval wrapper: native harness task/scoring implementations, custom weights.

One process/device per invocation; independent task/checkpoint invocations may be
scheduled concurrently by the existing runner. No benchmark downloads here.
"""
import argparse
import json
import os
from pathlib import Path
import sys
os.environ.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  HF_HUB_DISABLE_TELEMETRY='1', WANDB_MODE='offline')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def harness_class():
    import torch
    from lm_eval.models.huggingface import HFLM

    class BDOTHarness(HFLM):
        """Reuse harness tokenization/scoring; each right-padded row is one document.

        No token-ID-based splitting: an EOS inside a prompt is ordinary context.
        Causal attention makes trailing padding invisible to every scored token.
        """
        attention_backend = 'sdpa'
        truncated_requests = 0
        likelihood_requests = 0

        def _model_call(self, inps, attn_mask=None, labels=None):
            if attn_mask is not None or labels is not None:
                raise ValueError('Only causal likelihood requests are supported')
            b, length = inps.shape
            kwargs = dict(input_ids=inps, use_cache=False,
                          position_ids=torch.arange(length, device=inps.device).expand(b, -1))
            if self.attention_backend == 'fa4':
                kwargs.update(attention_mask={'full_attention': None},
                    fa4_cu_seqlens=torch.arange(b+1, device=inps.device, dtype=torch.int32)*length,
                    fa4_max_seqlen=length)
            with torch.inference_mode(), torch.autocast(self.device.type,
                    dtype=self.mixed_precision_dtype, enabled=self.mixed_precision_dtype is not None):
                return self.model(**kwargs).logits

        def _loglikelihood_tokens(self, requests, **kwargs):
            self.likelihood_requests += len(requests)
            self.truncated_requests += sum(len(c)+len(t)>self.max_length+1 for _,c,t in requests)
            return super()._loglikelihood_tokens(requests, **kwargs)

        def generate_until(self, *args, **kwargs):
            raise ValueError('Generation requires a separately validated cache/generation adapter')

    return BDOTHarness


def make_harness(checkpoint,tokenizer_path=None,device='cpu',batch_size=1,max_length=1024,
                 attention_backend='sdpa', precision='fp32'):
    import torch
    from transformers import AutoTokenizer
    from train import ModelArguments,build_model
    if attention_backend not in ('sdpa', 'fa4') or precision not in ('fp32', 'bf16'):
        raise ValueError('Unsupported inference backend/precision')
    if attention_backend == 'fa4' and (not device.startswith('cuda') or precision != 'bf16'):
        raise ValueError('FA4 requires CUDA/BF16; no silent fallback')
    if batch_size < 1 or max_length < 2:
        raise ValueError('Invalid batch/context size')
    model=build_model(ModelArguments(model_name_or_path=str(checkpoint),attention_backend=attention_backend)).to(device).eval()
    # Do not call model.float(): it would also cast FP64 checkpoint counters.
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('Expected checkpoint FP32 master parameters; use explicit BF16 autocast')
    tokenizer=AutoTokenizer.from_pretrained(str(tokenizer_path or checkpoint),local_files_only=True)
    if tokenizer.pad_token is None:tokenizer.pad_token=tokenizer.eos_token
    recipe = Path(checkpoint)/'data_recipe.json'
    if recipe.is_file():
        from training_data import tokenizer_identity
        saved = json.loads(recipe.read_text())
        if tokenizer_identity(tokenizer) != saved['tokenizer'] or max_length > saved['block_size']:
            raise ValueError('Tokenizer or context differs from checkpoint training contract')
    elif max_length > model.config.max_position_embeddings:
        raise ValueError('Context exceeds model capacity')
    lm = harness_class()(pretrained=model,tokenizer=tokenizer,backend='causal',device=device,
        batch_size=batch_size,max_length=max_length,softmax_dtype='float32',
        mixed_precision_dtype='bfloat16' if precision=='bf16' else None)
    lm.attention_backend = attention_backend
    return lm


def json_default(value):
    if hasattr(value, 'tolist'):return value.tolist()
    if hasattr(value, 'item'):return value.item()
    return str(value)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--tokenizer')
    p.add_argument('--output',required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--batch-size',type=int,default=1);p.add_argument('--max-length',type=int,default=1024)
    p.add_argument('--tasks',nargs='+',default=['hellaswag','arc_easy','arc_challenge','piqa','winogrande','lambada_openai'])
    p.add_argument('--attention-backend', choices=['sdpa','fa4'], default='sdpa')
    p.add_argument('--precision', choices=['fp32','bf16'], default='fp32')
    p.add_argument('--dataset-root', required=True);p.add_argument('--manifest', required=True)
    p.add_argument('--seed', type=int, default=42);p.add_argument('--num-fewshot', type=int, default=0)
    p.add_argument('--limit', type=int)
    a=p.parse_args()
    if int(os.environ.get('WORLD_SIZE','1')) != 1:
        p.error('Use one process per invocation; distribute independent tasks across devices')
    out=Path(a.output)
    if out.exists():p.error('Output already exists')
    if a.num_fewshot < 0 or a.limit is not None and a.limit < 1:p.error('Invalid shots/limit')
    from eval.benchmark_data import local_dataset_paths, task_configs, benchmark_task_manager
    paths=local_dataset_paths(a.dataset_root,a.manifest)
    if set(a.tasks)-paths.keys():p.error('Task missing from verified local manifest')
    from importlib.metadata import version
    if version('lm-eval') != '0.4.13':raise ValueError('Require pinned lm-eval==0.4.13')
    import hashlib, time
    started=time.monotonic()
    from lm_eval import evaluator
    harness=make_harness(a.checkpoint,a.tokenizer,a.device,a.batch_size,a.max_length,
                         a.attention_backend,a.precision)
    model=harness.model
    state_before={k:v.clone() for k,v in model.named_buffers() if k in
                  ('balance_bias','routing_counter','completed_steps','training_totals')}
    result=evaluator.simple_evaluate(model=harness,tasks=task_configs(a.tasks,paths),
        num_fewshot=a.num_fewshot,random_seed=a.seed,numpy_random_seed=a.seed,
        torch_random_seed=a.seed,fewshot_random_seed=a.seed,log_samples=True,
        apply_chat_template=False,limit=a.limit,bootstrap_iters=0,task_manager=benchmark_task_manager())
    import torch
    for k,v in state_before.items():torch.testing.assert_close(dict(model.named_buffers())[k],v,atol=0,rtol=0)
    with (Path(a.checkpoint)/'model.safetensors').open('rb') as weights:
        weights_sha256=hashlib.file_digest(weights,'sha256').hexdigest()
    result['bdot_metadata']=dict(status='ok',arguments=vars(a),
        checkpoint=str(Path(a.checkpoint).resolve()),global_step=int(model.completed_steps),
        routing=model.options.to_dict(),master_dtype=str(next(model.parameters()).dtype),
        tokenizer_sha256=hashlib.sha256(harness.tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
        manifest_sha256=hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest(),
        weights_sha256=weights_sha256,
        packages={k:version(k) for k in ['torch','transformers','datasets','lm-eval']+
                  (['flash-attn-4'] if a.attention_backend=='fa4' else [])},
        likelihood_requests=harness.likelihood_requests,truncated_requests=harness.truncated_requests,
        elapsed_seconds=time.monotonic()-started)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as handle:json.dump(result,handle,indent=2,default=json_default,allow_nan=False)

if __name__=='__main__':main()
