"""Offline lm-eval likelihood adapter for the trained dense/shared-KV models.

Canonical task prompts, tokenization, truncation and scoring stay in lm-eval.
Each right-padded row is one causal sequence; EOS does not split a request.
"""
import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import time

os.environ.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  HF_HUB_DISABLE_TELEMETRY='1', WANDB_MODE='offline')


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def harness_class():
    import torch
    from lm_eval.models.huggingface import HFLM

    class KVHarness(HFLM):
        truncated_requests = 0
        likelihood_requests = 0

        def _model_call(self, inps, attn_mask=None, labels=None):
            if attn_mask is not None or labels is not None:
                raise ValueError('Only causal likelihood requests are supported')
            with torch.inference_mode():
                return self.model(input_ids=inps, use_cache=False,
                    position_ids=torch.arange(inps.shape[1], device=inps.device)
                    .expand(inps.shape[0], -1)).logits

        def _loglikelihood_tokens(self, requests, **kwargs):
            self.likelihood_requests += len(requests)
            self.truncated_requests += sum(len(c)+len(t)>self.max_length+1 for _,c,t in requests)
            return super()._loglikelihood_tokens(requests, **kwargs)

        def generate_until(self, *args, **kwargs):
            raise ValueError('This screen evaluates likelihood only')

    return KVHarness


def make_harness(checkpoint, tokenizer_path=None, device='cpu', batch_size=1,
                 max_length=1024, precision='fp32'):
    import torch
    from transformers import AutoTokenizer
    from efficient_llm.kv_mlp import load_compressed
    if precision not in ('fp32', 'bf16') or batch_size < 1 or max_length < 2:
        raise ValueError('Invalid precision/batch/context')
    model = load_compressed(checkpoint, device,
        dtype=torch.bfloat16 if precision == 'bf16' else torch.float32).eval()
    if max_length > model.config.max_position_embeddings:
        raise ValueError('Context exceeds model capacity')
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path or checkpoint), local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return harness_class()(pretrained=model, tokenizer=tokenizer, backend='causal', device=device,
        batch_size=batch_size, max_length=max_length, softmax_dtype='float32',
        mixed_precision_dtype=None)


def json_default(value):
    if hasattr(value, 'tolist'): return value.tolist()
    if hasattr(value, 'item'): return value.item()
    return str(value)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--tokenizer', required=True)
    p.add_argument('--output', required=True); p.add_argument('--device', default='cpu')
    p.add_argument('--batch-size', type=int, default=8); p.add_argument('--max-length', type=int, default=1024)
    p.add_argument('--tasks', nargs='+', required=True)
    p.add_argument('--precision', choices=['fp32','bf16'], default='bf16')
    p.add_argument('--dataset-root', required=True); p.add_argument('--manifest', required=True)
    p.add_argument('--seed', type=int, default=42); p.add_argument('--num-fewshot', type=int, default=0)
    p.add_argument('--limit', type=int)
    a = p.parse_args(); out = Path(a.output); cp = Path(a.checkpoint)
    if int(os.environ.get('WORLD_SIZE','1')) != 1: p.error('Use one process per invocation')
    if out.exists(): p.error('Output already exists')
    if a.num_fewshot < 0 or a.limit is not None and a.limit < 1: p.error('Invalid shots/limit')
    if version('lm-eval') != '0.4.13': raise ValueError('Require lm-eval==0.4.13')
    parent = json.loads((cp.parent/'result.json').read_text())
    if parent['status'] != 'passed' or parent['smoke'] or parent['steps'] != 20752 or cp.name != 'final':
        raise ValueError('Require the completed long-recovery final checkpoint')
    weights = file_hash(cp/'model.safetensors')
    from eval.benchmark_data import local_dataset_paths, task_configs, benchmark_task_manager
    paths = local_dataset_paths(a.dataset_root, a.manifest)
    if set(a.tasks)-paths.keys(): p.error('Task missing from verified local manifest')
    started = time.monotonic()
    from lm_eval import evaluator
    lm = make_harness(cp, tokenizer_path=a.tokenizer, device=a.device, batch_size=a.batch_size,
                      max_length=a.max_length, precision=a.precision)
    if (lm.model.config.kv_sources != parent['kv_sources'] or
            lm.model.config.mlp_widths != parent['mlp_widths']):
        raise ValueError('Loaded architecture differs from trained checkpoint')
    result = evaluator.simple_evaluate(model=lm, tasks=task_configs(a.tasks, paths),
        num_fewshot=a.num_fewshot, random_seed=a.seed, numpy_random_seed=a.seed,
        torch_random_seed=a.seed, fewshot_random_seed=a.seed, log_samples=True,
        apply_chat_template=False, limit=a.limit, bootstrap_iters=0, task_manager=benchmark_task_manager())
    if weights != file_hash(cp/'model.safetensors'): raise ValueError('Checkpoint changed during evaluation')
    result['kv_metadata'] = dict(status='ok', arguments={**vars(a), 'attention_backend':'sdpa'},
        checkpoint=str(cp.resolve()), global_step=parent['steps'], kv_sources=lm.model.config.kv_sources,
        weight_dtype=str(next(lm.model.parameters()).dtype),
        tokenizer_sha256=hashlib.sha256(lm.tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
        manifest_sha256=file_hash(a.manifest), weights_sha256=weights,
        packages={k:version(k) for k in ['torch','transformers','datasets','lm-eval']},
        likelihood_requests=lm.likelihood_requests, truncated_requests=lm.truncated_requests,
        elapsed_seconds=time.monotonic()-started)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as handle:
        json.dump(result, handle, indent=2, default=json_default, allow_nan=False)


if __name__ == '__main__': main()
